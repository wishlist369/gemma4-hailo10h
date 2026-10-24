# g4host: Gemma4-on-Hailo 호스트 전처리 (보정/검증/런타임 공용)
#   토큰 → 임베딩(×√D), PLE(토큰 함수, 8960), 마스크(0/1, 슬라이딩/풀), RoPE cos/sin(부호 폴딩 sin)
#   텐서 레이아웃: NHWC [N,1,W,ch], W = 컨텍스트 폭(왼쪽 패딩)
import math
import numpy as np
import torch
from safetensors import safe_open


class G4Host:
    def __init__(self, model_dir, theta, cfg):
        self.cfg = cfg
        self.D = cfg["hidden_size"]; self.NH = cfg["num_attention_heads"]; self.NL = cfg["num_hidden_layers"]
        self.PLE = cfg["hidden_size_per_layer_input"]; self.win = cfg["sliding_window"]; self.eps = cfg["rms_norm_eps"]
        self.theta = {k: np.asarray(v, np.float32) for k, v in theta.items()}
        st = safe_open(f"{model_dir}/model.safetensors", "pt")
        g = lambda k: st.get_tensor(f"model.language_model.{k}")
        self.emb_w = g("embed_tokens.weight")                  # bf16 [V,D] (행 단위로 꺼내 씀)
        self.ple_w = g("embed_tokens_per_layer.weight")        # bf16 [V,NL*PLE]
        self.proj_w = g("per_layer_model_projection.weight").float()   # [NL*PLE, D]
        self.proj_norm_w = g("per_layer_projection_norm.weight").float()

    def embed(self, ids):
        ids = torch.as_tensor(ids)
        return self.emb_w[ids].float() * math.sqrt(self.D)    # [T,D]

    def ple(self, ids, emb):
        ids = torch.as_tensor(ids); T = len(ids)
        tok = (self.ple_w[ids].float() * math.sqrt(self.PLE)).reshape(T, self.NL, self.PLE)
        pr = (emb @ self.proj_w.T * self.D ** -0.5).reshape(T, self.NL, self.PLE)
        pr = pr * torch.pow(pr.pow(2).mean(-1, keepdim=True) + self.eps, -0.5) * self.proj_norm_w
        return ((pr + tok) * 2 ** -0.5).reshape(T, -1)          # [T, NL*PLE]

    def full_inputs(self, ids, W, types=("sliding_attention", "full_attention")):
        """FP 그래프(폭 W=cache_size)용 전체 입력. ids 길이 T<=W, 왼쪽 패딩."""
        T = len(ids); pad = W - T
        emb = self.embed(ids); ple = self.ple(ids, emb)
        X = {}
        ids_p = np.zeros((1, 1, W), np.int64); ids_p[0, 0, pad:] = np.asarray(ids); X["ids"] = ids_p     # 에뮬레이션용 토큰 id(왼쪽 패딩 0)
        X["npos"] = np.array([T], np.int64)                                                           # 에뮬레이션용 유효 토큰 수
        e = np.zeros((1, 1, W, self.D), np.float32); e[0, 0, pad:] = emb.numpy(); X["inputs_embeds"] = e
        p = np.zeros((1, 1, W, ple.shape[1]), np.float32); p[0, 0, pad:] = ple.numpy(); X["per_layer_inputs"] = p
        posw = np.arange(W) - pad; valid = posw >= 0
        r = np.arange(W)[:, None]; c = np.arange(W)[None, :]
        for t in types:
            m = (c <= r) & valid[None, :]
            if t == "sliding_attention":
                m &= (r - c) < self.win
            m |= np.eye(W, dtype=bool)
            X[f"mask_{t}"] = np.tile(m.astype(np.float32), (1, self.NH))[None, None]
            X[f"maskadd_{t}"] = np.where(m, 0.0, -np.inf).astype(np.float32)[None, None]      # 에뮬레이션용 4D 0/-inf 마스크 [1,1,W(쿼리),W(키)]
            cs, sn = self.rope(np.maximum(posw, 0), t)
            for side, heads in (("k", 1), ("q", self.NH)):
                X[f"{side}_cos_{t}"] = np.tile(cs, (1, heads))[None, None]
                X[f"{side}_sin_{t}"] = np.tile(sn, (1, heads))[None, None]
        return X

    def rope(self, pos, t):
        th = self.theta[t]
        ang = np.asarray(pos, np.float64)[:, None] * th[None, :]
        cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        sn[:, : len(th) // 2] *= -1                               # rotate_half 부호 폴딩
        return cs, sn
