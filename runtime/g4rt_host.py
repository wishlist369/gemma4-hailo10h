# g4rt_host: Gemma4-on-Hailo 런타임 호스트 측(numpy만, torch 없음) — 디바이스 호스트용
#   토크나이저(tokenizers) · 임베딩(bf16 테이블 룩업 ×√D) · PLE(토큰 PLE + 투영 RMSNorm) · 마스크/RoPE 입력 · tied lm_head(+softcap) · 샘플링
#   g4host.G4Host(torch)와 수치 동일해야 함(test_rt_host.py로 대조). 텐서 레이아웃 NHWC [1,1,W,ch], 왼쪽 패딩
import json, math, os
import numpy as np


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


class G4RtHost:
    TEMPLATE = "<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n"      # HF apply_chat_template(add_generation_prompt=True) 와 동일 문자열

    def __init__(self, hostdir, theta, cfg=None):
        self.dir = hostdir
        cfg = cfg or json.load(open(f"{hostdir}/config.json"))
        cfg = cfg.get("text_config", cfg); self.cfg = cfg
        self.D = cfg["hidden_size"]; self.NH = cfg["num_attention_heads"]; self.NL = cfg["num_hidden_layers"]
        self.PLE = cfg["hidden_size_per_layer_input"]; self.win = cfg["sliding_window"]; self.eps = cfg["rms_norm_eps"]
        self.softcap = cfg.get("final_logit_softcapping")
        self.eos = set(json.load(open(f"{hostdir}/generation_config.json")).get("eos_token_id", [1]))
        self.theta = {k: np.asarray(v, np.float32) for k, v in theta.items()}
        self.emb_u16 = np.load(f"{hostdir}/embed_bf16.npy", mmap_mode="r")      # [V,D] bf16 bits
        self.ple_u16 = np.load(f"{hostdir}/ple_bf16.npy", mmap_mode="r")        # [V,NL*PLE] bf16 bits
        self.proj = np.load(f"{hostdir}/ple_proj.npy")                          # [NL*PLE, D] fp32
        self.proj_norm = np.load(f"{hostdir}/ple_proj_norm.npy")                # [PLE]
        self.final_norm = np.load(f"{hostdir}/final_norm.npy")                  # [D]
        self._lm = None
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(f"{hostdir}/tokenizer.json")

    # ---------- 토크나이저 ----------
    def encode(self, prompt):
        return self.tok.encode(self.TEMPLATE.format(prompt=prompt), add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True)

    # ---------- 임베딩/PLE ----------
    def embed(self, ids):
        return bf16_to_f32(self.emb_u16[np.asarray(ids)]) * math.sqrt(self.D)          # [T,D]

    def ple(self, ids, emb):
        T = len(ids)
        tok = (bf16_to_f32(self.ple_u16[np.asarray(ids)]) * math.sqrt(self.PLE)).reshape(T, self.NL, self.PLE)
        pr = (emb @ self.proj.T * self.D ** -0.5).reshape(T, self.NL, self.PLE)
        pr = pr * (np.mean(pr * pr, -1, keepdims=True) + self.eps) ** -0.5 * self.proj_norm
        return ((pr + tok) * 2 ** -0.5).reshape(T, -1).astype(np.float32)             # [T, NL*PLE]

    # ---------- 그래프 입력(폭 W, 왼쪽 패딩) ----------
    def rope(self, pos, t):
        th = self.theta[t]
        ang = np.asarray(pos, np.float64)[:, None] * th[None, :]
        cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        sn[:, : len(th) // 2] *= -1                                                      # rotate_half 부호 폴딩
        return cs, sn

    def inputs(self, ids, W, types=("sliding_attention", "full_attention"), with_embeds=True):
        T = len(ids); pad = W - T; assert pad >= 0, f"ctx {W} < tokens {T}"
        X = {"npos": np.array([T], np.int64)}
        if with_embeds:
            emb = self.embed(ids); ple = self.ple(ids, emb)
            e = np.zeros((1, 1, W, self.D), np.float32); e[0, 0, pad:] = emb; X["inputs_embeds"] = e
            p = np.zeros((1, 1, W, ple.shape[1]), np.float32); p[0, 0, pad:] = ple; X["per_layer_inputs"] = p
        posw = np.arange(W) - pad; valid = posw >= 0
        r = np.arange(W)[:, None]; c = np.arange(W)[None, :]
        for t in types:
            m = (c <= r) & valid[None, :]
            if t == "sliding_attention":
                m &= (r - c) < self.win
            m |= np.eye(W, dtype=bool)
            X[f"mask_{t}"] = np.tile(m.astype(np.float32), (1, self.NH))[None, None]
            cs, sn = self.rope(np.maximum(posw, 0), t)
            for side, heads in (("k", 1), ("q", self.NH)):
                X[f"{side}_cos_{t}"] = np.tile(cs, (1, heads))[None, None]
                X[f"{side}_sin_{t}"] = np.tile(sn, (1, heads))[None, None]
        return X

    # ---------- lm_head(tied) + 샘플링 ----------
    def lm_table(self):
        if self._lm is None:
            self._lm = bf16_to_f32(np.asarray(self.emb_u16))                            # [V,D] fp32 (1.6GB)
        return self._lm

    def logits(self, h_normed):
        """최종 norm까지 끝난 hidden [D] → 로짓 [V] (softcap 포함)"""
        z = self.lm_table() @ np.asarray(h_normed, np.float32).reshape(-1)
        if self.softcap:
            z = np.tanh(z / self.softcap) * self.softcap
        return z

    def final_norm_apply(self, h):
        """청크가 잔차(norm 전)를 내준 경우 호스트에서 최종 RMSNorm"""
        h = np.asarray(h, np.float32).reshape(-1)
        return h * (np.mean(h * h) + self.eps) ** -0.5 * self.final_norm

    @staticmethod
    def sample(z, temperature=0.0, top_k=64, top_p=0.95, rng=None):
        if temperature <= 0:
            return int(np.argmax(z))
        rng = rng or np.random.default_rng()
        z = z / temperature
        if top_k:
            idx = np.argpartition(-z, top_k)[:top_k]; zz = z[idx]
        else:
            idx = np.arange(z.size); zz = z
        p = np.exp(zz - zz.max()); p /= p.sum()
        if top_p < 1.0:
            o = np.argsort(-p); cum = np.cumsum(p[o]); keep = o[: max(1, int(np.searchsorted(cum, top_p)) + 1)]
            mask = np.zeros_like(p); mask[keep] = p[keep]; p = mask / mask.sum()
        return int(idx[rng.choice(p.size, p=p)])
