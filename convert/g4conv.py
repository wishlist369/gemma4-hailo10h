# g4conv: Gemma 4 E2B(HF) → Hailo HN(dict) + params(npz) 직접 생성기
# Qwen2 내부 HAR fp.hn을 "부품 틀"로 써서 역할별 params를 복사하고 모양만 Gemma에 맞춘다.
# 사용(CT142 exp venv): python g4conv.py --layers 0-4 --out p1/pilot   → pilot.hn.json, pilot_params.npz, pilot_meta.json
import argparse, json, copy, math
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/hailo/g4/model")
ap.add_argument("--qwen", default="/root/hailo/g4/ref/har/Qwen2-1_5B-Instruct.fp.hn")
ap.add_argument("--layers", default="0-4")             # 블록 범위(포함)
ap.add_argument("--ctx", type=int, default=2048)       # fp 그래프 시퀀스 = cache_size
ap.add_argument("--head", choices=["hidden", "lm", "chunk"], default="hidden")  # 출력: 마지막 토큰 hidden(최종 norm 후) / lm_head / chunk(전 토큰 잔차+공유KV)
ap.add_argument("--lm_splits", type=int, default=8)
ap.add_argument("--out", default="/root/hailo/g4/p1/pilot")
ap.add_argument("--qknorm", choices=["ln", "l2", "none"], default="ln")
ap.add_argument("--vocab", default="")
ap.add_argument("--tap", default="")
ap.add_argument("--ple", default="slice", choices=["slice", "split"])   # PLE 공급: slice=단일 입력+블록별 slice 레이어 / split=블록별 입력(ple_{i}) — 컴파일 매핑 병목 회피(10-23)
ap.add_argument("--fold", choices=["weights", "scale"], default="weights")
ap.add_argument("--lnact", action="store_true")   # fold=scale 시 gamma 채널곱 뒤에 선형 activation 레이어를 끼워 DFC의 conv 폴딩 차단 시도
ap.add_argument("--rot", choices=["none", "random", "hadamard"], default="none")
ap.add_argument("--rotgamma", choices=["fold", "scale", "perm"], default="perm")
ap.add_argument("--smooth", type=float, default=0.0)   # SmoothQuant α: LN 출력 채널/ s_j, 소비 conv 열× s_j (s = amax^α / wmax^(1-α)). 0=끔
ap.add_argument("--smooth2", type=float, default=0.0)   # R2용 SmoothQuant α: 1/s를 ln_in_w/ln_ffn_w gamma 벡터에 합쳐(폴딩 차단 유지) 소비 conv 열×s. amax_eff=|gamma|·a_pure. 0=끔
ap.add_argument("--permg", action="store_true")        # 잔차 채널 전역 치환(거대함 점수 오름차순) → 이상치 열을 같은 가중치 그룹에
ap.add_argument("--lnstats", default="/root/hailo/g4/out/lnstats.npz")
ap.add_argument("--no_kvout", action="store_true")
ap.add_argument("--mask_per_block", action="store_true")
ap.add_argument("--rope_per_block", action="store_true")   # cos/sin 입력도 블록별 분리(kv6s: 공유 cos/sin도 Context-Partition topology error, 10-24)   # 마스크 입력을 블록별로 분리(6블록+ 그룹: 공유 마스크가 넓은 컨텍스트 범위에 걸리면 DFC Context-Partition topology error, 10-24)
ap.add_argument("--kvcopy", choices=["dup", "conv", "norm"], default="dup")   # 공유 K/V 출력 복사 레이어 종류
ap.add_argument("--lsmode", choices=["norm", "conv", "none"], default="norm")   # layer_scalar 구현: norm=normalization(기본) / conv=대각 1x1 conv / none=그래프에서 생략(1블록 실험용, meta ls_out에 기록 → p1_eval이 곱함)   # 실험용: 공유 K/V 출력 생략(KV 원천 블록 13/14 정밀도 급락 원인 분리)
ap.add_argument("--ffsplit", type=int, default=6144)   # MLP intermediate가 이보다 크면(블록 15~ 12288) 이 폭으로 쪼개 부분합(0=끄기)
ap.add_argument("--split", default="")   # 채널 분할 소비: 예 "16,112" = 잔차 LN 출력의 마지막 16채널(최거대)·그 앞 112채널·나머지를 각각 conv 후 합(각 구간 독자 8비트 범위). --permg 필수   # perm: 앞 conv 출력행을 gamma 크기순으로 치환 → 회전 conv(8비트 입력, 그룹 가중치)에 gamma 폴딩   # 후단 가지: gamma를 회전 conv에 접기 / 별도 채널곱(a16) 뒤 순수 ±1/√N 회전 conv(4비트 정확)   # 오프라인 QuaRot: 잔차 스트림 직교회전(x'=Rx)으로 거대 채널 분산. 후단 norm 가지엔 R·gamma 회전 conv 삽입   # LN gamma를 conv 가중치에 접기 / 별도 채널곱(normalization) 레이어로     # 중간 텐서를 출력으로(오차 국소화): 예 block0__post_attn_w   # 평가용 압축 어휘: npz(in__ids) 목록(쉼표) → 등장 토큰만 임베딩 테이블에 (메모리 1.6GB→MB)   # QK/V-norm 구현: LN분해 / L2정규화+√hd폴딩 / 생략(구조테스트)
a = ap.parse_args()

cfg = json.load(open(f"{a.model}/config.json"))["text_config"]
D = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; HD_S = cfg["head_dim"]; HD_F = cfg["global_head_dim"]
NL = cfg["num_hidden_layers"]; PLE = cfg["hidden_size_per_layer_input"]; TYPES = cfg["layer_types"]
FIRST_SHARED = NL - cfg["num_kv_shared_layers"]
C = a.ctx
lo, hi = map(int, a.layers.split("-"))
BLOCKS = list(range(lo, hi + 1))
SC = "gemma4"
# KV 공유: 공유층은 FIRST_SHARED 직전의 같은 타입 마지막 층의 K/V를 씀
prev_types = TYPES[:FIRST_SHARED]
KV_SRC = {}
for i in range(NL):
    if i >= FIRST_SHARED:
        KV_SRC[i] = len(prev_types) - 1 - prev_types[::-1].index(TYPES[i])
    else:
        KV_SRC[i] = i
KV_EXT = sorted({KV_SRC[i] for i in BLOCKS if KV_SRC[i] not in BLOCKS})          # 이 청크가 밖에서 받아야 할 K/V 소스 블록
KV_OUT = sorted({KV_SRC[j] for j in range(NL) if KV_SRC[j] in BLOCKS and j not in BLOCKS and KV_SRC[j] != j})   # 뒤 청크에 내줄 K/V 소스 블록
CHUNK = lo > 0

# ---------- 가중치 ----------
st = safe_open(f"{a.model}/model.safetensors", "pt")
def W(k): return st.get_tensor(f"model.language_model.{k}").float().numpy()

# ---------- 잔차 회전 R (직교) ----------
ROT = None
if a.rot == "random":
    ROT, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((D, D)))
    ROT = ROT.astype(np.float32)
elif a.rot == "hadamard":        # H_1536 = H_12(Paley, q=11) ⊗ H_128(Sylvester), 원소 ±1/√1536 → 4비트 가중치로 정확히 표현
    q = 11; qr_ = {(i * i) % q for i in range(1, q)}
    chi = lambda x: 0 if x % q == 0 else (1 if x % q in qr_ else -1)
    Qm = np.array([[chi(j - i) for j in range(q)] for i in range(q)])
    H12 = np.ones((12, 12), np.int64); H12[1:, 1:] = Qm + np.eye(q, dtype=np.int64); H12[1:, 0] = -1
    assert np.array_equal(H12 @ H12.T, 12 * np.eye(12, dtype=np.int64))
    H = np.array([[1]])
    for _ in range(7):
        H = np.block([[H, H], [H, -H]])
    ROT = (np.kron(H12, H) / math.sqrt(D)).astype(np.float32)
    assert ROT.shape == (D, D) and np.allclose(ROT @ ROT.T, np.eye(D), atol=1e-4)
STATS = np.load(a.lnstats) if (a.smooth > 0 or a.smooth2 > 0 or a.permg) else None
PG = np.argsort(STATS["score"]).astype(np.int64) if a.permg else None     # 잔차 채널 치환: 새 채널 n = 옛 채널 PG[n]
def rin(w):   # 잔차를 입력으로 받는 선형층 [out, D]: 치환 → (회전)
    if PG is not None:
        w = w[:, PG]
    return w @ ROT.T if ROT is not None else w
def rout(w):  # 잔차에 더해지는 가지의 출력행 [D, in] 치환(LN은 치환 등변)
    return w[PG] if PG is not None else w
def pvec(v):  # 잔차 공간 채널 벡터(gamma 등) 치환
    return np.asarray(v)[PG] if PG is not None else np.asarray(v)

def smooth2_s(i, key, gamma, consumers_raw):
    """R2 SmoothQuant 계수 s[D]: conv 실제 입력 = gamma⊙LN → amax_eff=|gamma|·a_pure(lnstats), wmax=소비 가중치 열 max. s=amax^α/wmax^(1-α)"""
    if a.smooth2 <= 0:
        return None
    assert a.fold == "scale" and PG is None and ROT is None, "--smooth2 는 --fold scale 전용, permg/rot 불가"
    amax = np.abs(np.asarray(gamma, np.float64)) * np.asarray(STATS[key][i], np.float64) + 1e-8
    wmax = np.max(np.abs(np.concatenate(consumers_raw, 0)), axis=0) + 1e-8
    return np.clip(amax ** a.smooth2 / wmax ** (1 - a.smooth2), 1e-3, 1e4).astype(np.float32)

SPLITS = [int(x) for x in a.split.split(",") if x]
def consumer(name, ln, Wm, act="linear", role="block0__conv2"):
    """잔차 LN 출력을 먹는 선형층. --split 시 채널 구간별 conv(_p0 보통 / _p1.. 거대) + ew_add(a16), 활성은 합 뒤 별도 activation 레이어"""
    if not SPLITS:
        return conv(name, ln, Wm, act=act, role=role)
    bounds, hi = [], D
    for sz in SPLITS:
        bounds.append((hi - sz, hi)); hi -= sz
    bounds.append((0, hi)); bounds = bounds[::-1]                    # [(0,N),(N,N+112),(N+112,D)]
    Wd = LAYERS[nm(ln)]["out"][2]
    parts = []
    for k, (lo, hi_) in enumerate(bounds):
        sl = add(f"{name}_sl{k}", "slice", [ln], hi_ - lo, out_w=Wd,
                 params={"height_slice": [0, 1, 1], "width_slice": [0, Wd, 1], "features_slice": [lo, hi_, 1], "groups": 1})
        parts.append(conv(f"{name}_p{k}", sl, Wm[:, lo:hi_], role=role))
    acc = parts[0]
    for k in range(1, len(parts)):
        acc = ew(f"{name}_add{k}" if k < len(parts) - 1 or act != "linear" else name, "ew_add", acc, parts[k], "block0__ew_add4")
    if act != "linear":
        acc = add(name, "activation", [acc], Wm.shape[0], out_w=Wd, params={"activation": act})
    return acc

def smooth(name, ln, consumers, amax):
    """SmoothQuant: LN 출력 채널 j를 s_j로 나누는 normalization(1/s; DFC가 LN gamma로 융합) 삽입, 소비 가중치 열에 s_j 곱해 반환.
    consumers: [W_oi(치환·gamma 폴딩 완료)], amax: 치환된 순서의 채널별 max|LN출력|"""
    if a.smooth <= 0:
        return ln, consumers
    wmax = np.max(np.abs(np.concatenate(consumers, 0)), axis=0) + 1e-8
    amax = np.asarray(amax, np.float64) + 1e-8
    s_ = np.clip((amax ** a.smooth) / (wmax ** (1 - a.smooth)), 1e-3, 1e4).astype(np.float32)
    ln2 = scale(name + "_s", ln, (1.0 / s_))
    return ln2, [w * s_[None, :] for w in consumers]

# ---------- Qwen 부품 틀 ----------
Q = json.load(open(a.qwen))["layers"]
def proto(role):
    v = Q[f"Qwen2-1_5B-Instruct/{role}"]
    return copy.deepcopy(v.get("params", {})), copy.deepcopy(v.get("quantization_params", {})), copy.deepcopy(v.get("compilation_params", {}))

LAYERS = {}   # name -> dict(type, input, params, out_ch, out_w, extra)
P = {}
def nm(s): return f"{SC}/{s}"

def add(name, typ, inputs, out_ch, out_w=C, role=None, params=None, extra=None):
    pr, qp, cp = proto(role) if role else ({}, {}, {})
    if params:
        pr.update(params)
    LAYERS[nm(name)] = dict(type=typ, input=[nm(i) for i in inputs], params=pr, qp=qp, cp=cp,
                            out=[-1, 1, out_w, out_ch], extra=extra or {})
    return name

def conv(name, inp, w_oi, act="linear", role="block0__conv2"):
    """w_oi: torch식 [out, in] → hailo kernel (1,1,in,out)"""
    o, i = w_oi.shape
    add(name, "conv", [inp], o, out_w=LAYERS[nm(inp)]["out"][2], role=role,
        params={"kernel_shape": [1, 1, i, o], "activation": act})
    P[f"{nm(name)}/kernel:0"] = np.ascontiguousarray(w_oi.T.reshape(1, 1, i, o)).astype(np.float32)
    P[f"{nm(name)}/bias:0"] = np.zeros(o, np.float32)
    P[f"{nm(name)}/padding_const_value:0"] = np.float32(0)
    return name

def rms(name, inp, groups=1):
    ch = LAYERS[nm(inp)]["out"][3]
    add(name, "layer_normalization", [inp], ch, out_w=LAYERS[nm(inp)]["out"][2], role="block0__layer_normalization1",
        params={"groups": groups})
    P[f"{nm(name)}/epsilon:0"] = np.float32(cfg["rms_norm_eps"])
    return name

def headnorm(name, inp, groups, hd):
    """헤드별 RMSNorm. 반환 (레이어명, 뒤따르는 채널곱에 접을 스칼라 보정)"""
    if a.qknorm == "ln":
        return rms(name, inp, groups=groups), 1.0
    if a.qknorm == "l2":      # RMSNorm(x) = sqrt(hd) * x/||x||
        ch = LAYERS[nm(inp)]["out"][3]
        add(name, "l2_normalization", [inp], ch, out_w=LAYERS[nm(inp)]["out"][2],
            params={"reduce_axes": [3], "groups": groups})
        return name, math.sqrt(hd)
    return inp, 1.0            # none: 정규화 생략(구조 테스트 전용)

def scale(name, inp, vec):
    """채널별 곱 = normalization 레이어 (kernel (1,1,C,1), bias)"""
    ch = LAYERS[nm(inp)]["out"][3]
    vec = np.broadcast_to(np.asarray(vec, np.float32), (ch,))
    add(name, "normalization", [inp], ch, out_w=LAYERS[nm(inp)]["out"][2],
        params={"elementwise_add": False, "activation": "linear"})
    P[f"{nm(name)}/kernel:0"] = vec.reshape(1, 1, ch, 1).astype(np.float32).copy()
    P[f"{nm(name)}/bias:0"] = np.zeros(ch, np.float32)
    return name

def gperm(gamma):
    """perm 모드: gamma 절대값 오름차순 치환(P). 앞 conv 출력행을 P로 재배열하면 LN(Py)=P·LN(y)"""
    return np.argsort(np.abs(np.asarray(gamma))) if (ROT is not None and a.rotgamma == "perm") else None

def post_scale(name, inp, gamma, perm=None):
    """후단 norm 가지(gamma ⊙ LN(y))를 잔차에 더하기 전: 회전 시 R·diag(gamma) conv, 아니면 채널곱"""
    gamma = np.asarray(gamma, np.float32)
    if ROT is None:
        return scale(name, inp, gamma)
    if a.rotgamma == "fold":
        return conv(name, inp, ROT * gamma[None, :], role="block0__conv5")
    if a.rotgamma == "perm":       # 입력 채널 j는 치환된 채널 perm[j]: W' = R[:, perm] * gamma[perm]
        return conv(name, inp, (ROT[:, perm] * gamma[perm][None, :]).astype(np.float32), role="block0__conv5")
    g = scale(name + "_g", inp, gamma)                       # scale: gamma 채널곱(a16) → 순수 회전 conv
    return conv(name, g, ROT.copy(), role="block0__conv5")

def ew(name, typ, a_, b_, role, params=None, out_ch=None):
    ch = out_ch or max(LAYERS[nm(a_)]["out"][3], LAYERS[nm(b_)]["out"][3])
    add(name, typ, [a_, b_], ch, out_w=LAYERS[nm(a_)]["out"][2], role=role, params=params)
    return name

# ---------- 입력 ----------
def inp(name, ch, conv_type, orig):
    LAYERS[nm(name)] = dict(type="input_layer", input=[], params=None, qp={}, cp={}, out=[-1, 1, C, ch],
                            extra={"conversion_type": conv_type, "orig": orig})
    return name

emb = W("embed_tokens.weight") * math.sqrt(D)          # Gemma ScaledWordEmbedding
EMB_RAW = emb
emb = rin(emb)                                           # 치환/회전 공간 임베딩(호스트 룩업 테이블도 이것)
VOCAB_IDS = None
if CHUNK:                                                # 청크: 앞 청크의 hidden(잔차)을 직접 입력(변환 없음, a16)
    E = inp("input_layer1", D, None, "hidden_in")
else:
    E = inp("input_layer1", D, "embedding", "inputs_embeds")
    if a.vocab:
        ids_all = np.concatenate([np.load(f)["in__ids"].reshape(-1) for f in a.vocab.split(",")])
        VOCAB_IDS = np.unique(np.concatenate([[0, 1, 2], ids_all])).astype(np.int64)   # pad/eos/bos 포함
        emb = emb[VOCAB_IDS]
    P[f"{nm(E)}/embed:0"] = emb.astype(np.float32)
need_s = any(TYPES[i] == "sliding_attention" for i in BLOCKS)
need_f = any(TYPES[i] == "full_attention" for i in BLOCKS)
MASK, COS, SIN = {}, {}, {}
idx = 2
for t, hd, need in (("sliding_attention", HD_S, need_s), ("full_attention", HD_F, need_f)):
    if not need:
        continue
    if not a.mask_per_block:
        MASK[t] = inp(f"input_layer{idx}", NH * C, "mask", f"mask_{t}"); idx += 1
        P[f"{nm(MASK[t])}/tile:0"] = np.array([1, 1, NH], np.int32)
    for side, heads in (("k", 1), ("q", NH)):
        if a.rope_per_block: break
        COS[(t, side)] = inp(f"input_layer{idx}", heads * hd, "cos", f"{side}_cos_{t}"); idx += 1
        SIN[(t, side)] = inp(f"input_layer{idx}", heads * hd, "sin", f"{side}_sin_{t}"); idx += 1
if a.ple == "slice":                                     # 단일 PLE 입력 [W, NL*PLE] + 블록별 slice 레이어(DFC가 auto_spatial_reshape로 분해 → 매핑 병목 의심, 10-23)
    PLEIN = inp(f"input_layer{idx}", NL * PLE, None, "per_layer_inputs"); idx += 1
else:                                                    # split: 블록별 [W, PLE] 입력(호스트가 열 슬라이스해서 공급, slice 레이어 없음)
    PLEIN_B = {}
    for i in BLOCKS:
        PLEIN_B[i] = inp(f"input_layer{idx}", PLE, None, f"ple_{i}"); idx += 1
KR, VN = {}, {}     # 블록별 RoPE 후 K, norm 후 V
KR_OUT, VN_OUT = {}, {}   # 공유 K/V 출력 전용 복제 가지(kvcopy=dup)
for src in KV_EXT:   # 밖에서 받는 공유 K/V (RoPE 적용된 K, v_norm 적용된 V)
    hd_s = HD_S if TYPES[src] == "sliding_attention" else HD_F
    KR[src] = inp(f"input_layer{idx}", hd_s, None, f"kv_k_{src}"); idx += 1
    VN[src] = inp(f"input_layer{idx}", hd_s, None, f"kv_v_{src}"); idx += 1
    # 입력층이 matmul에 직결되면 DFC가 'skipping partial sum on transposed matmul ... has input layer' 경로로 떨어져
    # 최적화 2차 통계 수집에서 메모리 29GB 폭증(k05~k11 OOM, 2026-10-23). 항등 normalization을 끼워 일반 블록(k_rope/v_norm_w)과 같은 입구로.
    KR[src] = scale(f"kv_k_{src}_w", KR[src], 1.0)
    VN[src] = scale(f"kv_v_{src}_w", VN[src], 1.0)

# RoPE theta (HF rotary 모듈에서 그대로 → 부분회전/비례 RoPE도 자동 반영)
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
tcfg = AutoConfig.from_pretrained(a.model).get_text_config()
rot = Gemma4TextRotaryEmbedding(tcfg)
THETA = {}
for t in ("sliding_attention", "full_attention"):
    inv = getattr(rot, f"{t}_inv_freq").float().numpy()
    THETA[t] = np.concatenate([inv, inv]).astype(np.float32)        # [hd] (rotate-half 레이아웃)
    assert getattr(rot, f"{t}_attention_scaling") == 1.0
for (t, side), n in COS.items():
    heads = 1 if side == "k" else NH
    P[f"{nm(n)}/theta:0"] = THETA[t].copy(); P[f"{nm(n)}/factor:0"] = np.ones(1, np.float32)
    P[f"{nm(n)}/tile:0"] = np.array([1, 1, heads], np.int32)
for (t, side), n in SIN.items():
    heads = 1 if side == "k" else NH
    th = THETA[t].copy(); th[: len(th) // 2] *= -1                  # rotate_half 부호 폴딩
    P[f"{nm(n)}/theta:0"] = th; P[f"{nm(n)}/factor:0"] = np.ones(1, np.float32)
    P[f"{nm(n)}/tile:0"] = np.array([1, 1, heads], np.int32)

# ---------- 블록 ----------
res = E
for i in BLOCKS:
    t = TYPES[i]; hd = HD_S if t == "sliding_attention" else HD_F
    p = f"layers.{i}."
    b = f"block{i}__"
    if a.mask_per_block:                                  # 블록별 마스크 입력(호스트는 같은 타입 마스크를 복제 공급)
        MB = inp(f"input_layer{idx}", NH * C, "mask", f"mask_{t}_b{i}"); idx += 1
        P[f"{nm(MB)}/tile:0"] = np.array([1, 1, NH], np.int32)
    else:
        MB = MASK[t]
    if a.rope_per_block:                                  # 블록별 cos/sin 입력(k쪽은 자기 K/V를 만드는 블록만)
        RC, RS = {}, {}
        for side, heads in (("k", 1), ("q", NH)):
            if side == "k" and KV_SRC[i] != i: continue
            RC[side] = inp(f"input_layer{idx}", heads * hd, "cos", f"{side}_cos_{t}_b{i}"); idx += 1
            RS[side] = inp(f"input_layer{idx}", heads * hd, "sin", f"{side}_sin_{t}_b{i}"); idx += 1
            P[f"{nm(RC[side])}/theta:0"] = THETA[t].copy(); P[f"{nm(RC[side])}/factor:0"] = np.ones(1, np.float32); P[f"{nm(RC[side])}/tile:0"] = np.array([1, 1, heads], np.int32)
            th_ = THETA[t].copy(); th_[: len(th_) // 2] *= -1
            P[f"{nm(RS[side])}/theta:0"] = th_; P[f"{nm(RS[side])}/factor:0"] = np.ones(1, np.float32); P[f"{nm(RS[side])}/tile:0"] = np.array([1, 1, heads], np.int32)
        CQ, SQ = RC["q"], RS["q"]; CK, SK = RC.get("k"), RS.get("k")
    else:
        CQ, SQ, CK, SK = COS[(t, "q")], SIN[(t, "q")], COS[(t, "k")], SIN[(t, "k")]
    ln1 = rms(b + "ln_in", res)
    w_in = W(p + "input_layernorm.weight")
    s_in = smooth2_s(i, "a_in", w_in, [W(p + "self_attn.q_proj.weight"), W(p + "self_attn.k_proj.weight"), W(p + "self_attn.v_proj.weight")])
    if a.fold == "scale":
        ln1 = scale(b + "ln_in_w", ln1, w_in if s_in is None else w_in / s_in); w_in = np.ones_like(w_in) if s_in is None else s_in   # smooth2: gamma/s 를 norm에, s 를 소비 가중치 열에
        if a.lnact:
            ln1 = add(b + "ln_in_act", "activation", [ln1], D, out_w=C, params={"activation": "linear"})
    Wq = rin(W(p + "self_attn.q_proj.weight") * w_in[None, :])
    Wk = rin(W(p + "self_attn.k_proj.weight") * w_in[None, :]); Wv = rin(W(p + "self_attn.v_proj.weight") * w_in[None, :])
    ln1, (Wq, Wk, Wv) = smooth(b + "ln_in", ln1, [Wq, Wk, Wv], pvec(STATS["a_in"][i]) if STATS is not None else None)
    q = consumer(b + "q_proj", ln1, Wq)
    qn, qg = headnorm(b + "q_norm", q, NH, hd)
    qs = scale(b + "q_norm_w", qn, np.tile(W(p + "self_attn.q_norm.weight"), NH) * qg)
    qsh = add(b + "q_rot", "feature_shuffle", [qs], NH * hd, role="block0__feature_shuffle2",
              params={"groups": NH, "groups_slice": [hd // 2, hd, 1]})
    q1 = ew(b + "q_cos", "ew_mult", qs, CQ, "block0__ew_mult2")
    q2 = ew(b + "q_sin", "ew_mult", qsh, SQ, "block0__ew_mult4")
    qr = ew(b + "q_rope", "ew_add", q1, q2, "block0__ew_add2")
    if KV_SRC[i] == i:
        k = consumer(b + "k_proj", ln1, Wk)
        v = consumer(b + "v_proj", ln1, Wv)
        kn, kg = headnorm(b + "k_norm", k, 1, hd)
        ks = scale(b + "k_norm_w", kn, W(p + "self_attn.k_norm.weight") * kg)
        ksh = add(b + "k_rot", "feature_shuffle", [ks], hd, role="block0__feature_shuffle1",
                  params={"groups": 1, "groups_slice": [hd // 2, hd, 1]})
        k1 = ew(b + "k_cos", "ew_mult", ks, CK, "block0__ew_mult1")
        k2 = ew(b + "k_sin", "ew_mult", ksh, SK, "block0__ew_mult3")
        KR[i] = ew(b + "k_rope", "ew_add", k1, k2, "block0__ew_add1")
        vn_, vg = headnorm(b + "v_norm", v, 1, hd)
        VN[i] = scale(b + "v_norm_w", vn_, vg)   # 항등(또는 √hd): 16→8 변환이 matmul 가중치 입구에 걸리지 않게
        if i in KV_OUT and not a.no_kvout and a.kvcopy == "dup":   # 공유 K/V 출력 전용 가지: k_rope/v_norm_w(matmul 입력)에 fan-out을 만들면 DFC가 signed_output+shortcut을 넣어 블록이 붕괴(0.627) → LN 출력에서 K/V 계산을 복제
            ke = consumer(b + "k_proj_out", ln1, Wk); ve = consumer(b + "v_proj_out", ln1, Wv)
            kne, kge = headnorm(b + "k_norm_out", ke, 1, hd)
            kse = scale(b + "k_norm_out_w", kne, W(p + "self_attn.k_norm.weight") * kge)
            kshe = add(b + "k_rot_out", "feature_shuffle", [kse], hd, role="block0__feature_shuffle1", params={"groups": 1, "groups_slice": [hd // 2, hd, 1]})
            k1e = ew(b + "k_cos_out", "ew_mult", kse, CK, "block0__ew_mult1")
            k2e = ew(b + "k_sin_out", "ew_mult", kshe, SK, "block0__ew_mult3")
            KR_OUT[i] = ew(b + "k_rope_out", "ew_add", k1e, k2e, "block0__ew_add1")
            vne, vge = headnorm(b + "v_norm_out", ve, 1, hd)
            VN_OUT[i] = scale(b + "v_norm_out_w", vne, vge)
    kr, vn = KR[KV_SRC[i]], VN[KV_SRC[i]]
    # QK^T (scaling=1.0 — QK-norm이 대신함)
    s = add(b + "matmul1", "matmul", [qr, kr], NH * C, role="block0__matmul1",
            params={"kernel_shape": [1, 1, NH * hd, NH * C], "groups": NH, "input_tiles": [[1, 1, 1], [1, 1, NH]]})
    # softmax 분해 체인 — Qwen fp.hn 이름/original_names 규약 그대로(최적화기가 체인을 이름으로 묶음)
    so = {"orig": f"/model/layers.{i}/self_attn/Softmax"}
    m1 = add(b + "ew_mult2_softmax1", "ew_mult", [s, MB], NH * C, role="block0__ew_mult2_softmax1", extra=so)
    mx = add(b + "reduce_max_softmax1", "reduce_max", [m1], NH, role="block0__reduce_max_softmax1", params={"groups": NH}, extra=so)
    ex = add(b + "ew_sub_softmax1", "ew_sub", [m1, mx], NH * C, role="block0__ew_sub_softmax1",
             params={"input_repeats": [[1, 1, 1], [1, 1, C]]}, extra=so)
    m3 = add(b + "ew_mult3_softmax1", "ew_mult", [ex, MB], NH * C, role="block0__ew_mult3_softmax1", extra=so)
    sm = add(b + "reduce_sum_softmax1", "reduce_sum", [m3], NH, role="block0__reduce_sum_softmax1", params={"groups": NH}, extra=so)
    pr = add(b + "ew_mult_softmax1", "ew_mult", [sm, m3], NH * C, role="block0__ew_mult_softmax1",
             params={"input_repeats": [[1, 1, C], [1, 1, 1]]}, extra=so)
    av = add(b + "matmul2", "matmul", [pr, vn], NH * hd, role="block0__matmul2",
             params={"kernel_shape": [1, 1, NH * C, NH * hd], "groups": NH, "input_tiles": [[1, 1, 1], [1, 1, NH]]})
    g_at = pvec(W(p + "post_attention_layernorm.weight")); P_at = gperm(g_at)
    Wo = rout(W(p + "self_attn.o_proj.weight")); Wo = Wo[P_at] if P_at is not None else Wo
    o = conv(b + "o_proj", av, Wo)
    on = rms(b + "post_attn_norm", o)
    os_ = post_scale(b + "post_attn_w", on, g_at, P_at)
    h1 = ew(b + "res_attn", "ew_add", res, os_, "block0__ew_add4")
    # MLP (GELU-tanh)
    ln2 = rms(b + "ln_ffn", h1)
    w_pre = W(p + "pre_feedforward_layernorm.weight")
    s_ff = smooth2_s(i, "a_ffn", w_pre, [W(p + "mlp.gate_proj.weight"), W(p + "mlp.up_proj.weight")])
    if a.fold == "scale":
        ln2 = scale(b + "ln_ffn_w", ln2, w_pre if s_ff is None else w_pre / s_ff); w_pre = np.ones_like(w_pre) if s_ff is None else s_ff
        if a.lnact:
            ln2 = add(b + "ln_ffn_act", "activation", [ln2], D, out_w=C, params={"activation": "linear"})
    Wg = rin(W(p + "mlp.gate_proj.weight") * w_pre[None, :]); Wu = rin(W(p + "mlp.up_proj.weight") * w_pre[None, :])
    ln2, (Wg, Wu) = smooth(b + "ln_ffn", ln2, [Wg, Wu], pvec(STATS["a_ffn"][i]) if STATS is not None else None)
    g_ff = pvec(W(p + "post_feedforward_layernorm.weight")); P_ff = gperm(g_ff)
    Wd = rout(W(p + "mlp.down_proj.weight")); Wd = Wd[P_ff] if P_ff is not None else Wd
    FF = Wg.shape[0]; NP = max(1, -(-FF // a.ffsplit)) if a.ffsplit > 0 else 1
    if NP == 1:
        g = consumer(b + "gate_proj", ln2, Wg, act="gelu", role="block0__conv6")
        u = consumer(b + "up_proj", ln2, Wu, role="block0__conv7")
        gu = ew(b + "gate_mul", "ew_mult", g, u, "block0__ew_mult5")
        dn = conv(b + "down_proj", gu, Wd, role="block0__conv8")
    else:   # MLP 12288(블록 15~): ew_mult(gate*up)→down_proj가 DFC에서 *_mult_reduce로 융합되며 커널 검증 실패(APU/subcluster 제약) → ffsplit 폭 조각(6144 블록과 동형)의 부분합
        parts = []
        for j in range(NP):
            lo, hi = j * FF // NP, (j + 1) * FF // NP; sb = f"{b}s{j}__"     # 이름은 __gate_proj 등 접미사 유지(p1_quant sel 접미사 매칭)
            gj = consumer(sb + "gate_proj", ln2, Wg[lo:hi], act="gelu", role="block0__conv6")
            uj = consumer(sb + "up_proj", ln2, Wu[lo:hi], role="block0__conv7")
            guj = ew(sb + "gate_mul", "ew_mult", gj, uj, "block0__ew_mult5")
            parts.append(conv(sb + "down_proj", guj, Wd[:, lo:hi], role="block0__conv8"))
        dn = parts[0]
        for j in range(1, NP):
            dn = ew(f"{b}down_sum{j}", "ew_add", dn, parts[j], "block0__ew_add5")
    fn = rms(b + "post_ffn_norm", dn)
    fs = post_scale(b + "post_ffn_w", fn, g_ff, P_ff)
    h2 = ew(b + "res_ffn", "ew_add", h1, fs, "block0__ew_add5")
    # PLE
    pg = conv(b + "ple_gate", h2, rin(W(p + "per_layer_input_gate.weight")), act="gelu", role="block0__conv6")
    if a.ple == "slice":
        psl = add(b + "ple_slice", "slice", [PLEIN], PLE, role=None,
                  params={"height_slice": [0, 1, 1], "width_slice": [0, C, 1], "features_slice": [i * PLE, (i + 1) * PLE, 1], "groups": 1})
    else:
        psl = PLEIN_B[i]
    pm = ew(b + "ple_mul", "ew_mult", pg, psl, "block0__ew_mult5")
    g_pl = pvec(W(p + "post_per_layer_input_norm.weight")); P_pl = gperm(g_pl)
    Wp = rout(W(p + "per_layer_projection.weight")); Wp = Wp[P_pl] if P_pl is not None else Wp
    pp = conv(b + "ple_proj", pm, Wp)
    pn = rms(b + "ple_norm", pp)
    ps = post_scale(b + "ple_w", pn, g_pl, P_pl)
    h3 = ew(b + "res_ple", "ew_add", h2, ps, "block0__ew_add5")
    ls_val = float(W(p + "layer_scalar")[0])
    if a.lsmode == "norm": res = scale(b + "layer_scalar", h3, ls_val)
    elif a.lsmode == "conv": res = conv(b + "layer_scalar", h3, (ls_val * np.eye(D)).astype(np.float32), role="block0__conv5")
    else: res = h3; LS_OUT = ls_val

# ---------- 출력 ----------
OUTS = []
KVOUTS = []
try: LS_OUT
except NameError: LS_OUT = 1.0
if a.head == "chunk":                                    # 청크 출력: 전 토큰 hidden(잔차 그대로, a16) + 뒤 청크용 K/V
    OUTS = [res]
    for src in ([] if a.no_kvout else KV_OUT):   # K/V를 본 경로 텐서 그대로 출력하면 그 블록 정밀도가 붕괴(b13 0.627/b14 0.695 → 출력 제거 시 0.952/0.992) → 항등 복사 레이어(a16)로 분리해 출력
        if a.kvcopy == "dup":
            KVOUTS += [KR_OUT[src], VN_OUT[src]]
        elif a.kvcopy == "conv":   # 항등 1x1 conv: normalization 복사는 출력의 signed 요구가 k_rope까지 전파(signed_output True → matmul K 경로 붕괴, b13kc3 0.627)
            KVOUTS += [conv(f"kv_k_out_{src}", KR[src], np.eye(LAYERS[nm(KR[src])]["out"][3], dtype=np.float32), role="block0__conv5"),
                       conv(f"kv_v_out_{src}", VN[src], np.eye(LAYERS[nm(VN[src])]["out"][3], dtype=np.float32), role="block0__conv5")]
        else:
            KVOUTS += [scale(f"kv_k_out_{src}", KR[src], 1.0), scale(f"kv_v_out_{src}", VN[src], 1.0)]
    OUTS += KVOUTS
last = add("last_token", "slice", [res], D, out_w=1, role="moved_slice1",
           params={"width_slice": [C - 1, C, 1]}) if a.head != "chunk" else None
fnrm = rms("final_norm", last) if last else None
if a.head == "chunk":
    pass
elif a.head == "hidden":
    gf = pvec(W("norm.weight"))
    fw = conv("final_norm_w", fnrm, np.diag(gf) @ ROT.T, role="block0__conv5") if ROT is not None else scale("final_norm_w", fnrm, gf)   # 회전 복원 포함
    OUTS = [fw]
else:
    wl = rin(EMB_RAW / math.sqrt(D) * W("norm.weight")[None, :])      # tied lm_head, final norm 폴딩 (+회전)
    V = wl.shape[0]; sp = V // a.lm_splits
    for d in range(a.lm_splits):
        OUTS.append(conv(f"lm_head_d{d}", fnrm, wl[d * sp:(d + 1) * sp], role="conv225_d0"))

if a.tap:                                   # 탭: 해당 레이어를 유일 출력으로, 뒤쪽 레이어는 제거
    assert nm(a.tap) in LAYERS, f"tap {a.tap} 없음"
    keep, stack = set(), [nm(a.tap)]
    while stack:                            # 탭의 조상만 유지
        n_ = stack.pop()
        if n_ in keep: continue
        keep.add(n_); stack.extend(LAYERS[n_]["input"])
    for n_ in [x for x in LAYERS if x not in keep]:
        del LAYERS[n_]
    OUTS = [a.tap]

# ---------- HN 조립 (output 리스트/shape 자동 계산) ----------
for o_ in OUTS:
    LAYERS[nm(f"output_{o_}")] = dict(type="output_layer", input=[nm(o_)], params=None, qp={}, cp={},
                                      out=LAYERS[nm(o_)]["out"], extra={})
succ = {n: [] for n in LAYERS}
for n, v in LAYERS.items():
    for i_ in v["input"]:
        succ[i_].append(n)
for n in [n for n, v in LAYERS.items() if v["type"] == "input_layer" and not succ[n]]:   # 소비자 없는 입력 제거(공유 KV 청크의 k_cos/k_sin 등; HN 스키마 output minItems 1)
    del LAYERS[n]; del succ[n]
    for k in [k for k in P if k.startswith(n + "/")]:
        del P[k]
hl = {}
for n, v in LAYERS.items():
    d = {"type": v["type"], "input": v["input"], "output": succ[n],
         "input_shapes": [LAYERS[i_]["out"] for i_ in v["input"]] if v["input"] else [v["out"]],
         "output_shapes": [v["out"] for _ in succ[n]] or [v["out"]],
         "original_names": [v["extra"].get("orig", n.split("/")[-1])],
         "compilation_params": v["cp"], "quantization_params": v["qp"]}
    if v["type"] == "input_layer":
        d.update({"transposed": False, "engine": "nn_core", "io_type": "standard"})
        if v["extra"].get("conversion_type"):
            d.update({"conversion_type": v["extra"]["conversion_type"], "emulate_conversion": True})
    elif v["type"] == "output_layer":
        d.update({"transposed": False, "engine": "nn_core", "io_type": "standard"})
    else:
        d["params"] = v["params"]
    hl[n] = d
hn = {"name": SC,
      "net_params": {"version": "1.0", "stage": "HN", "clusters_placement": [[]], "clusters_to_skip": [],
                     "output_layers_order": [nm(o) for o in OUTS], "is_transformer": False, "transposed_net": False,
                     "prefill_size": min(96, C // 2), "cache_size": C, "net_scopes": [SC], "lora_adapters": [SC]},
      "layers": hl}
json.dump(hn, open(f"{a.out}.hn.json", "w"))
np.savez(f"{a.out}_params.npz", **P)
if ROT is not None:
    np.save(f"{a.out}_rot.npy", ROT)
if PG is not None:
    np.save(f"{a.out}_perm.npy", PG)
meta = {"blocks": BLOCKS, "ctx": C, "rot": ROT is not None, "split": SPLITS, "chunk": CHUNK, "kv_ext": KV_EXT, "kv_out": ([] if a.no_kvout else KV_OUT), "ls_out": LS_OUT, "head": a.head, "perm": (PG.tolist() if PG is not None else None),
        "out_perm": (PG is not None and ROT is None and (not a.tap or a.tap.endswith(("res_attn", "res_ffn", "res_ple", "layer_scalar", "post_attn_w", "post_ffn_w", "ple_w")))), "vocab_ids": (VOCAB_IDS.tolist() if VOCAB_IDS is not None else None), "ple": PLE, "ple_mode": a.ple, "inputs": {v["extra"]["orig"]: n for n, v in LAYERS.items() if v["type"] == "input_layer"},
        "outputs": [nm(o) for o in OUTS], "theta": {k: v.tolist() for k, v in THETA.items()}}
json.dump(meta, open(f"{a.out}_meta.json", "w"), indent=1)
print(f"layers {len(hl)}  params {len(P)}  ({sum(x.size for x in P.values())/1e6:.0f}M floats)  inputs {list(meta['inputs'])}")
