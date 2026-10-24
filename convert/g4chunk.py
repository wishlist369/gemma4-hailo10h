# g4chunk: 청크(블록 lo..hi) 평가/보정 입력·정답 생성 (HF, exp venv)
#   입력: hidden_in(블록 lo 직전 잔차), per_layer_inputs, mask/cos/sin(에뮬레이션용 npos), 공유 K/V(kv_k_{src}, kv_v_{src})
#   정답: want = 청크 출력(마지막 토큰 잔차, head=chunk) 또는 최종 norm hidden(head=hidden)
# 사용: python g4chunk.py --meta chunk/c1_meta.json --out chunk/c1_ref.npz [--calib N --out_calib chunk/c1_calib.npz]
import argparse, json, math, random, glob
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from g4host import G4Host

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/hailo/g4/model")
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)                       # ref npz
ap.add_argument("--calib", type=int, default=0)               # >0 이면 보정셋도 생성
ap.add_argument("--out_calib", default="")
ap.add_argument("--texts", default="/root/hailo/g4/calib_texts")
ap.add_argument("--prompt", default="대한민국의 수도는 어디야? 한 문장으로 답하고, 그 도시의 유명한 음식도 하나 알려줘.")
ap.add_argument("--calib_full", action="store_true")   # 보정셋을 길이 C 꽉 찬 랜덤 윈도우만으로(fp_answers·짧은 창 제외 → 좌측 제로패딩 0)
a = ap.parse_args()
meta = json.load(open(a.meta)); C = meta["ctx"]; BLOCKS = meta["blocks"]; lo, hi = BLOCKS[0], BLOCKS[-1]
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
lm = model.model.language_model; cfg = lm.config
for i in range(0, hi + 1):
    lm.layers[i].float()
lm.norm.float(); lm.per_layer_model_projection.float(); lm.per_layer_projection_norm.float()
cfgd = json.load(open(f"{a.model}/config.json"))["text_config"]
H = G4Host(a.model, meta["theta"], cfgd)
types = [t for t in ("sliding_attention", "full_attention") if any(k.startswith(f"mask_{t}") for k in meta["inputs"])]   # mask_{t} 또는 블록별 mask_{t}_b{i}
NEED_KV = sorted({int(k.split("_")[-1]) for k in meta["inputs"] if k.startswith("kv_k_")})

def run(ids_list):
    """HF로 블록 0..hi 실행. 반환: hidden_in(lo 직전), 청크 출력 잔차(hi 후), 최종 norm hidden, 공유 K/V dict"""
    ids = torch.tensor(ids_list)[None]; T = ids.shape[1]
    with torch.no_grad():
        emb = lm.embed_tokens.weight[ids].float() * math.sqrt(cfg.hidden_size)
        tok_ple = (lm.embed_tokens_per_layer.weight[ids].float() * math.sqrt(cfg.hidden_size_per_layer_input)
                   ).reshape(1, T, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)
        ple = lm.project_per_layer_inputs(emb, tok_ple)
        pos = torch.arange(T)[None]
        kw = dict(config=cfg, inputs_embeds=emb, attention_mask=torch.ones_like(ids), past_key_values=None, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        pe = {t: lm.rotary_emb(emb, pos, t) for t in set(cfg.layer_types)}
        h = emb; shared = {}; kv = {}; hidden_in = None
        for i in range(0, hi + 1):
            if i == lo:
                hidden_in = h.clone()
            L = lm.layers[i]
            h = L(h, ple[:, :, i, :], shared_kv_states=shared, position_embeddings=pe[cfg.layer_types[i]],
                  attention_mask=masks[cfg.layer_types[i]], position_ids=pos, past_key_values=None)
            h = h[0] if isinstance(h, tuple) else h
            if i in NEED_KV:                                   # 이 블록이 store_full_length_kv → shared[type] 갱신됨
                k_, v_ = shared[cfg.layer_types[i]]            # [1, kv_heads=1, T, hd]
                kv[i] = (k_[0, 0].float(), v_[0, 0].float())   # [T, hd]
        out = h; final = lm.norm(h)
    return hidden_in, out, final, kv, ple

def pack(ids_list):
    hidden_in, out, final, kv, ple = run(ids_list)
    T = len(ids_list); pad = C - T
    X = H.full_inputs(ids_list, C, types)                    # ids/npos/mask/cos/sin/ple/inputs_embeds
    hin = np.zeros((1, 1, C, cfg.hidden_size), np.float32); hin[0, 0, pad:] = hidden_in[0].numpy(); X["hidden_in"] = hin
    for src, (k_, v_) in kv.items():
        kk = np.zeros((1, 1, C, k_.shape[-1]), np.float32); kk[0, 0, pad:] = k_.numpy(); X[f"kv_k_{src}"] = kk
        vv = np.zeros((1, 1, C, v_.shape[-1]), np.float32); vv[0, 0, pad:] = v_.numpy(); X[f"kv_v_{src}"] = vv
    want = (out if meta.get("head") == "chunk" else final)[0, -1].numpy()
    return X, want

ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
if len(ids) > C: ids = ids[:1] + ids[-(C - 1):]; print("ref prompt trimmed to", C)   # 긴 프롬프트: BOS + 마지막 C-1 토큰
X, want = pack(ids)
np.savez(a.out, want=want, ids=np.array(ids), **{f"in__{k}": v for k, v in X.items()})
print("ref saved", a.out, "want", want.shape, {k: v.shape for k, v in X.items() if k in ("hidden_in",) or k.startswith("kv_")})

if a.calib > 0:
    random.seed(0)
    seqs = [r["out_ids"] for r in json.load(open("/root/hailo/g4/out/fp_answers.json"))]
    docs = [open(f).read() for f in sorted(glob.glob(f"{a.texts}/*.md"))]
    body = tok("\n\n".join(docs))["input_ids"]
    if a.calib_full: seqs = []
    while len(seqs) < a.calib:
        Ls = C if a.calib_full else random.choice([256, 512, 900, 1400, C]); s0 = random.randrange(0, max(1, len(body) - Ls))
        seqs.append([2] + body[s0:s0 + Ls - 1])
    seqs = [s[-C:] for s in seqs[: a.calib]]
    acc = {}
    for s in seqs:
        Xs, _ = pack(s)
        for k, v in Xs.items():
            acc.setdefault(k, []).append(v)
    acc = {k: np.concatenate(v, 0) for k, v in acc.items()}
    np.savez(a.out_calib, lens=np.array([len(s) for s in seqs]), **{f"in__{k}": v for k, v in acc.items()})
    print("calib saved", a.out_calib, "lens", [len(s) for s in seqs])
