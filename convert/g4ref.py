# g4ref: HF Gemma4 기준 정답 + HN 입력 텐서 생성 (exp venv)
#  - 프롬프트를 C 길이로 왼쪽 패딩, HN 입력 12종(NHWC [1,1,C,ch])을 호스트 규칙대로 만든다 (= 런타임 전처리의 파이썬 원형)
#  - HF로 블록 범위만 돌려 마지막 토큰 hidden(최종 norm 후) 정답 저장
# 사용: python g4ref.py --meta p1/pilot_meta.json --out p1/ref.npz
import argparse, json, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/hailo/g4/model")
ap.add_argument("--meta", default="/root/hailo/g4/p1/pilot_meta.json")
ap.add_argument("--out", default="/root/hailo/g4/p1/ref.npz")
ap.add_argument("--tap", default="")
ap.add_argument("--prompt", default="대한민국의 수도는 어디야? 한 문장으로 답하고, 그 도시의 유명한 음식도 하나 알려줘.")
a = ap.parse_args()
meta = json.load(open(a.meta)); C = meta["ctx"]; BLOCKS = meta["blocks"]

torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
lm = model.model.language_model
for i in BLOCKS:
    lm.layers[i].float()
lm.norm.float(); lm.per_layer_model_projection.float(); lm.per_layer_projection_norm.float()
cfg = lm.config
ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
T = ids.shape[1]; pad = C - T
print("prompt tokens", T)

with torch.no_grad():
    emb = lm.embed_tokens.weight[ids].float() * math.sqrt(cfg.hidden_size)            # [1,T,D] (스케일 포함, fp32)
    tok_ple = (lm.embed_tokens_per_layer.weight[ids].float() * math.sqrt(cfg.hidden_size_per_layer_input)
               ).reshape(1, T, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)
    ple = lm.project_per_layer_inputs(emb, tok_ple)                                   # [1,T,NL,PLE]
    # HF 정답: 블록 범위만 직접 실행
    pos = torch.arange(T)[None]
    mask_kwargs = dict(config=cfg, inputs_embeds=emb, attention_mask=torch.ones_like(ids), past_key_values=None, position_ids=pos)
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    masks = {"full_attention": create_causal_mask(**mask_kwargs), "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)}
    pe = {t: lm.rotary_emb(emb, pos, t) for t in set(cfg.layer_types)}
    taps = {}
    if a.tap:   # 탭 이름 → HF 모듈 훅 (마지막 토큰 벡터)
        bi = int(a.tap.split("__")[0].replace("block", "")); Lb = lm.layers[bi]; kind = a.tap.split("__", 1)[1]
        hookmap = {"matmul2": (Lb.self_attn.o_proj, "in"), "o_proj": (Lb.self_attn.o_proj, "out"),
                   "post_attn_w": (Lb.post_attention_layernorm, "out"), "res_attn": (Lb.pre_feedforward_layernorm, "in"),
                   "post_ffn_w": (Lb.post_feedforward_layernorm, "out"), "ple_w": (Lb.post_per_layer_input_norm, "out"),
                   "ple_gate": (Lb.per_layer_input_gate, "out"), "q_norm_w": (Lb.self_attn.q_norm, "out"), "k_norm_w": (Lb.self_attn.k_norm, "out"),
                   "ln_in": (Lb.input_layernorm, "out"), "gate_mul": (Lb.mlp.down_proj, "in"), "down_proj": (Lb.mlp.down_proj, "out"),
                   "res_ffn": (Lb.per_layer_input_gate, "in"), "up_proj": (Lb.mlp.up_proj, "out"), "gate_proj": (Lb.mlp.gate_proj, "out"), "ple_proj": (Lb.per_layer_projection, "out"), "layer_scalar": None}
        if hookmap[kind] is not None:
            mod, side = hookmap[kind]
            def hk(m, inp, out):
                taps["v"] = (inp[0] if side == "in" else out).detach().float()
            mod.register_forward_hook(hk)
    h = emb; shared = {}
    for i in BLOCKS:
        L = lm.layers[i]
        h = L(h, ple[:, :, i, :], shared_kv_states=shared, position_embeddings=pe[cfg.layer_types[i]],
              attention_mask=masks[cfg.layer_types[i]], position_ids=pos, past_key_values=None)
        if isinstance(h, tuple):
            h = h[0]
        if a.tap and i == bi and kind == "layer_scalar":
            taps["v"] = h.detach().float()          # 블록 출력(layer_scalar 적용 후)
    want = lm.norm(h)[0, -1].numpy()
    if a.tap:
        tv = taps["v"]
        if kind in ("ple_gate", "gate_proj"):
            tv = torch.nn.functional.gelu(tv, approximate="tanh")           # HN ple_gate는 conv+gelu
        want = tv[0, -1].reshape(-1).numpy()
        print("tap", a.tap, "shape", want.shape)

# ---------- HN 입력 (호스트 전처리 공용 함수) ----------
from g4host import G4Host
cfgd = json.load(open(f"{a.model}/config.json"))["text_config"]
H = G4Host(a.model, meta["theta"], cfgd)
types = [t for t in ("sliding_attention", "full_attention") if f"mask_{t}" in meta["inputs"]]
X = H.full_inputs(ids[0].tolist(), C, types)
np.savez(a.out, want=want, ids=ids[0].numpy(), pad=pad, **{f"in__{k}": v for k, v in X.items()})
print("saved", a.out, {k: v.shape for k, v in X.items()}, "want norm", float(np.linalg.norm(want)))
