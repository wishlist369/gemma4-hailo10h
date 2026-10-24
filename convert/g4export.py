# g4export.py (DFC 호스트 exp venv): 런타임 호스트 테이블 추출 → host/ (bf16 원시 비트 uint16 npy, fp32 npy, 토크나이저)
#   embed_tokens [V,D] bf16(임베딩 룩업 + tied lm_head), embed_tokens_per_layer [V, NL*PLE] bf16, per_layer_model_projection [NL*PLE, D] fp32,
#   per_layer_projection_norm [NL*PLE] fp32, norm.weight [D] fp32, config.json, tokenizer*
import json, os, shutil, sys
import numpy as np, torch
from safetensors import safe_open
src = sys.argv[1] if len(sys.argv) > 1 else "/root/hailo/g4/model"; dst = sys.argv[2] if len(sys.argv) > 2 else "/root/hailo/g4/host"
os.makedirs(dst, exist_ok=True)
st = safe_open(f"{src}/model.safetensors", "pt")
g = lambda k: st.get_tensor(f"model.language_model.{k}")
def bf16_bits(t): return t.contiguous().view(torch.int16).numpy().view(np.uint16)
np.save(f"{dst}/embed_bf16.npy", bf16_bits(g("embed_tokens.weight")))
np.save(f"{dst}/ple_bf16.npy", bf16_bits(g("embed_tokens_per_layer.weight")))
np.save(f"{dst}/ple_proj.npy", g("per_layer_model_projection.weight").float().numpy())
np.save(f"{dst}/ple_proj_norm.npy", g("per_layer_projection_norm.weight").float().numpy())
np.save(f"{dst}/final_norm.npy", g("norm.weight").float().numpy())
for f in os.listdir(src):
    if f.startswith("tokenizer") or f in ("config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
print({f: os.path.getsize(f"{dst}/{f}") // 2**20 for f in sorted(os.listdir(dst))})
