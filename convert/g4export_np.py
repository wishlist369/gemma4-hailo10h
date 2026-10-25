# g4export_np.py: 런타임 호스트 테이블 추출 — torch 없이 numpy만(safetensors 헤더 직접 파싱). g4export.py와 같은 출력.
#   사용: python g4export_np.py <HF 모델 디렉터리(google/gemma-4-E2B-it)> <host 출력 디렉터리>
#   embed_bf16.npy [V,D] / ple_bf16.npy [V,NL*PLE] = bf16 원시 비트(uint16), ple_proj.npy·ple_proj_norm.npy·final_norm.npy = fp32, + config/tokenizer 파일 복사
import glob, json, os, shutil, struct, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
PFX = "model.language_model."
NEED = ["embed_tokens.weight", "embed_tokens_per_layer.weight", "per_layer_model_projection.weight", "per_layer_projection_norm.weight", "norm.weight"]

index = {}                                                   # 텐서명 → (파일, 헤더 항목, 데이터 시작 오프셋)
for f in sorted(glob.glob(f"{src}/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k != "__metadata__": index[k] = (f, v, 8 + n)

def raw(name):
    f, v, base = index[PFX + name]; a, b = v["data_offsets"]
    dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[v["dtype"]]
    return np.memmap(f, dtype=dt, mode="r", offset=base + a, shape=tuple(v["shape"])), v["dtype"]

def bf16_bits(name):
    x, dt = raw(name); assert dt == "BF16", f"{name}: {dt}"
    return x

def f32(name):
    x, dt = raw(name)
    return (np.asarray(x, np.uint32) << 16).view(np.float32) if dt == "BF16" else np.asarray(x, np.float32)

for n in NEED: assert PFX + n in index, f"missing tensor {PFX + n}"
np.save(f"{dst}/embed_bf16.npy", bf16_bits("embed_tokens.weight"))
np.save(f"{dst}/ple_bf16.npy", bf16_bits("embed_tokens_per_layer.weight"))
np.save(f"{dst}/ple_proj.npy", f32("per_layer_model_projection.weight"))
np.save(f"{dst}/ple_proj_norm.npy", f32("per_layer_projection_norm.weight"))
np.save(f"{dst}/final_norm.npy", f32("norm.weight"))
for f in os.listdir(src):
    if f.startswith("tokenizer") or f in ("config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
print({f: os.path.getsize(f"{dst}/{f}") // 2**20 for f in sorted(os.listdir(dst))})
