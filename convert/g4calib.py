# g4calib: 보정/평가 데이터 생성 (exp venv). 채팅 샘플(FP 답변) + 한국어 장문(임의 텍스트) 혼합, 길이 다양
# 출력: <out>.npz  — 입력별 [N,1,W,ch] + ids 목록
import argparse, json, glob, random
import numpy as np
from transformers import AutoTokenizer
from g4host import G4Host

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/hailo/g4/model")
ap.add_argument("--meta", default="/root/hailo/g4/p1/pilot_meta.json")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--texts", default="/root/hailo/g4/calib_texts")
ap.add_argument("--out", default="/root/hailo/g4/p1/calib")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
meta = json.load(open(a.meta)); W = meta["ctx"]
cfg = json.load(open(f"{a.model}/config.json"))["text_config"]
tok = AutoTokenizer.from_pretrained(a.model)
H = G4Host(a.model, meta["theta"], cfg)
types = [t for t in ("sliding_attention", "full_attention") if f"mask_{t}" in meta["inputs"]]

seqs = []
for r in json.load(open("/root/hailo/g4/out/fp_answers.json")):
    seqs.append(r["out_ids"])                                    # 프롬프트+FP 답변 (채팅 템플릿 포함)
docs = [open(f).read() for f in sorted(glob.glob(f"{a.texts}/*.md"))]
body = tok("\n\n".join(docs))["input_ids"]
while len(seqs) < a.n:
    L = random.choice([256, 512, 900, 1400, W])
    s = random.randrange(0, max(1, len(body) - L))
    seqs.append([2] + body[s:s + L - 1])                         # BOS + 장문 조각
seqs = [s[-W:] for s in seqs[: a.n]]
X = {}
for s in seqs:
    for k, v in H.full_inputs(s, W, types).items():
        X.setdefault(k, []).append(v)
X = {k: np.concatenate(v, 0) for k, v in X.items()}
np.savez(a.out, lens=np.array([len(s) for s in seqs]), **{f"in__{k}": v for k, v in X.items()})
print("calib", a.out, "lens", [len(s) for s in seqs], {k: v.shape for k, v in X.items()})
