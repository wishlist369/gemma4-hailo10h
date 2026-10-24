# hf_bench.py (DFC 호스트): 벤치 프롬프트 세트 → HF bf16 greedy N토큰 참조(prompt_ids, hf_ids) 저장. 사용: python hf_bench.py bench_prompts.json out/hf_bench.json 24
import sys, json, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(4)
M = "/root/hailo/g4/model"; P = json.load(open(sys.argv[1])); OUT = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16); model.eval()
T = "<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n"; res = []
for i, p in enumerate(P):
    ids = tok(T.format(prompt=p), add_special_tokens=False, return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=N, do_sample=False)
    hf = out[0, ids.shape[1]:].tolist()
    res.append({"prompt": p, "prompt_ids": ids[0].tolist(), "hf_ids": hf, "hf_text": tok.decode(hf, skip_special_tokens=True)})
    print(f"[{i}] {len(hf)} tok {time.time()-t0:.0f}s: {res[-1]['hf_text'][:60]!r}", flush=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False)
print("saved", OUT, len(res))
