# har_split.py (CT142 docker): KV=on 멀티 scope HAR에서 한 scope(예 gemma4__tbt)만 독립 네트워크 HAR로 추출 — base_layer 공유 가중치를 자기 이름으로 복제
# 사용: python har_split.py <src.har> <dst.har> <scope>
# har_split_cid.py: + env NET_NAME=<name> → hn/native/fp name·metadata model_name 교체(HailoRT CacheManager가 core_op 이름으로 캐시를 등록 — 청크마다 달라야 동시 상주)
# + env CACHE_OFF=<n> → 모든 cache_id에 n을 더함(청크 HEF끼리 cache_id 0..가 겹쳐 동시 configure 실패하는 문제 회피, 10-24)
import tarfile, json, io, sys, numpy as np, time
src, dst, scope = sys.argv[1], sys.argv[2], sys.argv[3]; SCOPES = scope.split(","); scope = SCOPES[0]
with tarfile.open(src) as t:
    members = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
hn = json.loads(members["gemma4.hn"]); L = hn["layers"]
keep = {n: v for n, v in L.items() if any(n.startswith(sc + "/") for sc in SCOPES)}
bases = {}
for n, v in keep.items():
    b = v.get("base_layer")
    if b and b != n: bases[n] = b
    v.pop("base_layer", None)                      # 베이스 레이어엔 키 자체가 없음(스키마: 있으면 string)
for n, v in keep.items():
    for i in v.get("input", []): assert any(i.startswith(sc + "/") for sc in SCOPES), f"cross-scope input {n} <- {i}"
hn["layers"] = keep
NET = __import__("os").environ.get("NET_NAME", "")
if NET: hn["name"] = NET
import os; COFF = int(os.environ.get("CACHE_OFF", "0")); ncid = 0
for v in keep.values():
    if "cache_id" in v: v["cache_id"] = int(v["cache_id"]) + COFF; ncid += 1
print("cache_id offset", COFF, "layers", ncid, "ids", sorted({v["cache_id"] for v in keep.values() if "cache_id" in v}))
npar = hn["net_params"]; npar["net_scopes"] = list(SCOPES); npar["lora_adapters"] = []
npar["output_layers_order"] = [o for o in npar.get("output_layers_order", []) if any(o.startswith(sc + "/") for sc in SCOPES)]
print("layers kept", len(keep), "with shared base:", len(bases), "sample base map:", list(bases.items())[:2]); print("net_params:", {k: v for k, v in npar.items() if k != "output_layers_order"})
def remap(npz_bytes):
    z = np.load(io.BytesIO(npz_bytes), allow_pickle=True); out = {}; copied = 0
    for k in z.files:
        if "/" not in k or not k.split("/")[0].startswith("gemma4"): out[k] = z[k]; continue   # meta keys
        if any(k.startswith(sc + "/") for sc in SCOPES): out[k] = z[k]
    for n, b in bases.items():
        for k in z.files:
            if k.startswith(b + "/"):
                nk = n + "/" + k[len(b) + 1:]
                if nk not in out: out[nk] = z[k]; copied += 1
    buf = io.BytesIO(); np.savez(buf, **out); return buf.getvalue(), len(out), copied
q_b, nq, cq = remap(members["gemma4.q.npz"]); ho_b, nho, cho = remap(members["gemma4.ho.npz"])
print(f"q.npz {nq} keys (copied from base {cq}); ho.npz {nho} keys (copied {cho})")
members["gemma4.hn"] = json.dumps(hn).encode(); members["gemma4.q.npz"] = q_b; members["gemma4.ho.npz"] = ho_b
alls = members["gemma4.alls"].decode()
if len(SCOPES) > 1:
    lines = []
    for ln in alls.splitlines():
        if "gemma4/" in ln: lines += [ln.replace("gemma4/", sc + "/") for sc in SCOPES]
        else: lines.append(ln)
    members["gemma4.alls"] = "\n".join(lines).encode()
else: members["gemma4.alls"] = alls.replace("gemma4/", scope + "/").encode()
# fp 단계 산출물(native/fp hn, npz/fpo/stats)도 scope를 치환 — hailo join 등이 'gemma4' scope를 찾다 KeyError 나는 것 방지
def rescope_hn(b):
    h = json.loads(b); h["layers"] = {n.replace("gemma4/", scope + "/", 1): v for n, v in h["layers"].items()}
    for v in h["layers"].values():
        for key in ("input", "output"):
            if key in v: v[key] = [x.replace("gemma4/", scope + "/", 1) for x in v[key]]
    if NET: h["name"] = NET
    npx = h.get("net_params", {}); npx["net_scopes"] = [scope]
    if "output_layers_order" in npx: npx["output_layers_order"] = [x.replace("gemma4/", scope + "/", 1) for x in npx["output_layers_order"]]
    return json.dumps(h).encode()
def rescope_npz(b):
    z = np.load(io.BytesIO(b), allow_pickle=True); out = {(k.replace("gemma4/", scope + "/", 1) if k.startswith("gemma4/") else k): z[k] for k in z.files}
    buf = io.BytesIO(); np.savez(buf, **out); return buf.getvalue()
for nm_ in ("gemma4.native.hn", "gemma4.fp.hn"):
    if nm_ in members: members[nm_] = rescope_hn(members[nm_])
for nm_ in ("gemma4.npz", "gemma4.fpo.npz", "gemma4.stats.npz"):
    if nm_ in members: members[nm_] = rescope_npz(members[nm_])   # 모델 스크립트의 레이어 scope도 치환(파서: Invalid scope name gemma4)
print("alls scope refs:", alls.count("gemma4/"), "->", members["gemma4.alls"].decode().count(scope + "/"))
if NET:
    md = json.loads(members["gemma4.metadata.json"]); md["model_name"] = NET; members["gemma4.metadata.json"] = json.dumps(md).encode()
    print("net name ->", NET)
with tarfile.open(dst, "w") as t:
    for name, data in members.items():
        ti = tarfile.TarInfo(name); ti.size = len(data); ti.mtime = int(time.time()); t.addfile(ti, io.BytesIO(data))
print("wrote", dst)
