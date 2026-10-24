# g4tbt.py (CT141): 플랜 A 런타임 — 청크별 prefill 그룹(96토큰 블록) + tbt 그룹(1토큰) KV 캐시 상주 생성.
#   사용: python -u g4tbt.py --chunks k00kv,... [--prompt ...] [--max_new 32] [--gap 0.003] [--order cache_first|cur_first] [--upd before|after] [--ref ref.npz]
#   청크 n: /opt/g4/n_prefill.hef, /opt/g4/n_tbt.hef, /opt/g4/n_meta.json. 모든 그룹 상주(configure 1회). 공유 KV(kv_ext)는 k04류 청크의 kv_out 현재 행을 그대로 다음 청크 입력으로.
import argparse, json, os, sys, time, numpy as np
from hailo_platform import VDevice, HEF, FormatType
from g4rt_host import G4RtHost
ap = argparse.ArgumentParser()
ap.add_argument("--chunks", required=True); ap.add_argument("--dir", default="/opt/g4"); ap.add_argument("--host", default="/opt/g4/host")
ap.add_argument("--prompt", default="대한민국의 수도는 어디야? 한 문장으로 답하고, 그 도시의 유명한 음식도 하나 알려줘.")
ap.add_argument("--max_new", type=int, default=32); ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--gap", type=float, default=0.003); ap.add_argument("--order", default="cache_first"); ap.add_argument("--upd", default="before")
ap.add_argument("--ref", default="")        # 디버그: ref npz의 ids로 prefill+tbt 돌려 마지막 청크 출력 hidden을 want와 비교(생성 안 함)
ap.add_argument("--mode", default="tbtonly", choices=["tbtonly", "prefill"])   # tbtonly: prefill 그룹 없이 프롬프트도 tbt로 1토큰씩(핸드오프 미해결 우회)
ap.add_argument("--P", type=int, default=96); ap.add_argument("--cs", type=int, default=512); ap.add_argument("--timeout_ms", type=int, default=10000)
a = ap.parse_args()
P, CS, NH = a.P, a.cs, 8
names = a.chunks.split(","); metas = [json.load(open(f"{a.dir}/{n}_meta.json")) for n in names]
host = G4RtHost(a.host, metas[0]["theta"]); D = host.D
TYPES = ("sliding_attention", "full_attention")

class Net:
    """한 네트워크 그룹(prefill 또는 tbt) 바인딩. 입력은 meta inputs 접미사로 매칭."""
    def __init__(self, vdev, hef_p, meta, width):
        self.meta = meta; self.width = width
        self.im = vdev.create_infer_model(hef_p, HEF(hef_p).get_network_group_names()[0])
        for n in self.im.input_names: self.im.input(n).set_format_type(FormatType.FLOAT32)
        for n in self.im.output_names: self.im.output(n).set_format_type(FormatType.FLOAT32)
        self.im._set_enable_kv_cache(True)
        t0 = time.time(); self.cim = self.im.configure(); self.cfg_ms = (time.time() - t0) * 1e3
        self.raw = self.cim._configured_infer_model; self.b = self.cim.create_bindings()
        self.outs = {n: np.zeros(self.im.output(n).shape, np.float32) for n in self.im.output_names}
        for n in self.outs: self.b.output(n).set_buffer(self.outs[n])
        self.in_map = {}
        for orig, hn in meta["inputs"].items():
            suf = "/" + hn.split("/")[-1]; c = [n for n in self.im.input_names if n.endswith(suf)]
            assert c, f"{orig}({hn}) vstream 없음: {self.im.input_names}"; self.in_map[orig] = c[0]
        outs = list(self.im.output_names)
        self.out_order = [self._match_out(ln, outs, i) for i, ln in enumerate(meta["outputs"])]
    @staticmethod
    def _match_out(ln, outs, i):
        suf = ln.split("/")[-1]; c = [n for n in outs if n.endswith(suf) or n.endswith(suf.replace("output_", ""))]
        return c[0] if c else outs[i]
    def run(self, X, n_new, first=True, last=True):
        # HailoRT CacheManager는 VDevice 전역: update_cache_offset 1회가 등록된 전 청크 캐시를 같이 민다 → 토큰당 첫 청크에서만 호출(10-25)
        for orig, hn in self.in_map.items():
            arr = np.ascontiguousarray(np.asarray(X[orig], np.float32).reshape(self.im.input(hn).shape)); self.b.input(hn).set_buffer(arr)
        if a.upd == "before" and first: self.raw.update_cache_offset(n_new); time.sleep(a.gap)
        self.cim.run([self.b], a.timeout_ms)
        if last: time.sleep(a.gap)
        if a.upd == "after" and last: self.raw.update_cache_offset(n_new); time.sleep(a.gap)
        return [self.outs[n] for n in self.out_order]

def mask_block(nq, n_cache_valid, cur_valid, q_idx, win):
    """[nq, NH*CS] 0/1 마스크. 키 슬롯 = [캐시 CS-width개 | 현재 width개](order). 캐시: 뒤쪽 n_cache_valid개 유효(최신), 현재: 인과. win=슬라이딩 창(토큰 거리) — 캐시 슬롯 j(뒤에서)와 쿼리 거리 반영"""
    width = nq; nc = CS - width; m = np.zeros((nq, CS), np.float32)
    for r in range(nq):
        qi = q_idx[r]
        cache = np.zeros(nc, np.float32)
        if n_cache_valid > 0:
            # 캐시 슬롯 nc-1-j = 현재 블록 첫 토큰보다 (j+1) 앞 토큰 → 쿼리 qi 와의 거리 = qi + j + 1
            j = np.arange(n_cache_valid); dist = (qi if qi >= 0 else 0) + j + 1
            cache[nc - 1 - j] = (dist < win).astype(np.float32)
        cur = np.zeros(width, np.float32)
        if qi >= 0: cur[: min(qi + 1, cur_valid)] = 1.0; cur[: qi + 1] *= ((qi - np.arange(qi + 1)) < win)
        else: cur[0] = 1.0
        m[r] = np.concatenate([cache, cur]) if a.order == "cache_first" else np.concatenate([cur, cache])
    return np.tile(m, (1, NH))

def block_inputs(meta, ids_blk, pos0, n_prev, width, emb, ple, kv_cur):
    """청크 하나의 prefill(width=P) 또는 tbt(width=1) 입력 dict. ids_blk: 이 블록 토큰(n개, 우측 패딩), pos0: 첫 토큰 위치, n_prev: 캐시에 든 토큰 수"""
    n = len(ids_blk); X = {}
    if "inputs_embeds" in meta["inputs"]: e = np.zeros((width, D), np.float32); e[:n] = emb; X["inputs_embeds"] = e
    for i in meta["blocks"]:
        if f"ple_{i}" in meta["inputs"]:
            p_ = np.zeros((width, host.PLE), np.float32); p_[:n] = ple[:, i * host.PLE:(i + 1) * host.PLE]; X[f"ple_{i}"] = p_
    if "per_layer_inputs" in meta["inputs"]:                     # --ple slice: 단일 PLE 입력 [W, NL*PLE](디바이스가 블록별 slice)
        p_ = np.zeros((width, ple.shape[-1]), np.float32); p_[:n] = ple[:n]; X["per_layer_inputs"] = p_
    pos = np.arange(pos0, pos0 + width); q_idx = [r if r < n else -1 for r in range(width)]
    for t in TYPES:
        mk = [k for k in meta["inputs"] if k.startswith(f"mask_{t}")]      # mask_{t} 또는 블록별 mask_{t}_b{i}(같은 마스크 복제)
        if not mk: continue
        win = host.win if t == "sliding_attention" else 10**9
        m_ = mask_block(width, n_prev, n, q_idx, win)
        for k in mk: X[k] = m_
        cs, sn = host.rope(pos, t)
        for k_ in meta["inputs"]:                                        # 공유(k_cos_{t}) 또는 블록별(k_cos_{t}_b{i}) cos/sin
            if k_.startswith(f"k_cos_{t}"): X[k_] = cs
            elif k_.startswith(f"k_sin_{t}"): X[k_] = sn
            elif k_.startswith(f"q_cos_{t}"): X[k_] = np.tile(cs, (1, NH))
            elif k_.startswith(f"q_sin_{t}"): X[k_] = np.tile(sn, (1, NH))
    for k in meta["inputs"]:
        if k.startswith("kv_"):
            src = kv_cur[k]; z = np.zeros((width, src.shape[-1]), np.float32); z[:n] = src.reshape(-1, src.shape[-1])[:n]; X[k] = z
    return X

def chain(nets, ids_blk, pos0, n_prev, width, hidden0, kv_cur):
    """청크 체인 1블록 실행. 반환 (마지막 청크 출력 hidden [width, D], 걸린 ms)"""
    t0 = time.time(); hidden = hidden0
    for net in nets:
        emb_or_hidden = hidden
        X = block_inputs(net.meta, ids_blk, pos0, n_prev, width, emb_or_hidden, PLE_CUR, kv_cur)
        if "hidden_in" in net.meta["inputs"]: X["hidden_in"] = np.zeros((width, D), np.float32); X["hidden_in"][:len(ids_blk)] = np.asarray(hidden).reshape(-1, D)[:len(ids_blk)]
        outs = net.run(X, len(ids_blk), first=net is nets[0], last=net is nets[-1])
        hidden = outs[0].reshape(width, -1)
        for i, src in enumerate(net.meta.get("kv_out", [])):
            kv_cur[f"kv_k_{src}"] = outs[1 + 2 * i].reshape(width, -1)[:len(ids_blk)].copy(); kv_cur[f"kv_v_{src}"] = outs[2 + 2 * i].reshape(width, -1)[:len(ids_blk)].copy()
    return hidden, (time.time() - t0) * 1e3

with VDevice() as vdev:
    t0 = time.time()
    PF = [Net(vdev, f"{a.dir}/{n}_prefill.hef", m, P) for n, m in zip(names, metas)] if a.mode == "prefill" else []
    TB = [Net(vdev, f"{a.dir}/{n}_tbt.hef", m, 1) for n, m in zip(names, metas)]
    print(f"configured {len(names)} chunks ({a.mode}) in {time.time()-t0:.1f}s; cfg ms prefill {[round(x.cfg_ms) for x in PF]} tbt {[round(x.cfg_ms) for x in TB]}", flush=True)
    REF = np.load(a.ref) if a.ref else None
    ids = [int(x) for x in REF["ids"]] if REF is not None else host.encode(a.prompt); T = len(ids)
    emb_all = host.embed(ids); PLE_ALL = host.ple(ids, emb_all)
    REF_HID = REF_KV = None
    if REF is not None:
        W0 = REF["in__hidden_in"].shape[-2]; pad0 = W0 - T
        if "hidden_in" in metas[0]["inputs"]: REF_HID = REF["in__hidden_in"].reshape(W0, -1)[pad0:]            # [T, D] 블록 입력 hidden(HF)
        REF_KV = {k[4:]: REF[k].reshape(W0, -1)[pad0:] for k in REF.files if k.startswith("in__kv_")}         # kv_k_13 → [T, hd]
    # --- prefill: 96블록(마지막 블록은 우측 패딩)
    kv_cur = {}; pos = 0; t_pf = time.time(); last_hidden = None
    if a.mode == "tbtonly":
        for t in range(T):
            PLE_CUR = PLE_ALL[t:t + 1]
            if REF_KV: kv_cur.update({k: v[t:t + 1] for k, v in REF_KV.items()})
            x0 = REF_HID[t:t + 1] if REF_HID is not None else emb_all[t:t + 1]
            h, ms = chain(TB, [ids[t]], t, t, 1, x0, kv_cur); last_hidden = h[0].copy()
            if t % 10 == 0 or t == T - 1: print(f"prefill(tbt) tok {t}/{T} {ms:.1f} ms", flush=True)
    else:
      while pos < T:
        n = min(P, T - pos); blk = ids[pos:pos + n]; PLE_CUR = PLE_ALL[pos:pos + n]
        h, ms = chain(PF, blk, pos, pos, P, emb_all[pos:pos + n], kv_cur)
        last_hidden = h[n - 1].copy(); pos += n
        print(f"prefill block pos {pos-n}..{pos-1} ({n} tok) {ms:.0f} ms", flush=True)
    print(f"prefill total {T} tok {(time.time()-t_pf)*1e3:.0f} ms", flush=True)
    if a.ref:
        want = np.load(a.ref)["want"].reshape(-1); got = last_hidden.reshape(-1)
        print(f"REF after prefill-only (T={T}): cos {float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want))):.5f} rel {np.linalg.norm(got-want)/np.linalg.norm(want):.3e}", flush=True)
        sys.exit(0)
    # --- tbt 생성
    out_ids = []; rng = np.random.default_rng(0); h = last_hidden; t_all = time.time()
    for step in range(a.max_new):
        hn_ = host.final_norm_apply(h) if metas[-1]["head"] == "chunk" else h
        z = host.logits(hn_); nxt = host.sample(z, a.temp, rng=rng); out_ids.append(nxt); ids.append(nxt)
        if nxt in host.eos: break
        t0 = time.time(); e = host.embed([nxt]); PLE_CUR = host.ple([nxt], e)
        h, ms = chain(TB, [nxt], len(ids) - 1, len(ids) - 1, 1, e, kv_cur); h = h[0]
        print(f"[{step}] id {nxt} {host.decode([nxt])!r} tbt {ms:.1f} ms", flush=True)
    print("=== output:", host.decode(out_ids)); print(f"{len(out_ids)} tokens, {(time.time()-t_all)/max(1,len(out_ids))*1e3:.0f} ms/token", flush=True)
