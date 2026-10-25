# g4chat.py: 대화형 CLI — 12청크 tbt 그룹을 한 번만 configure하고 프롬프트를 반복 입력(턴마다 KV 새로 시작, 단일 턴).
#   사용: python g4chat.py [--dir <HEF·meta 디렉터리>] [--host <호스트 테이블 디렉터리>] [--prompt "..."(1회 실행)] [--max_new 128] [--temp 0]
#   생성 루프·입력 구성은 g4tbt.py와 동일(그 파일에서 생성). 컨텍스트 = 프롬프트+생성 ≤ cache_size-1 (511)
import argparse, json, os, sys, time, numpy as np
from hailo_platform import VDevice, HEF, FormatType
from g4rt_host import G4RtHost
ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="g00,g03,g06,g09,g12,g15,g18,g21,g24,g27,g30,g33"); ap.add_argument("--dir", default="."); ap.add_argument("--host", default="./host")
ap.add_argument("--prompt", default="")
ap.add_argument("--max_new", type=int, default=128); ap.add_argument("--temp", type=float, default=0.0)
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
    def close(self):
        if self.cim is not None: self.cim.shutdown()
        self.cim = self.raw = self.b = None
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


def generate(TB, prompt):
    global PLE_CUR
    ids = host.encode(prompt); T = len(ids); budget = CS - 1 - T
    if budget <= 0: print(f"[prompt too long: {T} tokens, max {CS - 2}]"); return
    emb_all = host.embed(ids); PLE_ALL = host.ple(ids, emb_all); kv_cur = {}; t_pf = time.time(); h = None
    for t in range(T):                                     # 프롬프트도 1토큰 경로(prefill 핸드오프 미구현)
        PLE_CUR = PLE_ALL[t:t + 1]
        h, _ = chain(TB, [ids[t]], t, t, 1, emb_all[t:t + 1], kv_cur); h = h[0]
        print(f"\r  [prompt {t + 1}/{T}]", end="", flush=True)
    print(f"\r  [prompt {T} tok, {time.time() - t_pf:.1f}s]", flush=True)
    rng = np.random.default_rng(); out_ids = []; shown = ""; t0 = time.time()
    for step in range(min(a.max_new, budget)):
        hn_ = host.final_norm_apply(h) if metas[-1]["head"] == "chunk" else h
        nxt = host.sample(host.logits(hn_), a.temp, rng=rng)
        if nxt in host.eos: break
        out_ids.append(nxt); ids.append(nxt)
        txt = host.decode(out_ids)
        if not txt.endswith("\ufffd"): print(txt[len(shown):], end="", flush=True); shown = txt   # 멀티바이트 조각은 완성될 때까지 보류
        e = host.embed([nxt]); PLE_CUR = host.ple([nxt], e)
        h, _ = chain(TB, [nxt], len(ids) - 1, len(ids) - 1, 1, e, kv_cur); h = h[0]
    dt = time.time() - t0
    print(f"\n  [{len(out_ids)} tok, {len(out_ids) / max(dt, 1e-9):.2f} tok/s]", flush=True)

with VDevice() as vdev:
    t0 = time.time(); TB = []
    try:
        for i, (n, m) in enumerate(zip(names, metas)):
            TB.append(Net(vdev, f"{a.dir}/{n}_tbt.hef", m, 1)); print(f"\r  configuring {i + 1}/{len(names)}", end="", flush=True)
        print(f"\r  {len(names)} network groups ready ({time.time() - t0:.1f}s). Gemma 4 E2B on Hailo-10H — empty line or Ctrl-D to quit.", flush=True)
        if a.prompt: generate(TB, a.prompt)
        else:
            while True:
                try: p = input("\n> ").strip()
                except EOFError: break
                if not p: break
                generate(TB, p)
    except KeyboardInterrupt: print()
    finally:
        for net in reversed(TB): net.close()
