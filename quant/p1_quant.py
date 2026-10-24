# P1 파일럿 양자화: g4conv HN + Hailo 내부 Qwen 레시피(Gemma판) → optimize → BIT_EXACT/QUANTIZED vs HF 정답
# 사용: python p1_quant.py <base> <calib.npz> <ref.npz> [tag]
import json, os, sys, time
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

# --- 패치: smart_softmax_stats가 어텐션까지만 잘린 서브모델로 Saitama 추론할 때, 서브모델이 안 쓰는 입력(PLE)이 데이터에 있으면
#     ValueError("Mismatch between input nodes and data sources")로 죽음(Qwen은 모든 입력이 어텐션 경로라 안 걸림). 여분 키는 버리고 부족할 때만 에러.
from hailo_model_optimization.saitama.framework.model import model as _smodel
_IU = _smodel.InferUtils
_orig_reorg = _IU.reorganize_dict_inputs.__func__
def _tolerant_reorg(cls, inputs, static_mapping):
    need = set(src for src, _ in static_mapping.data_sources) | set(static_mapping.cache_layers)
    if set(inputs.keys()) - need:
        inputs = {k: v for k, v in inputs.items() if k in need}
    return _orig_reorg(cls, inputs, static_mapping)
_IU.reorganize_dict_inputs = classmethod(_tolerant_reorg)

# --- 패치: a16 그룹가중치 conv(HailoConvA16QuantWeightGroup)의 kernel_scale 계산(ConvStrippedOp.calc_kernel_scale_external)이
#     그룹 conv에서도 [in_ch × out_ch·groups] dense float64 행렬을 먼저 만들고 대각 블록만 잘라 쓴다 → MLP 12288 블록(15~)의 down_proj가
#     12288×147456×8B ≈ 14.5GB(+복사) 로 29GB OOM(2026-10-23, 블록 6144는 3.6GB 스파이크). 블록별로 직접 계산(결과 동일, 메모리 ~1/100).
from hailo_model_optimization.acceleras.atomic_ops import conv_stripped_op as _cso
import tensorflow as tf
_orig_ckse = _cso.ConvStrippedOp.__dict__["calc_kernel_scale_external"].__func__
def _ckse_blockwise(input_scales, output_scale, total_rshift, kernel_shape, is_depthwise, group_sizes, group_kernel_indices=None):
    groups = len(group_sizes) if group_sizes is not None else 1
    if is_depthwise or groups <= 1 or group_kernel_indices is None:
        return _orig_ckse(input_scales, output_scale, total_rshift, kernel_shape, is_depthwise, group_sizes, group_kernel_indices)
    left = 1 / tf.cast(input_scales[0], tf.float64)
    right = tf.cast(output_scale, tf.float64) / 2 ** tf.cast(total_rshift, tf.float64)
    if tuple(left.shape) == (1,):
        left = left[0]
    if left.shape.rank == 0:
        left = tf.repeat(left, kernel_shape[-2])
    if right.shape.rank == 0:
        right = tf.repeat(right, kernel_shape[-1])
    if (int(left.shape[0]), int(right.shape[0])) == tuple(int(x) for x in kernel_shape[-2:]):
        return _orig_ckse(input_scales, output_scale, total_rshift, kernel_shape, is_depthwise, group_sizes, group_kernel_indices)
    in_idx, out_idx = group_kernel_indices
    blocks = []
    for g in range(groups):
        blk = tf.matmul(tf.expand_dims(left[in_idx[g]:in_idx[g + 1]], 1), tf.expand_dims(right[out_idx[g]:out_idx[g + 1]], 0))
        blocks.append(tf.tile(blk, [max(group_sizes) - group_sizes[g] + 1, 1]))
    return tf.cast(tf.concat(blocks, axis=-1), tf.float32)
_cso.ConvStrippedOp.calc_kernel_scale_external = staticmethod(_ckse_blockwise)

base, calib_p, ref_p = sys.argv[1:4]
tag = sys.argv[4] if len(sys.argv) > 4 else "q"
if os.environ.get("MEMTRAP"):   # 진단: RSS가 MEMTRAP GB를 넘는 순간 전 스레드 스택을 stderr로 덤프하고 종료(갑작스런 거대 할당 위치 추적)
    import threading, faulthandler
    def _memtrap():
        lim = float(os.environ["MEMTRAP"]) * 1e9; pg = os.sysconf("SC_PAGE_SIZE")
        while True:
            time.sleep(0.1)
            rss = int(open("/proc/self/statm").read().split()[1]) * pg
            if rss > lim:
                sys.stderr.write(f"\n=== MEMTRAP rss {rss/1e9:.1f}GB > {lim/1e9:.0f}GB ===\n"); sys.stderr.flush()
                faulthandler.dump_traceback(all_threads=True); sys.stderr.flush(); os._exit(99)
    threading.Thread(target=_memtrap, daemon=True).start()
    # 어떤 a16 그룹가중치 conv가 create_decompose_16bits에서 거대 kernel_scale 행렬을 만드는지 추적
    from hailo_model_optimization.acceleras.hailo_layers import hailo_conv_a16_quant_weight_group as _a16
    _orig_anc = _a16.HailoConvA16QuantWeightGroup.apply_native_calibration
    def _anc(self, **kw):
        def _d(o): return {k: (getattr(o, k, None) if not callable(getattr(o, k, None)) else "fn") for k in ("input_shape", "output_shape", "groups", "group_sizes", "filters")}
        try:
            cs = self.conv_sub_layer; ks = getattr(getattr(cs, "kernel", None), "shape", None)
            sys.stderr.write(f"[MEMTRAP] a16wg {getattr(self, 'full_name', '?')} self={_d(self)} conv_sub={type(cs).__name__} kernel={ks} {_d(cs)} rss={int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9:.1f}GB\n"); sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[MEMTRAP] a16wg introspect fail {e}\n")
        return _orig_anc(self, **kw)
    _a16.HailoConvA16QuantWeightGroup.apply_native_calibration = _anc
QUAROT = sys.argv[5] if len(sys.argv) > 5 else "enabled"
PROJ = os.environ.get("PROJ") or (sys.argv[6] if len(sys.argv) > 6 else "a8_w4_a16")          # q/k/v/o_proj 정밀도(env PROJ 우선)
EXTRA16 = [x for x in (os.environ.get("EXTRA16") or (sys.argv[7] if len(sys.argv) > 7 else "")).split(",") if x and x != "-"]   # "-" = 없음(쉘 빈인자 유실 방지)   # a16_w16 추가 접미사
LN8 = [x for x in (os.environ.get("LN8") or (sys.argv[8] if len(sys.argv) > 8 else "")).split(",") if x and x != "-"]       # LN 자체를 a8_w8(8비트 경로)로
TOKEQ = sys.argv[9] if len(sys.argv) > 9 else "enabled"
hn = json.load(open(f"{base}.hn.json")); meta = json.load(open(f"{base}_meta.json"))
EMU_MODE = os.environ.get("EMU", "0")      # 0: 전부 pass-through(FP 텐서) / 1: 전부 에뮬(id·npos; 마스크도 npos=인과) / 2: 마스크만 명시 0/1 4D(슬라이딩 가능), 나머지 에뮬
EMU = EMU_MODE in ("1", "2")
for v in hn["layers"].values():
    if v["type"] == "input_layer" and "emulate_conversion" in v:
        if EMU_MODE == "0" or (EMU_MODE == "2" and v.get("conversion_type") == "mask"):
            v["emulate_conversion"] = False

def lasttok(out, want):
    """infer 출력 → 벡터. 탭 출력처럼 폭(토큰) 축이 남아 있으면 마지막 토큰만"""
    g = np.asarray(out[0] if isinstance(out, list) else out).reshape(-1)
    if g.size != want.size and g.size % want.size == 0:
        g = g.reshape(-1, want.size)[-1]
    return g

def feed(z, meta):
    """npz(in__*) → 입력 레이어명 → 텐서. EMU면 변환 전 원시값(id/위치/가산마스크)"""
    d = {}
    for orig, lname in meta["inputs"].items():
        if EMU and orig == "inputs_embeds":
            ids_ = z["in__ids"]
            if meta.get("vocab_ids"):          # 압축 어휘 → 행 인덱스로 재매핑
                vi = np.asarray(meta["vocab_ids"]); pos = np.searchsorted(vi, ids_)
                assert np.array_equal(vi[pos], ids_), "calib/ref 토큰이 압축 어휘에 없음"
                ids_ = pos
            d[lname] = ids_
        elif EMU and orig.startswith("mask_"):
            d[lname] = z["in__npos"] if EMU_MODE == "1" else z["in__" + orig]   # 1: 유효토큰수(인과 마스크 자동생성) / 2: 명시 0/1 [N,1,C,NH*C]
        elif EMU and ("_cos_" in orig or "_sin_" in orig):
            d[lname] = z["in__npos"]
        elif orig.startswith("ple_") and ("in__" + orig) not in z.files:      # 블록별 PLE 입력(--ple split): 전체 PLE에서 열 슬라이스
            i = int(orig[4:]); P_ = int(meta.get("ple", 256))
            d[lname] = np.ascontiguousarray(z["in__per_layer_inputs"][..., i * P_:(i + 1) * P_])
        elif ("_cos_" in orig or "_sin_" in orig) and ("in__" + orig) not in z.files:   # 블록별 cos/sin(--rope_per_block): 타입 공유 배열 재사용
            d[lname] = z["in__" + orig.rsplit("_b", 1)[0]]
        elif orig.startswith("mask_") and ("in__" + orig) not in z.files:   # 블록별 마스크 입력(g4conv --mask_per_block): 타입 공유 마스크 in__mask_{t} 재사용
            d[lname] = z["in__" + orig.rsplit("_b", 1)[0]]
        else:
            d[lname] = z["in__" + orig]
    return d
SC = hn["name"]; L = hn["layers"]
if os.environ.get("OPTLVL", "0") != "0":      # optimization_level>=1(bias correction block-by-block): 캐시 npz의 float64를 float32로 (DFC 'Non-matching data types of stacked cache files' 회피)
    import hailo_model_optimization.algorithms.block_by_block.cache_utils as _cu
    _np = _cu.np
    class _NP32:
        def __getattr__(self, k): return getattr(_np, k)
        @staticmethod
        def _cast(kw):
            out = {}
            for k, v in kw.items():
                v = _np.asarray(v); out[k] = v.astype(_np.float32) if v.dtype == _np.float64 else v
            return out
        def savez(self, f, *a, **kw): return _np.savez(f, *a, **self._cast(kw))
        def savez_compressed(self, f, *a, **kw): return _np.savez_compressed(f, *a, **self._cast(kw))
    _cu.np = _NP32()
if os.environ.get("KV", "on") == "off":
    hn["net_params"].pop("prefill_size", None); hn["net_params"].pop("cache_size", None)   # 키 삭제(스키마상 None 불가) → duplicate_llm 스킵 → 평가용 단일 그래프
def Q(layers, mode):
    """quantization_param 한 줄. 대상 없으면 빈 문자열(빈 리스트는 파서 에러)"""
    return f"quantization_param([{layers}], {mode})" if layers else ""

import fnmatch
def selg(*globs):
    return ", ".join(n for n in L if L[n]["type"] not in ("input_layer", "output_layer")
                     and any(fnmatch.fnmatch(n.split("/")[-1], g) for g in globs))

def sel(*suffixes):
    return ", ".join(n for n in L if L[n]["type"] not in ("input_layer", "output_layer")
                     and any(n.endswith("__" + s) or n.endswith("/" + s) for s in suffixes))

ctx = meta["ctx"]
cz0 = np.load(calib_p); NCAL = len(cz0["lens"]) if "lens" in cz0.files else int(cz0["in__npos"].shape[0])   # lens 없는 npz(청크 ref 재사용)도 허용
PREFILL = min(96, ctx // 2)        # 캐시 입력 폭 = ctx - prefill > 0 이어야 함(micro ctx64 → 32)
KV = os.environ.get("KV", "on")    # KV=off → prefill/tbt 복제 없이 평가용 HAR(generic infer로 BIT_EXACT 비교 가능)
alls = f"""
pre_quantization_optimization(llm_modifications, policy=disabled)
pre_quantization_optimization(ew_add_fusing, policy=disabled)
{f"set_kv_cache_global_params({PREFILL}, {ctx})" if KV == "on" else ""}
model_optimization_flavor(compression_level=0, optimization_level={os.environ.get("OPTLVL", "0")})
model_optimization_config(globals, multiproc_policy=disabled{", output_encoding_vector=enabled, input_encoding_vector=enabled" if os.environ.get("ENCVEC") == "1" else ""})
model_optimization_config(calibration, batch_size=1, calibset_size={NCAL}, use_saitama=True)
{Q(f"{SC}/input_layer1", "precision_mode=a16_w16")}
{Q(sel('res_attn', 'res_ffn', 'res_ple', 'layer_scalar', *([] if os.environ.get("GM8") == "1" else ['gate_mul']), 'post_attn_w', 'post_ffn_w', 'ple_w', 'ple_mul', *EXTRA16), "precision_mode=a16_w16")}
{Q(sel('gate_proj', 'up_proj'), {"1": "precision_mode=a8_w4_a16", "in": "precision_mode=a16_w4_a16", "in8": "precision_mode=a16_w8_a16", "w8": "precision_mode=a8_w8_a16", "w8o": "precision_mode=a8_w8"}.get(os.environ.get("MLP16", ""), "precision_mode=a8_w4"))}
{Q(sel("q_proj", "k_proj", "v_proj", "o_proj", "k_proj_out", "v_proj_out"), f"precision_mode={PROJ}")}
{Q(sel('ple_gate', 'ple_proj'), "precision_mode=a16_w4_a16")}
{Q(sel('down_proj'), "precision_mode=" + (os.environ.get("DOWNP") or ("a8_w4_a16" if os.environ.get("GM8") == "1" else "a16_w4_a16")))}
{Q(selg("*__down_sum*"), "precision_mode=a16_w16") if selg("*__down_sum*") else ""}
{Q(selg("kv_k_out_*", "kv_v_out_*"), "precision_mode=" + os.environ.get("KVOUTP", "a8_w8")) if selg("kv_k_out_*", "kv_v_out_*") else ""}
{Q(sel('matmul1', 'matmul2'), "precision_mode=" + os.environ.get("MM16")) if os.environ.get("MM16") and sel('matmul1', 'matmul2') else ""}
{Q(sel('q_proj', 'k_proj', 'v_proj', 'o_proj', 'k_proj_out', 'v_proj_out', 'gate_proj', 'up_proj', 'down_proj', 'ple_gate', 'ple_proj'), "weight_group_size=" + os.environ.get("WGS", "128")) if os.environ.get("WGS", "128") != "0" else ""}
{Q(f"{SC}/last_token" if f"{SC}/last_token" in L else "", "precision_mode=a16_w16")}
{Q(sel('post_attn_w', 'post_ffn_w', 'ple_w', 'final_norm_w'), "precision_mode=" + os.environ.get("ROTMODE", "a8_w8_a16")) if meta.get("rot") else ""}
{Q(selg("*_proj_p0"), "precision_mode=a8_w4_a16") if meta.get("split") else ""}
{Q(selg("*_proj_p0"), "weight_group_size=128") if meta.get("split") else ""}
{Q(selg("*_proj_p[1-9]"), "precision_mode=" + os.environ.get("SPLITMODE", "a8_w8_a16")) if meta.get("split") else ""}
{Q(selg("*_proj_add*", "*_proj_sl*") + (", " + sel("q_proj", "k_proj", "v_proj", "up_proj") if sel("q_proj", "k_proj", "v_proj", "up_proj") else ""), "precision_mode=a16_w16") if meta.get("split") else ""}
{Q(sel('post_attn_w', 'post_ffn_w', 'ple_w', 'final_norm_w'), "weight_group_size=128") if meta.get("rot") else ""}
{Q(", ".join(v for k, v in meta["inputs"].items() if k == "per_layer_inputs" or k.startswith("ple_")), "precision_mode=a16_w16")}
{Q(", ".join(v for k, v in meta["inputs"].items() if k.startswith("kv_")), "precision_mode=a16_w16")}
{Q(sel(*LN8), "precision_mode=a8_w8") if LN8 else ""}
{"" if os.environ.get("GEQ") == "1" else "pre_quantization_optimization(equalization, policy=disabled)"}
{f"pre_quantization_optimization(matmul_correction, correction_type=zp_comp_block_3, layers=[{sel('matmul1')}])" if sel('matmul1') and os.environ.get("MMC") != "off" else ""}
{f"pre_quantization_optimization(matmul_equalization, layers=[{sel('matmul1')}], matmul_bias=disabled, policy={'enabled' if os.environ.get('MEQ') == '1' else 'disabled'})" if sel('matmul1') else ""}
pre_quantization_optimization(quarot, equalize_inputs=enabled, equalize_outputs=disabled, policy={QUAROT})
pre_quantization_optimization(layer_norm_decomposition, {('bit_decomposition_mode=' + os.environ['BDM'] + ', ') if os.environ.get('BDM') not in (None, 'omit') else ('' if os.environ.get('BDM') == 'omit' else 'bit_decomposition_mode=uniform_precision, ')}eq_consumer={'True' if os.environ.get('LNEQ') == '1' else 'False'}, equalization={'enabled' if os.environ.get('LNEQ') == '1' else 'disabled'}, square_reduce_sum_groups=16, token_equalization={TOKEQ})
{"" if os.environ.get("SSS") == "off" else "pre_quantization_optimization(smart_softmax_stats, force_zero_centered=True, optimize_bias=enabled, optimize_bias_algorithm=MSE, use_saitama=True)"}
{("quantization_param([" + ", ".join(n for n, v in L.items() if v["type"] == "conv") + "], precision_mode=a16_w16)") if os.environ.get("ALL16") == "1" else ""}
{("quantization_param([" + ", ".join(n for n, v in L.items() if v["type"] == "conv") + "], precision_mode=a8_w8)") if os.environ.get("W8") == "1" else ""}
{("quantization_param([" + ", ".join(n for n, v in L.items() if v["type"] == "conv") + "], precision_mode=" + os.environ.get("ALLCONV") + ")") if os.environ.get("ALLCONV") else ""}
"""
if os.environ.get("WBITS") == "8":            # 가중치 4→8비트 일괄(a8_w4→a8_w8, a16_w4_a16→a16_w8_a16)
    alls = alls.replace("_w4", "_w8")
open(f"{base}_{tag}.alls", "w").write(alls)
t0 = time.time()
r = ClientRunner(hn=hn, hw_arch="hailo10h")
r.load_params(f"{base}_params.npz")
r.load_model_script(alls)
cz = np.load(calib_p)
calib = feed(cz, meta)
ref = np.load(ref_p); data = feed(ref, meta); want = ref["want"].reshape(-1)
if meta.get("out_perm") and meta.get("perm"):
    want = want[np.asarray(meta["perm"])]          # 출력이 치환된 잔차 공간이면 정답도 치환
if EMU:   # 에뮬레이션 입력 의미 검증: 양자화 전 FP에서 cos≈1 이어야 함
    with r.infer_context(InferenceContext.SDK_NATIVE) as c0:
        o0 = r.infer(c0, data, batch_size=1)
    g0 = lasttok(o0, want)
    print(f"NATIVE(emu inputs): cos {float(g0 @ want / (np.linalg.norm(g0) * np.linalg.norm(want))):.6f}", flush=True)
    if os.environ.get("FPONLY") == "1":      # 구조/수학 검증만
        sys.exit(0)
r.optimize(calib)
r.save_har(f"{base}_{tag}.q.har")
print(f"optimize OK {time.time()-t0:.0f}s", flush=True)

EVAL = os.environ.get("EVAL", "all")      # all: FP_OPTIMIZED+QUANTIZED+BIT_EXACT / fp: FP_OPTIMIZED만(5블록 에뮬 교착 회피) / none: 양자화만
CTXS = [("FP_OPTIMIZED", InferenceContext.SDK_FP_OPTIMIZED), ("QUANTIZED", InferenceContext.SDK_QUANTIZED), ("BIT_EXACT", InferenceContext.SDK_BIT_EXACT)]
for name, c in {"all": CTXS, "fp": CTXS[:1], "none": []}[EVAL]:
    try:
        t1 = time.time()
        with r.infer_context(c) as ctx_:
            out = r.infer(ctx_, data, batch_size=1)
        got = lasttok(out, want)
        cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
        print(f"{name}: cos {cos:.5f}  rel {np.linalg.norm(got-want)/np.linalg.norm(want):.3e}  ({time.time()-t1:.0f}s)", flush=True)
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}: {str(e)[:500]}", flush=True)
