# Technical notes

Notes from porting Gemma 4 E2B to Hailo-10H with DFC 5.4 and HailoRT 5.4.0. They're mostly things that aren't documented anywhere and cost me hours or days to find.

## 1. Running several KV-cache HEFs at the same time

### Setup

A KV-cache network, compiled with `set_kv_cache_global_params(prefill, cache_size)`, has `CACHE`-type inputs and outputs. The compiled HEF holds two network groups:

- `<net>__prefill`, for 96-token blocks;
- `<net>__tbt`, for one token at a time.

Both groups share the cache buffers. My setup was 35 layers split into 12 chunk HEFs, every chunk configured on one VDevice at once.

### Symptom

The first chunk configures fine. The second, different chunk fails with one of two errors:

- `Cache size mismatch` (`HAILO_INTERNAL_FAILURE`);
- `set_cache_output_channel` returns `HAILO_NOT_FOUND`.

Configuring the *same* HEF 12 times works. That's misleading: it made me think "12 resident groups are fine" for a while.

### Cause (HailoRT v5.4.0, `libhailort/src/core_op/resource_manager/cache_manager.cpp`)

- **Cache registration is keyed by name.** `CacheManager::m_core_op_managers` uses the core-op name, which is the network name from the HN. Every chunk exported by my converter was named `gemma4`. So the second HEF's caches never got registered under their own entry.
- **Backing buffers are keyed globally by `cache_id`.** `StorageManager` does this, and it's how prefill and tbt share the same buffers.
- **The DFC reuses IDs.** `duplicate_llm_to_network_groups.py` assigns `cache_id` from **0 in every HEF**, so different chunks collide on IDs with different sizes.

### Fix

Each chunk needs:

- a **unique network name**: HN `name`, native/fp HN names, and `model_name` in the HAR metadata;
- a **disjoint `cache_id` range**.

`compile/har_split_cid.py` does both while extracting the tbt scope:

```
NET_NAME=g4_g03 CACHE_OFF=12 python har_split_cid.py kv03_r2.q.har g03_tbt.har gemma4__tbt
```

### `update_cache_offset` is VDevice-global

`ConfiguredInferModel::update_cache_offset(n)` advances the caches of **every** registered core op, not just the one you call it on.

- Call it **once per token**, before running the first chunk.
- If you call it once per chunk, every cache advances 12 times per token. The output stays plausible-looking but wrong. On a 2-chunk test (layers 0–5), the cosine against HF went from 0.986 to 0.774.
- The runtime sleeps about 2–3 ms after the update and after the last chunk (`--gap`). I haven't checked whether that sleep is necessary.

## 2. Building the HN directly instead of going through ONNX

`ClientRunner(hn=dict, hw_arch="hailo10h")` + `load_params(npz)` lets you hand the DFC a graph you wrote yourself.

`convert/g4conv.py` writes Gemma's graph following the conventions in Hailo's public `qwen2_1.5b_instruct.q.har`:

| Op | How it's expressed |
|---|---|
| RMSNorm | `layer_normalization(rms_norm=True)` |
| Linear layers | 1×1 conv |
| RoPE | `feature_shuffle` (rotate-half) + multiply by cos/sin inputs. The sign is folded into the sin input. |
| QK matmul | `matmul(transpose, groups=q_heads, input_tiles=[1,1,gqa])` |
| Softmax | decomposed as reduce_max → sub/exp → **0/1 multiplicative mask** → reduce_sum → mult |

Layer names matter: the optimizer finds some chains, the softmax decomposition in particular, by name.

Gemma-specific pieces:

- **QK/V norm:** per-head RMS groups.
- **Sandwich norms.**
- **GELU-tanh.**
- **Per-layer-embedding branch:** the PLE table lookup stays on the host; the gate and projection run on the chip.
- **Partial RoPE on full-attention layers.**
- **`layer_scalar`:** folded into the following norm weight.

## 3. Quantization recipe that held up

The recipe is called R2. Its full details are in `quant/p1_quant.py`.

| Part | Precision |
|---|---|
| Projections | a8 / w4 (group 128) |
| Residual adds and the MLP gate multiply | a16 |
| Norm weights (`ln_in_w`, `ln_ffn_w`, `final_norm_w`) | kept on the activation side |
| KV outputs | a8/w8 |

How I checked quality:

- **Use emulation, not just the quantized-sim score.** I measured with `SDK_BIT_EXACT` emulation and on the real device. Hailo's `SDK_QUANTIZED` estimate was consistently optimistic.
- **Calibrate on full-window sequences:** 12 sequences × 512 tokens.

## 4. Where decode time goes

| Measurement | Result |
|---|---|
| 1 block, tbt | about 5.5–6 ms per step |
| Same 3-block chunk with fewer contexts (18 → 9) | only about 8% faster |
| Each network-group switch | about +10–12 ms |
| 12 groups × (3 blocks × about 5.7 ms + switch) + host | **435 ms/token** |

The per-block cost barely moves with the context count; it looks like a fixed per-block cost on the chip.

Projections if larger groups compiled:

| Groups | Projected tok/s |
|---|---|
| 12 | 2.3 (measured) |
| 6 | about 3.6 |
| 4 | about 3.9 |
| 3 | about 4.1 |

The 6-block chunk currently hits `BackendAllocatorException: Compilation failed with unexpected crash`. It's deterministic and happens at the last context, at both 0.5 and 0.75 utilization caps.

The compile settings that worked for 3-block tbt chunks are in `compile/compile_l0_u75.alls`:

```
performance_param(compiler_optimization_level=0)
context_switch_param(max_compute_16bit_utilization=0.6, max_compute_utilization=0.75, max_control_utilization=0.75, max_memory_utilization=0.75)
resources_param(max_compute_16bit_utilization=0.8, max_utilization=0.9)
```

## 5. Host side

| Piece | Size or detail |
|---|---|
| Token embedding | 262144 × 1536, bf16, 0.8 GB |
| Per-layer embedding table | 262144 × 35×256, bf16, 4.7 GB |
| Lookups | memory-mapped, so RSS stays reasonable |
| Final norm and tied lm_head (+ softcap 30) | numpy on the host |

The host side has no torch dependency at runtime.
