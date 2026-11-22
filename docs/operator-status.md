# Operator inventory, vLLM parity, and measured performance

Baseline commit: `ce200f0` (upstream `InferMatrixLab/vllm-rlt` main). Measurements
recorded 2026-11-22 on the `lyx` host. This document is a measurement record plus an
optimization priority list. It does not change any operator implementation.

Scope: the operator layer only, that is `vllm_rlt/kernels/*`, `csrc/*`, and the
`vllm_rlt/layers/*` modules that sit on the CUDA decode path. Scheduling, KV capacity
policy, and serving semantics are out of scope.

## Why compare against vLLM at all

vllm-rlt deliberately depends on nothing except `torch`/`triton`, so the vLLM comparison
is not "do we call vLLM". It is "for the same logical operator, does vLLM's shipped
implementation show that a materially faster formulation exists, and what does it cost us
to reach it". Two findings drive the whole document:

1. Where the operator is a single fused native kernel in vLLM and an unfused PyTorch
   expression here, the whole-op gap is an order of magnitude and it is *not* GPU-bound.
2. Where both sides are Triton kernels over the same paged layout, vllm-rlt is already at
   or better than vLLM parity for MHA decode, and loses only in GQA decode at larger
   batch.

## Operator inventory

KV pool layout in this repository is `[num_blocks, num_layers, block_size, num_kv_heads,
head_dim]`; every paged kernel receives a per-layer 4-D view `[block, token, kv_head,
dim]` cut from that pool.

| file:symbol | kind | shape / dtype contract | CUDA decode-path caller |
| --- | --- | --- | --- |
| `vllm_rlt/kernels/activation.py` `silu_and_mul` | native CUDA (`csrc/activation/silu_and_mul.cu`) + Torch composite fallback | contiguous fp32/fp16/bf16, equal shapes | `vllm_rlt/models/ouro.py` |
| `vllm_rlt/kernels/triton_attention.py` `paged_attention` | Triton, split-KV + merge | `q[rows,heads,dim]`, pool 4-D, `head_dim<=256` | `kernels/paged_attention.py` -> `core/kv_cache_manager.py`, `worker/cuda_graph.py` |
| `vllm_rlt/kernels/triton_attention.py` `paged_prefill_attention` | Triton, packed causal | packed `q[total_tokens,heads,dim]`, needs `q.stride(2)==1` | `core/kv_cache_manager.py`, `worker/cuda_graph.py` |
| `vllm_rlt/kernels/triton_attention.py` `store_fp8` | Triton, fused quantize+scatter | pool fp8 e4m3 + fp32 per-vector scales | `kv_cache_manager._store_layer` |
| `vllm_rlt/kernels/paged_attention.py` `torch_paged_attention(_windowed)` | pure PyTorch reference | any | torch/NPU backend, tests, oracles |
| `vllm_rlt/kernels/flash_attention.py` `FlashPagedAttention` | vendor flash-attn 2/3/4 | fp16/bf16, FA4-only prefill | backend registry |
| `vllm_rlt/kernels/ascend_attention.py` `AscendAttention` | PyTorch SDPA | NPU, bf16 only | `--attention-backend ascend_eager` |
| `vllm_rlt/kernels/graph_kv.py` `write_graph_kv` | Triton | skips inactive rows on device | CUDA-graph KV write |
| `vllm_rlt/kernels/routing.py` `masked_kv_store` | Triton | bool write mask | non-graph KV write |
| `vllm_rlt/kernels/routing.py` `slots/metadata/gather/scatter/gather_segments/scatter_segments/finalize_kernel` | Triton | int64 descriptor rows, int32 tables | async state, model runner |
| `vllm_rlt/kernels/metadata.py` `expand_kv_addresses` | Triton | int64 positions, int32 tables | `worker/buffers.py` |
| `vllm_rlt/kernels/tensor_transfer.py` `tensor_transfer_kernel` | Triton | flat or 8-wide segmented int64 descriptors | `worker/tensors.py` |
| `vllm_rlt/layers/normalization.py` `RMSNorm` | **pure PyTorch, no kernel** | fp32 internal reduction | every Ouro/Nanbeige/Huginn layer |
| `vllm_rlt/layers/rotary_embedding.py` `RotaryEmbedding` / `apply_rotary_pos_emb` | **pure PyTorch, no kernel** | fp32 angles, split-half layout | every attention layer |

Every operator in this table is Triton or native CUDA except the two layer modules in the
last two rows. Those two are the only operators on the CUDA decode path with no fused
implementation at all, and they are also the two that account for the largest measured
per-step cost (see below).

### Attention backend capability matrix

Registered backends: `torch`, `triton`, `ascend_eager`, and `flash_attn`,
`flash_attn_2`, `flash_attn_3`, `flash_attn_4`. Auto priority is
`flash_attn_4 -> flash_attn_3 -> flash_attn_2 -> triton -> torch`.

| backend | decode | prefill | windowed | packed prefill | FP8 KV | CUDA graphs |
| --- | --- | --- | --- | --- | --- | --- |
| torch | yes | no | yes | no | per-vector + static | no |
| triton | yes | yes | yes | yes (tile-probed) | per-vector + static | yes |
| flash_attn_2/3 | yes | no | no | no | no | yes |
| flash_attn_4 | yes | yes | yes | yes | static scalar only | yes |
| ascend_eager | yes | packed only | no | yes | no | no |

## What differs from vLLM, operator by operator

Compared against vLLM `0.33.0` (`vllm/v1/attention/ops/*`, `csrc/*`).

| operator | vLLM | vllm-rlt | Consequence |
| --- | --- | --- | --- |
| RMSNorm | single native kernel incl. fused-add variants (`csrc/layernorm_kernels.cu`) | four eager PyTorch ops: `.float()`, `square().mean()`, `rsqrt`, cast, multiply | **8 kernels, ~11x slower, flat in token count** |
| RoPE apply | one native kernel, `cos_sin_cache` lookup inside (`csrc/pos_encoding_kernels.cu`) | 7 kernels to build cos/sin + 10 more to apply (`chunk`, `cat`, `neg`, multiply, add) | **17 kernels, ~7x slower** |
| SiLU-and-Mul | native, one kernel | native (`csrc/activation/silu_and_mul.cu`), one kernel | parity in kernel count; **3x slower** at 1-128 tokens because the dispatch entry does more host work |
| decode attention (MHA) | `decode_attention_fwd` (legacy, split-K + reducev) and `unified_attention` (production) | own per-head split-KV + merge; split count derived from the batch shape (`_split_count`) | **rlt is fastest or at parity** at every measured MHA point (0.83x-1.05x of vLLM) |
| decode attention (GQA) | `unified_attention` keeps the KV tile resident across the query group; `triton_decode_attention` batches `BLOCK_H` query heads per KV head | `_paged_attention_group_kernel` reuses `_paged_prefill_kernel`'s many-rows-one-tile structure with one program per **KV head** | **gap closed**: 3.31x faster than before at rows=128, now 0.67x/0.95x/1.05x of vLLM at rows 8/32/128 (see `docs/experiments/V006-gqa-decode-grouped-reuse.md`) |
| packed prefill | `unified_attention` with `tl.dot` tiles, tensor descriptors, optional 3D split | own packed kernel, `BLOCK_M/BLOCK_N=64`, `tl.dot` + optional rounding compensation | **rlt 1.2-1.5x faster**; both are prefill-only cost |
| KV write | `triton_reshape_and_cache_flash` (Triton fallback) / native `reshape_and_cache_flash` | `masked_kv_store` / `write_graph_kv` | **rlt ~2x faster** than vLLM's Triton variant; all are launch-bound |
| quantized KV | MXFP4/FP8 per-tensor and per-token-head paths, native | FP8 e4m3 per-vector only, fused Triton writer | narrower coverage, no measured disadvantage on the paths that exist |
| sliding window | `window_size` left/right with R-SWA prefix support | `window` constexpr, masking only | feature parity for the supported case |

Two structural differences are worth stating separately because they change what an
optimization can even achieve:

- **No `tl.dot` in the per-head decode kernel.** vLLM's `unified_attention` uses tensor-core
  dots for the query tile; rlt's per-head decode kernel computes `tl.sum(keys * query)` per
  split. At `head_dim=128` and one query row this is the right call for MHA (measured
  faster), which is why GQA now goes through the separate grouped kernel that does use
  `tl.dot` over the whole query group.
- **Host-side launch dominance.** vLLM ships fused native replacements for the elementwise
  ops precisely because their per-call work is below the launch cost. rlt keeps them in
  eager PyTorch, so the cost scales with *kernel count*, not with tokens.

## Measurement environment

| item | value |
| --- | --- |
| host | `lyx` (`js4.blockelite.cn:22200`), 8x A100-SXM4-80GB, SM80, no TMA, all GPUs idle |
| Python / torch / triton | 3.12.16 / `2.14.0+cu130` / `3.8.0` |
| vLLM | `0.33.0`, wheel install; `torch.ops._C` requires `import vllm._C_stable_libtorch` |
| flashinfer | `0.7.2.post1` (upper-bound reference only) |
| flash-attn | **not installed** on this host; FA2/3/4 backends were not measured |
| GPU timers | CUDA events (device) and wall clock (host); medians of >=50 repetitions, >=20 warmup |
| CUDA graphs | `torch.cuda.CUDAGraph` capture + replay x32 |
| code under test | `InferMatrixLab/vllm-rlt` @ `ce200f0`, tarball export in `/root/kun/rlt-perf/src` |
| vLLM source for diffing | tag `v0.33.0` in `/root/kun/rlt-perf/vllm-src` |
| checkpoint | `ByteDance/Ouro-1.4B`, `model.safetensors` sha256 `58872a72...9885af`, at `/data/rlt-perf/weights/Ouro-1.4B` |
| benchmark drivers | `/root/kun/rlt-perf/bench/*.py` (not part of the repository), raw JSON in `/root/kun/rlt-perf/out/` |

Correctness for every latency number is checked against
`torch_paged_attention`, the repository's own FP32 reference, on the identical logical
problem. Observed vs vLLM: rlt decode `max_err <= 3e-5`, vLLM decode `<= 4e-3`; both are
BF16-rounding level, so a faster-but-wrong backend would have shown up.

## Measured results

### Decode attention, Ouro-1.4B geometry (16 heads / 16 kv heads / head_dim 128, block 16)

ms per call, eager launch, rows = decode batch.

| case | rlt-triton | vLLM legacy decode | vLLM unified | flashinfer |
| --- | --- | --- | --- | --- |
| L512 r1 | **0.066** | 0.103 | 0.056 | 0.023 |
| L512 r8 | **0.068** | 0.105 | 0.057 | 0.025 |
| L512 r32 | **0.102** | 0.187 | 0.104 | 0.094 |
| L2048 r1 | **0.067** | 0.103 | 0.077 | 0.023 |
| L2048 r8 | **0.100** | 0.160 | 0.119 | 0.098 |
| L2048 r32 | **0.339** | 0.405 | 0.368 | 0.322 |
| L8192 r1 | 0.076 | 0.134 | 0.386 | **0.058** |
| L8192 r8 | **0.358** | 0.395 | 0.446 | 0.350 |
| L8192 r32 | 1.271 | 1.334 | 1.415 | **1.243** |

rlt is at or ahead of vLLM parity for MHA decode everywhere except the single-row
`L8192` case, where flashinfer's `fa2` backend is 1.3x faster. Under CUDA graphs the
split-KV + merge pair collapses from 0.066 ms to 0.010 ms at `L512 r1`.

### Decode attention, GQA (registered in the model configs)

Before V006 (the per-head kernel, `NUM_SPLITS=16`), the gap was up to 3.1x. After moving GQA
onto the grouped kernel that reuses the paged-prefill structure:

| geometry | case | before | after | vLLM unified |
| --- | --- | --- | --- | --- |
| Nanbeige 48q/8kv | L2048 r8 | 0.173 | **0.080** | 0.119 |
| Nanbeige 48q/8kv | L2048 r32 | 0.544 | **0.172** | 0.180 |
| Nanbeige 48q/8kv | L2048 r128 | 2.158 | **0.653** | 0.621 |
| GQA 32q/8kv | L2048 r32 | 0.544 | **0.171** | 0.172 |
| GQA 32q/8kv | L2048 r128 | 2.158 | **0.650** | 0.640 |

The gap is closed at r8/r32 and down to 1.05x at r128, where vLLM's `unified_attention` needs
no merge pass at all. The residual is a kernel-launch cost, not redundant work: stage 1 alone
measures 0.98-0.99x of vLLM's stage 1. At short context (512) the operator stays 1.23-1.37x
behind for the same launch reason; `docs/experiments/V006-gqa-decode-grouped-reuse.md` records
both the phase attribution and the two candidate closures that were measured and rejected.
The `flashinfer` column this table used to carry is dropped: flashinfer is not installed on
`lyx`, so those numbers could not be re-measured and are no longer comparable.

### Packed prefill (MHA, head_dim 128)

| case | rlt | vLLM unified |
| --- | --- | --- |
| q512 kv512 s1 | **0.046** | 0.058 |
| q512 kv512 s16 | **0.302** | 0.405 |
| q512 kv4096 s16 | **3.217** | 4.749 |
| q2048 kv4096 s1 | **0.776** | 1.020 |
| q2048 kv4096 s16 | **9.919** | 14.817 |

rlt's packed prefill is consistently 1.2-1.5x faster. Verified independently at `seqs=16`
against the FP32 oracle: `max_err` 9.8e-4 (`q512 kv512`), 4.9e-4 (`q512 kv4096`).

### Elementwise operators (Ouro widths, head_dim 128 / hidden 2048)

| op | tokens | rlt (eager) | vLLM native | rlt kernels | vLLM kernels | rlt under CUDA graph |
| --- | --- | --- | --- | --- | --- | --- |
| RMSNorm | 1 / 128 / 1024 | 0.075 / 0.076 / 0.076 | 0.007 | 8 | 1 | 0.020-0.026 |
| RoPE table + apply | 1 / 128 / 1024 | 0.184 / 0.186 / 0.185 | 0.026 | 17 | 1 | 0.045 |
| SiLU-and-Mul 5632 | 1-128 | 0.0198 | 0.0066 | 1 | 1 | n/m |
| SiLU-and-Mul 5632 | 1024 | 0.0363 | 0.0247 | 1 | 1 | n/m |
| KV write (masked) | 1-128 rows | 0.0176 | 0.0379 | 1 | 1 | n/m |

All three elementwise rows are **flat in token count from 1 to 1024 tokens**. That is the
signature of launch-bound cost, not arithmetic: the work grows 1024x and the time does not
move. CUDA graphs recover 3-4x but do not reach the fused-kernel number, because the
graph still contains 8 and 17 nodes.

The SiLU gap at small token counts is host-side, not compute: `torch.ops.vllm_rlt` dispatch
plus the reference validation adds ~13 us to a 6.5 us kernel. Above 1024 tokens the native
kernel work dominates and the gap narrows to 1.5x.

### End-to-end, Ouro-1.4B, 128-token prompt / 64-token decode / batch 1

| configuration | ms per decode step | tokens/s |
| --- | --- | --- |
| eager (default `cuda_graphs=False`) | 79.8 | 12.5 |
| `cuda_graphs=True` | **24.0** | **41.7** |

3.3x. Verified over 3 repetitions with spread under 0.3%.

### Per-decode-step operator composition (ops-only profile, CUDA graphs on)

47 decode steps, 5585 kernel launches per step. Device-busy time 21.85 ms/step, which
matches the 24.0 ms/step end-to-end figure, so the step is device-bound after graphs and
**not** host-bound.

| group | ms/step | share | launches/step |
| --- | --- | --- | --- |
| linear (gemv, all projections) | 6.30 | 28.8% | 488 |
| elementwise (misc: add, mul, cast) | 5.01 | 22.9% | 2496 |
| linear (gemm) | 3.78 | 17.3% | 207 |
| **RMSNorm core** (mean, rsqrt, pow) | **2.65** | **12.1%** | **1189** |
| copy / cast / gather | 1.49 | 6.8% | 404 |
| **RoPE rotate_half** (neg, cat) | **1.23** | **5.6%** | **396** |
| rlt attention (decode + merge) | 0.89 | 4.1% | 194 |
| SiLU | 0.25 | 1.1% | 98 |
| KV write | 0.16 | 0.7% | 96 |
| transfer | 0.04 | 0.2% | 8 |

Per step this is 24 layers x 4 loops = 96 attention layers, so the counts divide as
expected: 96 `_paged_attention_kernel`, 96 `_paged_attention_merge_kernel`, 96
`_write_kv`, 480 gemv.

Repeating the profile at a 2048-token prompt moves attention from 0.89 to 0.95 ms/step
(4.1% -> 4.2%) and leaves everything else essentially unchanged. **Attention is not the
bottleneck of this engine at any context length measured here.** The elementwise layer
modules are.

## Optimization priority

Ordered by measured cost on the real decode path, then by risk. Items 4 and 5 were
implemented in V006; each item states the experiment that confirmed or rejected it.

| # | target | measured cost | evidence | expected gain | risk |
| --- | --- | --- | --- | --- | --- |
| 1 | Keep CUDA graphs on by default, or make the default workload work under them | 3.3x end-to-end (79.8 -> 24.0 ms/step) | A/B on the real checkpoint, 3 reps | 3.3x | config-only; must confirm graph coverage for all supported schedules |
| 2 | Fuse RMSNorm (fp32 reduce + scale + weight) into one kernel | 2.65 ms/step (12.1%) + part of the 5.01 ms elementwise tail | 8 kernels vs 1; 11x slower at every token count; flat in tokens | 2.0-2.4 ms/step, ~9% | numerically sensitive: vLLM's fused kernel differs from the current FP64-verified reference by up to 6.3e-2 on BF16, so a fused kernel must keep the existing FP32 accumulation order |
| 3 | Fuse RoPE table build + apply | 1.23 ms/step rotate_half + part of the elementwise/copy tail, 17 kernels | 7x slower; `cos`/`sin` only 4 launches/step but `neg`/`cat` are 396 | 0.9-1.2 ms/step, ~5% | must keep FP32 angle computation; the module already forces this deliberately |
| 4 | ~~Batch the GQA query group inside decode attention~~ **done in V006** | was up to 3.1x loss vs vLLM | `Nanbeige r128` 2.158 -> 0.653 | delivered 3.31x on that case; now 0.67x/0.95x/1.05x of vLLM at r8/r32/r128 | MHA did not regress; residual is kernel-launch cost, see V006 |
| 5 | ~~Retune `NUM_SPLITS`~~ **done in V006** | was 1.20-1.6x depending on shape | split sweep 1..16 at 108 SMs; 16 was never optimal | split count is now derived from the batch shape (`_split_count`) | kept power-of-two and shape-only dependence, so CUDA graphs and batch invariance still hold |
| 6 | Cut `tl.dot`-free single-row long-context decode | 1.3x behind flashinfer at `L8192 r1` | 0.076 vs 0.058 | small | low priority: single-row long context is not a throughput regime |

Items 2 and 3 are the real operator-level work. Item 1 is free and already measured; it
should be resolved before any kernel rewrite, because with graphs off every operator
conclusion is confounded by launch count.

### What not to spend effort on

- **Attention kernel micro-tuning for MHA.** rlt already beats both vLLM Triton paths at
  every MHA point measured, and attention is 4% of the step. Optimizing it further cannot
  move end-to-end throughput meaningfully.
- **Packed prefill.** Already 1.2-1.5x faster than vLLM and prefill is a one-shot cost.
- **KV write.** 0.16 ms/step, 0.7%, and already ~2x faster than vLLM's Triton variant.
- **Split-KV merge kernel.** 0.22 ms/step. Even eliminating it entirely is smaller than
  one of the RMSNorm launches.

## Untested / out of scope

- **flash-attn backends (2/3/4) were not measured**: the package is not installed on this
  host. The capability matrix above is from source, not from execution. Any FA-vs-Triton
- **FP8 KV cache paths were not benchmarked.** Only their existence and shape contracts
  were read. `store_fp8` is one launch and is not on the critical path in BF16 runs.
- **Nanbeige / Huginn / Dory end-to-end were not run**; only their attention geometries
  were used for operator sizing. Dory's 72-sublayer / `head_dim=256` / sliding-window
  combination and Huginn's 32-loop recurrence could change the per-step composition.
- **PD and speculative paths were not measured.** Their operator mix differs (more
  transfers, draft rows).
- **No Nsight Compute counters.** All attribution here is duration-based from Nsight
  Systems-level traces. Occupancy, memory throughput, and instruction-level claims would
  need `ncu`, which was not used.
- The `q512/kv512/s16` prefill oracle in the first pass reported 0.35 max error; the
  cause was the harness reading the wrong query row for a packed batch, not the kernel.
  Corrected values are the ones reported above.

## Reproducing

```bash
# code + vLLM source + checkpoint already staged on lyx
ssh lyx
cd /root/kun/rlt-perf

# operator micro-benchmarks
CUDA_VISIBLE_DEVICES=0 python bench/bench_decode.py \
  --rows 1,8,32 --lengths 512,2048,8192 --geometry ouro2048 --graphs
CUDA_VISIBLE_DEVICES=0 python bench/bench_prefill.py \
  --seqs 1,16 --q-lens 512,2048 --kv-lens 512,4096
CUDA_VISIBLE_DEVICES=0 python bench/bench_elementwise.py
CUDA_VISIBLE_DEVICES=0 python bench/bench_kvwrite.py --rows 1,8,32,128
CUDA_VISIBLE_DEVICES=0 python bench/bench_attrib2.py    # kernel counts per op

# end-to-end + CUDA-graph A/B (real checkpoint)
CUDA_VISIBLE_DEVICES=0 python bench/bench_e2e.py --prompt 128 --decode 64 --batch 1
CUDA_VISIBLE_DEVICES=0 python bench/bench_graphs.py

# per-step composition (ops-only profile, graphs on)
PROMPT=128 CUDA_VISIBLE_DEVICES=0 python bench/bench_e2e_profile.py
```

Raw results: `/root/kun/rlt-perf/out/*.json`. Traces:
`/root/kun/rlt-perf/out/profiles*/rank-*/.cycle-*/*.raw/trace.json` plus the packaged
`cycle-*.tar.gz`. Exports are ~700 MB unpacked and are not committed.
