# Fused RMSNorm and RoPE: experiment record

Version V001. Branch `perf/fuse-norm-rope`, commit `4944676`. Parent: `ce200f0`
(upstream main). Host `lyx`, 8x A100-SXM4-80GB, torch `2.14.0+cu130`, triton `3.8.0`,
checkpoint `ByteDance/Ouro-1.4B` (sha256 `58872a72...9885af`).

## Objective

The operator survey ([operator-status.md](operator-status.md)) found that RMSNorm and
RoPE are the only operators on the CUDA decode path with no fused implementation, and
they cost 12.1% + 5.6% of device time per decode step plus a large share of the
elementwise/copy tail. Hypothesis: replacing the eager expressions with fused kernels
removes launches and elementwise work, and — if the rounding order is preserved — changes
no output.

Expected outcome: 3-5 ms/step (~15-20%) with task-equivalent output.

## Implementation

| file | change |
| --- | --- |
| `vllm_rlt/kernels/fused_norm_rope.py` | new: `_rmsnorm_kernel`, `_rotary_apply_kernel`, `fused_rmsnorm`, `fused_rotary_apply`, `fused_precision_ok`, `fuse_norm_rope` |
| `vllm_rlt/layers/normalization.py` | `RMSNorm.forward` dispatches to the fused kernel when the flag is on and the dtype/shape contract holds |
| `vllm_rlt/layers/rotary_embedding.py` | `apply_rotary_pos_emb` dispatches to the fused kernel under the same flag |
| `tests/test_fused_norm_rope.py` | new: 55 parity/contract/graph tests |
| `tests/test_ouro.py` | one added layer-level test |

The flag `VLLM_RLT_FUSE_NORM_ROPE` defaults to **off**; the eager path stays the default
and the reference. The fused `cos`/`sin` table is *not* rebuilt inside the kernel — the
layer already builds one per decode step and every layer reuses it, so building it inside
would move work rather than remove it.

## Three things that made this not a mechanical port

1. **vLLM's RoPE pairs interleaved dimensions.** `IS_NEOX=false` is vLLM's default. Every
   model in this repository uses split-half `rotate_half`. Calling vLLM's kernel with its
   default pairing measured |diff| ≈ 7.9 on random inputs — a different rotation, not a
   rounding difference. `is_neox=True` gets the pairing right but still differs by up to
   0.0156 in BF16, because of item 2.
2. **Triton contracts `a*b + c*d` into one FP32 multiply-add.** Eager PyTorch computes two
   BF16 products and a BF16 add: three roundings. The contracted form rounds once, moving
   results by up to 0.0156. The kernel therefore pushes each product through an explicit
   `cvt.rn.*` inline-PTX barrier that the compiler cannot fold. With the barrier, RoPE is
   bitwise equal to eager.
3. **The FP16 and BF16 barriers must be separate `@triton.jit` functions.** Branching on a
   runtime flag inside one helper makes Triton unify `fp16` and `bf16` return types and the
   kernel fails with `Return type mismatch`. First implementation failed exactly here; FP16
   coverage only worked after splitting the helper.

## Result: what was achieved and what was not

**RoPE: bitwise exact.** FP16 and BF16, head_dim 64/128/256, MHA and GQA
(16/16, 32/8, 48/8, 16/4), rows 1-33, non-zero and sparse positions. All `torch.equal`.

**RMSNorm: NOT bitwise exact, and the earlier claim that it was is corrected here.** The
fused kernel's FP32 row sum runs in one Triton block whose tree order differs from
PyTorch's for wide rows. Measured over 160 shape/weight combinations (8 seeds, rows
1/4/33/64, widths 2048/2560/3072/5120/5632, scaled weights and inputs):

| metric | value |
| --- | --- |
| cases not bitwise equal | 4 / 160 (2.5%) |
| maximum difference | 2 BF16 ULP |
| maximum fraction of elements differing | 0.0286% |
| fused vs FP64 oracle | same max distance as eager (0.0269) |
| eager vs FP64 oracle | same max distance as eager (0.0269) |
| FP16 cases not bitwise equal | 11 / 25 (excluded by `fused_precision_ok`) |
| FP32 cases not bitwise equal | 18 / 25 (excluded) |

Both implementations sit at the same distance from the FP64 oracle, so neither is more
accurate; eager is one particular rounding of an equally valid sum. Because a 1-ULP
activation difference is much larger in relative terms for FP16/FP32, those dtypes keep the
eager expression entirely rather than accepting a wider deviation.

**End-to-end:** 64 generated tokens from a 128-token prompt are token-identical with the
flag on and off, in both eager and CUDA-graph mode. That is the property that matters for
output equivalence, and it holds.

## Benchmark

Unprofiled, 3 repetitions, spread < 1%.

| configuration | eager norm/rope | fused | speedup |
| --- | --- | --- | --- |
| `cuda_graphs=False` | 82.87 ms/step | 60.49 ms/step | **1.370x** |
| `cuda_graphs=True` | 24.09 ms/step | 16.64 ms/step | **1.447x** |

Ops-only profile, 47 decode steps, CUDA graphs on:

| group | baseline ms/step | fused ms/step |
| --- | --- | --- |
| linear (gemv) | 6.30 | 6.33 |
| elementwise (misc) | 5.01 | **0.61** |
| linear (gemm) | 3.78 | 3.76 |
| RMSNorm core | 2.65 | — |
| RoPE rotate_half | 1.23 | — |
| copy/cast | 1.49 | — |
| **fused kernels** | — | **1.76** |
| attention (rlt) | 0.89 | 0.88 |
| SiLU | 0.25 | 0.25 |
| KV write | 0.16 | 0.16 |
| **total** | **21.85** | **13.86** (1.577x) |
| **launches/step** | **5585** | **1929** |

Microbenchmark, per operator call:

| op | eager kernels | fused kernels | eager ms | fused ms | eager graph ms | fused graph ms |
| --- | --- | --- | --- | --- | --- | --- |
| RMSNorm (t=1..32) | 8 | 1 | 0.107-0.112 | 0.028 | 0.020-0.027 | 0.0056 |
| RoPE apply | 10 | 1 | 0.127-0.128 | 0.036 | 0.032-0.034 | 0.010 |

## Correctness gates run

| suite | result |
| --- | --- |
| `tests/test_fused_norm_rope.py` + `test_normalization.py` + `test_rotary_embedding.py` | **70 passed** |
| `tests/test_ouro.py`, `tests/test_nanbeige.py` | 32 passed, 16 skipped |
| `tests/test_dory.py` | 66 passed |
| ruff | clean |

## Analysis

- The gain is larger than the 12.1% + 5.6% device share predicted from the survey
  (−8.0 ms/step of 21.85), because the eager path also fed the `elementwise(misc)` and
  `copy` buckets: those drop from 5.01 → 0.61 and 1.49 → 0. This is why the survey's
  per-bucket attribution understated the elementwise opportunity.
- Attention and the projections did not move (6.30 → 6.33, 3.78 → 3.76), confirming the
  change is confined to the two operators.
- After fusion the largest remaining item is the token-run gemv at 6.33 ms/step (45.7%)
  and gemm at 3.76 (27.1%). Those are cuBLAS calls, not custom kernels, so the next
  operator-level opportunity is elsewhere (GQA decode for GQA models, or launch count in
  the graph).
- The per-step device time (13.86 ms) is now below the measured end-to-end 16.64 ms/step;
  the gap is the non-graph host work around the replay.

## Status

**Improved.** Retained on `perf/fuse-norm-rope`, still behind an opt-in flag because the
RMSNorm difference, however small, is real and the repository's precision policy asks for
explicit validation of changed arithmetic. Promoting the default requires a decision on
whether a ≤2-ULP BF16 deviation in the normalizer is acceptable for the shipped models;
that is a policy call, not a measurement.

## Recovery

```bash
git checkout ce200f0            # baseline, no fused path
git checkout perf/fuse-norm-rope # this experiment
VLLM_RLT_FUSE_NORM_ROPE=0 python ...   # fused code present, eager path used
```

Reproduce:

```bash
ssh lyx && cd /root/kun/rlt-perf
CUDA_VISIBLE_DEVICES=0 python bench/bench_fused_e2e.py      # A/B + token equality
VLLM_RLT_FUSE_NORM_ROPE=1 PROMPT=128 python bench/bench_e2e_profile.py
cd fused && CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_fused_norm_rope.py -q
```

Raw results: `/root/kun/rlt-perf/out/fused_e2e.json`,
`/root/kun/rlt-perf/out/profiles-fused/`. Baseline profile:
`/root/kun/rlt-perf/out/profiles-graph/`.

## Next

The RMSNorm ULP gap could be closed by matching PyTorch's reduction order exactly (a
segmented or multi-pass sum mirroring `at::native::reduce_kernel`'s tree), which would make
the fused path bitwise exact and remove the reason it stays opt-in. That is a separate
experiment: it trades some of the 0.028 ms gain for exactness and should be measured, not
assumed.
