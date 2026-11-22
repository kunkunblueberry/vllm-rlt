# Prompt-length sweep (256 / 512 / 1024): measurement record

Version V004 (analysis only, no code change). Branch `perf/fuse-norm-rope`, commit
`1206fd2`. Host `lyx`, 8x A100-SXM4-80GB, torch `2.14.0+cu130`, Ouro-1.4B BF16, batch 1,
CUDA graphs on, 32 output tokens per run, 3 repetitions.

## Why this re-measurement was warranted

Every earlier operator number used `M=1`. That is the size of a decode step (one token per
request), so it is correct for decode, but it is *not* representative of prefill, where the
matmul's `M` equals the number of prompt tokens in the chunk. The reviewer's objection - one
token is too extreme to characterise compute performance - is right for prefill. Both
regimes are measured below and they behave very differently.

## Operator level: the regime flips at prefill's M

Measured with CUDA events on the real projection shapes (`K=2048`), cold L2:

| shape | M | us | TFLOP/s | achieved GB/s | us per row |
| --- | --- | --- | --- | --- | --- |
| qkv (N=6144) | 1 | 31.49 | 0.80 | 799 | 31.49 |
| qkv | 256 | 43.78 | **147.17** | 575 | **0.171** |
| qkv | 1024 | 140.54 | **183.36** | 179 | **0.137** |
| gate_up (N=11264) | 1 | 39.68 | 1.16 | 1163 | 39.68 |
| gate_up | 256 | 74.24 | **159.09** | 622 | **0.290** |
| gate_up | 1024 | 269.06 | **175.59** | 172 | **0.263** |

ncu confirmation that the bottleneck moves from memory to compute:

| metric | M=1 | M=256 | M=512 | M=1024 |
| --- | --- | --- | --- | --- |
| `sm__throughput` (compute) | 28.9% | **65.5%** | **73.2%** | **75.1%** |
| `dram__throughput` (memory) | 55-59% | 36.0% | 21.0% | **12.8%** |
| `launch__waves_per_multiprocessor` | 0.79 | ~1.6 | 1.63 | 3.26 |

At M=1024 the same kernel moves from ~57% of DRAM peak to ~13%, and from 29% to 75% SM
throughput. **So at prefill sizes it is compute-bound, exactly as the objection implies -
and the earlier memory-bound conclusion does not apply there.**

Two facts still matter for interpretation:

- **Raising M is enormously efficient per row.** Per-row cost drops from 31.5 us (M=1) to
  0.137 us (M=1024) for `qkv`: **230x cheaper per row**, because the weight read is
  amortised over 1024 rows instead of one.
- **Absolute efficiency is still far from peak**: 183 TFLOP/s against the 312 TFLOP/s A100
  BF16 peak is **59%**. That is the honest number for "how much compute efficiency this
  kernel reaches at prefill sizes".

## Engine level: prompt length, and the chunk-size trap

Total generation time for a 256/512/1024-token prompt plus 32 output tokens (median of 3):

| prompt | default chunk=128 total | full chunk total |
| --- | --- | --- |
| 256 | 846.5 ms | **758.9 ms** |
| 512 | 1016.1 ms | **787.0 ms** |
| 1024 | 1371.9 ms | **828.6 ms** |

Incremental prefill cost, i.e. the marginal cost of more prompt tokens:

| step | default chunk=128 | full chunk |
| --- | --- | --- |
| 256 -> 512 | **+662.5 us/token** | **+109.8 us/token** |
| 512 -> 1024 | **+694.9 us/token** | **+81.2 us/token** |

**This is the headline finding.** With `prefill_chunk_size=128` (the default), a 1024-token
prompt is prefilled as 8 separate 128-token chunks, so the matmul never sees `M > 128` and
the engine pays **6-8x more per prompt token** than when the prompt is prefilled as one
chunk. The default configuration leaves the entire benefit of large `M` on the table.

## What this changes about the earlier conclusions

| earlier claim | still valid? | correction |
| --- | --- | --- |
| "decode is memory-bound, 55-59% of DRAM peak" | **yes, for decode** | unchanged: decode is `M=1` by definition |
| "the matmuls are memory-bound" | **no for prefill** | at `M>=256` they are compute-bound (75% SM) |
| "the projection fusion gain is 1.09x" | **decode only** | it was measured at batch 1 |
| "raising M is the largest lever" | **confirmed** | but the engine caps M at 128 by default |

The earlier work is not invalidated - decode really is memory-bound and the fusion really
does help there - but the framing was incomplete: it described only the decode regime and
generalised the bottleneck to all matmuls.

## Correctness status

No code changed in this experiment. The full suite was already at 1360 passed / 0 failed on
the code under test (`0e7b253`); nothing here touches the implementation.

## What is still missing

- **Prefill throughput with graphs vs without.** These runs used CUDA graphs; prefill chunks
  are commonly outside graph capture, so the numbers may not represent a chunked deployment
  that falls back to eager.
- **Attention cost inside prefill.** Total time includes prefill attention, which grows
  quadratically; the split between projection time and attention time at 1024 tokens was not
  profiled. The inference above attributes the chunk effect to matmul M, and the operator
  data supports that, but an ops-only profile of the 1024-token prefill would confirm the
  attribution directly.
- **`M` at concurrency.** Increasing `max_num_seqs` also raises the matmul M during decode;
  that was not measured here and is the natural next experiment.
- **A larger chunk-size sweep** (128/256/512/1024/2048) would locate where the chunk penalty
  flattens, instead of only comparing the default against prompt-length chunks.

## Status

Analysis retained. Recommended follow-up, in order of expected value: (1) raise the default
`prefill_chunk_size` or make it prompt-aware, (2) profile the 1024-token prefill to split
projection vs attention, (3) sweep `max_num_seqs` for decode `M`.
