# Where the matmul time goes: measurement record

Version V003 (analysis only, no code change). Branch `perf/fuse-norm-rope`, commit
`6b2bc86`. Host `lyx`, 8x A100-SXM4-80GB (SM80, 40 MB L2), torch `2.14.0+cu130`,
triton `3.8.0`, Ouro-1.4B at BF16, batch 1, CUDA graphs on, 128-token prompt.

Instruments: Nsight Compute 12.6 (hardware counters, per kernel) and Nsight Systems
(device timeline), cross-checked against unprofiled CUDA-event timing.

## The question

The projections are 70% of the decode step. This records which stage the time is actually
spent in: arithmetic, memory access, synchronisation, or waiting.

## Answer in one paragraph

The matmuls are **memory-bound and already reasonably efficient**. On the real path an
`M=1, N=6144, K=2048` BF16 GEMM reads exactly its 25.18 MB of weights (no re-reads, no
duplication) and sustains **55-59% of the DRAM peak that ncu reports**, with wave occupancy
of 0.79 and 6.25% achieved warps. The dominant warp stall is `wait` (2.22 cycles/issue),
which is the classic signature of memory latency not being hidden by enough concurrent
warps, not of a maths-pipe or sync problem. The step is **not** synchronisation-bound and
**not** host-bound under graphs. The one structural loss that remains is that batch 1 runs
GEMMs with `M=1`, so the arithmetic side is ~0: the kernel exists only to stream weights.
Padding M up to 128 costs only ~38% more time for 128x the arithmetic, which is why raising
concurrency is the lever with the most headroom.

## Evidence

### 1. Arithmetic is not the limit

| metric (real path, M=1, N=6144) | value |
| --- | --- |
| `dram__bytes_read` | 25.18 MB = exactly the weight matrix |
| `dram__throughput` (% of ncu peak) | **55-59%** |
| `sm__throughput` (% of ncu peak) | 28.9% |
| `sm__sass_thread_inst_executed_op_ffma` | 0 (tensor cores, no FFMA) |
| Duration | 21-23 us |
| `launch__waves_per_multiprocessor` | 0.79 |
| `sm__warps_active` (% of peak) | 6.25% |
| `dram__bytes.sum.per_second` | 1.18 TB/s |

A kernel reading exactly what it needs and reaching ~57% of DRAM with 6% warp occupancy is
latency-limited on memory, not compute-limited. Consistent with this, the same GEMM takes
23.04 us at M=1 and 31.81 us at M=128 while reading 25.18 vs 25.71 MB: **128x the
arithmetic for 1.38x the time**, because the time is the weight stream.

### 2. Warp stalls say "waiting on memory", not "waiting on sync"

`smsp__average_warps_issue_stalled_*_per_issue_active` on the real GEMM:

| stall reason | cycles/issue |
| --- | --- |
| `wait` | **2.22** |
| `long_scoreboard` (global load latency) | 0.95 |
| `short_scoreboard` | 0.57 |
| `barrier` | 0.11 |
| `mio_throttle` | 0.02 |
| `lg_throttle` | 0.01 |
| `math_pipe_throttle` | 0.00 |
| `not_selected` | 0.00 |

`drain`, `math_pipe_throttle` and `not_selected` are at zero, so no tail effect, no tensor
core saturation, and no scheduler starvation. `barrier` at 0.11 rules out a
synchronisation-bound kernel. The mass sits in `wait` + `long_scoreboard`, i.e. instruction
issue is stalled on dependencies that more resident warps would cover.

### 3. It is not synchronisation, and not host waiting

Under CUDA graphs the device timeline contains **77 689 kernels over 47 decode steps**
(1653/step) and every inter-kernel gap is tiny:

| metric | value |
| --- | --- |
| gaps > 5 us in the whole window | **0** |
| gaps < 0.1 us | 65 003 |
| gaps 0.1-1 us | 569 |
| `cudaStreamSynchronize` while generating | none inside the graph |

Reconciling the three clocks, which was necessary because they disagreed at first:

| measurement | value |
| --- | --- |
| sum of kernel durations (nsys) | 12.41 ms/step |
| sum of inter-kernel gaps (nsys) | 7.37 ms/step |
| nsys device span | 19.75 ms/step |
| **CUDA-event end-to-end, unprofiled** | **22.53 ms/token** |
| unprofiled wall clock | 22.52 ms/token |

The event measurement and the wall clock agree to 0.1%, so the 22.5 ms is real device time,
not profiler distortion. The `busy + gaps = span` identity reconciles to 0.03 ms. So the
7.37 ms/step is **not** host idle and **not** a gap between graph replays: it is the
accumulated launch-to-launch spacing of the ~846 graph nodes per step (mean 5.28 us where a
gap is positive). That is graph-node scheduling cost inside the replay, which is real device
time but not useful work.

### 4. Where the step actually goes

| component | ms/step | share |
| --- | --- | --- |
| matmuls (gemv 3.33 + gemm 3.06 + gemm 1.77) | **8.16** | **65.7%** |
| inter-node spacing inside the graph | 7.37 | (not work) |
| fused RMSNorm | 0.91 | 7.3% |
| fused RoPE apply | 0.83 | 6.7% |
| attention (decode + merge) | 0.86 | 6.9% |
| elementwise (sampling, masks, etc.) | 0.82 | 6.6% |
| KV write | 0.16 | 1.3% |
| kernel busy total | 12.41 | 100% |

Launch counts confirm the expected composition: 192 gemv + 201 gemm = 393 matmul launches
per step against the 4 matmuls x 24 layers x 4 loops = 384 predicted.

### 5. Roofline check against the weight stream

Per decode step the model streams 2.47 GB x 4 loops = **9.87 GB** of weights.

| achieved bandwidth | implied time |
| --- | --- |
| 1335 GB/s (measured on gate_up, the widest matrix) | 7.39 ms |
| 1100 GB/s (measured on qkv) | 8.97 ms |
| 1770 GB/s (measured peak via 2 GB copy) | 5.57 ms |

Measured matmul time is 8.16 ms/step, which lands between the 1100 and 1335 GB/s rows. So
the matmuls are running at **62-74% of the achievable memory bandwidth**, and the remaining
gap to the 5.57 ms floor is the narrow matrices (`N=2048`: `o` and `down`) which measure
lower bandwidth than the wide ones.

## What the numbers rule out

| hypothesis | verdict | evidence |
| --- | --- | --- |
| compute / tensor-core bound | **no** | 28.9% SM throughput, 0 FFMA, 0 math-pipe throttle |
| redundant or duplicated memory traffic | **no** | `dram__bytes_read` = exactly the weight size |
| synchronisation-bound | **no** | `barrier` stall 0.11, no `cudaStreamSynchronize` inside the graph, 0 gaps > 5 us |
| host-launch-bound (with graphs) | **no** | gaps are in-nanosecond range; eager mode differs by 2.6x and is a separate story |
| profiler distortion | **no** | event timing 22.53 vs wall clock 22.52 ms/token |
| the time is a real memory bottleneck | **partly** | 55-59% of DRAM peak: memory *is* the limiter, but not at peak efficiency |

## Where the remaining headroom is, ranked by evidence

1. **Narrow-matrix efficiency (`N=2048`, i.e. `o_proj` and `down_proj`).** These are half of
   the 8 matmuls per layer and read only 8.4 MB and 23.1 MB. Measured `o` takes 15.9 us for
   8.4 MB. Fusing them is not possible (different inputs), so the options are a different
   tile, or batching across rows so the same weight read serves more queries. *Evidence
   level: measured* that they are slower per byte; *not measured* which alternative helps.
2. **Raise M.** Time barely moves between M=1 (23.0 us) and M=128 (31.8 us) on the same
   weight. Every extra row is nearly free in time, so throughput per unit of weight read
   scales with concurrency until the memory system saturates. *Evidence level: measured on
   the operator; the engine-level consequence for batch > 1 was not measured.*
3. **Reduce graph node count.** 7.37 ms/step, 37% of the device span, is inter-node spacing.
   1653 nodes/step at 24 layers x 4 loops means ~17 nodes per layer-loop. Fewer, larger
   kernels would shrink this. *Evidence level: measured as a quantity; the mapping from
   specific fusions to node-count reduction is inferred.*
4. **Small-matrix bandwidth floor.** `qkv` at 55-59% of peak vs `gate_up` at ~65% shows the
   effect of having less in flight. Already exploited by the V002 fusion; further gains
   would need larger fused units, and there are none left (the remaining projections have
   distinct inputs).

## What data is still missing

- **DRAM peak disagreement.** ncu reports 55-59% of *its* peak while an independent 2 GB
  copy measures 1770 GB/s (~87% of that peak). A `dram__throughput` reading therefore cannot
  be converted to GB/s without knowing which peak is used. To close this, capture
  `dram__bytes.sum` together with `gpu__time_duration.sum` for the *same* kernel and divide:
  that ratio is peak-independent and is what the roofline table above uses.
- **Per-kernel bandwidth for every matmul shape.** Only `qkv` (25.18 MB) and `gate_up`
  (46.19 MB) were profiled individually; `o` and `down` were inferred from aggregate timing.
  A full ncu pass over the four shapes at M=1 would settle item 1 above.
- **L2 hit rate.** `lts__t_sector_hit_rate` was not collected, so the claim that weights are
  not cached is supported only by `dram__bytes_read` matching the matrix size.
- **Multi-batch engine behaviour.** The M-sweep is operator-level. Whether the engine's
  scheduler actually reaches higher M, and what that does to per-token latency, needs a
  concurrency sweep (batch 1/2/4/8) with the same instruments.
- **Eager mode's 60 ms/step.** That regime is host-bound and was not profiled at kernel
  level; the statements above are for the graph configuration only.
