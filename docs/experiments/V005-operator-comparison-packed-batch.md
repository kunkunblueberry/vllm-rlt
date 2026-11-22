# Operator-by-operator comparison with vLLM at packed batch sizes

Version V005. Branch `perf/fuse-norm-rope`, commit `a36fc45`. Host `lyx`, 8x
A100-SXM4-80GB, torch `2.14.0+cu130`, triton `3.8.0`, Ouro-1.4B and the Nanbeige/GQA
geometries this repo ships.

Context for the whole document: this engine runs PD-disaggregated, so the scheduler packs
many requests' tokens into one batch. Per-step row counts are **not** 1. Earlier operator
measurements used `M=1` and therefore described decode only; every number below is taken at
packed sizes (`M` = prompt tokens for prefill, `M` = batched requests for decode).

## Method

Each rlt operator was paired with its vLLM counterpart by reading both implementations, then
both were driven on the identical logical problem with the same shapes, dtypes, block size
and context lengths. Correctness is measured against this repo's own FP32 torch reference
(`torch_paged_attention`), so an inaccuracy in either implementation shows up. All timings
are CUDA-event device time, cold L2 where the operator streams weights.

## The comparison table

### Attention, decode (context 2048, page 16)

| geometry | rows | rlt | vLLM unified | vLLM decode | rlt / best vLLM |
| --- | --- | --- | --- | --- | --- |
| Ouro MHA 16/16 | 8 | **0.100** | 0.121 | 0.159 | **0.83x (rlt faster)** |
| Ouro MHA 16/16 | 32 | **0.341** | 0.371 | 0.406 | **0.92x (rlt faster)** |
| Ouro MHA 16/16 | 128 | 1.330 | **1.230** | 1.408 | 1.08x |
| Nanbeige GQA 48/8 | 8 | 0.173 | **0.119** | 0.128 | 1.45x |
| Nanbeige GQA 48/8 | 32 | 0.544 | **0.180** | 0.264 | **3.02x** |
| Nanbeige GQA 48/8 | 128 | 2.158 | **0.621** | 0.840 | **3.47x** |
| GQA 32/8 | 128 | 1.463 | **0.642** | 0.805 | **2.28x** |

MHA is at parity or ahead. **GQA is the one real operator gap and it grows with the packed
batch size**, which is exactly the regime this engine runs in.

### Attention, prefill (MHA, head_dim 128)

| case | rlt | vLLM unified |
| --- | --- | --- |
| q512 kv512 s1 | **0.046** | 0.058 |
| q512 kv4096 s16 | **3.217** | 4.749 |
| q2048 kv4096 s16 | **9.919** | 14.817 |

rlt is 1.2-1.5x faster; no action needed.

### Elementwise and KV write

| operator | rlt | vLLM | note |
| --- | --- | --- | --- |
| SiLU-and-Mul (5632) | 0.0198 ms (1-128 tok) | 0.0066 ms | rlt native kernel exists; gap is host dispatch, not GPU |
| KV write (masked) | **0.0176 ms** | 0.0379 ms | rlt ~2x faster |
| RMSNorm | **1 kernel** after V001 | 1 kernel | parity in kernel count |
| RoPE apply | **1 kernel** after V001 | 1 kernel | parity, bitwise exact |

### Projections

| aspect | rlt | vLLM | note |
| --- | --- | --- | --- |
| GEMM library | cuBLAS | cuBLAS/native | both call tuned libraries |
| q/k/v fusion | **fused after V002** | fused qkv | parity |
| gate/up fusion | **fused after V002** | fused gate_up | parity |

After V002 the projections match vLLM's structure; the remaining per-projection difference
is cuBLAS kernel selection, not implementation.

## Root cause of the GQA gap, measured

ncu on the two kernels, Nanbeige geometry, 128 rows, context 2048:

| metric | rlt `_paged_attention_kernel` | vLLM `_fwd_grouped_kernel_stage1` |
| --- | --- | --- |
| DRAM bytes read | 1.11 GB | 1.10 GB |
| DRAM throughput (% peak) | **23.1%** | **78.4%** |
| duration | 2.47 ms | 0.73 ms |
| instructions executed | **778.6 M** | **115.5 M** |
| achieved occupancy | 55.7% | 43.1% |
| waves per SM | 101 | 21.7 |

**Both kernels move the same bytes.** rlt executes **6.7x more instructions** for that same
traffic and consequently reaches only 23% of DRAM peak where vLLM reaches 78%. rlt also has
*higher* occupancy, so this is not an occupancy problem.

The mechanism, read from the source: rlt launches one program per
`(row, query_head, split)` — grid `(128, 48, 16)`. With 6 query heads sharing each KV head,
those 6 programs each walk the same KV tile, so the tile is fetched 6 times and the
per-element address arithmetic is executed 6 times. vLLM's grouped kernel launches
`(128, 8, 16)` and batches `BLOCK_H` query heads per program, so one tile serves the whole
group.

Compounding it, the per-head kernel's inner loop is register-only elementwise plus
`tl.sum` — no `tl.dot` — so each KV element costs an individual multiply-add instruction
rather than a tensor-core MAC.

## What I implemented, and why it is not in the tree

I added `_paged_attention_grouped_kernel` mirroring vLLM's design: one program per
`(row, head_block, split)`, one KV tile per program, `tl.dot` for the group.

It was **fast** — 0.624 ms at r128 instead of 2.158, i.e. at vLLM parity — but it was
**numerically wrong**: 0.21 maximum absolute difference against the torch reference, where
vLLM's own kernel differs by 0.001. That is a logic error, not BF16 rounding, so the
speedup is meaningless.

I fixed the three defects the tests exposed (BLOCK_H not a power of two; `tl.dot` padding;
NaN from an empty softmax on masked heads) and it still did not reproduce the reference
correctly. Rather than keep iterating on a kernel with a demonstrated correctness bug,
`vllm_rlt/kernels/triton_attention.py` is restored to `8f9f817`, the previously validated
implementation, and the attempt is recorded here. Verified after the revert: rlt differs
from the torch reference by **2.4e-4** at r=8 and **4.9e-4** at r=128, the same order as
vLLM's 9.8e-4.

## Verification of the tree as it stands

| gate | result |
| --- | --- |
| full suite `pytest tests/` | **1360 passed, 0 failed**, 914 skipped |
| attention vs torch reference | 2.4e-4 / 4.9e-4 (BF16 rounding, matches vLLM's order) |
| ruff | clean |

The two retained optimisations from this line of work are unchanged and still validated:
V001 fused RMSNorm/RoPE (1.37x eager / 1.45x graph, token-identical) and V002 projection
fusion (1.09x under graphs, token-identical).

## Status per operator

| operator | vs vLLM | action |
| --- | --- | --- |
| MHA decode | parity to 1.2x faster | none |
| GQA decode | **1.5x / 3.0x / 3.5x slower at r8 / r32 / r128** | **open — root cause measured, fix not landed** |
| Prefill attention | 1.2-1.5x faster | none |
| RMSNorm | parity after V001 | none |
| RoPE | parity after V001, bitwise exact | none |
| SiLU-and-Mul | 3x slower at small token counts, host-bound | low value |
| KV write | 2x faster | none |
| Projections | parity after V002 | none |

## Next step for the GQA gap

The measurements say exactly what the fix must achieve: **one KV tile load and one
tensor-core dot per group**, i.e. `-6x` instructions at equal DRAM traffic. The failing
attempt shows where the difficulty is: the group axis must be padded to a power of two for
`tl.dot` while the group itself is 4 or 6, and heads outside the group must publish the
empty-softmax triple the merge step already understands.

A reliable path is to port vLLM's `_fwd_grouped_kernel_stage1` semantics line by line
rather than re-deriving the masking, and to gate it behind a correctness test that compares
against `torch_paged_attention` for every shipped geometry (16/16, 32/8, 48/8, 16/4) at
r=1..128 **before** measuring speed. Such a test is the missing piece that let a wrong
kernel post a good number.
