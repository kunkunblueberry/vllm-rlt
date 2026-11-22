# Short-context GQA decode: the second path, and what it must not cost

Version V007. Branch `perf/fuse-norm-rope`, base commit `fcf12e8`. Host `lyx`, 8x
A100-SXM4-80GB (108 SMs), page 16, head_dim 128.

V006 closed the GQA gap for long contexts and left one measured defect: at context <= 512
the operator is 1.35-1.52x behind vLLM's single-pass kernel. This entry establishes why, what
a second path must be, and - the part worth reading first - which fixes are wrong.

## The gap is not what it first looked like

Two hypotheses were measured and rejected before anything shipped.

**Rejected: per-call allocation of the split partials.** Timed with per-call CUDA-event
synchronization, the shipped operator took 0.0722 ms at rows=8/length=128 while the identical
two launches from pre-allocated buffers took 0.0403 ms, which looked like ~0.031 ms of
allocator + free work. Caching the buffers then made the operator *slower* (0.0872 ms), and
the reason is a property of the measurement, not of the code: per-call event timing
synchronizes every iteration, so when the two kernels write and re-read the same reused
buffer, the readiness wait lands inside that iteration's window. An untouched, hoisted buffer
measured 0.0411 ms against 0.0585 ms for a freshly allocated one and 0.0626 ms for a reused
one. The "saved 0.031 ms" was that artifact, and the memoization was reverted.

**Rejected: inlining the merge for `splits == 1`.** Correct, and slightly slower
(0.1800 -> 0.1876 ms at 512/128). It also could not apply where it was needed: with the split
count derived from the batch shape, `splits` is never 1 at rows=8/32, so the branch never
fired in the cases that were short of performance.

**Kept: measure decode the way the loop runs it.** Back-to-back enqueues between one event
pair, since a decode step does not synchronize after attention. That gives the honest numbers
used below and confirms the gap is real, not measurement noise.

## What the gap actually is

GQA 48/8, BF16, page 16, back-to-back per-call time:

| context | rows | rlt (split pair) | vLLM unified | ratio |
| --- | --- | --- | --- | --- |
| 128 | 8 | 0.0834 | 0.0559 | 1.49x |
| 128 | 128 | 0.0848 | 0.0589 | 1.44x |
| 512 | 8 | 0.0838 | 0.0559 | 1.50x |
| 512 | 32 | 0.0842 | 0.0603 | 1.40x |
| 2048 | 8 | 0.0844 | 0.1159 | **0.73x (rlt faster)** |
| 2048 | 128 | 0.6512 | 0.6192 | 1.05x |

Splitting the KV is what pays for long context and what costs short context. At length 128
the split stage alone is 0.027 ms and the merge 0.015 ms: the work is trivial and the operator
is two launches where vLLM needs one. At 2048 the split is what keeps one row's serial walk
off the critical path, and rlt is faster than vLLM at rows=8.

## The second path, measured

A single-pass GQA kernel, written from the repo's own pieces - the tile loop, head mapping and
masking of `_paged_attention_group_kernel`, with the merge arithmetic applied in registers
instead of a second kernel. It is correct at every geometry (max abs diff <= 1.56e-2 against
the FP32 oracle, the same quantisation level the split path shows) and, measured back-to-back:

| context | rows | single-pass | split pair | vLLM | single / vLLM |
| --- | --- | --- | --- | --- | --- |
| 128 | 8 | **0.0317** | 0.0835 | 0.0551 | **0.57x** |
| 128 | 128 | **0.0536** | 0.0842 | 0.0590 | **0.91x** |
| 512 | 8 | **0.0323** | 0.0849 | 0.0556 | **0.58x** |
| 512 | 32 | **0.0547** | 0.0846 | 0.0601 | **0.91x** |
| 512 | 128 | 0.1744 | 0.1796 | 0.1686 | 1.03x |
| 2048 | 8 | 0.1248 | **0.0849** | 0.1161 | 1.07x |
| 2048 | 32 | 0.1692 | **0.1685** | 0.1770 | 0.96x |

So the two regimes are complementary and both are now better than vLLM at their own end:
single-pass wins through context 512 (up to 1.7x faster than vLLM), the split pair wins at
2048. That is the case for a real second path rather than further tuning of one kernel.

## What the dispatch must satisfy, and the design

The choice depends on a row's context length, and context lengths live on the device: the
CUDA-graph path fills `context_lengths` into a preallocated buffer
(`_copy_graph_inputs`), so a host-side `if max(lengths) > X` would either force a device sync
per step or be wrong under capture. The rule that follows is that the decision must be made
inside the kernel from a tensor the kernel already receives.

The design that meets it:

1. **One kernel, two exits.** Extend the grouped kernel with an `OUT` pointer and two
   constexprs. After the tile loop, a row whose length is `<= SINGLE_MAX` divides by its own
   normalizer and stores the result; any other row stores the split partials exactly as
   before. Same compile, same file, one new constexpr pair.
2. **Both passes are always launched.** Pass A (`SHORT=True`, one split) finishes short rows
   and returns for long ones; pass B (`SHORT=False`, `SPLITS` splits) returns immediately for
   short rows and does the split work for long ones. Because the work per row is
   length-determined, the kernel launches stay shape-determined, so a captured graph stays
   valid and no host value is ever read.
3. **Pass A must run first.** Pass B's merge writes every row it owns; if it ran first, a
   short row's correct output would be overwritten. Ordering is part of the contract.
4. **The long path keeps the grouped kernel's own split arithmetic** - tile-aligned equal
   parts of the covered range, including the `WINDOW` case - so a row's result still does not
   depend on the batch it ran in, which `test_gqa_decode_is_batch_invariant` pins.

Estimated cost of the dispatch: a pass that does nothing still costs its launch, so a batch of
`R` rows pays two launches of `(R, kv_heads, splits)` instead of one plus the merge. That is
the same launch count as the current split pair, which is why the mixed case does not regress;
it is stated as an estimate because the mixed-length measurement is the one piece not yet
taken.

## Status

The prototype above is measured; the dispatch is designed and **not shipped**. An
implementation attempt was abandoned mid-edit after the file was left inconsistent, and the
working tree was restored to `fcf12e8` rather than shipping an unverified kernel. What remains
is mechanical against the design above:

- add `OUT`, `SINGLE`, `SHORT` to `_paged_attention_group_kernel`, branch the store, and keep
  `WINDOW` handling inside the long branch;
- add `_launch_grouped` that orders pass A before pass B and then merges;
- gate with `tests/test_gqa_decode.py` (all geometries x rows x lengths 1/33/512/2048 plus
  windowing and zero-length rows), which already covers both dispatch sides at the threshold;
- re-measure the table above plus a mixed-length batch (some rows 128, some 2048) to confirm
  the launch-count argument.

## Reproduce

```bash
# honest per-call timing (back-to-back, one sync)
python /root/kun/rlt-perf/batched_short.py

# the single-pass prototype, with the correctness column
python /root/kun/rlt-perf/single_pass.py

# why the allocation hypothesis was wrong
python /root/kun/rlt-perf/why_slower.py
```
