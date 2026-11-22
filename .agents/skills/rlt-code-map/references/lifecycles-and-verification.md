# Lifecycles and verification

A map is only useful if its claims can be checked. This file gives, per lifecycle
node, (a) the exact symbol to re-locate, (b) the cheapest way to observe it, and
(c) what that observation does **not** prove.

## Test harness facts you must know first

- `pytest` config: `pyproject.toml:44-49` — `testpaths=["tests"]`, `pythonpath=["."]`,
  markers `gpu` and `npu`.
- GPU tests are **skipped by default**: `tests/conftest.py` skips anything marked
  `gpu` unless `--run-gpu` is passed *and* `CUDA_VISIBLE_DEVICES` is set; same for
  `--run-npu` with `ASCEND_RT_VISIBLE_DEVICES` (`conftest.py:15-25`).
- `conftest.py:28-33` forces `torch.set_num_threads(1)` for the session.
- Consequence: a green CPU-only suite proves the state machine and the
  bookkeeping. It does **not** prove kernels, stream ordering, graph capture,
  overlap, or throughput.

```bash
# CPU-only (default, fast, safe on any machine)
python -m pytest -q

# Targeted
python -m pytest -q tests/test_kv_cache.py tests/test_split_prefill.py

# GPU, only with an explicitly selected device
CUDA_VISIBLE_DEVICES=0 python -m pytest -q --run-gpu tests/test_cuda_graph.py
```

## Node-by-node

Legend: **relocate** = symbol to grep for if the anchor moved; **observe** =
cheapest evidence; **does not prove** = the claim it cannot support.

### Request lifecycle

| Node | Relocate | Observe | Does not prove |
| --- | --- | --- | --- |
| Ingress validation and loop bounds | `StageEngine._add_request` (`engine/stage_engine.py:442`) | construct an `LLM` on CPU with a toy model (`entrypoints/cli.py:19` builds one) and send a bad prompt; expect `ValueError` before any allocation | that a valid request is scheduled promptly |
| Admission | `Scheduler._admit` (`core/scheduler.py:474`) | `tests/test_scheduler_selection.py`, `tests/test_prefill_livelock.py` | GPU-side timing of admission |
| Batch building | `Scheduler.take_phases` (`:704`) | `tests/test_mixed_execution.py`, `tests/test_split_prefill.py` | that the chosen batch is optimal |
| Loop-boundary count | `_planned_loop_count` (`:798`) | assert `ExecutionBatch.loop_count` after `build_execution_batch` (`worker/executor.py:16`) | anything about graph replay coverage |
| Exit decision | `_should_exit` (`:1232`), `_delayed_signal` (`:1247`) | `tests/test_engine_core_contract.py`, `tests/test_async_pipeline.py` | that a *trained* gate produces the same depth (that is accuracy, see `tests/test_gsm8k_accuracy.py`) |
| Finish and pump | `StageEngine._finish` (`engine/stage_engine.py:819`), `Scheduler.finish` (`core/scheduler.py:196`) | `tests/test_stage_engine.py`, `tests/test_engine.py` | the memory-ordering of any still-in-flight device work |
| Stale-result rejection | identity checks (`core/scheduler.py:1304`, `:1424`) | `tests/test_async_speculative_lifecycle.py` | cross-process ID reuse (needs an equivalent generation contract) |

### KV lifecycle

Candidate files: `tests/test_kv_cache.py`, `tests/test_prefix_growth.py`,
`tests/test_prepared_kv.py`, `tests/test_fp8_kv_cache.py`,
`tests/test_split_prefill.py`. Read the file before citing a specific test name.

| Node | Relocate | Observe | Does not prove |
| --- | --- | --- | --- |
| Allocation / growth budget | `KVCacheManager.allocate` (`core/kv_cache_manager.py:420`), `ensure_capacity` (`:463`), `required_blocks` (`:327`) | CPU test asserting `num_free_blocks`/`num_used_blocks` and per-plane tables | that the pages stay valid while a transfer is in flight |
| Metadata planning | `MetadataBuilder.build` (`worker/metadata.py:172`), `plan_attention_metadata` (`attention/metadata.py:48`) | inspect `KVLayout.rows`/`write_blocks`/`context_lengths` for a small batch; padding sentinels (`-1`, zero-length) are the interesting cases | kernel correctness |
| Write / attend | `_write_prepared` (`:848`), `_attend_prepared` (`:970`) | `tests/test_attention_module.py`, `tests/test_attention.py`, `tests/test_attention_64bit.py` | FP8 backends (`tests/test_fa4_fp8.py`, GPU) |
| Copy-on-exit | `finalize_token` (`:1053`), `finalize_many` (`worker/model_runner.py:1323`) | `exit_written` before, `mark_finalized` after; `tests/test_engine_core_contract.py` | ordering of the async `finalize_kernel` vs later consumers (GPU only) |
| Prefix reuse | `publish_prefix` (`:378`), `lookup_prefix` (`:407`) | `tests/test_prefix_growth.py` | that a hit is *cheaper* — measure it |
| Truncation (spec rollback) | `truncate_suffix` (`:481`) | `tests/test_speculative.py` (CPU paths) | GPU-owned frontier cases |
| Snapshot / restore | `snapshot` (`:625`), `restore` (`:656`) | preemption tests, `tests/test_kv_cache.py` | that a resumed request reproduces the exact token sequence (needs RNG, see below) |
| Transfer lease | `pin_transfer` (`:511`), `unpin_transfer` (`:517`) | `tests/test_transfer_connector.py` | real NIXL transport behavior |

### Submission lifecycle

| Node | Relocate | Observe | Does not prove |
| --- | --- | --- | --- |
| Slot rotation | `SubmissionQueue.begin/complete/drain` (`worker/submission_queue.py:26/40/44`) | `tests/test_submission_queue.py` | that rotating pools never wrap early under real stream timing |
| CPU prepare | `prepare_on_cpu` (`model_runner.py:851`) | `tests/test_input_preparation.py`, `tests/test_async_state.py` | device-side costs |
| GPU prepare | `prepare_on_gpu` (`:972`) | same files, plus `tests/test_fast_execution_plan.py` for the packed/static choices | copy bandwidth |
| Execute | `_execute_batch` (`:1124`) | `tests/test_ouro.py`, `tests/test_nanbeige.py`, `tests/test_huginn.py`, `tests/test_dory.py` compare against reference implementations in `tests/reference/` | speed |
| Event + readback | `model_runner.py:1266-1283`, `_readback_slot` (`:1170`) | `tests/test_async_pipeline.py`, `tests/test_cuda_graph.py` (GPU) | that the host copy is not on the critical path |
| Recycle / RNG | `_free_runner_slot` (`:1412`), `suspend` (`:1426`), `release` (`:1434`) | `tests/test_sampler.py` for RNG; preemption tests for slot reuse | async release ordering under graphs |
| Graphs | `RecurrentGraphs.select` (`worker/cuda_graph.py:251`) | `tests/test_cuda_graph.py`, `tests/test_async_speculative_graphs.py` (GPU only) | eager-path equivalence for shapes not captured |

### PD and serving

| Node | Relocate | Observe | Does not prove |
| --- | --- | --- | --- |
| Coordinator handshake | `PDEngine._message` (`pd/engine.py:184`), `_admit` (`:335`) | `tests/test_pd.py` — most of it runs without a real transport | multi-node behavior, NIXL bandwidth |
| Connector staging | `PDStageConnector.prefill_submitted` (`pd/connector.py:188`), `progress` (`:349`) | `tests/test_transfer_connector.py` | real KV-arrival timing |
| HTTP + SSE | `serving/server.py:95`, `serving/worker.py:146` | `tests/test_serving.py` | tail latency under real clients |
| Profiling lifecycle | `Profiler.start/stop/wait` (`profiling.py:218/230/246`) | `tests/test_profiling.py` | that traces contain the frames you expect |

## The three "equivalence" traps

1. **Token equality is not step equality.** A cheaper path may produce identical
   tokens with a different number of steps/loops; assert both `token_ids` and
   `exit_depths` (`request.py:44`, `RequestOutput.exit_depths` `request.py:218`).
2. **RNG belongs to the request.** Suspension keeps the generator, termination
   drops it (`model_runner.py:1426-1441`). A "resume" test that reseeds proves
   nothing about resuming.
3. **CPU kernels are Torch references.** Backend parity has to be checked on the
   device with the backend actually selected; see `tests/test_backend_selection.py`
   for which combinations are even legal, and `docs/runtime-history.md` for what
   was historically qualified.

## Proving something is not stale

Before asserting "the code does X":

```powershell
# 1. Does the symbol still exist where the map says?
Select-String -Path vllm_rlt\core\scheduler.py -Pattern 'def _admit'

# 2. Who calls it now?
Get-ChildItem -Recurse -File vllm_rlt -Include *.py |
  Select-String -Pattern '_admit\(|update_after_submission\(' |
  ForEach-Object { "$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }

# 3. Did the docs claim change?
git log --oneline -5 -- docs/design.md docs/runtime.md
```

If a `path:line` anchor and its symbol disagree, the symbol wins: update the map
(and say so) rather than the reader's expectation.
