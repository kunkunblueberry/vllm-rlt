# Performance surface: where the time is paid

This file locates cost, it does not certify it. Every claim about speed needs a
measurement under matched conditions ([rlt-perf-opt](../../rlt-perf-opt/SKILL.md):
unprofiled baseline first, then ops-only and full profiles).

## Cost hierarchy per output token

Ordered by typical magnitude for a single-request or small-batch decode step. The
point of the order is to stop optimizing a lower row while a higher one dominates.

1. **Recurrent core math** — `depth` iterations × physical layers × attention.
   `worker/model_runner.py:1124` → `models/*.py` `recurrent_prepared` →
   `core/kv_cache_manager.py:848` (write) + `:970` (attend) →
   `attention/backend.py:50` → kernel.
   - Multi-loop submissions (`loop_count > 1`, `core/scheduler.py:798`) keep the
     hidden state on device between loops; only the submission boundary syncs.
2. **Kernel choice and shapes** — packed vs per-query vs windowed vs graph.
   `kernels/paged_attention.py:204` (Triton), `kernels/flash_attention.py:34`
   (FA), `kernels/triton_attention.py:176`/`:386`, `worker/cuda_graph.py:251`
   (graph bucket selection). Head-dim/block-size/packed support decide which of
   these is even reachable (`attention/backend.py:85-160`).
3. **Metadata preparation and copies per loop** —
   `MetadataBuilder.build` (`worker/metadata.py:172`), `Workspace.store_metadata`
   (`worker/buffers.py:96`), packed expansion (`kernels/metadata.py:8`), UVA
   upload (`worker/buffers.py:130`, `worker/tensors.py:35`). This is pure per-step
   overhead proportional to rows × layers, and it is what static buffers, UVA and
   `AsyncState` exist to reduce.
4. **Host↔device synchronisation and readback** — `compute_done` event +
   non-blocking host copy (`worker/model_runner.py:1266-1283`), lease wait
   (`:1170`), `Submission.collect` (`:131`), slot recycle `event.synchronize()`
   (`:1417`). Each of these is a place where the GPU can go idle.
5. **Scheduling and host bookkeeping** — admission
   (`core/scheduler.py:474`), batch assembly (`:704`), result application
   (`:1285`/`:1409`), per-request Python loops in
   `worker/model_runner.py:881-970`. Cost scales with `max_num_seqs` and
   `max_num_batched_tokens`, not with model size.
6. **Process/IPC and serving overhead** — `StageClient` pipe polling
   (`engine/llm_engine.py:256`), tokenizer and SSE framing
   (`serving/worker.py:146`), PD coordinator polling (`pd/engine.py:441`).
7. **Transport (PD only)** — descriptor build (`pd/transport.py:29/66/95`),
   submit/poll (`:182`/`:203`), backpressure at `max_inflight_bytes` (`:185`).

## Existing knobs (configuration, not behavior changes)

All in `vllm_rlt/config.py` unless noted; CLI mirrors them via
`vllm_rlt/entrypoints/runtime_args.py:12`.

| Knob | Anchor | Effect |
| --- | --- | --- |
| `ExecutionConfig.cuda_graphs` | `config.py:147` | replay recurrent core / coda / prelude instead of eager launches (`worker/cuda_graph.py`) |
| `cuda_graph_max_batch_size`, `cuda_graph_max_graphs`, `cuda_graph_memory_reserve_bytes` | `config.py:150-153` | graph bucket ladder and reserved memory (`cuda_graph.py:22`); too small → eager fallbacks (`:251-285`) |
| `async_scheduling` | `config.py:141` | submit round *r* before collecting *r−1* (`engine/stage_engine.py:934`); requires `ouro_delayed`/`random_lookahead`/`trace` or a fixed-loop model (`stage_engine.py:195-209`) |
| `multi_stream` | `config.py:144` | separate core/boundary/copy/readback streams (`model_runner.py:222-246`) |
| `static_buffers`, `pad_to_power_of_two` | `config.py:145-146` | reuse host/device buffers; padding trades compute for graph-friendly shapes (`model_runner.py:323-349`, `:371`) |
| `use_uva` | `config.py:155` | mapped pinned memory instead of staging copies (`worker/tensors.py:23`, `buffers.py:82`) |
| `packed_prefill` | `config.py:162` | packed prefill metadata + kernel path; `None` resolves at construction (`engine/stage_engine.py:254-260`) |
| `CacheConfig.layout` / `block_size` / `num_blocks` / `kv_cache_memory_bytes` / `gpu_memory_utilization` | `config.py:11-25` | plane count, block granularity, capacity (`core/memory.py:128`) |
| `CacheConfig.kv_cache_dtype="fp8_e4m3"` | `config.py:25` | halves KV bytes; needs backend support or a calibrated scale (`engine/stage_engine.py:248-253`) |
| `CacheConfig.incremental_allocation`, `watermark_ratio` | `config.py:20-22` | allocation granularity and admission headroom |
| `SchedulerConfig.max_num_seqs`, `max_num_batched_tokens`, `prefill_chunk_size` | `config.py:57-64` | batch row/token budget, prefill chunking |
| `SchedulerConfig.mode`, `policy`, `max_prefill_batches_before_decode`, `min_coda_batch_size` | `config.py:60-66` | selection order and fairness (`core/scheduling_policy.py`) |
| `SchedulerConfig.wavefront_chunked_prefill` | `config.py:68` | WCPB staged prefill (`core/scheduler.py:859`) |
| `SpeculativeConfig.*` | `config.py:186` | draft/target depth and `num_speculative_tokens` |
| `attention_backend` | `engine/stage_engine.py:226` | `auto` picks by capability priority (`attention/backend.py:113`) |

Illogical combinations are rejected at construction (`engine/stage_engine.py:48-360`),
so read that block before assuming a knob is independent.

## Measurement entry points

| Tool | Where | Notes |
| --- | --- | --- |
| Engine profiler | `vllm_rlt/profiling.py:207` `Profiler`, `:63` `ProfileSession` | owned by the engine; start/stop must run on the engine thread |
| CLI flags | `docs/profiling.md` (`--profile`, `--profile-activities`, `--profile-record-shapes`, `--profile-with-stack`, `--profile-memory`, `--profile-wait/warmup/active/repeat`) | ops-only = default activities, full = shapes+stack+memory |
| Python/HTTP control | `LLM.start_profile` (`entrypoints/llm.py:103`), `serving/server.py:184` | `scheduled=True` uses the configured step schedule |
| Artifacts | `profiling_artifacts.py:375` `export_profile`, `:405` `ProfileArtifactWorker` | export/compression runs off the engine thread; `artifacts_complete` can be false with a job error |
| Benchmarks | `benchmarks/cdb_runtime.py` (continuous-batching runtime), `benchmarks/preemption.py`, `benchmarks/gsm8k.py`, `benchmarks/kernels/silu_and_mul.py` | reuse these rather than adding a new harness (see rlt-perf-opt) |

## Attribution map: profile row → anchor

| What you see | Look here first |
| --- | --- |
| Many small `index_put`/copy kernels between layers | `_store_layer` (`core/kv_cache_manager.py:891`) and `_write_prepared` `:848` |
| Attention kernel variants alternating | dispatch in `attention/backend.py:50-66`, packed gating `worker/metadata.py:185` |
| Long gaps on core stream | event dependency in `worker/model_runner.py:1234-1238`, `_wait_for_requests` `:1196` |
| Repeated `cudaMemcpy`/`tensor_transfer_kernel` | `worker/buffers.py:130`, `worker/tensors.py:67`, `worker/async_state.py:54` |
| Eager fallback after warmup | `RecurrentGraphs.select` (`worker/cuda_graph.py:251`), `_cache_full` `:53` |
| Host `synchronize` inside a step | `Submission.collect` (`model_runner.py:131`), `_readback_slot` `:1170`, `_free_runner_slot` `:1417` |
| Idle device with queued-looking requests | scheduler returned nothing: `take_phases` (`core/scheduler.py:704`), capacity checks `:750`, admission `:474` |
| Time between tokens not in any kernel | client/pipe polling (`engine/llm_engine.py:256`), serving tick (`serving/worker.py:234`) |
| PD stalls | `can_schedule` (`pd/connector.py:175`), `progress()` `:349`, `max_inflight_bytes` (`pd/transport.py:185`) |
| Early-exit cost spike | `finalize_many` (`model_runner.py:1323`) — copies per exited request × skipped planes |

## Cautions

- A profile changes the thing you are measuring. Report unprofiled throughput and
  latency separately; never quote profiled latency as the result.
- CPU-only runs (`pytest` without `--run-gpu`) exercise the state machine with
  synchronous no-ops; they say nothing about overlap or kernel time
  (`tests/conftest.py`).
- `write()`/`attend()` in `KVCacheManager` are test adapters; timing them
  measures descriptor rebuilding, not the engine path.
- Historical numbers in `docs/runtime-history.md` and `docs/experiments/V00*.md`
  do not qualify the current revision.
- One measurement cannot establish a global optimum; state the tested
  configuration and the tradeoff accepted (`docs/developer-guide.md`).
