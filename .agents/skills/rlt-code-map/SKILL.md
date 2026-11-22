---
name: rlt-code-map
description: Navigate vllm-rlt as an executable map before changing, reviewing or profiling it — stage/loop layering, who calls whom, which component owns each piece of state, the request/KV/submission lifecycles, and where the performance-critical paths and measurement entry points are. Use when onboarding to this repository, tracing a call chain, deciding which module owns a behavior, or locating the code behind a latency number. Not a substitute for measurement (rlt-perf-opt) or code review (rlt-refactor).
---

# vllm-rlt Code Map

This is a navigation contract, not a source encyclopedia. It answers: **who calls
whom, where data comes from, who owns each state, how state is transformed, who
consumes it, where its lifetime ends, and which boundaries must not be crossed.**

Use it to get from a symptom, a PR diff or a profile flame to the right module
without re-reading the tree. Anchors are plain `path:line` text: they name the
symbol's current location, and they are cheap to re-locate (see
[Refreshing the map](#refreshing-the-map)) — prefer re-locating a symbol over
trusting a stale number.

Load order: this file (one screen) → the one [reference](#references) for the
layer you are changing → `docs/design.md` / `docs/runtime.md` only for the intent
behind a contract you are about to modify.

## The system model in one paragraph

**A stage is the deployment and resource boundary; a loop is the scheduling and
execution unit inside a stage** (`docs/unified_execution.md`). One stage is one
device and one `StageEngine` in one spawned worker process. Requests move through
phases (`WAITING → PREFILL → RECURRENT … → CODA …`), and the scheduler may
re-group different requests at different loop depths into one recurrent batch.
Three operations exist: `PRELUDE` (prompt token → hidden), `RECURRENT` (hidden →
hidden at a given depth), `CODA` (hidden → logits → one token). Prompt prefill runs
the model's full depth; each decode token advances one loop at a time; early exit
copies KV into the skipped depth planes. Everything else in the repository is
either scheduling that progression, moving its state to/from the device, or
serving it.

## Layer ownership

| Layer | Owner (symbol at anchor) | Owns | Must not own |
| --- | --- | --- | --- |
| Ingress | `entrypoints/cli.py:55`, `entrypoints/llm.py:14`, `entrypoints/serve.py:72`, `entrypoints/pd_serve.py:90` | argument/CLI parsing, tokenizer, HTTP app, process count | engine state, KV, scheduling |
| Client boundary | `engine/llm_engine.py:19` `StageClient` | the pipe to the worker, external request→wire handling, blocking `step()` calls | any device tensor or `Request` |
| Stage orchestration | `engine/stage_engine.py:45` `StageEngine` | the per-step handoff: schedule → submit → collect → apply; pending tickets; exit-signal carry; connector notifications | queue order, batch composition, kernel choice |
| Scheduling | `core/scheduler.py:132` `Scheduler` + `core/scheduling_policy.py:184` | request registry, per-phase queues, admission/allocation budget, progress application, exit decisions | device tensors, tensor shapes |
| Logical KV | `core/kv_cache_manager.py:169` `KVCacheManager` | physical page pool, per-request/per-depth block tables, validity bookkeeping, prefix reuse, leases, snapshot/restore | kernel launches' semantics, attention math |
| Execution prep | `worker/model_runner.py:153` `ModelRunner` | device tensors, staging buffers, streams/events, graphs, readback | request logical progress, KV capacity policy |
| Execution dispatch | `worker/executor.py:87` `Executor` / `:141` `ModelRuntime` | mapping `SchedulerOutput` → `ExecutionBatch` → model call; graph vs eager choice | scheduler policy, output delivery |
| Attention | `attention/backend.py:40` `AttentionBackend` (+`BackendCapabilities`) | capability declaration and kernel routing for a given backend | block allocation, KV lifetime |
| Model | `models/ouro.py`, `nanbeige.py`, `huginn.py`, `dory.py` | architecture ops (`prelude_prepared` / `recurrent_prepared` / `coda_prepared`) | scheduling, allocation, streams |
| Transport | `pd/connector.py:49` + `pd/transport.py:115` | KV/hidden staging, wire submission, lease release ordering | scheduling decisions, decode math |
| Profiling | `profiling.py:207` `Profiler` + `profiling_artifacts.py:405` | trace capture/export lifecycle | any inference decision |

Cross-layer seams that are easy to get wrong:

- `StageEngine` is the only component that turns a `SchedulerUpdate`
  (`core/scheduler.py:90`) into side effects — runner finalization, connector
  notification, request completion (`engine/stage_engine.py:795`).
- `ModelRunner.submit` is the **single submission boundary**
  (`worker/model_runner.py:1207`). Everything device-facing for one batch happens
  inside it; callers get a ticket, not a tensor.
- `Scheduler` decides; `PreemptionManager` (`engine/preemption.py:13`) only
  executes the state preservation it is asked for.

## The spine: one `StageEngine.step()`

`engine/stage_engine.py:579` (`step`) → `:590` (`_step`). Two shapes exist; the
synchronous one is the reference reading order.

1. **`_step_async` or synchronous branch** — `:934` (async), `:590` (sync, the
   path below).
2. **Choose work** — `_schedule()` `:636` → `Scheduler.schedule`
   `core/scheduler.py:949` → policy `SchedulingPolicy.schedule`
   (`core/scheduling_policy.py:50/60/97/139/165`). Admission runs from inside the
   policy (`Scheduler._admit` `core/scheduler.py:474`); batch assembly is
   `Scheduler.take_phases` `:704`.
3. **Commit the batch to the device** — `Executor.submit`
   `worker/executor.py:98` → `build_execution_batch` `:16` (SchedulerOutput →
   immutable `ExecutionBatch`/`StateSpan`, `request.py:153`/`:126`) →
   `ModelRunner.submit` `worker/model_runner.py:1207`.
4. **Host prepare** — `prepare_on_cpu` `:851` → `_prepare_cpu_inputs` `:881`
   (snapshot positions/depths, choose static workspace, or
   `AsyncState.prepare` `worker/async_state.py:203`, or
   `MetadataBuilder.build` `worker/metadata.py:172`).
5. **Device prepare** — `prepare_on_gpu` `:972` (UVA/pinned upload, packed
   metadata expansion, `_prepare_forward` `:1011` binds device views).
6. **Execute** — `_execute_batch` `:1124` → `ModelRuntime.forward`
   `worker/executor.py:172` → model `*_prepared` → per layer
   `KVCacheManager._write_prepared` `core/kv_cache_manager.py:848` then
   `_attend_prepared` `:970` → `AttentionBackend.attend_metadata`
   `attention/backend.py:50`.
7. **Mark completion** — `compute_done` event recorded, per-request completion
   event pushed into `ModelRunner.events`, submission queued
   (`worker/model_runner.py:1266-1286`).
8. **Collect** — either immediately (`Submission.collect`,
   `worker/model_runner.py:131`) or, when async, one round later in
   `_step_async` (`engine/stage_engine.py:936-937` `_collect_coda`).
9. **Apply** — `Scheduler.update_from_output` `core/scheduler.py:1285`
   (sync) or `update_after_submission` `:1409` + later
   `_deliver_coda` `engine/stage_engine.py:919` (async) → `SchedulerUpdate` →
   `StageEngine._apply_update` `:795` (finalize early exits, connector
   notifications, `_finish` `:819`).

Nothing in steps 3–7 may change logical request progress; nothing in step 9 may
launch work.

## Three lifecycles

Each node lists its source entry and the state it owns. Verification pointers are
in [references/lifecycles-and-verification.md](references/lifecycles-and-verification.md).

### Request lifecycle

```
add_request ─► WAITING ─► PREFILL (or PREFILLING_PRELUDE/RECURRENT) ─► RECURRENT×N ─► CODA ─► [PRELUDE]… ─► FINISHED
                              │                                                     │
                              └── admission + KV allocation (_admit)                └── exit decision (_should_exit)
```

- Phase enum: `request.py:11`; ingress validation and loop bounds:
  `engine/stage_engine.py:442`; registration: `Scheduler.add_request`
  `core/scheduler.py:176`.
- Phase transitions are written only by `Scheduler` (`enqueue`
  `core/scheduler.py:182`), except the PD import path
  (`StageEngine.accept_import` `engine/stage_engine.py:751`) and
  transfer suspension (`suspend_for_transfer` `core/scheduler.py:186`).
- Exit: `_should_exit` `core/scheduler.py:1232` reads
  `request.remaining_probability` and `exit_threshold`; delayed/async lookahead
  uses `needs_exit_signal` `:1262` and `_delayed_signal` `:1247` with scores
  carried in `StageEngine._pending_exit_signals` (`engine/stage_engine.py:353`).
- End of life: `Scheduler.finish` `core/scheduler.py:196` frees KV, drops tensors
  and removes the request; device-owned resources are released separately by
  `ModelRunner.release` `worker/model_runner.py:1434` (see the submission
  lifecycle — the scheduler can free logical pages before the device event is
  reached).

### KV lifecycle

```
allocate ─► plan metadata ─► write (per layer) ─► attend ─► finalize (copy-on-exit) ─► truncate/prefix publish ─► free
```

- `KVCacheManager.allocate` `core/kv_cache_manager.py:420` (page count
  = `pages × storage_depths`, depth planes for `LAST_EXITED`), growth
  `ensure_capacity` `:463`, `required_blocks` `:327` for budgets.
- Metadata planning: `MetadataBuilder.build` `worker/metadata.py:172` →
  `KVLayout` `:72` (`rows`, `write_blocks`, `write_offsets`, attention views);
  device binding by `bind` `:278` or UVA staging `Workspace.store_metadata`
  `worker/buffers.py:96` / `RoutingBuffers` `worker/async_state.py:18`.
- Write/attend per layer: `core/kv_cache_manager.py:848` / `:970`, called from
  each model's attention block (e.g. `models/ouro.py:194-195`). Under CUDA graphs
  the same two methods are replaced by `worker/cuda_graph.py:108` / `:135`.
- Early-exit copy-on-exit: `finalize_many` `worker/model_runner.py:1323`
  (`finalize_kernel` `kernels/routing.py:148`) or `finalize_token`
  `core/kv_cache_manager.py:1053`; bookkeeping `mark_finalized` `:594`.
- Reuse: `publish_prefix` `:378` / `lookup_prefix` `:407` / `poll_prefixes`
  `:368`; speculative rollback `truncate_suffix` `:481`; preemption
  `snapshot` `:625` / `restore` `:656`; transfer leases `pin_transfer` `:511`.
- End of life: `free` `:501`. A transfer lease must outlive any remote writer —
  `release_transfer` `engine/stage_engine.py:733` is the only path that may
  unpin after a completed transfer.

### Submission lifecycle

```
begin ─► prepare_on_cpu ─► prepare_on_gpu ─► execute ─► compute_done event ─► readback (optional) ─► collect ─► retire
```

- Slot/rotation: `SubmissionQueue.begin/complete/drain`
  `worker/submission_queue.py:26/40/44`; readback leases `_readback_slot`
  `worker/model_runner.py:1170`; results are host copies plus borrowed
  `ExecutionBatch` (`worker/output.py:31`).
- Per-request completion event: `model_runner.events[request_id]`
  (`worker/model_runner.py:1284`); consumed by scheduler progress checks
  (`request_events=` on `update_from_output`/`update_after_submission`) and by
  `_free_runner_slot` `:1412`, which synchronizes before a slot is recycled.

## Boundary rules (do not cross these)

1. **Policy vs mechanism.** Eligibility, ranking, victim choice and queue order
   live in `Scheduler`/policy; snapshot/restore lives in `PreemptionManager`;
   `Scheduler.suspend_callback`/`resume_callback` (`core/scheduler.py:161-162`)
   are the only channel.
2. **Logical KV vs device operations.** `KVCacheManager` owns allocation,
   references, block tables and validity; it does not own streams, events or
   kernel selection. `ModelRunner` owns tensors and ordering; it must not decide
   what is valid.
3. **One authoritative owner per mutable state.** Queues and logical progress:
   `Scheduler`. Device tensors/events/slots: `ModelRunner`. Page pool: cache
   manager. Transport leases: connector. RNG: `RngRegistry` (`worker/sampler.py:84`),
   shared by the CODA and speculative paths through `StageEngine.rng`
   (`engine/stage_engine.py:303`) — never re-seed a resumed request.
4. **Identity beats request ID.** Results are matched with
   `self.requests.get(rid) is not request` (`core/scheduler.py:1304`,
   `engine/stage_engine.py:722`, `:735`, `:757`) because IDs are reused after
   cancellation.
5. **Submission ≠ completion ≠ validity.** A submitted loop has an event, not a
   token; delivered tokens come from `collect()`; prefix publication requires
   completed writes on every plane (`core/kv_cache_manager.py:378`).
6. **`position` is not `depth`.** `position` = prompt length + outputs − 1
   (`request.py:118`); `depth` is the recurrent loop index (`request.py:137`).
   CODA executes no recurrent loop and uses depth 0 for boundary-layer
   addressing while `request.loops_done` keeps the real exit depth
   (`worker/executor.py:41-47`).
7. **Backends declare capabilities; callers read them.** Route on
   `BackendCapabilities` (`attention/backend.py:18`), never on a version string.
8. **Test-only adapters are not the hot path.** `KVCacheManager.write` `:827`
   and `attend` `:954` rebuild a descriptor per call and are used by tests only;
   production goes through `_write_prepared`/`_attend_prepared`. Optimizing or
   reading `write()` as if it were the engine path is a false lead.
9. **Async paths keep their own invariants.** In async scheduling the CPU
   frontier intentionally lags device state (`worker/cuda_graph.py:304-306`), and
   `_step_async` must submit round *r* before collecting *r−1*
   (`engine/stage_engine.py:989-1008`). Requiring sync points "for clarity"
   destroys the overlap this codebase exists for.

## Performance surface (inventory only)

Full detail, including where each cost is paid and which metric to read, is in
[references/performance-surface.md](references/performance-surface.md).

Rough cost hierarchy per token, cheapest reading first: kernel math inside one
recurrent loop → metadata preparation and copies per loop → host↔device sync
and readback → scheduling/host bookkeeping → transport (PD only).

| Area | Entry point |
| --- | --- |
| Recurrent core (per loop, per layer) | `worker/model_runner.py:1124` → model `recurrent_prepared` → `core/kv_cache_manager.py:848/970` → `attention/backend.py:50` |
| Attention kernels | `kernels/paged_attention.py:204`, `kernels/triton_attention.py:176/386`, `kernels/flash_attention.py:34` |
| CUDA graphs / launch overhead | `worker/cuda_graph.py:240` (recurrent), `:423` (coda), `:536` (prelude) |
| Metadata staging and copies | `worker/buffers.py:96/130/155`, `worker/async_state.py:65`, `worker/tensors.py:35/282` |
| Streams and sync | `worker/model_runner.py:222-246` (streams), `:1266` (events), `:1170` (readback), `:1412` (slot recycle) |
| Async CPU/GPU pipelining | `engine/stage_engine.py:934` (`_step_async`), `:882` (`_step_async_speculative`) |
| Speculation | `worker/speculative.py:186`, `worker/async_speculative.py:336` |
| Scheduling overhead | `core/scheduler.py:474` (admission), `:704` (batch build), `:1285`/`:1409` (apply) |
| PD transport | `pd/connector.py:349`, `pd/transport.py:182/203` |
| Measurement (not a cost source) | `profiling.py:207` `Profiler`, `docs/profiling.md`, `benchmarks/cdb_runtime.py`, `benchmarks/preemption.py`, `benchmarks/gsm8k.py` |

Do not conclude a performance story from this map. Measure with
[rlt-perf-opt](../rlt-perf-opt/SKILL.md) (unprofiled baseline first, then
ops-only/full profiles), then use these anchors to attribute what the profile
shows.

## Conflicts and traps when reading this repository

- The authoritative *intent* lives in `docs/` (`design.md`, `runtime.md`,
  `scheduler.md`, `unified_execution.md`, `async-speculative.md`,
  `split-prefill-wavefront.md`); the authoritative *behavior* is the code. When
  they disagree, read the code and report the doc as stale rather than "fixing"
  behavior from the doc.
- `docs/runtime-history.md` and `docs/experiments/V00*.md` keep dated
  measurements. They do not qualify the current revision.
- Feature gating is dense and deliberate at `StageEngine.__init__`
  (`engine/stage_engine.py:48-360`): model caps (`requires_boundary_kv`,
  `supports_cuda_graphs`, `supports_async_scheduling`, `fixed_loop_count`),
  layout, exit mode, backend capabilities and memory reserve are cross-checked
  there. Read that block before assuming a combination is legal.
- `SHARED` layout keeps one plane and overwrites it; `LAST_EXITED` keeps one
  plane per depth (`core/kv_cache_manager.py:239`, `_plane` `:683`). Any
  arithmetic over block tables must go through `_plane(depth)` /
  `plane_block_tables` `:550`.
- Huginn/Dory (`models/huginn.py`, `models/dory.py`) accept only a narrow subset:
  boundary-KV, no prefix caching, restricted speculation, sync execution for
  Huginn's caps. `models/dory.py:266` (`kv_cache_layout`) and
  `requires_boundary_kv` drive those rejections.
- Three processes exist in the default `LLM` path, not one: the caller, the
  spawned `StageWorker` (`engine/worker.py:23/207/251`). A stack trace from the
  caller's `step()` does not contain scheduler code.

## Refreshing the map

The map was produced by symbol outlining plus targeted reads. Re-derive the
outline instead of trusting memory:

```powershell
# Symbol outline for the whole package (cheap, no imports)
Get-ChildItem -Recurse -File vllm_rlt -Include *.py |
  Where-Object { $_.FullName -notmatch '__pycache__' } | Sort-Object FullName |
  ForEach-Object { "### " + $_.FullName; Select-String -Path $_.FullName `
    -Pattern '^(class |def |    def |async def |    async def )' |
    ForEach-Object { "{0,4}: {1}" -f $_.LineNumber, $_.Line } }
```

```powershell
# Locate one symbol you believe you know
Get-ChildItem -Recurse -File vllm_rlt -Include *.py |
  Select-String -Pattern 'def build\(' | ForEach-Object { "$($_.Path):$($_.LineNumber)" }
```

When you change a contract in this map's scope (a new owner, a new phase, a new
stage of a lifecycle), update this file or the matching reference in the same
change, and say which anchors moved.

## References

| Reference | Read when |
| --- | --- |
| [processes-and-ingress.md](references/processes-and-ingress.md) | Entry points, process/thread topology, IPC message kinds, PD handoff sequence, HTTP serving path, blocking points. |
| [scheduling.md](references/scheduling.md) | Admission, batch composition, phase policies, mixed/WCPB prefill, exit policies, preemption, speculative scheduling. |
| [kv-and-execution.md](references/kv-and-execution.md) | KV layout/planes, metadata builders, attention dispatch, UVA staging, graphs, streams, finalize copy-on-exit. |
| [performance-surface.md](references/performance-surface.md) | Locating where a latency/throughput number comes from, and which knobs exist (without changing behavior). |
| [lifecycles-and-verification.md](references/lifecycles-and-verification.md) | Proving a claim about a lifecycle node: CPU-only checks, GPU markers, targeted test files, and what each does not prove. |
