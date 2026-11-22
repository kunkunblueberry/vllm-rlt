# Processes, ingress and the serving path

Answers: what process am I in, who owns the engine, how does a request get in, and
what can stall the loop. Anchors are `path:symbol-line`.

## Entry points

Console scripts (`pyproject.toml:33-35`):

| Command | Module entry | Builds |
| --- | --- | --- |
| `vllm-rlt` | `vllm_rlt/entrypoints/cli.py:55` `main` | `LLM(...)` then `generate()` |
| `vllm-rlt-serve` | `vllm_rlt/entrypoints/serve.py:72` `main` | `create_app(load_engine)` + tokenizer |
| `vllm-rlt-pd-serve` | `vllm_rlt/entrypoints/pd_serve.py:90` `main` | `create_app(load_engine)` over a `PDEngine` |

Library entry points: `LLM` (`vllm_rlt/entrypoints/llm.py:14`), `StageClient`
(`vllm_rlt/engine/llm_engine.py:19`), `PDEngine` (`vllm_rlt/pd/engine.py:54`).
Shared CLI → config translation: `add_runtime_args` (`vllm_rlt/entrypoints/runtime_args.py:12`)
and `runtime_configs` (`:103`); profiling flags `add_profile_args`/`profile_config_from_args` (`:151`/`:173`).

## One local stage: object graph

```
caller process                          spawned worker process
──────────────                          ──────────────────────
LLM ─► StageClient ──(mp Pipe)──────► StageWorker.run
        (add/abort/control)            └─► StageEngine.step()
                                             ├─ Scheduler ── policy
                                             ├─ KVCacheManager
                                             ├─ ModelRunner (streams, graphs, buffers)
                                             ├─ SpeculativeRunner (optional)
                                             ├─ Executor ─► ModelRuntime ─► model
                                             └─ connector (optional: PD)
```

- `LLM.__init__` (`vllm_rlt/entrypoints/llm.py:17`): model load plan
  (`:33`), `StageClient(...)` (`:40`).
- `StageClient.__init__` (`vllm_rlt/engine/llm_engine.py:26`) → `start_stage_worker`
  (`vllm_rlt/engine/worker.py:251`): `spawn` context (`:255`), `Pipe` (`:256`),
  `Process(target=worker_main)` (`:257-262`).
- `worker_main` (`vllm_rlt/engine/worker.py:207`) → `StageWorker`
  (`:23`): loads the model, constructs `StageEngine` (`:67`), optional connector
  factory (`:68-72`), then `send("ready")` (`:89`). The child sets
  `torch.set_num_threads(1)` (`:38`).
- The child's loop is `StageWorker.run` (`vllm_rlt/engine/worker.py:179`):
  drain ≤`max_control_messages` (`:181-184`) → `connector.progress()` (`:190`) →
  `engine.step()` (`:194`) → `time.sleep(0.001)` when idle (`:201`).
- There is no thread pool in the worker: one thread owns the engine, the KV pool,
  all device streams and the profiler. `start_profile`/`stop_profile`/`profile
  step` must run on that thread (`docs/profiling.md`).

## IPC message kinds

Parent → child (`StageClient.send`, `vllm_rlt/engine/llm_engine.py:103`); handler
`StageWorker.command` (`vllm_rlt/engine/worker.py:94`):

| kind | sent at | handled at |
| --- | --- | --- |
| `control` | `engine/llm_engine.py:168` | `engine/worker.py:96` (method table `:98-106`) |
| `add` | `engine/llm_engine.py:213` | `engine/worker.py:156` |
| `abort` | `engine/llm_engine.py:237` | `engine/worker.py:163` |
| `shutdown` / `stop` | `engine/llm_engine.py:283` | `engine/worker.py:147` / `:165` |
| PD kinds `connect/reserve/prefill/cancel/cancel_safe/commit/ack` | `pd/engine.py` (see below) | routed to the connector first (`engine/worker.py:154`) → `PDStageConnector.command` (`pd/connector.py:104`) |

Child → parent (`StageWorker.send`, `vllm_rlt/engine/worker.py:91`): `ready` (`:89`), `reply`
(`:135`/`:143`), `output` (`:174`), `stopped` (`:204`), `fatal` (`:233`),
`connected` (`pd/connector.py:110`).

`StageClient.step` (`vllm_rlt/engine/llm_engine.py:254`) sends nothing: it pulls
queued `output` messages via `poll` (`:112`) and, when a caller asks for progress
with no output ready, blocks in `channel.poll(0.01)` (`:256`). That 10 ms is the
client-side latency floor for offline `LLM` stepping.

## PD (prefill/decode disaggregation)

Topology: coordinator (`PDEngine`, `vllm_rlt/pd/engine.py:54`) in the host process
plus one `StageClient`/`StageWorker` per role per device
(`vllm_rlt/pd/engine.py:124-150`). Roles come from `StageConfig`
(`vllm_rlt/engine/stage_config.py:7`) — a prefill-only stage does not produce
tokens and must export every request (`engine/stage_engine.py:381-390`).

One request, in order:

1. `PDEngine.add_request` (`pd/engine.py:276`) → `Pending` (`:331`) →
   `_admit` (`:335`) selects one P and one D peer by credit (`:350-367`) and sends
   `reserve` to **D** (`:384`).
2. D: `PDStageConnector.command` (`pd/connector.py:104`) receives `reserve` with `receiving=True`
   → `StageEngine.reserve_transfer` (`engine/stage_engine.py:679`): `_add_request`
   (`:689`), `Scheduler.reserve_transfer` (`core/scheduler.py:422`), phase
   `RECEIVING` (`engine/stage_engine.py:707-709`) → notify `reserved`
   (`pd/connector.py:291`).
3. P: coordinator sends `prefill` with D's tables/slot/cached tokens
   (`pd/engine.py:202-214`); P calls `reserve(receiving=False)`
   (`pd/connector.py:298`) → notify `prefill_started` (`:305`).
4. P computes prompt chunks in its own worker loop; each chunk reaches
   `_notify_connector_prefill` (`engine/stage_engine.py:661`) →
   `prefill_submitted` (`pd/connector.py:188`, hidden staging `:206-213`) →
   `queue_chunk` (`:220`, descriptors from `pd/transport.py:29/66/95`).
5. Transport is driven from `progress()` (`pd/connector.py:349`) →
   `NixlConnector.poll` (`pd/transport.py:203`) / `submit` (`:182`), reached from
   the worker loop (`engine/worker.py:190`). Compute release is
   `release_transfer_compute` (`pd/connector.py:384` → `engine/stage_engine.py:716`);
   `commit` (`pd/connector.py:433`) marks the end of submission, not of arrival.
6. D: `received` counter reaches `expected` → `accept_import`
   (`pd/connector.py:416` → `engine/stage_engine.py:751`), which marks the
   imported prefix, restores `hidden_state` and re-queues the request in
   `WAITING`. It then competes with local arrivals in the same admission path
   and enters `CODA` when admitted (`core/scheduler.py:507-511`).
7. `ack` back to P → `acknowledge` (`pd/connector.py:342`) → `released`
   (`:436-442`), which unblocks the pinned pages
   (`StageEngine.release_transfer` `engine/stage_engine.py:733`).

Cancellation ordering matters: `cancel` waits for in-flight writes, `cancel_safe`
is the advisory variant (`pd/connector.py:307`/`:319`). A cancelled destination
must not recycle pages while the source may still write.

## HTTP serving path

- Middleware `guard` (`vllm_rlt/serving/server.py:42`): host/content-type/
  readiness/inflight checks (`:49-65`), then `asyncio.wait_for(handler, request_timeout)`
  (`:67`).
- `complete` (`vllm_rlt/serving/server.py:95`): `request.json()` (`:100`) →
  `CompletionRequest.parse` (`vllm_rlt/serving/protocol.py:81`, called
  `server.py:101`) → `EngineWorker.submit` (`vllm_rlt/serving/worker.py:77`) which
  creates a `RequestChannel` (`:82`) and sets `wake` (`:85`) → handler awaits
  `channel.receive()` (`server.py:108`).
- Engine side is a **single executor thread**: `EngineWorker._call`
  (`vllm_rlt/serving/worker.py:108`) → `run_in_executor` → `_tick`
  (`:146`): tokenize (`:157`), resolve params (`:158`), `engine.add_request`
  (`:164`), `engine.step()` (`:170`), incremental decode (`:177-181`), then
  `TokenEvent` (`:192-197`) appended on the loop side (`:253-254`).
- SSE: stream prepared once (`server.py:110-117`), `completion` (`serving/protocol.py:200`)
  and `sse` (`:215`) written under `write_timeout` (`server.py:126`),
  `[DONE]` at `:156`. Errors before the first token become an HTTP status
  (`:163`); after it they abort the transport (`:168`).
- The engine thread is the only place that may touch the engine, tokenizer or
  profiler; the asyncio thread owns `channels`/`pending`/`cancellations`
  (`serving/worker.py:59-66`). Crossing that line is a correctness bug, not a
  style issue.

## Blocking and polling points (performance-relevant)

| Site | Behavior |
| --- | --- |
| `vllm_rlt/engine/llm_engine.py:256` | `poll(0.01)` — 10 ms stall when stepping with no queued output |
| `vllm_rlt/engine/llm_engine.py:127` | `_receive(timeout=…)` blocking `Pipe.poll` for control replies (`:196-201`) |
| `vllm_rlt/engine/worker.py:182`, `:199` | non-blocking command poll each iteration |
| `vllm_rlt/engine/worker.py:201` | idle `sleep(0.001)` |
| `vllm_rlt/serving/worker.py:227` | idle `await wake.wait()` (event-driven), one `run_in_executor` hop per tick (`:234`) |
| `vllm_rlt/pd/engine.py:173-177` | `_wait` 1 ms poll loop (used by control/admission paths) |
| `vllm_rlt/pd/engine.py:441` | `sleep(0.0005)` when a coordinator step produced no outputs |
| `vllm_rlt/pd/transport.py:185-186` | `submit` returns False when `max_inflight_bytes` is reached (backpressure, not a sleep) |
| `vllm_rlt/engine/stage_engine.py:616-620` | `_step` returns `[]` instead of raising while PD imports are pending |

Data-plane gating that can look like idling: `can_schedule`
(`pd/connector.py:175`) blocks an exporting request while chunks/events are
pending, and `_waiting_for_connector` (`engine/stage_engine.py:640`) classifies
those waits as progress.

## Where to look next

- Request/KV/submission state machines → [scheduling.md](scheduling.md).
- Device-side work inside one submission → [kv-and-execution.md](kv-and-execution.md).
- Turning these anchors into wall-clock attribution → [performance-surface.md](performance-surface.md).
