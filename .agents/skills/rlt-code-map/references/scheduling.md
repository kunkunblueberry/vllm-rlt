# Scheduling: admission, batch composition, progress

`Scheduler` (`vllm_rlt/core/scheduler.py:132`) is the **control center**: it owns
`requests`, the per-phase `queues` (`:157-158`) and every logical progress field
on `Request` (`vllm_rlt/request.py:38`). It never touches device tensors.
Policy lives in `vllm_rlt/core/scheduling_policy.py`; the split exists so that
"when to admit/yield" can change without touching "how a batch is built".

## Phase machine

`RequestPhase` (`vllm_rlt/request.py:11`):

```
WAITING ──admit──► PREFILL ──────────────────► CODA ──► PRELUDE ──► RECURRENT ──► CODA ──► …
                   PREFILLING_PRELUDE ─┐        ▲                                   │
                   PREFILLING_RECURRENT ┴────────┘                                   │
PRELUDE ─► RECURRENT* ─(exit)─► CODA ─► PRELUDE …            (decode; * loop count from exit policy)
RECEIVING / TRANSFERRING  — PD import/export only, never model operations
SPECULATIVE  — one draft/verify round, expands into model calls locally
FINISHED     — terminal; request removed from `Scheduler.requests`
```

Rules that the code enforces:

- `enqueue` (`core/scheduler.py:182`) is the only writer of `request.phase`
  besides the PD import (`StageEngine.accept_import`
  `engine/stage_engine.py:751`), transfer suspension (`suspend_for_transfer`
  `core/scheduler.py:186`) and the async-speculative fast finish
  (`engine/stage_engine.py:819-829`).
- A request may be in exactly one queue. Removal is defensive
  (`while rid in queue: queue.remove(rid)`, `core/scheduler.py:198-200`,
  `:307-309`) because delayed EOS or preemption can race a queued entry.
- `FINISHED` + `requests.pop` happen in `Scheduler.finish`
  (`core/scheduler.py:196`); `StageEngine._finish`
  (`engine/stage_engine.py:819`) is what calls the runner release first.

## Admission (`_admit`, `core/scheduler.py:474`)

```
order waiting ─► active-slot check ─► resume-or-allocate ─► enter phase queue
```

- Ordering: `_order_waiting_requests` (`:219`) — FCFS keeps queue order; `priority`
  policy sorts by `sampling_params.priority` in place (lower first).
- Slot limit: `_ensure_active_slot` (`:313`) — with `policy="priority"` and a
  `suspend_callback`, a higher-priority arrival may evict one victim.
- Resume: `_try_resume` (`:332`) → `ResumeResult` (`:103`) `NOT_PREEMPTED` /
  `RESTORED` / `BLOCKED`; `BLOCKED` counts a bypass, so a long resident request
  cannot starve short ones forever (`max_admission_bypasses`).
- Budget: `_plan_admission` (`:376`) produces `AdmissionPlan` (`:110`) =
  capacity tokens + prefix claim + growth reservation + admission headroom
  (`total_budget_blocks` `:123`). `_commit_admission` (`:449`) allocates and only
  then enters prefill — never partially.
- Capacity pressure retries with a waiting-only victim (`:530-534`) and recomputes
  the plan, because releasing blocks invalidates prefix claims.
- Imported PD requests skip allocation: `RECEIVING` → unpin → `CODA` (`:507-511`).

## Batch composition (`take_phases`, `core/scheduler.py:704`)

- Budgets: `max_num_batched_tokens` is consumed by selected token counts;
  `max_num_seqs` bounds rows. `selected` (a set of request IDs) is passed to
  capacity checks so choosing A cannot let B evict A (`:758-762`).
- Eligibility: `execution_ready` (`:164`) — the connector's `can_schedule`
  predicate — is honoured per candidate; a blocked queue must not hide another
  runnable phase (`has_ready` `:694`).
- Per-request span: `_make_scheduled_request` (`:552`) builds an immutable
  `ScheduledRequest` (`:25`) — never new Requests, never new KV metadata. Prompt
  spans clamp to `prefill_chunk_size` and the remaining prompt; decode spans are
  one position at `request.loops_done`.
- Loop boundary: `_planned_loop_count` (`:798`) decides how many recurrent
  iterations this submission covers. Atomic prefill covers `total_ut_steps`;
  decode covers 1; `PRELUDE`/`CODA`/`SPECULATIVE` cover 0. This is the field that
  becomes `ExecutionBatch.loop_count` and therefore the number of model calls per
  submission.
- Fairness accounting: `policy.record_batch` counts **nonempty scheduled batches**
  (`core/scheduling_policy.py:31-36`), not attempts or GPU completions.

### Staged and wavefront prefill

- `requires_staged_prefill` (`core/scheduling_policy.py:221`): true for
  `mode="mixed"` (producing tokens) or `wavefront_chunked_prefill`. Staged prefill
  splits a prompt into `PREFILLING_PRELUDE` (embeddings) and
  `PREFILLING_RECURRENT` (per-depth recurrence) so recurrent decode work can join.
- Per-depth frontier: `Request.prefill_boundaries` (`vllm_rlt/request.py:87`) fills the
  `prefill_depth_ranges` field (`vllm_rlt/request.py:54`); advanced by
  `_apply_split_prefill_progress` (`core/scheduler.py:980`) and friends (`:1049`, `:1077`, `:1177`).
- Wavefront: `_take_wavefront_phase` (`:859`) round-robins chunks across requests
  and depths inside one token budget; it needs `last_exited` KV
  (`engine/stage_engine.py:210-211`). Design intent:
  `docs/split-prefill-wavefront.md`.
- Hidden-state residency is the scarce resource here, not KV: a chunk's hidden
  states are kept in `ModelRunner.prefill_states` until the chunk reaches full
  depth (`request.py:50-59` documents the frontier bookkeeping).

## Policies (`create_scheduling_policy`, `core/scheduling_policy.py:184`)

| Selection | Class | Order |
| --- | --- | --- |
| `mode=refill`, no staged prefill | `RefillPolicy` `:59` | recurrent (if preferred) → prelude → coda (batch threshold) → recurrent → admit → prefill → … |
| `mode=no_refill` | `NoRefillPolicy` `:92` | phase state machine FILL → CORE → CODA, bounded by `max_prefill_batches_before_decode` |
| `mode=mixed` | `MixedPolicy` `:162` | recurrent + PREFILLING_RECURRENT in one batch first, then refill ordering |
| staged prefill, non-mixed | `SplitPrefillPolicy` `:156` | refill ordering over split prompt phases |
| speculation | `SpeculativePolicy` `:128` | coda → speculative round → admit → prompt phases |
| `produces_tokens=False` (PD prefill stage) | `PromptOnlyPolicy` `:39` | admit + prompt phases only |

`prefer_recurrent=True` is set inside `StageEngine._step_async`
(`engine/stage_engine.py:934`, argument at `:942`) when the previous core is still
in flight, so an independent boundary op can overlap instead of fragmenting the batch.

## Exit policies

- Fixed depth: `request.target_loops` (`request.py:79`) → `_should_exit`
  (`core/scheduler.py:1232`) is a depth comparison.
- Trained gate (`ouro`): `remaining_probability *= 1 - sigmoid(score)` per loop,
  exit when `1 - remaining_probability >= exit_threshold` and depth ≥ `min_loops`.
  Score arrives as `ModelRunnerOutput.gate_scores` (`worker/output.py:31`).
- `ouro_delayed` / `random_lookahead`: the decision for loop *n* consumes the
  score of loop *n−1*, which is why submission and delivery are separated
  (`update_after_submission` `:1409` + `StageEngine._pending_exit_signals`
  `engine/stage_engine.py:353` + `_delayed_signal` `core/scheduler.py:1247`).
  `needs_exit_signal` (`:1262`) is the cheap predicate that avoids a readback when
  no score could possibly end the request.
- `trace`: replay configured depths, no gate compute
  (`ExitConfig(depths_by_request=…)`, `vllm_rlt/config.py:92`;
  `_trace_exit` `core/scheduler.py:1244`).
- Early exit implies **copy-on-exit KV**: `update.exited` → `finalize_many`
  (`engine/stage_engine.py:798-799`) for `LAST_EXITED` planes.

## Preemption

`scheduler.enable_preemption` wires `PreemptionManager` (`engine/preemption.py:13`):

- Candidate safety: `_is_preemption_candidate` (`core/scheduler.py:233`) — never
  the requesting request, never `RECEIVING`, never one with pending outputs or
  speculative rounds, never transfer-leased, never in the current batch.
- Ranking: `_select_preemption_victim` (`:258`) — lower priority first, then the
  larger KV footprint by `required_blocks(position + 1)`.
- Effect: `preempt` (`:285`) calls `suspend_callback` (device snapshot, optional
  KV offload) and re-queues the victim in `WAITING`; restoration goes back
  through admission, and the restored phase comes from the snapshot.
- RNG survives suspension but not termination (`worker/model_runner.py:1426`,
  `:1434`) — a resumed request must reproduce an uninterrupted token sequence.

## Speculative scheduling

- A speculative request is scheduled in `SPECULATIVE` with
  `num_speculative_tokens + 1` candidates (`_make_scheduled_request`
  `core/scheduler.py:606-611`), at most 2 rounds in flight (`:737-742`).
- The round itself is executed by `Executor.submit` → `SpeculativeRunner.submit`
  (`worker/executor.py:98`, `worker/speculative.py:348`) or
  `AsyncSpeculativeRunner.submit` (`worker/async_speculative.py:773`). It
  expands into multiple model calls; it is **not** a fourth `Operation`.
- Progress uses placeholders: `num_output_placeholders` /
  `num_speculative_rounds` (`request.py:46-47`) are incremented at submission
  (`update_after_submission` `:1449-1455`) and decremented when the round is
  delivered (`update_from_output` `:1369-1379`). `num_scheduled_outputs`
  (`request.py:114`) is what the rest of the scheduler reads.
- Rollback: a rejected suffix is truncated with `KVCacheManager.truncate_suffix`
  (`core/kv_cache_manager.py:481`), except when GPU-owned frontiers are ahead
  (`core/scheduler.py:1394-1402`).

## Two apply paths (keep them distinct)

| Path | Called from | Meaning |
| --- | --- | --- |
| `update_from_output` (`:1285`) | `StageEngine._update` / `_apply_update` | results delivered; may advance depth, sample tokens, finish requests |
| `update_after_submission` (`:1409`) | `StageEngine._step_async` / `_step_async_speculative` | device work enqueued; advances only what the host already knows |

Both share the prefill-progress helpers and both use object identity to drop
results for cancelled/replaced requests (`:1304`, `:1424`). Merging them is a
semantic change, not a cleanup.

## Where to look next

- What a selected batch becomes on device → [kv-and-execution.md](kv-and-execution.md).
- How to prove a scheduling claim (livelock, fairness, exit) →
  [lifecycles-and-verification.md](lifecycles-and-verification.md).
