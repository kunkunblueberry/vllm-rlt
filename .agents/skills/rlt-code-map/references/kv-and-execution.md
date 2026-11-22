# KV and execution: from a batch to kernels

Covers what happens between `ModelRunner.submit` and the attention kernel, and
which component owns each piece of it. Split:

- **Logical KV** (`KVCacheManager`, `vllm_rlt/core/kv_cache_manager.py:169`):
  which pages a request owns, which rows are valid, who may read/write which
  position at which depth.
- **Device work** (`ModelRunner`, `vllm_rlt/worker/model_runner.py:153`): tensors,
  metadata staging, streams, events, graphs, readback.
- **Attention math** (`AttentionBackend`, `vllm_rlt/attention/backend.py:40`):
  what a given backend can do and which kernel receives the rows.

## Physical layout

- `storage_depths = max_loops if layout == "last_exited" else 1`
  (`kv_cache_manager.py:239`). Pool shape
  `(num_blocks, num_layers, block_size, num_kv_heads, head_dim)` (`:261-264`);
  FP8 pools use `float8_e4m3fn` plus a per-vector `float32` scale pool
  (`:266-278`).
- A request owns `pages × storage_depths` physical blocks; its `block_tables` is
  per plane (`allocate` `:420`, `ensure_capacity` `:463`).
- Plane selection is **always** `_plane(depth)` (`:683`) — depth for
  `LAST_EXITED`, 0 for `SHARED`. Never index `block_tables` by depth directly;
  `plane_block_tables` (`:550`) exists because that mistake is easy.
- `SHARED` overwrites one plane per loop, so a prompt must finish all depths for
  one position before the next position advances (`docs/design.md`); `LAST_EXITED`
  allows depth-major packed prefill.

## Metadata: who builds the addresses

```
MetadataBuilder.build(ids, depths, positions)     worker/metadata.py:172
      └─► KVLayout                                metadata.py:72
             rows / write_blocks / write_offsets
             attention: AttentionMetadataValues   attention/metadata.py:37
      └─► bind | store_metadata | routing.transfer   (device views)
```

- Kernel-facing structs: `KVMetadata` (`worker/metadata.py:22`) and
  `DeviceKVMetadata` (`:45`) carry `position_ids`, `write_blocks`,
  `write_offsets`, `block_tables`, `context_lengths`, optional `write_mask`.
- Attention plan: `plan_attention_metadata` (`attention/metadata.py:48`) sizes
  table width from the max position (`:64`), pads tables with `-1` (`:90`), sets
  `context_lengths = position + 1` (`:95`) and builds `cu_seqlens_q`/`max_seqlen_q`
  for packed rows (`:68-85`).
- Four staging mechanisms exist; which one runs decides the CPU cost per step:
  | Path | When | Anchor |
  | --- | --- | --- |
  | `Workspace.store_metadata` + `upload_metadata` | static buffers, graph-free | `worker/buffers.py:96`/`:130` |
  | `AsyncState.prepare` (+`RoutingBuffers.materialize_metadata`) | async scheduling on CUDA | `worker/async_state.py:203`/`:65` |
  | `MetadataBuilder.prepare_device` | device-owned positions (async speculation) | `worker/metadata.py:93` |
  | `build`/`bind` directly | eager fallback, one-shot batches | `worker/metadata.py:172`, `:278` |
- Packed prefill legality: `build` rejects `packed` unless layout is `last_exited`
  and the backend declares `packed_prefill` (`worker/metadata.py:185-188`);
  `PackedMetadataBuffer` (`worker/buffers.py:223`) expands write addresses on
  device (`kernels/metadata.py:8`).

## Call path inside one submission

```
ModelRunner.submit                          worker/model_runner.py:1207
 ├─ SubmissionQueue.begin                   worker/submission_queue.py:26
 ├─ prepare_on_cpu                          model_runner.py:851
 │   └─ _prepare_cpu_inputs                 :881   (depths/positions snapshot,
 │        ├─ AsyncState.prepare             worker/async_state.py:203
 │        ├─ _stage_metadata                model_runner.py:785 → MetadataBuilder.build
 │        └─ Workspace.store_metadata       worker/buffers.py:96
 ├─ prepare_on_gpu                          :972   (routing.transfer / upload_metadata,
 │                                                  _prepare_forward :1011, bind :1030)
 ├─ _execute_batch                          :1124
 │   └─ ModelRuntime.forward                worker/executor.py:172
 │        ├─ model.prelude_prepared         e.g. models/ouro.py:270
 │        ├─ model.recurrent_prepared       e.g. models/ouro.py:296
 │        │    ├─ KVCacheManager._write_prepared   kv_cache_manager.py:848
 │        │    │    └─ _store_layer                 :891  (index copy / store_fp8 / masked_kv_store)
 │        │    └─ KVCacheManager._attend_prepared  :970
 │        │         ├─ _attend_windowed              :1021 (sliding window)
 │        │         └─ AttentionBackend.attend_metadata  attention/backend.py:50
 │        │              ├─ packed → kernel.prefill  backend.py:68
 │        │              └─ per-query → kernel()      backend.py:66
 │        └─ model.coda_prepared            e.g. models/ouro.py:306
 ├─ compute_done event + readback copy      model_runner.py:1266-1283
 └─ Submission(...)                         model_runner.py:1289
```

- `write()` (`kv_cache_manager.py:827`) and `attend()` (`:954`) are **test-only
  adapters** that rebuild a descriptor per call. Production attention blocks call
  `_write_prepared` / `_attend_prepared` directly (`models/ouro.py:194-195`,
  `nanbeige.py:166-167`, `huginn.py:170-171`, `dory.py:400-401`).
- Backend entry points: torch `kernels/paged_attention.py:93`, Triton
  `kernels/paged_attention.py:204` (`kernels/triton_attention.py:176` decode,
  `:386` packed prefill, `:571` FP8 store), FlashAttention
  `kernels/flash_attention.py:34` (`prefill` `:112`, decode `__call__` `:159`,
  `windowed` `:214`), Ascend `kernels/ascend_attention.py:69`.

## Attention dispatch rules

- `AttentionBackend.attend_metadata` (`attention/backend.py:50`): a kernel-level
  `attend_metadata` wins (Ascend), else `cu_seqlens_q is not None` → `prefill`,
  else per-row `__call__`.
- Windowed attention only via `_attend_windowed`
  (`kv_cache_manager.py:1021`), requires the `sliding_window` capability
  (`:1031-1034`); packed + window needs `packed_sliding_window` (`:984-985`).
- Backend capabilities are declared, not inferred: `BackendCapabilities`
  (`attention/backend.py:18`), per-backend tables (`:85-112`), auto priority
  `:113`, requirement filtering and FP8 calibration fallback in `create_backend`
  (`:177`). `StageEngine` passes the requirements it actually needs
  (`engine/stage_engine.py:232-247`).

## Graphs, streams, buffers

- Streams are created in `ModelRunner.__init__` (`model_runner.py:222-246`):
  `core_stream` (recurrent), `boundary_stream` (prelude/coda/finalize),
  `copy_stream`, `readback_stream`; `multi_stream=False` aliases them all.
- Graph owners: `RecurrentGraphs` (`worker/cuda_graph.py:240`), `CodaGraphs`
  (`:423`), `PreludeGraphs` (`:536`), device-owned variant
  `DeviceRecurrentGraphs` (`:354`). Selection key and fallback rules:
  `RecurrentGraphs.select` (`:251`) — non-contiguous rows, row count over the
  limit, or a full cache run eager instead.
- Graph capture sizes: `graph_capture_sizes` (`cuda_graph.py:22`),
  `speculative_capture_sizes` (`:40`); lazy runtime capture synchronizes the device
  once (`_capture` `:66-90`).
- Under a graph, KV access is re-implemented by `_DeviceCache`
  (`worker/cuda_graph.py:102`): `_write_prepared` `:108`, `_attend_prepared` `:135`.
  A change to `KVCacheManager` KV semantics must be mirrored there.
- UVA staging primitives: `UvaBuffer` (`worker/tensors.py:97`), `UvaBufferPool`
  (`:139`), `StagedInputTensor` (`:229`), `TensorTransfer` (`:282`),
  `StagedWriteTensor` (`:368`), `copy_tensor`/`tensor_transfer_kernel` (`:35`).

## Finalize, readback, release

- Copy-on-exit: `finalize_many` (`model_runner.py:1323`) batches the copy with
  `finalize_kernel` (`kernels/routing.py:148`) on `boundary_stream`; the scalar
  path is `finalize_token` (`kv_cache_manager.py:1053`). Precondition:
  `exit_written` for every recurrent layer (`:582`), error
  `_require_exit_written` (`:605`).
- Readback never blocks unless a buffer must be recycled:
  `_readback_slot` (`model_runner.py:1170`) reuses a lease whose event has
  completed, synchronizing only when every slot is busy.
- Slot recycling waits for the request's last event: `_free_runner_slot`
  (`model_runner.py:1412`), called by `suspend` `:1426` (keeps RNG) and
  `release` `:1434` (drops RNG).
- Error paths must drain before memory is reused: `_synchronize_submission`
  (`model_runner.py:1186`), `submissions.drain()` (`:1304`), and the engine-level
  abort sweep (`engine/stage_engine.py:601-612`).

## Invariants worth restating

1. Allocated capacity ≠ valid history. Attention lengths must exclude unwritten
   or rejected positions (`docs/design.md`; `truncate_suffix`
   `kv_cache_manager.py:481`).
2. A row reads history at its **own** depth in `LAST_EXITED`; a batch does not need
   one shared depth.
3. The final prompt token's hidden state is needed by CODA, which is why prefix
   reuse publishes only the contiguous valid prefix (`publish_prefix` `:378`).
4. `SHARED` never needs copy-on-exit; `_record_finalized` is a no-op there
   (`:618-623`).

## Where to look next

- Which knobs change these paths without changing behavior →
  [performance-surface.md](performance-surface.md).
- Which tests pin each of these steps →
  [lifecycles-and-verification.md](lifecycles-and-verification.md).
