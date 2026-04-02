# Weight Streaming: What Has Actually Been Implemented for Prefetch/Eviction

This file documents the code that currently exists in this worktree to support
weight-streaming-style prefetch and eviction. It replaces the earlier
speculative design that described an FX pass and custom ops. The implementation
that landed is different:

- It is wired into Inductor wrapper codegen, not an FX graph rewrite.
- It injects Python runtime calls around generated kernel launches.
- It can schedule SSD prefetches and evictions from the wrapper.
- It contains runtime primitives for DRAM->VRAM H2D prefetch and wait, but
  those primitives are not yet emitted by Inductor codegen.

The rest of this document is intentionally concrete and describes only what the
current code does.

---

## 1. Files Changed

### New package: `torch/_inductor/weight_streaming/`

This package contains the schedule loader, wrapper/codegen helpers, and the
runtime singleton.

#### `torch/_inductor/weight_streaming/plan.py`

Defines the schedule dataclasses and the JSON loader:

- `TensorEntry`
- `PrefetchOp`
- `EvictOp`
- `ColdStartOp`
- `IOSchedule`
- `load_io_schedule(path, nodes_csv="")`

#### `torch/_inductor/weight_streaming/codegen.py`

Defines the codegen-side adapter and line emitter:

- `_is_gpu_node(node)`
- `ScheduleAdapter`
- `WeightStreamingLine`

#### `torch/_inductor/weight_streaming/runtime.py`

Defines `WeightStreamRuntime`, a singleton that owns:

- async SSD-read worker threads
- pinned CPU buffers
- a dedicated CUDA H2D stream
- H2D completion events
- loaded GPU tensors
- per-tensor location tracking

#### `torch/_inductor/weight_streaming/__init__.py`

Exports:

- `IOSchedule`
- `load_io_schedule`
- `WeightStreamRuntime`

### Modified existing Inductor files

#### `torch/_inductor/config.py`

Adds two config knobs:

- `weight_streaming_plan: str = ""`
- `weight_streaming_nodes_csv: str = ""`

If `weight_streaming_plan` is set, wrapper codegen will inject weight
streaming runtime calls.

#### `torch/_inductor/codegen/wrapper.py`

Adds `_inject_weight_streaming_io()` and calls it from
`PythonWrapperCodegen._generate()` after wrapper IR passes have run and before
the final wrapper body is emitted.

### New tests

#### `test/inductor/test_weight_streaming.py`

Adds tests for:

- schedule JSON loading
- optional CSV-based node enrichment
- GPU/CPU node classification
- trace-index to wrapper-kernel-index remapping
- runtime lifecycle
- runtime eviction behavior
- runtime H2D behavior
- a real SSD->DRAM->VRAM file-backed path

---

## 2. The Schedule Format That Is Actually Consumed

The current loader does not consume the earlier `pre_ops` / `post_ops` plan
shape. It consumes a JSON object with these top-level keys:

```json
{
  "nodes": [
    {"idx": 0, "name": "aten::empty", "resource_kind": "cpu_thread"},
    {"idx": 1, "name": "triton_fused_mm_0", "resource_kind": "gpu_stream"}
  ],
  "prefetches": [
    {
      "tensor_name": "param_0_1",
      "before_node": 1,
      "after_node": -1,
      "eager_start": false
    }
  ],
  "evictions": [
    {
      "tensor_name": "param_0_1",
      "evict_after_node": 1
    }
  ],
  "cold_starts": [
    {
      "tensor_name": "param_0_1",
      "attach_before_node": 0
    }
  ],
  "tensor_metadata": {
    "param_0_1": {
      "kind": "WEIGHT",
      "size_bytes": 16384,
      "dtype": "float16",
      "shape": [64, 64],
      "safetensors_file": "model-00001-of-00004.safetensors",
      "safetensors_offset": 0
    }
  }
}
```

### Meaning of each section

#### `nodes`

The full scheduler trace. The adapter uses this to decide which trace nodes
map to wrapper kernel indices.

Each node is expected to have:

- `idx`
- `name`
- optionally `resource_kind`

#### `prefetches`

Each entry becomes a `PrefetchOp`:

- `tensor_name`
- `before_node`
- optional `after_node`
- optional `eager_start`

Important: the current code only uses `tensor_name` and `before_node` for
wrapper injection. `after_node` and `eager_start` are parsed and stored, but
they are not used by the current adapter or wrapper codegen.

#### `evictions`

Each entry becomes an `EvictOp`:

- `tensor_name`
- `evict_after_node`

#### `cold_starts`

Each entry becomes a `ColdStartOp`:

- `tensor_name`
- `attach_before_node`

Important: `attach_before_node` is parsed and stored, but the current wrapper
injection does not use it to place the operation near a specific node. All cold
start ops are emitted before the first kernel.

#### `tensor_metadata`

Each tensor becomes a `TensorEntry` with:

- `name`
- `kind`
- `size_bytes`
- `dtype`
- `shape`
- `file` from `safetensors_file`
- `file_offset` from `safetensors_offset`

Important: `size_bytes` is stored in the dataclass, but the runtime currently
computes the number of bytes to map from `shape` and `dtype`. It does not use
`size_bytes` during file reads.

### Optional CSV enrichment

`load_io_schedule(path, nodes_csv=...)` can enrich `nodes[*]["resource_kind"]`
from a `runtime_nodes.csv` file if the JSON nodes do not already carry
`resource_kind`.

This is done by:

1. Reading `runtime_nodes.csv` with `csv.DictReader`.
2. Building `{idx: resource_kind}`.
3. Copying `resource_kind` onto matching JSON nodes.

This is the purpose of `config.weight_streaming_nodes_csv`.

---

## 3. How Inductor Uses the Schedule

The integration point is wrapper codegen, not graph rewriting.

### Config gating

`torch._inductor.config.weight_streaming_plan` is the top-level switch.

If it is a non-empty string, `PythonWrapperCodegen._generate()` calls
`self._inject_weight_streaming_io()`.

### When injection happens

Injection happens after `self.run_wrapper_ir_passes(is_inference)` has already
built the wrapper line list and before the final wrapper body is emitted.

That matters because the implementation works by rewriting `self.lines`, not by
rewriting the FX graph or scheduler IR.

### `_inject_weight_streaming_io()`

The wrapper injection flow is:

1. Load the schedule with `load_io_schedule(...)`.
2. Build a `ScheduleAdapter(schedule)`.
3. Emit two header lines:
   - `from torch._inductor.weight_streaming.runtime import WeightStreamRuntime`
   - `_ws_rt = WeightStreamRuntime.instance()`
4. Walk `self.lines`.
5. Count only `KernelCallLine` instances as kernel indices.
6. Insert `WeightStreamingLine` objects before or after those kernel lines.
7. Replace `self.lines` with the augmented list.

### What code is emitted

Cold-start ops become:

```python
_ws_rt.cold_start_prefetch("tensor_name")
```

Prefetches become:

```python
_ws_rt.ssd_prefetch("tensor_name")
```

Evictions become:

```python
_ws_rt.evict("tensor_name")
```

Important: the wrapper does not currently emit:

- `_ws_rt.h2d_prefetch(...)`
- `_ws_rt.wait_h2d(...)`
- `_ws_rt.evict_vram(...)`
- `_ws_rt.evict_dram(...)`

So the wrapper currently supports:

- cold-start population into DRAM
- background SSD->DRAM prefetch
- post-kernel eviction from both DRAM and VRAM

It does not yet swap actual kernel operands from normal model weights to
runtime-fetched GPU tensors.

### `WeightStreamingLine`

This is a very small wrapper IR helper. It subclasses `WrapperLine` and simply
emits literal Python call strings into the generated wrapper body.

There is no custom operator registration and no special lowering path here.

---

## 4. Trace-to-Wrapper Mapping

The scheduler trace and the wrapper body do not have the same indexing scheme.

- The scheduler trace contains CPU and GPU nodes.
- The generated Inductor wrapper only has explicit insertion points around GPU
  kernel dispatches.

`ScheduleAdapter` exists to bridge that mismatch.

### GPU/CPU classification

The adapter treats a node as GPU if either:

- `resource_kind` is `gpu_stream` or `gpu`
- or, if `resource_kind` is missing, `name` starts with one of:
  - `triton_`
  - `cublas`
  - `Memcpy`
  - `cudaMemcpy`
  - `nccl`
  - `cublasLt`
  - `cusparse`
  - `cudnn`
  - `cutlass`

Every GPU trace node gets assigned a wrapper kernel index:

- first GPU node -> wrapper kernel `0`
- second GPU node -> wrapper kernel `1`
- etc.

### Prefetch remapping

For each `PrefetchOp`, the adapter uses `before_node` and maps it with
`_snap_to_gpu_next(trace_idx)`.

Behavior:

- if `before_node` is already a GPU trace node, use that exact wrapper kernel
  index
- if `before_node` is a CPU trace node, snap forward to the next GPU trace node
- if there is no later GPU node, drop the prefetch

The result is stored in `before_kernel`, a mapping of:

```python
wrapper_kernel_idx -> list[PrefetchOp]
```

### Eviction remapping

For each `EvictOp`, the adapter uses `evict_after_node` and maps it with
`_snap_to_gpu_prev(trace_idx)`.

Behavior:

- if `evict_after_node` is already a GPU trace node, use that exact wrapper
  kernel index
- if `evict_after_node` is a CPU trace node, snap backward to the previous GPU
  trace node
- if there is no earlier GPU node, drop the eviction

The result is stored in `after_kernel`, a mapping of:

```python
wrapper_kernel_idx -> list[EvictOp]
```

### Cold starts

`cold_starts` are not remapped by node index in the current code. The adapter
simply preserves the list as `startup_ops`, and wrapper codegen emits every
startup op before any kernel dispatch.

---

## 5. Runtime Implementation Details

`WeightStreamRuntime` is a singleton. Generated wrapper code assumes it has
already been initialized and accesses it with:

```python
_ws_rt = WeightStreamRuntime.instance()
```

### Important current behavior

The wrapper code does not initialize the runtime. That means some external
driver still has to do something equivalent to:

```python
from torch._inductor.weight_streaming.plan import load_io_schedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime

schedule = load_io_schedule(plan_path, nodes_csv=nodes_csv_path)
WeightStreamRuntime.initialize(schedule, torch.device("cuda"))
```

If that has not happened, the generated wrapper will assert when it calls
`WeightStreamRuntime.instance()`.

### Internal state

The runtime owns:

- `_schedule`
- `_device`
- `_ssd_pool = ThreadPoolExecutor(max_workers=4)`
- `_inflight_reads: dict[str, Future]`
- `_dram_buffers: dict[str, torch.Tensor]`
- `_h2d_stream = torch.cuda.Stream(device=device)`
- `_h2d_events: dict[str, torch.cuda.Event]`
- `_vram_tensors: dict[str, torch.Tensor]`
- `_location: dict[str, str]`

Each tensor named in `schedule.tensors` starts in location `"ssd"`.

### `register_dram_tensor`

`register_dram_tensor(tensor_name, tensor)` is a manual escape hatch for the
case where the caller already has the tensor in CPU memory. It stores the
tensor in `_dram_buffers` and marks the location as `"dram"`.

This is not automatically called by Inductor codegen today.

### `ssd_prefetch`

`ssd_prefetch(tensor_name)` does the following:

1. Return immediately if the tensor is not currently tracked as `"ssd"`.
2. Return immediately if the tensor already has an inflight read.
3. Look up tensor metadata in `self._schedule.tensors`.
4. Return immediately if there is no metadata or no file path.
5. Submit `_read_from_file(entry)` to the SSD thread pool.
6. Record the returned `Future` in `_inflight_reads[tensor_name]`.

This gives actual asynchronous host-side prefetch behavior, but it is not
implemented with explicit `pread`. The current implementation uses a background
thread plus `torch.UntypedStorage.from_file(...)`.

### `cold_start_prefetch`

`cold_start_prefetch(tensor_name)` is a wrapper-visible helper for startup
loads.

Behavior:

1. If the tensor is already in `"dram"` or `"vram"`, do nothing.
2. Otherwise call `ssd_prefetch(tensor_name)`.
3. If that created an inflight read, wait for it immediately with
   `.result()`.
4. Store the resulting pinned tensor in `_dram_buffers`.
5. Mark the tensor location as `"dram"`.

So cold-start prefetch is synchronous from the perspective of wrapper
execution: it ensures the tensor is in DRAM before the first kernel runs.

### `h2d_prefetch`

`h2d_prefetch(tensor_name)` exists in the runtime even though wrapper code does
not currently emit it.

Behavior:

1. If there is an inflight SSD read, wait for it and move the result into
   `_dram_buffers`.
2. Else, if the tensor is still marked `"ssd"` and has a file entry, call
   `_read_from_file(entry)` synchronously.
3. Read the CPU tensor from `_dram_buffers`.
4. Launch `cpu_tensor.to(self._device, non_blocking=True)` on the dedicated
   `_h2d_stream`.
5. Create a `torch.cuda.Event()`.
6. Record the event on `_h2d_stream`.
7. Save the event in `_h2d_events[tensor_name]`.
8. Save the GPU tensor in `_vram_tensors[tensor_name]`.
9. Mark the location as `"vram"`.
10. Return the GPU tensor.

This is the runtime primitive intended for DRAM->VRAM overlap.

### `wait_h2d`

`wait_h2d(tensor_name)` also exists but is not wired into wrapper codegen.

Behavior:

1. Look up `_h2d_events[tensor_name]`.
2. If present, call `torch.cuda.current_stream(self._device).wait_event(event)`.
3. Delete the event entry.
4. Return `_vram_tensors[tensor_name]`.

This is the synchronization point that would make H2D prefetch safe for
consumers on the main compute stream.

### `evict_vram`

`evict_vram(tensor_name)`:

1. Pop the tensor from `_vram_tensors`.
2. If present, call `gpu_tensor.untyped_storage().resize_(0)`.
3. If the location was `"vram"`, change it to `"dram"`.

The `resize_(0)` detail matters. It is explicitly trying to release the
underlying CUDA allocation even if some Python reference still exists in the
compiled wrapper.

### `evict_dram`

`evict_dram(tensor_name)`:

1. Remove the CPU tensor from `_dram_buffers`.
2. If the location was `"dram"` or `"vram"`, change it to `"ssd"`.

### `evict`

`evict(tensor_name)` is the method that wrapper codegen currently emits.

It simply calls:

1. `evict_vram(tensor_name)`
2. `evict_dram(tensor_name)`

That means every scheduled eviction currently drops both copies. The schedule
does not currently distinguish:

- evict only from VRAM
- evict only from DRAM

even though the runtime has separate methods for those operations.

### `_read_from_file`

The file-read path is:

1. Map the schedule dtype string through `_DTYPE_MAP`.
2. Compute `numel` from the recorded `shape`.
3. Compute `nbytes = numel * element_size(dtype)`.
4. Create a file-backed `torch.UntypedStorage` with:
   `torch.UntypedStorage.from_file(entry.file, False, entry.file_offset + nbytes)`.
5. Slice that storage to `[entry.file_offset : entry.file_offset + nbytes]`.
6. Wrap the sliced storage in a `torch.TypedStorage`.
7. Create a tensor with `.set_(...)`.
8. Reshape to the recorded shape.
9. Call `.pin_memory()` and return the pinned CPU tensor.

Two important implications:

- The current implementation relies on file-backed storage plus a pinned-memory
  copy, not an explicit async `pread`.
- The runtime derives `nbytes` from `dtype` and `shape`, not from
  `TensorEntry.size_bytes`.

---

## 6. What Prefetch/Eviction Behavior Exists Today

The currently implemented end-to-end behavior is:

1. Parse a scheduler-generated JSON schedule.
2. Optionally enrich node `resource_kind` from `runtime_nodes.csv`.
3. Convert trace-node indices into wrapper kernel indices.
4. Inject Python runtime calls around generated GPU kernel dispatches.
5. Use a background thread pool to prefetch tensors from file into pinned DRAM.
6. Use runtime eviction methods to free tracked GPU/CPU copies later.

Concretely, the wrapper can now do this kind of thing:

```python
_ws_rt.cold_start_prefetch("tok_embeddings.weight")
_ws_rt.ssd_prefetch("layers.0.attn.q_proj.weight")
call_kernel_0(...)
call_kernel_1(...)
_ws_rt.evict("layers.0.attn.q_proj.weight")
```

That is the real support that has landed for prefetch/eviction.

---

## 7. What Has Not Been Wired Up Yet

This section is important because the previous version of this document
overstated what existed.

### Not implemented in the compile path

- No FX pass rewrites weight uses.
- No `torch.library` custom ops were added.
- No graph-level replacement of `get_attr` or placeholder weights exists.
- No emitted `_ws_rt.h2d_prefetch(...)` calls exist.
- No emitted `_ws_rt.wait_h2d(...)` calls exist.
- No emitted `evict_vram` or `evict_dram` calls exist; wrapper emits only
  `evict`.
- No wrapper-side runtime initialization exists.

### Parsed but not used for scheduling decisions

- `PrefetchOp.after_node`
- `PrefetchOp.eager_start`
- `ColdStartOp.attach_before_node`

These fields survive parsing, but they do not currently influence wrapper
placement.

### Limitations of the current schedule mapping

- CPU trace nodes are not first-class insertion points in the wrapper.
- Prefetches targeted at CPU nodes are snapped forward to the next GPU kernel.
- Evictions targeted at CPU nodes are snapped backward to the previous GPU
  kernel.
- Prefetches beyond the last GPU node are dropped.
- Evictions before the first GPU node are dropped.

### Limitations of the current runtime

- No pinned-buffer reuse or pool management.
- No cap on aggregate pinned memory.
- No rate limiting for inflight SSD reads or H2D copies beyond the fixed thread
  pool width.
- No explicit check that `size_bytes` matches `shape * dtype`.
- No schedule compatibility validation between compile time and runtime
  initialization.

### Most important functional gap

The runtime already knows how to do:

- SSD->DRAM prefetch
- DRAM->VRAM prefetch
- wait-on-H2D
- VRAM-only eviction
- DRAM-only eviction

But the generated Inductor wrapper currently uses only:

- `cold_start_prefetch`
- `ssd_prefetch`
- `evict`

So the codebase has real schedule-driven prefetch/eviction support, but it is
currently wrapper-level host-side scheduling support, not a complete
weight-substitution path for compiled kernels.

---

## 8. Tests Added

`test/inductor/test_weight_streaming.py` covers the following:

- `TestIOSchedule`
  - schedule JSON loading
  - CSV enrichment of `resource_kind`
  - basic dataclass construction

- `TestScheduleAdapter`
  - GPU/CPU classification using `resource_kind`
  - fallback GPU classification using name prefixes
  - exact mapping for GPU-targeted prefetches/evictions
  - snapping behavior for CPU-targeted prefetches/evictions
  - multiple ops attached to the same wrapper kernel
  - dropping out-of-range prefetches/evictions
  - cold-start preservation

- `TestWeightStreamRuntime` (CUDA-only)
  - singleton initialize/reset behavior
  - VRAM eviction releasing underlying storage
  - DRAM eviction behavior
  - combined eviction behavior
  - H2D prefetch and wait behavior
  - separate H2D stream creation
  - manual DRAM registration
  - cold-start no-op when already in DRAM
  - end-to-end file-backed SSD->DRAM->VRAM flow

There is not currently an end-to-end test that compiles a model, initializes
the runtime, and asserts that the generated wrapper emits the expected injected
weight-streaming calls.

---

## 9. Short Summary

What has actually changed for prefetch/eviction is:

- a new `torch._inductor.weight_streaming` package
- a real JSON schedule loader
- optional node classification enrichment from `runtime_nodes.csv`
- a trace-to-wrapper adapter that maps scheduler nodes to kernel indices
- wrapper codegen that injects weight streaming runtime calls
- a runtime singleton that can asynchronously prefetch from file into pinned
  DRAM, asynchronously copy to VRAM on a dedicated CUDA stream, and evict DRAM
  and/or VRAM state
- tests for the loader, adapter, and runtime

What has not changed yet is equally important:

- there is no FX-level weight replacement
- the wrapper does not emit H2D/wait calls
- the wrapper does not initialize the runtime
- scheduled evictions currently drop both DRAM and VRAM copies, not one level at
  a time
