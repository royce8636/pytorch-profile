# Weight Streaming: What The Current Code Actually Does

This file documents the implementation that exists in this worktree today.
It is intentionally descriptive, not aspirational.

The current implementation is:

- integrated into Inductor Python wrapper codegen, not an FX rewrite
- driven by a scheduler JSON plus optional CSV side inputs
- executed by injected Python runtime calls around generated compute lines
- able to emit SSD/DRAM and VRAM-level weight-streaming calls in the generated
  wrapper

It is not a separate post-processing pass over exported code, and it is not a
custom-op based system.

## 1. Main Files

### `torch/_inductor/weight_streaming/plan.py`

Defines the schedule dataclasses and loader:

- `TensorEntry`
- `PrefetchOp`
- `H2DPrefetchOp`
- `EvictVramOp`
- `EvictDramOp`
- `ColdStartOp`
- `IOSchedule`
- `load_io_schedule(path, nodes_csv="", tensor_csv="")`

### `torch/_inductor/weight_streaming/codegen.py`

Defines the wrapper-side helpers:

- `_is_gpu_node(node)`
- `ScheduleAdapter`
- `_op_to_call(op)`
- `WeightStreamingLine`

### `torch/_inductor/weight_streaming/runtime.py`

Defines `WeightStreamRuntime`, the singleton used by generated wrapper code.

### `torch/_inductor/codegen/wrapper.py`

Adds weight-streaming injection and code export:

- `PythonWrapperCodegen._inject_weight_streaming_io()`
- `PythonWrapperCodegen._export_weight_streaming_code(...)`

### `torch/_inductor/config.py`

Adds the config knobs used by the compile path:

- `weight_streaming_plan`
- `weight_streaming_nodes_csv`
- `weight_streaming_tensor_csv`
- `weight_streaming_output_code`

### `scripts/run_weight_streaming.py`

Provides the end-to-end driver that:

- resolves the schedule and CSV inputs
- initializes `WeightStreamRuntime`
- sets Inductor config knobs
- loads the model
- registers GPU weight tensors with the runtime
- compiles `pipe.unet` with `torch.compile(..., backend="inductor")`
- optionally exports the generated wrapper to `weight_streaming_generated.py`

## 2. Schedule Format That Is Actually Loaded

`load_io_schedule(...)` consumes the scheduler-style JSON, not the older
speculative `pre_ops` / `post_ops` shape.

The keys it reads are:

- `nodes`
- `io_operations`
- `spill_decisions`
- `cold_start_prefetches`
- `tensor_metadata` as JSON fallback when `tensor_csv` is not provided

A representative shape is:

```json
{
  "nodes": [
    {"idx": 0, "name": "aten::empty", "resource_kind": "cpu_thread"},
    {"idx": 1, "name": "triton_fused_mm_0", "resource_kind": "gpu_stream"}
  ],
  "io_operations": [
    {
      "type": "prefetch",
      "tensor_name": "param_0_1",
      "before_node": 1,
      "after_node": -1,
      "eager_start": true
    },
    {
      "type": "vram_prefetch_h2d",
      "tensor_name": "param_0_1",
      "before_node": 1
    },
    {
      "type": "vram_evict_d2h",
      "tensor_name": "param_0_1",
      "after_node": 1
    }
  ],
  "spill_decisions": [
    {
      "tensor_name": "param_0_1",
      "evict_after_node": 1
    }
  ],
  "cold_start_prefetches": [
    {
      "tensor_name": "param_cold",
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

### Parsed operation types

`io_operations` is split into:

- `type == "prefetch"` -> `PrefetchOp`
- `type == "vram_prefetch_h2d"` -> `H2DPrefetchOp`
- `type == "vram_evict_d2h"` -> `EvictVramOp`

`spill_decisions` becomes:

- `EvictDramOp`

`cold_start_prefetches` becomes:

- `ColdStartOp`

### Optional CSV enrichment

If `nodes_csv` is provided and JSON nodes do not already carry
`resource_kind`, `load_io_schedule(...)` enriches the nodes from
`runtime_nodes.csv`.

If `tensor_csv` is provided, the loader uses `pytorch_runtime_tensors.csv` for
tensor metadata and filters scheduled VRAM ops to `WEIGHT` tensors only.

Important current behavior:

- `tensor_csv` replaces the JSON tensor metadata instead of merging with it
- `_load_tensors_csv(...)` does not populate `file` or `file_offset`
- therefore, string-name SSD reads depend on JSON-backed tensor metadata or on
  tensors already registered in DRAM

### Fields parsed but not used for wrapper placement

These fields are loaded and preserved, but current wrapper placement does not
use them directly:

- `PrefetchOp.after_node`
- `PrefetchOp.eager_start`
- `ColdStartOp.attach_before_node`

## 3. How `run_weight_streaming.py` Drives The Flow

The current compile path is anchored in
[`scripts/run_weight_streaming.py`](/data/pytorch-source/scripts/run_weight_streaming.py).

The script:

1. resolves `jit_sim_prune_schedule.json`, optional `runtime_nodes.csv`, and
   optional `pytorch_runtime_tensors.csv`
2. loads the schedule with `load_io_schedule(...)`
3. initializes `WeightStreamRuntime.initialize(schedule, device)`
4. sets:
   - `inductor_config.weight_streaming_plan`
   - `inductor_config.weight_streaming_nodes_csv`
   - `inductor_config.weight_streaming_tensor_csv`
   - `inductor_config.weight_streaming_output_code`
5. loads the model
6. registers model weights with `register_model_weights(...)`
7. compiles `pipe.unet` with `torch.compile(..., backend="inductor")`

Two details matter:

- the generated wrapper expects the runtime singleton to already exist
- GPU weights are registered up front so the wrapper can later call
  `_ws_rt.h2d_prefetch(tensor)` and `_ws_rt.evict_vram(tensor)` on actual
  tensor objects

## 4. Where Inductor Injects Weight-Streaming Calls

Weight streaming is injected from
[`torch/_inductor/codegen/wrapper.py`](/data/pytorch-source/torch/_inductor/codegen/wrapper.py).

`PythonWrapperCodegen._generate()` does this in order:

1. build the normal wrapper lines
2. run wrapper IR passes
3. if `config.weight_streaming_plan` is set, call
   `_inject_weight_streaming_io()`
4. emit the final wrapper body
5. if `config.weight_streaming_output_code` is set, write the final generated
   Python wrapper to `weight_streaming_generated.py`

This means the exported file is:

- real generated Inductor wrapper code
- after normal wrapper passes
- after weight-streaming injection
- without any later readability-only editing pass

## 5. What `_inject_weight_streaming_io()` Emits Today

The wrapper injector first loads the schedule and constructs a
`ScheduleAdapter`.

It then emits these header lines:

```python
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime
_ws_rt = WeightStreamRuntime.instance()
```

### Compute lines used by the injector

The injector scans two kinds of wrapper lines as compute:

- `KernelCallLine`
- `ExternKernelOutLine`

This is important because VRAM restore/evict placement is computed across all
compute lines, not just Triton kernel lines.

### Schedule-driven SSD and DRAM calls

For schedule-driven insertion around rescaled kernel indices, the wrapper uses
only `KernelCallLine` instances.

The currently emitted schedule-driven calls are:

- before kernels:
  - `_ws_rt.ssd_prefetch("tensor_name")` from `PrefetchOp`
- after kernels:
  - `_ws_rt.evict_dram("tensor_name")` from `EvictDramOp`

### Cold-start calls

All `ColdStartOp`s are emitted before any compute line as:

```python
_ws_rt.cold_start_prefetch("tensor_name")
```

`attach_before_node` is currently ignored for placement. All cold starts are
attached at wrapper startup.

### Graph-input-based VRAM restore and eviction

Separately from the schedule adapter, the wrapper computes first-use and
last-use of graph inputs across all compute lines.

For each graph input:

- before first use, it emits:

```python
_ws_rt.h2d_prefetch(graph_input_tensor)
```

- after last use, it emits:

```python
_ws_rt.evict_vram(graph_input_tensor)
```

These calls use the actual graph-input tensor object, not a tensor-name string.

Important current behavior:

- this path is driven by graph-input first-use / last-use analysis
- it is not driven directly by `H2DPrefetchOp` or `EvictVramOp` from the
  schedule
- the runtime no-ops for tensors that were not registered as weights

### What the wrapper does not emit

The wrapper does not currently emit:

- `wait_h2d(...)`
- schedule-driven `H2DPrefetchOp` calls from `adapter.before_kernel`
- schedule-driven `EvictVramOp` calls from `adapter.after_kernel`
- runtime initialization

The lack of `wait_h2d(...)` is intentional for the tensor-object path used by
the current wrapper: `_h2d_prefetch_tensor(...)` records an event on the H2D
stream and immediately makes the current stream wait on that event.

## 6. How `ScheduleAdapter` Maps Trace Indices To Wrapper Indices

`ScheduleAdapter` exists because the scheduler trace and the generated wrapper
do not share the same indexing scheme.

The scheduler trace can contain interleaved CPU and GPU nodes. The wrapper only
has a limited number of compute insertion points, and kernel fusion means that
the wrapper often has fewer kernel calls than the original GPU trace did.

### GPU classification

A node is treated as GPU if either:

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

### Initial trace-level attachment

The adapter builds:

- `_before_kernel` from:
  - `PrefetchOp`
  - `H2DPrefetchOp`
- `_after_kernel` from:
  - `EvictVramOp`
  - `EvictDramOp`

### CPU-node snapping behavior

If an op targets a CPU trace node:

- prefetch-like ops snap forward to the next GPU node
- eviction-like ops snap backward to the previous GPU node

Out-of-range ops are dropped.

### Fusion-aware rescaling

After trace-level mapping, the adapter rescales to the actual wrapper kernel
count using linear interpolation:

```python
wrapper_idx = min(gpu_idx * n_wrapper_kernels // n_gpu, n_wrapper_kernels - 1)
```

This is the mechanism that adapts a dense scheduler trace to a smaller fused
Inductor wrapper.

Important current behavior:

- the adapter can map `H2DPrefetchOp` and `EvictVramOp`
- current wrapper injection only consumes the rescaled `PrefetchOp` and
  `EvictDramOp` results

## 7. What `WeightStreamRuntime` Actually Tracks

`WeightStreamRuntime` is a singleton with these main pieces of state:

- `_schedule`
- `_device`
- `_ssd_pool = ThreadPoolExecutor(max_workers=4)`
- `_inflight_reads`
- `_dram_buffers`
- `_h2d_stream`
- `_h2d_events`
- `_vram_tensors`
- `_location`
- `_weight_backups`
- `_weight_storage_nbytes`

For every tensor in `schedule.tensors`, the initial tracked location is `"ssd"`.

### `register_dram_tensor(...)`

Manual hook for preloaded CPU tensors. Stores the tensor in `_dram_buffers` and
marks it as `"dram"`.

### `register_weight(...)`

Used by the current driver path for real model weights.

It:

- takes a GPU tensor
- creates a pinned CPU backup with `gpu_tensor.data.cpu().pin_memory()`
- remembers the original GPU storage size

This enables the tensor-object VRAM restore/evict path used by the generated
wrapper.

## 8. Runtime Methods Used By The Current Wrapper

### `ssd_prefetch(tensor_name)`

If the tensor is still tracked as `"ssd"` and file metadata is available, it
submits `_read_from_file(entry)` to the thread pool and stores the `Future` in
`_inflight_reads`.

This is the current SSD -> DRAM async path.

### `cold_start_prefetch(tensor_name)`

If the tensor is not already in DRAM or VRAM, it:

1. calls `ssd_prefetch(...)`
2. waits for the result immediately
3. stores the pinned CPU tensor in `_dram_buffers`
4. marks the tensor as `"dram"`

So cold-start prefetch is synchronous from the wrapper's perspective.

### `h2d_prefetch(tensor_or_name)`

This has two paths.

#### Tensor-object path

This is the path used by the current wrapper for graph inputs.

If the tensor was registered with `register_weight(...)` and its storage has
been evicted, the runtime:

1. resizes the GPU storage back to its original size
2. launches `gpu_tensor.copy_(backup, non_blocking=True)` on `_h2d_stream`
3. records a CUDA event on `_h2d_stream`
4. makes the current stream wait on that event immediately

This path restores the weight in VRAM without needing a separate emitted
`wait_h2d(...)` call.

#### String-name path

This is the legacy path.

It:

1. completes any pending SSD read
2. or does a blocking `_read_from_file(...)` if the tensor is still on `"ssd"`
3. launches `cpu_tensor.to(device, non_blocking=True)` on `_h2d_stream`
4. records a CUDA event
5. stores the event and the GPU tensor for later `wait_h2d(...)`

### `evict_vram(tensor_or_name)`

Also has two paths.

#### Tensor-object path

If the tensor was registered with `register_weight(...)`, it frees GPU memory
by calling:

```python
gpu_tensor.untyped_storage().resize_(0)
```

#### String-name path

Pops the tensor from `_vram_tensors`, resizes its storage to `0`, and updates
location bookkeeping from `"vram"` to `"dram"`.

### `evict_dram(tensor_name)`

Drops the pinned CPU buffer from `_dram_buffers` and updates bookkeeping back
to `"ssd"`.

### `evict(tensor_name)`

Still exists as a convenience helper:

1. `evict_vram(tensor_name)`
2. `evict_dram(tensor_name)`

The current wrapper does not emit this combined helper anymore. It emits
`evict_vram(...)` and `evict_dram(...)` separately.

## 9. File-Backed Read Path

`_read_from_file(entry)` does not use explicit `pread`.

It currently:

1. maps the dtype string through `_DTYPE_MAP`
2. derives `nbytes` from `shape` and `dtype`
3. creates file-backed storage with `torch.UntypedStorage.from_file(...)`
4. slices to the recorded byte range
5. wraps it as a typed tensor
6. reshapes it
7. calls `.pin_memory()`
8. returns the pinned CPU tensor

Important current behavior:

- `size_bytes` is loaded, but the runtime computes the read size from
  `shape * dtype`
- file-backed SSD reads depend on `entry.file` and `entry.file_offset`

## 10. What `weight_streaming_generated.py` Is

When `config.weight_streaming_output_code` is set, wrapper codegen writes:

- `<output_dir>/weight_streaming_generated.py`

This file is produced by writing `result.getvalue()` from the final generated
wrapper buffer.

That means it is:

- the actual Inductor-generated Python wrapper
- after weight-streaming injection
- not a hand-edited export
- not a prettified or reduced representation

The visible `_ws_rt.*` calls in that file are the ones the wrapper genuinely
compiled with.

## 11. What The Current End-To-End System Supports

Today, the codebase supports this overall flow:

1. load a scheduler-produced plan
2. optionally enrich node kinds from `runtime_nodes.csv`
3. optionally load tensor metadata from `pytorch_runtime_tensors.csv`
4. initialize the runtime
5. register model weights
6. compile the model with Inductor
7. generate wrapper code that can:
   - synchronously cold-start weights into DRAM
   - schedule SSD -> DRAM prefetches
   - restore registered weights from DRAM -> VRAM before first use
   - evict registered weights from VRAM after last use
   - evict named tensors from DRAM after scheduled use

This is real generated-code behavior, not just a dry-run adapter.

## 12. Important Current Limitations

The current implementation still has important constraints:

- there is no FX-level weight replacement pass
- there are no `torch.library` custom ops
- wrapper-side runtime initialization is external to generated code
- `H2DPrefetchOp` and `EvictVramOp` are parsed and adapter-mapped, but the
  wrapper currently uses graph-input first-use / last-use analysis instead of
  directly emitting those scheduled VRAM ops
- `PrefetchOp.after_node`, `PrefetchOp.eager_start`, and
  `ColdStartOp.attach_before_node` are parsed but do not currently affect
  placement
- CPU trace nodes are not first-class insertion points; they are snapped to
  neighboring GPU nodes
- the SSD read path depends on tensor metadata carrying file information
- there is no pinned-buffer reuse policy or explicit global pinned-memory cap
- there is no compile-time/runtime compatibility validation beyond using the
  same schedule inputs in the driver

## 13. Tests In Tree

[`test/inductor/test_weight_streaming.py`](/data/pytorch-source/test/inductor/test_weight_streaming.py)
currently covers:

- scheduler-format schedule loading
- CSV node enrichment
- schedule dataclasses
- GPU/CPU classification
- trace-index snapping
- H2D and VRAM op adapter mapping
- runtime singleton behavior
- VRAM eviction behavior
- H2D behavior
- manual DRAM registration
- file-backed SSD -> DRAM -> VRAM flow

It does not yet provide a full end-to-end assertion over a compiled exported
wrapper file.
