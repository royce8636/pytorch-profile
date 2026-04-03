# Compiled Launch ID And Tensor ID Plan For Weight Streaming

## Context

The current weight-streaming injection uses two approximate mechanisms:

1. Launch placement: `ScheduleAdapter` linearly rescales GPU-node indices to
   wrapper kernel count. This is inexact because fusion collapses many trace
   nodes into fewer wrapper launches.
2. Tensor identity: SSD/DRAM ops use scheduler string names such as
   `param_45_46` that have no mapping to compiled code entities.

This plan introduces compiler-assigned stable identifiers:

- `compiled_launch_id`: one integer per wrapper compute launch site. This is
  intended to be fully implemented and operational at the end of Phase 2.
- `compiled_tensor_id`: one integer per weight graph input. This is emitted as
  metadata in Phase 3. Runtime behavior and the scheduler do not change until a
  future FQN-based phase.

## Bootstrap Workflow

Profiling must happen before a schedule exists. Metadata emission therefore has
to be decoupled from schedule injection.

Recommended flow:

1. Compile once with `weight_streaming_emit_ids=True` and
   `weight_streaming_emit_launch_markers=True`.
   - Launch/tensor IDs are assigned
   - sidecars are written
   - profiler-visible launch markers are emitted
   - no schedule is injected
2. Profile the compiled model.
   - launch markers appear in the trace as
     `ws_launch:{graph_id}:{launch_id}`
3. Build the schedule and crosswalk from the profiler trace plus sidecars.
4. Compile again with `weight_streaming_plan=<schedule>`.
   - the injector attaches ops by exact launch ID

## Phase 1: Launch Identity Foundation

### 1a. Launch Counter And Fields

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Add to `PythonWrapperCodegen.__init__`:

```python
self._next_launch_id: int = 0
```

Add helper:

```python
def _assign_launch_id(self) -> int:
    lid = self._next_launch_id
    self._next_launch_id += 1
    return lid
```

Launch IDs are always assigned. They are a structural property of the wrapper,
not a config-gated feature.

Add `launch_id: int = -1` to:

- `ExternKernelAllocLine`
- `ExternKernelOutLine`
- `ExternKernelMultiOutLine`
- `KernelCallLine`

Assign `launch_id` at each creation site:

- `generate_kernel_call()`
- `generate_extern_kernel_out()`
- `generate_extern_kernel_alloc()`
- `generate_extern_kernel_multi_out()`

Add a shared tuple:

```python
_COMPUTE_LINE_TYPES = (
    KernelCallLine,
    ExternKernelOutLine,
    ExternKernelAllocLine,
    ExternKernelMultiOutLine,
)
```

### 1b. Config Flags

File: [config.py](/data/pytorch-source/torch/_inductor/config.py)

Add:

- `weight_streaming_emit_ids: bool = False`
- `weight_streaming_emit_launch_markers: bool = False`
- `weight_streaming_tensor_crosswalk: str = ""`

Existing behavior remains:

- `weight_streaming_plan: str = ""` triggers schedule injection

### 1c. Launch Markers Via `LaunchMarkerLine`

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Add a new `WrapperLine` subclass:

```python
@dataclasses.dataclass
class LaunchMarkerLine(WrapperLine):
    wrapper: PythonWrapperCodegen
    inner: WrapperLine
    launch_id: int
    graph_id: int

    def codegen(self, code: IndentedBuffer) -> None:
        tag = f"ws_launch:{self.graph_id}:{self.launch_id}"
        code.writeline(
            f"_ws_m{self.launch_id} = record_function({tag!r}).__enter__()"
        )
        code.writeline("try:")
        with code.indent():
            self.inner.codegen(code)
        code.writeline("finally:")
        with code.indent():
            code.writeline(
                f"_ws_m{self.launch_id}.__exit__(None, None, None)"
            )
```

Apply this in `_generate()` when
`weight_streaming_emit_launch_markers` is enabled:

- add `from torch.profiler import record_function`
- wrap each compute line in `LaunchMarkerLine`

This must run independently of `weight_streaming_plan`.

### 1d. Sidecars And Index File

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

When `weight_streaming_emit_ids` is enabled, write:

- `compiled_launch_map_graph{gid}.json`
- `compiled_tensor_map_graph{gid}.json`
- `compiled_weight_streaming_index.json`

Launch map contents should include:

- `compiled_launch_id`
- `graph_id`
- `launch_kind`
- `kernel_name`
- `original_fxnode_name` when available
- `compilation_hash`

The index file should list all graph sidecars for the current output dir.

Recommended shape:

```json
{
  "graphs": [
    {
      "graph_id": 0,
      "launch_map": "compiled_launch_map_graph0.json",
      "tensor_map": "compiled_tensor_map_graph0.json",
      "compilation_hash": "a1b2c3d4e5f6g7h8"
    }
  ]
}
```

### 1e. Fix `ExternKernelMultiOutLine` Coverage

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Add `ExternKernelMultiOutLine` to:

- the compute-line collection in `_inject_weight_streaming_io()`
- the input-extraction branch used for first-use / last-use analysis

### 1f. Decoupled `_generate()` Flow

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Recommended order:

1. `self.run_wrapper_ir_passes(is_inference)`
2. `self._compute_compilation_hash()`
3. if `weight_streaming_emit_ids`: write sidecars and index
4. if `weight_streaming_emit_launch_markers`: wrap compute lines
5. if `weight_streaming_plan`: inject schedule-driven IO

This separates:

- compile-time metadata emission
- profiling instrumentation
- schedule injection

## Phase 2: Exact SSD/DRAM Placement By Launch ID

### 2a. Add Optional Launch-ID Fields To Schedule Ops

File: [plan.py](/data/pytorch-source/torch/_inductor/weight_streaming/plan.py)

Add:

- `before_launch_id: int = -1` to `PrefetchOp`, `H2DPrefetchOp`, `ColdStartOp`
- `after_launch_id: int = -1` to `EvictVramOp`, `EvictDramOp`

Parse those fields in `load_io_schedule()` when present.

### 2b. Retain Schedule In `ScheduleAdapter`

File: [codegen.py](/data/pytorch-source/torch/_inductor/weight_streaming/codegen.py)

Store:

```python
self._schedule = schedule
```

Add exact maps:

- `before_launch: dict[int, list[PreKernelOp]]`
- `after_launch: dict[int, list[PostKernelOp]]`

These should be keyed directly by `compiled_launch_id`.

### 2c. Use Launch IDs In Injection When Available

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

In `_inject_weight_streaming_io()`:

- inspect both pre and post ops for launch IDs
- when launch IDs are present, attach ops using
  `adapter.before_launch.get(line.launch_id, [])` and
  `adapter.after_launch.get(line.launch_id, [])`
- when absent, fall back to the existing rescaling path

The VRAM graph-structure path remains unchanged.

Important scope note:

- SSD/DRAM ops still use scheduler string names through `_op_to_call()`
- Phase 2 solves exact launch placement only

## Phase 3: Tensor Identity Metadata

This phase is foundation only.

It does:

- emit `compiled_tensor_id` metadata
- classify weight graph inputs
- define a crosswalk format
- provide a diagnostic crosswalk builder

It does not:

- change runtime registration
- change `_op_to_call()`
- change the scheduler’s emitted tensor keys

### 3a. Thread `static_input_idxs` To `GraphLowering`

Files:

- [graph.py](/data/pytorch-source/torch/_inductor/graph.py)
- [compile_fx.py](/data/pytorch-source/torch/_inductor/compile_fx.py)

Add `static_input_idxs: Sequence[int] = ()` to `GraphLowering.__init__` and
store it on the graph object. Pass it through both `GraphLowering`
construction sites in `compile_fx.py`.

### 3b. Validate That `static_input_idxs` Identifies Weights

Add a targeted validation check:

- iterate `V.graph.graph_input_names`
- for each index in `V.graph.static_input_idxs`, verify the corresponding
  `graph_inputs` entry is a `TensorBox`
- warn if this assumption fails

This is a confidence check, not a design blocker.

### 3c. Assign `compiled_tensor_id` To Weight-Only Tensor Inputs

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Use:

- `V.graph.static_input_idxs`
- `TensorBox`

to assign IDs only to tensor inputs that correspond to static inputs.

That excludes:

- user inputs
- symbolic inputs
- `None`
- `TorchBindObject`
- backward-only state
- generator state

### 3d. Write `compiled_tensor_map_graph{gid}.json`

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Each entry should include:

- `compiled_tensor_id`
- `graph_input_name`
- `graph_input_idx`
- `dtype`
- `shape`
- `compilation_hash`

### 3e. Crosswalk Format And Config

File: [config.py](/data/pytorch-source/torch/_inductor/config.py)

Add:

- `weight_streaming_tensor_crosswalk: str = ""`

Crosswalk format:

```json
{
  "compilation_hash": "a1b2c3d4e5f6g7h8",
  "mapping": {
    "param_45_46": 0,
    "param_47_48": 1
  }
}
```

In this phase, the crosswalk is for validation and annotation only.

### 3f. Diagnostic Crosswalk Builder

File: [crosswalk.py](/data/pytorch-source/torch/_inductor/weight_streaming/crosswalk.py)

Add a standalone utility that matches scheduler tensor names to
`compiled_tensor_id` by `dtype + shape`.

Provide two modes:

- `--strict`: fail on ambiguity or missing matches
- `--allow-partial`: emit only confident matches

This is explicitly diagnostic bootstrap tooling, not the production mapping
mechanism.

## Phase 4: Compilation Hash

File: [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)

Add `_compute_compilation_hash()` using:

- `graph_id`
- launch order
- launch type
- kernel identity

Use the hash in:

- launch map sidecars
- tensor map sidecars
- index file
- schedule validation
- crosswalk validation

## Phase 5: Future Work

These are intentionally out of scope for the initial implementation:

- FQN-based crosswalk using `GraphSignature.inputs_to_parameters` and
  `inputs_to_buffers`
- runtime registration by `compiled_tensor_id`
- op payloads carrying optional `compiled_tensor_id`
- `_op_to_call()` preferring ID-based runtime methods
- scheduler emitting `compiled_tensor_id` directly

## Files Modified

- [wrapper.py](/data/pytorch-source/torch/_inductor/codegen/wrapper.py)
- [config.py](/data/pytorch-source/torch/_inductor/config.py)
- [graph.py](/data/pytorch-source/torch/_inductor/graph.py)
- [compile_fx.py](/data/pytorch-source/torch/_inductor/compile_fx.py)
- [plan.py](/data/pytorch-source/torch/_inductor/weight_streaming/plan.py)
- [codegen.py](/data/pytorch-source/torch/_inductor/weight_streaming/codegen.py)
- [crosswalk.py](/data/pytorch-source/torch/_inductor/weight_streaming/crosswalk.py)

## Implementation Order

1. Phase 1: launch IDs, config flags, markers, sidecars, index, MultiOut fix
2. Phase 4: compilation hash
3. Phase 2: exact launch-ID SSD/DRAM placement
4. Phase 3: tensor metadata, weight classification, crosswalk tooling

## Verification

1. Profiling compile pass:
   - compile with `emit_ids=True` and `emit_launch_markers=True`
   - no `weight_streaming_plan`
   - verify sidecars, index, and launch markers
2. Injection compile pass:
   - compile with `weight_streaming_plan=<schedule>`
   - verify exact launch-ID placement when IDs are present
   - verify legacy rescaling still works when IDs are absent
3. Inspect `compiled_launch_map_graph0.json`
   - launch count matches compute-line count
4. Inspect `compiled_tensor_map_graph0.json`
   - tensor count is less than total graph inputs
5. Validate `static_input_idxs` confidence check
6. Run crosswalk builder in `--strict` mode on a model with unique shapes
7. Run crosswalk builder on SDXL and verify ambiguity is reported
8. Verify GPU peak memory matches existing behavior
9. Change a kernel and verify the `compilation_hash` changes
10. Compile with graph breaks and verify separate sidecars plus one index

## Remaining Implementation Findings

These are not core design blockers, but they should be handled in the final
implementation.

### 1. Marker Wrapping Versus Injection Order

If launch markers are wrapped before `_inject_weight_streaming_io()` runs, the
injector can no longer discover compute lines by a plain `_COMPUTE_LINE_TYPES`
scan, because those lines are now inside `LaunchMarkerLine`.

Acceptable fixes:

- make `emit_launch_markers` and `weight_streaming_plan` mutually exclusive
- wrap with `LaunchMarkerLine` after `_inject_weight_streaming_io()`
- or teach `_inject_weight_streaming_io()` to unwrap `LaunchMarkerLine.inner`

### 2. Index File Must Replace Stale Entries

The index file should not append forever.

Graph IDs are reused across recompiles, so the writer should replace entries by
`graph_id` or by `(graph_id, compilation_hash)` rather than accumulate stale
duplicates.

### 3. Keep Phase 1 And Phase 3 Rollout Honest

`weight_streaming_emit_ids` should not promise tensor sidecars before Phase 3
lands.

Recommended rollout wording:

- Phase 1: launch sidecars plus index
- Phase 3: tensor sidecars added to the same metadata path

### 4. Crosswalk Is Metadata Until Runtime Payloads Change

In the current implementation, SSD/DRAM ops are still emitted by scheduler
string name, and runtime SSD/DRAM methods are still string keyed.

So in the initial implementation:

- `compiled_tensor_id` is metadata
- the crosswalk is validation/annotation only
- exact operational tensor identity remains future work
