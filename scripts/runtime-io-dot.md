# Runtime-I/O DOT

This document explains what `runtime-io.dot` is, where its data comes from,
what the timings mean, and what files were changed to add it.

## What `runtime-io.dot` is

`runtime-io.dot` is a post-fusion runtime graph.

It is built from two profiler data sources:

- Kineto runtime events for timing and execution ordering
- Execution Trace Observer JSON for exact recorded op inputs and outputs

The goal is:

- keep the runtime graph aligned with what actually ran
- attach exact external tensor arguments where Execution Trace recorded them

The goal is not:

- reconstruct internal values inside a fused Triton kernel
- recover Python variable names for tensors
- replace the CSV export

## Where the timing comes from

The timing in `runtime-io.dot` comes only from Kineto.

The implementation reads raw Kineto events from:

- `prof.profiler.kineto_results.events()`

in:

- [profile_sdxl_turbo_common.py](./profile_sdxl_turbo_common.py)

The relevant helper is `get_raw_kineto_events(...)`.

For each DOT runtime node:

- `start_us` is computed as `(event.start_ns() - trace_start_ns) / 1000.0`
- `dur_us` is computed as `event.duration_ns() / 1000.0`

`trace_start_ns` comes from:

- `prof.profiler.kineto_results.trace_start_ns()`

This means:

- `start_us` is a trace-relative timestamp
- `dur_us` is the event duration reported by Kineto
- these values do not come from Execution Trace JSON
- these values do not come from `trace.csv`

## What the runtime edges mean

The graph backbone is the same runtime structure used by `kineto.dot`.

Edges are built from raw Kineto events:

- `cpu` dashed edge:
  parent/child nesting for synchronous CPU events on the same thread
- `launch` orange edge:
  CPU event to linked non-CPU event using Kineto `linked_correlation_id()`
- `linked` orange edge:
  CPU-to-CPU linked relationship when Kineto reports one
- `stream` dotted blue edge:
  ordering of non-CPU events on the same `(device_type, device_index, device_resource_id)`

This is runtime structure, not tensor dataflow structure.

## Where the tensor inputs/outputs come from

The tensor I/O overlay comes from Execution Trace Observer JSON, not from Kineto.

The script loads the JSON file written by:

- `torch.profiler.ExecutionTraceObserver`

and reads ET nodes with:

- `inputs.values`
- `inputs.shapes`
- `inputs.types`
- `outputs.values`
- `outputs.shapes`
- `outputs.types`
- `attrs[name="rf_id"]`

The underlying PyTorch implementation that writes those fields is:

- [execution_trace_observer.cpp](../torch/csrc/profiler/standalone/execution_trace_observer.cpp)

For tensors, ET records a 6-field value tuple:

- `tensor_id`
- `storage_id`
- `offset`
- `numel`
- `itemsize`
- `device`

In the exporter, a pink tensor node is created from that exact tuple.

The tensor node label shows:

- `tensor_id`
- `storage_id`
- `offset`
- `numel`
- `itemsize`
- `device`
- ET `type`
- ET `shape`

Edge labels:

- `in0`, `in1`, ...
- `out0`, `out1`, ...

For nested ET inputs, the label uses the flattened path, for example:

- `in3.0`

Scalar and non-tensor ET arguments are not turned into nodes. They stay inline
on the runtime node label as:

- `arg0=...`
- `arg1=...`

## How ET nodes are matched to runtime nodes

The exporter does not guess based on timing.

It matches ET nodes to Kineto runtime nodes by `rf_id`.

Concretely:

- ET node key: `attrs[name="rf_id"]`
- Kineto-side key: `event.correlation_id()` for CPU runtime events

The script assumes these refer to the same record-function identity. That
assumption is already validated by existing profiler tests in:

- [test_execution_trace.py](../test/profiler/test_execution_trace.py)

The matching policy is intentionally strict:

- if exactly one ET node and exactly one CPU runtime event share the same `rf_id`, the match is accepted
- otherwise the overlay is skipped for that runtime node

The DOT label then shows:

- `io_match=ok`
- `io_match=missing`
- `io_match=ambiguous`

This is why `runtime-io.dot` is exact where it draws I/O, instead of doing
best-effort timing/name inference.

## Why the tensor edges usually attach to CPU-side runtime nodes

Execution Trace Observer is built on `RecordFunction`, so its node identity is
CPU-side.

That means the exact ET tensor inputs naturally attach to the matched CPU
runtime node, for example a Triton launch record-function node.

The GPU kernel event still exists in the graph as a Kineto runtime node and is
connected by the normal Kineto `launch` edge.

So the structure is usually:

- CPU launch/runtime node with exact ET inputs
- `launch` edge
- GPU kernel runtime node with device timing

This is the most exact mapping available without adding new compiler/profiler
metadata.

## What it does not show

`runtime-io.dot` does not show:

- internal intermediates inside a fused Triton kernel
- Python variable names like `x`, `hidden_states`, `latent`
- guaranteed outputs for Triton nodes if ET did not record them

One important current limitation is that Triton ET nodes often have inputs but
no outputs. PyTorch's existing ET tests already encode that behavior.

So for Triton kernels:

- exact external inputs are often available
- exact outputs may be absent
- internal fused-kernel dataflow is not available

## Files modified for this feature

Runtime-I/O export itself changed these files:

- [scripts/profile_sdxl_turbo_common.py](./profile_sdxl_turbo_common.py)
  - added CLI support for `--dot-level runtime-io`
  - added output path handling for `*_runtime_io.dot`
  - added ET loading/parsing helpers
  - added strict `rf_id` join logic
  - added `write_runtime_io_profile_dot(...)`
  - wired the export into the main profiling flow
- [test/profiler/test_execution_trace.py](../test/profiler/test_execution_trace.py)
  - added a small tensor-tuple extraction regression
  - strengthened Triton ET tests so Triton inputs must include tensor-valued entries

Files used as existing data sources, but not modified for this feature:

- [torch/csrc/profiler/standalone/execution_trace_observer.cpp](../torch/csrc/profiler/standalone/execution_trace_observer.cpp)
  - existing ET JSON writer
- [torch/autograd/profiler.py](../torch/autograd/profiler.py)
  - existing profiler event materialization / Kineto integration

## CLI usage

Example:

```bash
PYTHONNOUSERSITE=1 /tmp/ptvenv/bin/python /data/pytorch-source/scripts/profile_sdxl_turbo_gpu.py \
  --output-dir /tmp/sdxl_gpu_inductor \
  --fusion inductor \
  --dot-level runtime-io \
  --disable-progress-bar
```

This writes:

- trace JSON
- trace CSV
- output PNG
- `*_runtime_io.dot`

If `--execution-trace` is not provided, the script uses the default ET path in
the selected output directory.

## How this differs from the other DOT modes

- `execution.dot`
  - built only from Execution Trace JSON
  - shows ET graph structure and ET tensor nodes
  - does not use Kineto timings for node timestamps
- `memory.dot`
  - built from PyTorch memory profiler data-flow graph
  - shows logical tensor flow for selected leaf ops
  - not an exact post-fusion runtime graph
- `kineto.dot`
  - built only from Kineto runtime events
  - most faithful to visible runtime execution
  - no tensor nodes
- `hybrid.dot`
  - Kineto runtime graph plus best-effort memory-graph tensor overlay
  - can miss or distort post-fusion boundaries
- `runtime-io.dot`
  - Kineto runtime graph plus exact ET-recorded external I/O where `rf_id` matches unambiguously

## Verification that was run

The implementation was verified with:

- Python syntax/bytecode compilation of the edited script and test file
- a standalone CPU repro that generated a `runtime-io.dot`
- a targeted unit test for ET tensor-tuple extraction

The Triton GPU-specific ET assertions were added, but full end-to-end Triton
test execution still depends on a working CUDA-enabled PyTorch runtime with
Triton available.
