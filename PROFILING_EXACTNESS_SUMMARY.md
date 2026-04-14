# Profiling Exactness Changes

This document summarizes the changes made to make the PyTorch profiling export
more exact for LLaMASim runtime bundles, with emphasis on where each piece of
evidence comes from, what is exact, what is still inferred, and what artifacts
are emitted for downstream graph modifiers.

## Goal

The original runtime export could build useful LLaMASim graphs, but important
relationships were either inferred from timing/correlation heuristics or not
explicitly labeled with provenance. The main goal was to make the critical
runtime dependencies auditable:

- CPU submit API -> GPU runtime work should be exact when trace evidence exists.
- GPU work -> CPU wait/sync API should be exact when wrapper or ATen evidence
  proves the waited-on work.
- Fallback heuristic edges should still exist, but must be labeled as heuristic.
- The bundle should expose counters and diagnostics so a user can see exactly
  how much of the graph is evidence-backed.
- Compiled launch/tensor sidecars should be emitted for compiled-tensor-space
  LLaMASim schedulers.

## Output Bundle Artifacts

The LLaMASim runtime bundle now contains the runtime graph plus sidecar metadata:

- `step_0_compute_graph.dot`
- `ggml_profile_node_records.csv`
- `runtime_nodes.csv`
- `runtime_edges.csv`
- `pytorch_runtime_tensors.csv`
- `manifest.json`
- `compiled_launch_map_graph*.json` when Inductor ID emission is enabled
- `compiled_tensor_map_graph*.json` when Inductor ID emission is enabled
- `compiled_weight_streaming_index.json` when Inductor ID emission is enabled

The compiled sidecars are required by
`/data/llamasim/graph_modifiers/jit_sim_prune_compiled_tensor`.
That modifier auto-detects `compiled_launch_map_graph*.json` and
`compiled_tensor_map_graph*.json` in the input bundle folder.

## Edge Kinds In The Runtime Graph

The final `runtime_edges.csv` currently emits only these edge kinds:

- `thread_order`: consecutive CPU leaf events on the same CPU thread.
- `stream_order`: consecutive GPU runtime events on the same GPU stream.
- `submit`: CPU CUDA runtime/driver launch or memcpy API -> GPU runtime event.
- `wait`: GPU runtime event -> CPU sync/wait API.
- `data_input`: tensor -> runtime node.
- `data_output`: runtime node -> tensor.

Only submit and wait edges currently carry explicit exactness fields.
`thread_order` and `stream_order` are deterministic ordering edges from Kineto
timestamps/resources. Tensor edges are evidence-backed from execution-trace
tensor metadata, with parent-to-GPU propagation when direct kernel tensor I/O is
not available.

## Exact Submit Edges

Submit edges were changed from a correlation-only path to an exact Chrome trace
`ac2g` endpoint path when trace JSON is available.

### Evidence Source

The exporter parses Chrome trace events with category `ac2g`. These encode the
CPU-to-GPU flow for a CUDA launch/memcpy correlation ID. For each paired
correlation ID, the exporter resolves:

- CPU endpoint: the CUDA runtime/driver API event.
- GPU endpoint: the GPU runtime/kernel/memcpy event.

The raw Kineto events are matched back to those endpoints by correlation ID,
timestamp, and event category.

### Exactness Rule

A submit edge is exact when:

- The Chrome trace has paired `ac2g` CPU and GPU endpoints for the correlation
  ID.
- Both endpoints can be matched to raw Kineto events without ambiguity.
- Both matched events are selected into the runtime graph.

The edge is then emitted as:

```csv
edge_kind=submit
submit_exact=true
```

When `ac2g` evidence is unavailable, the exporter can still fall back to the
older correlation-based submit builder, but those submit edges are marked:

```csv
submit_exact=false
```

### Diagnostics

The manifest now reports:

- `submit_edge_count`
- `exact_submit_edge_count`
- `heuristic_submit_edge_count`
- `submit_edge_provenance`
- `ac2g_paired_count`
- `ac2g_endpoint_matched_count`
- `ac2g_endpoint_match_error_count`

For the checked `sdxl-turbo-profile-0414` bundle, submit exactness was:

```text
submit_edge_count: 16217
exact_submit_edge_count: 16217
heuristic_submit_edge_count: 0
submit_edge_provenance: ac2g_endpoint_exact
```

## Exact Wait Edges From Wrapper Markers

The wrapper path was extended so wrapper-owned synchronizations can produce
exact wait edges instead of timing-based heuristic wait edges.

### Wrapper Line Types

The wrapper now has typed synchronization lines:

- `DeviceSynchronizeLine`
- `StreamSynchronizeLine`
- `WaitEventLine`
- `EventSynchronizeLine`
- `EventRecordLine`

These replace or supplement raw string emission for wrapper-owned syncs. Typed
lines let the marker pass know what kind of wait is being emitted.

### Launch Markers

Each compiled compute line can be wrapped with:

```text
ws_launch:{graph_id}:{launch_id}
```

The exporter uses these markers to map runtime CPU/GPU events back to static
compiled launch IDs.

### Sync Markers

Each typed sync line can be wrapped with:

```text
ws_sync:{graph_id}:{sync_id}:{kind}:{src_launch_ids}
```

The `kind` field identifies the synchronization type:

- `device_sync`
- `stream_sync`
- `wait_event`
- `event_sync`

The `src_launch_ids` field records the exact compiled launches that the wrapper
sync should wait for.

### Frontier Tracking

The wrapper marker pass tracks cumulative launch frontiers:

- A device sync records all prior launch IDs on all tracked streams.
- A stream sync records all prior launch IDs on that stream.
- An event record snapshots the current stream frontier into an event frontier.
- A stream wait-event records the frontier associated with that event and merges
  it into the waiting stream.
- A host event synchronize records the frontier associated with that event.

This is exact for wrapper-owned code because the wrapper is the code that emits
the launch and sync operations.

### Exporter Mapping

Kineto records `record_function("ws_sync:*")` as a CPU parent range. The runtime
graph keeps CPU leaves, so the exporter propagates the `ws_sync` metadata to
the matching leaf CUDA sync/wait event, such as:

- `cudaDeviceSynchronize`
- `cudaStreamSynchronize`
- `cudaEventSynchronize`
- `cudaStreamWaitEvent*`

Then the exporter resolves each `src_launch_id` to the corresponding GPU runtime
node through the `ws_launch` index and emits:

```csv
edge_kind=wait
wait_exact=true
wait_source=wrapper_ws_sync
wait_kind=<device_sync|stream_sync|wait_event|event_sync>
wait_source_launch_ids=<compiled launch ids>
```

### Failure Diagnostics

The manifest reports:

- `ws_sync_marker_count`
- `ws_sync_leaf_count`
- `ws_sync_unmatched_marker_samples`
- `ws_sync_unresolved_launch_samples`
- `ws_sync_empty_source_count`

The runtime summary prints explicit notes when wrapper markers are absent,
unmatched, or unable to resolve launch IDs.

## Exact Wait Edges From ATen Parent Scopes

Not all waits are wrapper-owned. Common ATen operations can launch GPU work and
then synchronously wait for it inside the same ATen parent scope. These are now
handled as exact when the parent scope proves the wait source.

### Allowlisted Parent Scopes

The exact ATen wait path currently applies to sync leaves under:

- `aten::copy_`
- `aten::_to_copy`
- `aten::nonzero`
- `aten::_local_scalar_dense`

These are operations where the parent CPU op commonly launches a GPU memcpy or
kernel and then synchronizes before returning to the caller.

### Evidence Source

For a sync leaf under an allowlisted ATen parent, the exporter collects CPU
descendant submit APIs under the same parent:

- `cudaLaunchKernel`
- `cuLaunchKernel`
- `cudaMemcpyAsync`
- `cudaMemcpy2DAsync`
- `cudaMemcpy3DAsync`
- `cudaMemsetAsync`

It then finds GPU events whose correlation IDs match those submit APIs.

### Exactness Rule

An ATen wait edge is exact only when:

- The sync leaf is under an allowlisted ATen parent.
- The same parent contains the CPU submit event for the candidate GPU work.
- The candidate GPU event correlation matches that submit event.
- The GPU event starts after the parent starts.
- The GPU event ends no later than the sync leaf returns.
- For `cudaStreamSynchronize`, the GPU event is on the synchronized stream when
  Kineto exposes the stream, or there is only one candidate stream.

The important safety guard is:

```python
if gpu_event.end_ns() > sync_event.end_ns():
    continue
```

This prevents marking a wait edge exact if the candidate GPU work had not
actually completed when the sync returned.

ATen parent-scope evidence is not used to mark `cudaDeviceSynchronize` exact,
because a device sync waits all streams and the local parent scope does not prove
all unrelated stream work.

### Exported Attributes

Exact ATen waits are emitted as:

```csv
edge_kind=wait
wait_exact=true
wait_source=aten_parent_scope_exact
wait_kind=stream_sync
wait_source_correlation_ids=<source correlation ids>
```

For the checked `sdxl-turbo-profile-0414` bundle, ATen exact wait coverage was:

```text
wait_edge_count: 9
exact_wait_edge_count: 8
aten_exact_wait_edge_count: 8
heuristic_wait_edge_count: 1
wait_source_counts: aten_parent_scope_exact=8, kineto_sync_frontier=1
wait_kind_counts: stream_sync=8, device_sync=1
```

## Heuristic Wait Fallbacks

Heuristic wait builders still exist for syncs that do not have wrapper or ATen
exact evidence.

### Linked Family Heuristic

The linked-family path groups CPU and GPU events by linked correlation ID. If a
CPU event overlaps the completion of a GPU family and has not already been
classified as submit or exact wait, it can be linked as a heuristic wait.

This emits:

```csv
wait_exact=false
wait_source=linked_family_heuristic
```

### Sync Frontier Heuristic

For remaining CUDA sync APIs, the fallback picks the latest completed GPU event
per relevant stream before the sync returns.

This emits:

```csv
wait_exact=false
wait_source=kineto_sync_frontier
wait_kind=<device_sync|stream_sync|event_sync>
```

In the checked `sdxl-turbo-profile-0414` bundle, the only heuristic wait was a
top-level `cudaDeviceSynchronize` linked to the latest completed GPU event.

## Profiler-Owned Sync Exclusion

The profiler driver performs a final synchronization to flush device work before
the profiler exits. That sync is infrastructure, not model behavior.

The final sync is now wrapped with:

```text
profiler_sync:device
```

The exporter excludes CUDA sync leaves nested under `profiler_sync:*` markers
from runtime node selection. This prevents the profiling flush from creating a
misleading model wait edge.

The manifest reports:

```text
excluded_wait_source_counts: profiler_sync_excluded=<count>
profiler_sync_excluded_count=<count>
```

## Compiled Launch And Tensor IDs

The runtime graph now carries compiled ID information when Inductor marker/ID
emission is enabled.

### Runtime Node Fields

`runtime_nodes.csv` includes:

- `compiled_graph_id`
- `compiled_launch_id`

These fields are populated from `ws_launch` markers when a runtime event is
inside or linked to a compiled launch marker.

### DOT Node Attributes

The DOT output also includes compiled launch attributes, so downstream tooling
can read IDs from either CSV sidecars or the DOT graph.

### Sidecar Files

The compiled wrapper emits:

- `compiled_launch_map_graph{gid}.json`
- `compiled_tensor_map_graph{gid}.json`
- `compiled_weight_streaming_index.json`

The launch map records static compiled launch IDs. The tensor map records
compiled tensor IDs, graph input names, sizes, dtypes, shapes, and the launch
IDs that use each compiled tensor.

These files are emitted into the LLaMASim bundle directory when:

```python
torch._inductor.config.weight_streaming_emit_ids = True
torch._inductor.config.weight_streaming_output_code = <bundle dir>
```

## Marker Configuration

The profile entrypoints now configure Inductor marker and sidecar emission when
a LLaMASim bundle is requested.

`configure_llamasim_inductor_markers(output_dir)` enables:

```python
weight_streaming_emit_ids = True
weight_streaming_emit_launch_markers = True
weight_streaming_emit_sync_markers = True
weight_streaming_output_code = str(output_dir)
```

If any of those compile-affecting settings change, it calls:

```python
torch._dynamo.reset()
```

This is needed because marker settings affect generated Inductor wrapper code.
Without a Dynamo reset, an already-cached compiled graph could be reused without
the new marker instrumentation.

The marker configuration is wired into:

- `scripts/profile_sdxl_turbo_common.py`
- `scripts/profile_qwen_image_common.py`
- `scripts/profile_accelerate_cpu_offload.py`
- `scripts/run_weight_streaming.py`

## SDXL GPU Default

`scripts/profile_sdxl_turbo_gpu.py` now defaults to:

```python
default_fusion="inductor"
default_dot_level="llamasim-runtime"
default_llamasim_output_dirname="llama_bundle"
```

This matters because compiled maps cannot be produced in eager mode. A default
SDXL GPU profile run now compiles the UNet, emits the LLaMASim runtime bundle,
and writes the compiled launch/tensor sidecars under the bundle directory.

With `--output-dir DIR`, the default bundle path is:

```text
DIR/llama_bundle
```

Without `--output-dir`, the default bundle path follows the trace path stem:

```text
/tmp/sdxl_turbo_gpu_inductor_llamasim_runtime
```

## Verification Changes

`scripts/verify_llamasim_bundle.py` was updated to rebuild expected runtime
edges using the same exact submit and wait logic as the exporter.

It can verify:

- Internal bundle consistency.
- Edge/node/tensor CSV consistency.
- Runtime edge kind validity.
- Manifest count consistency.
- Exact submit edges against Chrome trace `ac2g` endpoint evidence when source
  artifacts are available.
- Rebuilt runtime edges against the exported bundle when source artifacts are
  available.

Source-backed verification requires the original source artifacts, including the
Kineto map CSV. If the profile directory lacks `*_kineto_map.csv`, the verifier
can still pass internal consistency, but it will warn that source-backed
verification was not run.

## Current Checked Profile Example

For `sdxl-turbo-profile-0414/llama_bundle`, the checked profile had:

```text
submit_edges: exact=16217/16217, heuristic=0
wait_edges: exact=8/9, heuristic=1
wait_sources: aten_parent_scope_exact=8, kineto_sync_frontier=1
wait_kinds: stream_sync=8, device_sync=1
ws_sync markers: 0
profiler syncs excluded: 1
compiled launch/tensor maps: present
```

The one heuristic wait was the remaining top-level `cudaDeviceSynchronize`.
The eight exact waits were stream synchronizations inside ATen parent scopes.

## What Is Still Not Exact Or Not Modeled

The graph is more exact, but it is not a complete happens-before model of every
runtime cause.

Remaining limitations:

- Wrapper exact waits only appear when the sync is emitted through typed wrapper
  sync lines and `ws_sync` markers are present.
- Default SDXL currently has no wrapper `ws_sync` markers in the checked run, so
  exact waits came from ATen parent-scope evidence instead.
- Non-wrapper, non-ATen syncs still fall back to heuristics unless instrumented
  or explicitly excluded.
- Device sync exactness is conservative. ATen parent-scope evidence is not
  enough to prove all streams waited by a device sync.
- Library-internal waits inside cuDNN/cuBLAS/ATen are not split into separate
  graph nodes unless they appear as separate Kineto events.
- CPU cross-thread dependencies, locks, queues, dataloader handoff, and async
  file-read completion are not explicitly modeled.
- Allocator dependencies, memory-pool reuse constraints, and resource
  contention are not explicit graph edges.
- Tensor I/O exactness is not represented with an explicit exact/heuristic flag.
  Some GPU tensor I/O is propagated from parent execution-trace nodes when the
  GPU runtime event has no direct execution-trace tensor metadata.

## Practical Interpretation

After these changes, the critical runtime dependency fields are auditable:

- If `submit_exact=true`, the submit edge is backed by exact `ac2g` endpoint
  evidence.
- If `wait_exact=true` and `wait_source=wrapper_ws_sync`, the wait edge is backed
  by wrapper-emitted launch/sync provenance.
- If `wait_exact=true` and `wait_source=aten_parent_scope_exact`, the wait edge
  is backed by same-parent ATen submit/sync correlation plus the requirement
  that the GPU work completed before the sync returned.
- If `wait_exact=false`, the edge is intentionally marked as heuristic and the
  `wait_source` field explains which fallback produced it.

This makes exactness visible in the bundle rather than implicit in exporter
implementation details.
