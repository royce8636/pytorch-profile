# Weight Streaming Actual Implementation

This note records what was actually implemented for exact-ID weight streaming,
and how it differs from the original plan.

## Original Plan

The original plan was intentionally narrow:

1. Keep the scheduler's exact `compiled_launch_id` placement.
2. Add `compiled_tensor_id` to VRAM ops in the JSON schedule.
3. Resolve each VRAM op from `compiled_tensor_id` to the compiled wrapper graph input name, for example `compiled_tensor_id=9 -> arg13_1`.
4. Emit VRAM calls with tensor-object arguments:
   - `_ws_rt.h2d_prefetch(arg13_1)`
   - `_ws_rt.evict_vram(arg13_1)`
5. Keep SSD and DRAM operations as string-name calls:
   - `_ws_rt.ssd_prefetch("param_...")`
   - `_ws_rt.evict_dram("param_...")`
6. Remove the graph-structure first-use/last-use fallback for VRAM ops.
7. Avoid runtime changes, because the runtime already had a tensor-object path.

That plan fixed the first correctness issue: VRAM ops should operate on the real model `Parameter` tensors, not on disconnected legacy name dictionaries.

## What Was Actually Implemented

The final implementation is larger than the original plan because exact tensor resolution exposed additional timing and repeated-wrapper issues.

### Schedule Parsing

File: `torch/_inductor/weight_streaming/plan.py`

`H2DPrefetchOp` now carries:

- `before_launch_id`: the consumer launch where the tensor must be ready.
- `after_launch_id`: the launch after which an async H2D copy may start.
- `compiled_tensor_id`: graph-local ID used to resolve the wrapper input.
- `cross_iter`: whether the op reloads a tensor evicted in one wrapper invocation for use in the next invocation.

`EvictVramOp` now carries:

- `after_launch_id`
- `compiled_tensor_id`

`load_io_schedule()` parses these fields from `jit_sim_prune_schedule.json`. For H2D ops, `cross_iter=True` is inferred from:

```python
reason == "h2d_cross_iter_reload"
```

### Tensor-ID Mapping

File: `torch/_inductor/codegen/wrapper.py`

The wrapper now builds a shared graph-local tensor map:

```text
compiled_tensor_id -> graph_input_name
```

This is done by `_build_tensor_id_map()`. The same helper is used by:

- `_write_tensor_map_sidecar()`, when exporting `compiled_tensor_map_graph*.json`
- `_inject_weight_streaming_io()`, when resolving schedule ops during codegen

This matters because the sidecar and the injector must enumerate static graph inputs in exactly the same order. If the order diverges, the scheduler could emit `compiled_tensor_id=9` for one tensor while the injector maps ID `9` to a different wrapper input.

### Variable-Reference VRAM Calls

File: `torch/_inductor/weight_streaming/codegen.py`

`_vram_op_to_call()` resolves `compiled_tensor_id` to a Python variable and emits tensor-object calls:

```python
_ws_rt.h2d_prefetch(arg13_1)
_ws_rt.h2d_prefetch_async(arg13_1)
_ws_rt.evict_vram(arg13_1)
```

If the op lacks a usable `compiled_tensor_id`, codegen raises a hard error. There is no fallback to string-name VRAM calls.

SSD and DRAM ops still use `_op_to_call()` and remain string-name calls.

### Stable Wrapper Aliases

File: `torch/_inductor/codegen/wrapper.py`

The first implementation attempted to emit stable aliases like:

```python
_ws_ref_arg13_1 = arg13_1
```

through `self.prefix`. That placed aliases outside the wrapper function and created invalid generated Python in some cases.

The actual implementation emits aliases as a `WeightStreamingLine` inside the wrapper body, before cold-start prefetches and compute lines:

```python
_ws_ref_arg13_1 = arg13_1
```

Later weight-streaming calls use the alias:

```python
_ws_rt.evict_vram(_ws_ref_arg13_1)
_ws_rt.h2d_prefetch(_ws_ref_arg13_1)
```

The alias keeps a reference to the tensor object even if the generated wrapper later executes `del arg13_1`.

### Exact Launch Placement

File: `torch/_inductor/weight_streaming/codegen.py`

The schedule adapter now separates H2D placement into three maps:

- `before_launch`: synchronous pre-launch ops, including SSD prefetches and H2D ops that have no usable async start.
- `h2d_after_launch`: async H2D starts that should be emitted after the launch identified by `after_launch_id`.
- `h2d_wait_launch`: waits emitted before the consumer launch identified by `before_launch_id`.

This is a major change from the initial implementation. Initially, `after_launch_id` H2D ops were inserted into the pre-launch map for that same launch. That was wrong: an op with `after_launch_id=19` must be emitted after launch `19`, not before launch `19`.

### Async H2D Runtime Path

File: `torch/_inductor/weight_streaming/runtime.py`

The original plan said no runtime changes were needed. That turned out to be wrong once the scheduler started exporting earlier H2D start points.

The tensor-object H2D path now has an async variant:

```python
_ws_rt.h2d_prefetch_async(tensor)
_ws_rt.h2d_wait(tensor)
```

`h2d_prefetch_async()`:

1. Resolves the CPU backup by `id(tensor)`.
2. If there is a pending eviction for the same tensor, synchronizes the eviction event, frees the old storage, and removes the pending eviction.
3. Allocates GPU storage if needed.
4. Records the tensor storage on the H2D stream.
5. Copies from pinned CPU backup to GPU on `_h2d_stream` with `non_blocking=True`.
6. Records a CUDA event in `_h2d_events`.

`h2d_wait()`:

1. Looks up the recorded H2D event by `id(tensor)`.
2. Makes the current compute stream wait on that event.
3. Calls `record_stream()` so the caching allocator knows the current stream is now using the tensor storage.
4. Removes the event from `_h2d_events`.

This runtime change is necessary for real overlap. Without it, moving H2D calls earlier in the wrapper would still do synchronous copies and would not hide the transfer cost.

### Post-Launch H2D Ordering

File: `torch/_inductor/codegen/wrapper.py`

Post-launch emission now orders operations as:

1. VRAM evictions for the launch.
2. Async H2D starts anchored after that launch.
3. Other post-launch ops such as DRAM evictions.
4. `flush_ready_vram_evictions()` when VRAM evictions were queued.

This ordering matters for cases where a tensor is evicted and reloaded around nearby launches. If H2D starts before the matching eviction is queued, the runtime can observe the tensor as still resident, skip the copy, and then the later eviction can free storage before the next consumer uses it.

### Same-Launch H2D Fallback

Some schedule rows have:

```text
after_launch_id == before_launch_id
```

Example encountered during SDXL:

```text
arg46_1: after_launch_id=47, before_launch_id=47
```

This cannot be an async prefetch, because the start point and consumer point are the same launch. Treating it as async produced invalid generated code:

```python
_ws_rt.h2d_wait(_ws_ref_arg46_1)
launch 47 uses arg46_1
_ws_rt.h2d_prefetch_async(_ws_ref_arg46_1)
```

The implementation treats same-launch H2D as synchronous pre-launch restore:

```python
_ws_rt.h2d_prefetch(_ws_ref_arg46_1)
launch 47 uses arg46_1
```

The adapter has explicit `cross_iter` handling for cyclic reload rows, where `after_launch_id >= before_launch_id` may mean start in iteration N and wait in iteration N+1. The code path should be kept aligned so `cross_iter` rows are emitted with async semantics consistently.

### Repeated Wrapper / Cyclic Reload Repair

File: `torch/_inductor/codegen/wrapper.py`

SDXL calls the same compiled UNet wrapper repeatedly across denoising steps. That means exact launch IDs are static wrapper sites, but runtime execution is cyclic.

If a tensor is evicted after its last use in wrapper invocation N, it must be reloaded before its first use in invocation N+1. Some schedules created for a single linear trace did not include that wrap-around reload.

The injector now detects this pattern:

1. Build `graph_input_name -> used_launch_ids`.
2. For each compiled tensor, find its first and last use in the static wrapper.
3. If there is a tail eviction at or after the last use and no prefetch before the first use, synthesize a conservative H2D prefetch before the first use.

This was not in the original plan. It was added after runtime failures where a tensor was correctly evicted near the end of one invocation and then missing at the beginning of the next invocation.

### Cache Disabling For Runner Reproducibility

File: `scripts/run_weight_streaming.py`

The runner now saves and temporarily sets:

```python
inductor_config.force_disable_caches = True
```

This avoids accidentally reusing a stale Inductor wrapper after schedule or injection changes. It was not part of the original design, but it was needed for reliable debugging because stale generated code could make a fixed scheduler or injector look unchanged.

### IO Profiling Script

File: `ws_scripts/profile_weight_streaming_io.py`

A profiling runner was added to measure weight-streaming runtime overhead. It wraps runtime methods, records per-call CSV events, and emits a JSON summary.

It reports information such as:

- call count
- duration
- tensor shape and storage size before/after
- whether a restore happened
- whether SSD read metadata existed
- pending eviction counts
- queued/freed eviction counts

This script was added because the normal `--log-io` output only showed call order, not timing or stall attribution.

## What Changed Compared To The Plan

| Area | Original plan | Actual implementation |
| --- | --- | --- |
| VRAM tensor resolution | Resolve `compiled_tensor_id` to graph input name and emit tensor-object calls. | Implemented. Hard errors are raised for unresolved VRAM ops. |
| Runtime changes | No runtime changes expected. | Runtime changes were required for async tensor-object H2D and wait-before-use. |
| H2D placement | Emit H2D at exact launch ID from schedule. | Split into sync pre-launch H2D, async post-launch H2D start, and pre-consumer wait. |
| `after_launch_id` semantics | Not in the original plan. | Added and required. It means "emit after this launch", not "emit before this launch". |
| Same-launch H2D | Not considered. | `after_launch_id == before_launch_id` falls back to synchronous pre-launch H2D unless handled as a cross-iteration reload. |
| Cross-iteration H2D | Not considered. | `cross_iter=True` was added for reloads that start in one wrapper invocation and are consumed in the next. |
| Repeated wrapper behavior | Not explicitly handled. | Added cyclic reload repair for tail evictions. |
| Stable tensor references | Not in the plan. | Added `_ws_ref_*` aliases inside the wrapper body to survive generated `del arg*` statements. |
| SSD/DRAM path | Keep string-name calls. | Kept as planned. SSD still depends on file metadata and is separate from tensor-object VRAM handling. |
| Graph-structure fallback | Remove/avoid first-use/last-use replacement for exact VRAM ops. | Exact VRAM ops are used, but a limited graph-input usage scan remains for cyclic reload repair and tensor-map sidecar `used_by_launch_ids`. |
| Cache behavior | Not in the plan. | Runner disables Inductor caches to avoid stale generated wrappers. |
| Profiling | Not in the original compiled-ID plan. | Added `ws_scripts/profile_weight_streaming_io.py` to attribute runtime cost. |

## Bugs Encountered And Fixes

### 1. String-Name VRAM Calls Did Not Affect Real Weights

Problem:

```python
_ws_rt.h2d_prefetch("param_45_46")
```

went through the legacy by-name path and did not operate on the real compiled wrapper `Parameter` tensor.

Fix:

```python
_ws_rt.h2d_prefetch(_ws_ref_arg5_1)
```

now routes to the tensor-object path using `id(tensor)`.

### 2. Aliases Were Emitted In The Wrong Scope

Problem:

Aliases emitted through `self.prefix` could appear outside the wrapper body and cause generated Python indentation/scope errors.

Fix:

Aliases are emitted as a normal wrapper line inside the function body.

### 3. Tail Evictions Broke The Next Wrapper Invocation

Problem:

The compiled UNet wrapper is reused across denoising steps. A tensor evicted near the end of one call could be missing at the beginning of the next call.

Fix:

The injector synthesizes a conservative first-use H2D reload for tail-evicted tensors when the schedule lacks a cyclic reload.

### 4. H2D Was Not Hidden

Problem:

The scheduler eventually exported `start_ns` and `after_node`, but PyTorch still needed executable semantics for "start H2D early and wait later".

Fix:

Added async tensor-object H2D:

```python
_ws_rt.h2d_prefetch_async(_ws_ref_arg13_1)
...
_ws_rt.h2d_wait(_ws_ref_arg13_1)
```

### 5. `after_launch_id` Was Initially Interpreted On The Wrong Side

Problem:

An H2D row with:

```text
after_launch_id=19
before_launch_id=21
```

was initially emitted before launch 19. That allowed this bad order:

```python
h2d_prefetch_async(arg13_1)  # sees tensor resident, may no-op
launch 19
evict_vram(arg13_1)         # frees later
launch 21 uses arg13_1      # can see freed storage
```

Fix:

`after_launch_id` H2D starts are emitted after the launch, not before it:

```python
launch 19
evict_vram(arg13_1)
h2d_prefetch_async(arg13_1)
launch 20
h2d_wait(arg13_1)
launch 21 uses arg13_1
```

### 6. Same-Launch Async H2D Was Impossible

Problem:

For rows with:

```text
after_launch_id == before_launch_id
```

the generated code waited before the copy was even issued.

Fix:

Same-launch H2D is synchronous pre-launch H2D unless the row is explicitly intended as a cross-iteration reload.

## Current Executable Semantics

For a normal async H2D row:

```text
after_launch_id = A
before_launch_id = B
A != B
cross_iter = False
```

generated code is:

```python
launch A
_ws_rt.evict_vram(_ws_ref_argX_1)       # if scheduled at A
_ws_rt.h2d_prefetch_async(_ws_ref_argX_1)
...
_ws_rt.h2d_wait(_ws_ref_argX_1)
launch B
```

For same-launch H2D:

```text
after_launch_id = B
before_launch_id = B
cross_iter = False
```

generated code is:

```python
_ws_rt.h2d_prefetch(_ws_ref_argX_1)
launch B
```

For cross-iteration H2D:

```text
cross_iter = True
```

the adapter routes the row through async start/wait maps even when `after_launch_id >= before_launch_id`, because the static wrapper launches repeat every invocation.

## Verification Performed

Standalone adapter repro:

```bash
PYTHONPATH=/data/pytorch-source \
/home/royce/anaconda3/envs/ptvenv/bin/python \
/tmp/repro_ws_after_launch_order.py
```

Targeted unit tests:

```bash
PYTHONPATH=/data/pytorch-source \
/home/royce/anaconda3/envs/ptvenv/bin/python \
test/inductor/test_weight_streaming.py \
TestVramOpResolution.test_vram_ops_in_exact_launch_maps \
TestVramOpResolution.test_async_h2d_after_launch_not_before_launch \
TestVramOpResolution.test_same_launch_h2d_falls_back_to_sync_prelaunch
```

Result:

```text
Ran 3 tests in 0.098s
OK
```

End-to-end SDXL run:

```bash
PYTHONNOUSERSITE=1 \
/home/royce/anaconda3/envs/ptvenv/bin/python \
scripts/run_weight_streaming.py \
  --model /data/llamasim/models/sdxl-turbo/ \
  --export-code /tmp/ws_after_launch_fix_output2 \
  --height 512 \
  --width 512 \
  --steps 8 \
  --plan-dir sdxl-turbo-profile-0414/llama_bundle/jit_sim_prune_compiled_tensor_output/ \
  --log-io
```

Observed result:

```text
GPU memory: 4608.4 MB allocated, 7776.0 MB peak
warmup_seconds: 52.881832
inference_seconds: 0.878249
```

Generated code for the same-launch `arg46_1` case now correctly restores before the consumer launch:

```python
_ws_rt.ssd_prefetch('param_9192_9193')
_ws_rt.h2d_prefetch(_ws_ref_arg46_1)
_ws_m47 = record_function('ws_launch:0:47').__enter__()
```

## Remaining Caveats

1. The SSD path is still string-name based and only performs real file IO if the schedule tensor metadata includes backing file information.
2. The profiling script should be kept in sync with runtime method names. If async H2D timing is the target metric, it should explicitly wrap `h2d_prefetch_async()` and `h2d_wait()`.
3. The scheduler must still produce correct `after_launch_id`, `before_launch_id`, and `cross_iter` semantics. PyTorch now executes those fields more faithfully, so bad schedule placement will be exposed as real runtime stalls or missing-storage failures rather than hidden by a fallback.
4. Same-launch H2D works but is not hidden; it is intentionally synchronous because there is no launch gap available for overlap.
