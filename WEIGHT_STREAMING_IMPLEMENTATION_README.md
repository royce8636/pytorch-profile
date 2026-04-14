# Weight Streaming Exact-ID Implementation README

## Overview

This document explains the goal of the weight-streaming work, the method used
to make it correct, the concrete problems encountered during implementation,
and how each problem was solved.

It is intentionally detailed. It is meant to be a technical record of:

- what the system was trying to do
- why the original implementation was insufficient
- how exact launch identity and tensor identity were introduced
- how the profiling bundle, scheduler, and Inductor wrapper were connected
- what failed in practice
- what was fixed
- what is correct today
- what is still intentionally out of scope

This README is descriptive. It explains the current design and the engineering
path that led to it.

## The Goal

The goal of weight streaming is to reduce peak GPU memory usage during model
execution by moving selected weights between storage tiers at specific points
in execution.

At a high level, the system tries to:

1. profile a compiled model run
2. build a schedule that says when a weight should:
   - be prefetched from storage
   - be restored into GPU memory
   - be evicted from GPU memory
   - be spilled out of DRAM
3. inject those operations back into the generated compiled wrapper
4. execute the model with lower GPU peak memory

For SDXL Turbo, the concrete target was:

- preserve correct image generation
- reduce GPU peak memory materially
- attach prefetch/eviction calls at the correct compiled execution points
- avoid approximate placement and naming hacks

The key requirement that emerged during the work was:

- even if the schedule is suboptimal, execution must still follow the schedule
  faithfully

That requirement is what drove the exact-ID redesign.

## The Original Problems

The original weight-streaming injection had two fundamental approximation
problems.

### 1. Launch placement was approximate

The scheduler reasoned in terms of profiled GPU nodes. The compiled wrapper,
however, emits a smaller set of wrapper compute launches because fusion and
codegen collapse many traced GPU events into fewer wrapper call sites.

As a result, the old system used linear rescaling:

- take the profiled GPU-node index
- compress it onto the number of wrapper kernel launches
- attach the IO call near that approximated position

This was structurally wrong:

- one traced GPU node was not equal to one wrapper launch
- one wrapper launch could correspond to multiple runtime GPU leaf events
- repeated model invocations across diffusion steps multiplied the mismatch

This meant the runtime could be doing the right kind of operation at the wrong
time.

### 2. Tensor identity was approximate or missing

The scheduler emitted tensor names such as:

- `param_45_46`
- `cache_35_36`

Those names came from profiler/bundle metadata, not from compiled codegen.

The compiled wrapper, however, works with:

- graph input variable names such as `arg5_1`
- real tensor objects
- registered model parameters and buffers

There was no authoritative crosswalk between:

- scheduler tensor names
- compiled graph inputs
- runtime tensor objects

This had a particularly bad effect on VRAM ops:

- the schedule might say “H2D prefetch `param_45_46` before launch X”
- the wrapper would emit `_ws_rt.h2d_prefetch('param_45_46')`
- but the runtime’s by-name path did not refer to the actual registered model
  parameter object the compiled kernel uses

So the op could be attached at the right point but still target the wrong
runtime contract.

## The Core Design Change

The fix was to introduce compiler-assigned stable identities.

### `compiled_launch_id`

Each wrapper compute launch site gets a stable integer:

- local to one compiled graph
- emitted at codegen time
- persisted in sidecar metadata
- exposed in the profiler trace through `ws_launch:{graph_id}:{launch_id}`

This solved launch placement.

### `compiled_tensor_id`

Each compiled weight graph input gets a stable integer:

- local to one compiled graph
- derived from compiled graph input order, filtered to static tensor inputs
- emitted in a tensor sidecar
- later consumed by the scheduler and injector to recover the correct graph
  input variable

This solved the missing tensor identity needed for exact VRAM ops.

## The End-to-End Method

The exact-ID method is a two-pass workflow.

### Pass 1: Profiling compile

Compile the model with:

- `weight_streaming_emit_ids=True`
- `weight_streaming_emit_launch_markers=True`
- no schedule injection

This does three things:

1. assigns `compiled_launch_id` to each wrapper compute launch
2. assigns `compiled_tensor_id` to each compiled weight graph input
3. emits profiler-visible `record_function` scopes:
   - `ws_launch:{graph_id}:{launch_id}`

It also writes sidecar files:

- `compiled_launch_map_graph{gid}.json`
- `compiled_tensor_map_graph{gid}.json`
- `compiled_weight_streaming_index.json`

### Pass 2: Schedule building

The profiling bundle plus sidecars are then fed to `jit_sim_prune`.

The scheduler uses:

- `runtime_nodes.csv`
- `pytorch_runtime_tensors.csv`
- `compiled_launch_map_graph{gid}.json`
- `compiled_tensor_map_graph{gid}.json`

to produce `jit_sim_prune_schedule.json`.

The schedule contains:

- exact launch placement:
  - `before_launch_id`
  - `after_launch_id`
- exact tensor identity for VRAM ops:
  - `compiled_tensor_id`

### Pass 3: Injection compile

Compile the model again with:

- `weight_streaming_plan=<schedule>`

The Inductor wrapper injector then:

- loads the schedule
- validates `compilation_hash`
- finds the wrapper compute line with matching `launch_id`
- emits weight-streaming runtime calls before or after that line

The important distinction is:

- SSD/DRAM ops still use string-name runtime calls
- VRAM ops use variable-reference calls on real graph inputs

That distinction is deliberate and comes from how the runtime already works.

## Main Files And Their Roles

### PyTorch side

- `torch/_inductor/codegen/wrapper.py`
  - assigns launch IDs
  - writes sidecars
  - emits launch markers
  - injects weight-streaming calls

- `torch/_inductor/weight_streaming/plan.py`
  - defines schedule dataclasses
  - loads `jit_sim_prune_schedule.json`
  - enriches from CSV metadata

- `torch/_inductor/weight_streaming/codegen.py`
  - maps schedule ops to emitted runtime calls
  - contains exact launch maps and legacy rescale support

- `torch/_inductor/weight_streaming/runtime.py`
  - implements the runtime behavior of:
    - `ssd_prefetch(...)`
    - `cold_start_prefetch(...)`
    - `h2d_prefetch(...)`
    - `evict_vram(...)`
    - `evict_dram(...)`

- `scripts/profile_sdxl_turbo_gpu.py`
  - entrypoint for profiling and bundle creation

- `scripts/profile_sdxl_turbo_common.py`
  - shared profiling/bundle export logic

- `ws_scripts/run_weight_streaming_plan.py`
  - execution entrypoint for schedule-injected runs

### Llamasim side

- `/data/llamasim/graph_modifiers/jit_sim_prune/main.py`
  - loads the bundle
  - runs the solver
  - writes the schedule

- `/data/llamasim/graph_modifiers/jit_sim_prune/scheduler.py`
  - builds launch-ID and tensor-ID mappings
  - filters unsupported operations
  - produces exact schedule attachment metadata

## How Exact Launch Identity Works

The launch-ID solution has four parts.

### 1. Launch IDs are assigned during wrapper construction

Each compute line in the wrapper gets a monotonically increasing
`launch_id`.

The relevant compute line types are:

- `KernelCallLine`
- `ExternKernelOutLine`
- `ExternKernelAllocLine`
- `ExternKernelMultiOutLine`

This means launch identity is attached to the wrapper’s actual executable
structure, not reconstructed later from profiler leaf events.

### 2. Launch markers are emitted into generated code

The generated wrapper uses `record_function(...)` to wrap each compute launch:

```python
_ws_m42 = record_function("ws_launch:0:42").__enter__()
try:
    ...
finally:
    _ws_m42.__exit__(None, None, None)
```

This gives the profiler an exact bridge between:

- compiled wrapper launch site
- runtime execution scope

### 3. Sidecars describe launch identity explicitly

`compiled_launch_map_graph0.json` contains:

- `compiled_launch_id`
- `graph_id`
- launch kind
- kernel name
- original FX node name when available
- `compilation_hash`

This file answers:

- what does launch 42 refer to?
- which graph did it come from?
- which compile did it belong to?

### 4. The profiler bundle must carry launch IDs forward

The sidecar alone is not enough. The runtime bundle also needs to identify GPU
nodes with the launch they belong to.

This required `runtime_nodes.csv` to carry:

- `compiled_graph_id`
- `compiled_launch_id`

Once that happened, `jit_sim_prune` no longer needed to guess by count and
order.

## How Exact Tensor Identity Works

Launch placement alone was not enough. VRAM ops also needed correct tensor
identity.

### 1. `compiled_tensor_id` is assigned to compiled weight inputs

The tensor map is built from:

- `V.graph.graph_input_names`
- `V.graph.graph_inputs`
- `V.graph.static_input_idxs`

Only graph inputs that are:

- `TensorBox`
- static inputs

become `compiled_tensor_id` entries.

This intentionally restricts the mapping to compiled weight inputs, not all
graph inputs and not runtime intermediates.

### 2. The tensor sidecar records the mapping

`compiled_tensor_map_graph0.json` contains:

- `compiled_tensor_id`
- `graph_input_name`
- `graph_input_idx`
- dtype
- shape
- `compilation_hash`

This file is the authoritative compiled-side view of weight tensor identity.

### 3. The scheduler crosswalks profiler tensor names to compiled tensor IDs

The scheduler uses:

- profiler/bundle tensor metadata
- compiled tensor sidecar

to match scheduler names such as `param_...` to `compiled_tensor_id`.

This matching must be strict:

- zero candidates: fail
- multiple candidates: fail
- exactly one candidate: accept

No guessing is allowed for VRAM ops.

### 4. The injector resolves VRAM ops to graph input variable references

At injection time, the wrapper rebuilds the same compiled tensor map and uses:

- `compiled_tensor_id -> graph_input_name`

to emit calls like:

```python
_ws_rt.h2d_prefetch(arg5_1)
_ws_rt.evict_vram(arg5_1)
```

This is the important semantic fix. These calls go through the runtime’s
tensor-object dispatch, which uses `id(tensor)` and the registered model
weights.

By contrast, the old string-name form:

```python
_ws_rt.h2d_prefetch("param_45_46")
```

went through the by-name runtime path, which did not target the actual model
parameter tensor used by compiled kernels.

## Problems Encountered During Implementation

The work hit several distinct classes of problems.

### Problem 1: Metadata emission depended on schedule injection

At first, launch markers and sidecars were effectively tied to the path that
already assumed a plan existed.

That was backwards, because the plan can only be built after profiling.

#### Fix

Metadata emission was decoupled from schedule injection:

- emit IDs and markers in a profiling compile pass
- inject the schedule in a later execution compile pass

This produced the correct bootstrap workflow.

### Problem 2: Launch markers were not being emitted during profiling

The profiling script originally enabled sidecar emission but not launch-marker
emission.

That meant:

- `compiled_launch_map_graph0.json` existed
- but `runtime_nodes.csv` still had `compiled_launch_id = -1` for GPU nodes

As a result, the scheduler had no exact runtime-to-launch mapping and fell back
to count-based rescaling.

#### Fix

The profiling path was changed so llamasim-bundle export also enables:

- `weight_streaming_emit_launch_markers = True`

This caused `ws_launch:{graph_id}:{launch_id}` scopes to appear in the trace,
which allowed the exporter to annotate runtime nodes with exact launch IDs.

### Problem 3: Marker wrapping originally interfered with injection ordering

At one point, compute lines were wrapped with `LaunchMarkerLine` before weight
streaming injection ran.

The injector then stopped seeing raw compute lines because they were hidden
inside wrappers.

#### Fix

Injection ordering was corrected so weight-streaming injection happens before
marker wrapping.

This preserved both behaviors:

- correct injection
- correct profiler markers

### Problem 4: The scheduler assumed every GPU node had to match a launch ID

After exact launch IDs were exported into the bundle, many GPU nodes still
lacked `compiled_launch_id`.

This initially triggered a hard failure.

The reason was not that the bundle was wrong. The reason was that the bundle
contains the whole SDXL run, including:

- memcpys
- eager kernels
- non-UNet GPU work
- setup/runtime kernels outside the compiled wrapper scope

Those nodes legitimately do not belong to a compiled UNet launch.

#### Fix

`jit_sim_prune` was changed to accept partial exact coverage:

- use exact launch IDs for resolved GPU nodes
- warn about the unresolved remainder
- skip unresolved nodes instead of failing the whole solve

This allowed the scheduler to use exact launch IDs for the relevant compiled
subset while tolerating unrelated pipeline GPU work.

### Problem 5: CONTEXT tensors were being treated like compiled weight inputs

The bundle includes tensors named like:

- `cache_35_36`

These are `CONTEXT` tensors, meaning intermediates/runtime-carried values, not
compiled graph weight inputs.

The scheduler initially tried to annotate `compiled_tensor_id` for VRAM ops on
these tensors and failed.

#### Fix

The scheduler was changed so `compiled_tensor_id` is required only for VRAM
ops on `WEIGHT` tensors.

This is correct because:

- `compiled_tensor_map_graph0.json` only describes compiled weight graph inputs
- CONTEXT tensors can never resolve to that map

### Problem 6: Whole-pipeline weights were entering a single-graph UNet plan

The profiler bundle contains the whole SDXL pipeline run, not only the compiled
UNet graph.

That means some `param_*` weights in bundle metadata belong to:

- text encoders
- embeddings
- other non-UNet components

An example was `param_45_46`, which looked like a weight but did not belong to
the compiled UNet graph represented by `compiled_tensor_map_graph0.json`.

The scheduler initially treated these as if they had to crosswalk into the
UNet compiled tensor map, which failed.

#### Fix

`jit_sim_prune` was changed to filter unmapped `WEIGHT` ops when a compiled
tensor crosswalk is available.

This keeps the resulting plan scoped to the compiled graph it is actually going
to drive.

### Problem 7: Exact VRAM ops were emitted through the wrong runtime path

This was the key semantic bug.

The scheduler had exact VRAM placement, but the emitted code looked like:

```python
_ws_rt.h2d_prefetch("param_45_46")
_ws_rt.evict_vram("param_45_46")
```

Those are string-name calls. They use the runtime’s by-name path, which is not
the same thing as operating on the actual registered model parameter tensor.

To avoid incorrect behavior, an intermediate workaround filtered VRAM ops out of
exact placement entirely and replaced them with graph-structure first-use /
last-use behavior. That restored runtime correctness but broke plan fidelity.

#### Fix

The final fix was:

- stop filtering exact VRAM ops
- require `compiled_tensor_id` on VRAM ops
- resolve `compiled_tensor_id -> graph_input_name`
- emit variable-reference calls

So exact VRAM ops now become:

```python
_ws_rt.h2d_prefetch(arg5_1)
_ws_rt.evict_vram(arg5_1)
```

This preserves:

- exact launch placement
- correct tensor-object runtime behavior

### Problem 8: Schedules without tensor crosswalk had to fail

Once exact VRAM emission was implemented correctly, old schedules without
`compiled_tensor_id` on VRAM ops started failing during injection.

This was not a regression. It was the correct enforcement of the new contract.

#### Fix

The scheduler was updated to annotate:

- `vram_prefetch_h2d`
- `vram_evict_d2h`

with `compiled_tensor_id`.

Schedules that cannot do this now fail instead of silently taking a wrong path.

### Problem 9: The plan remained too large and performance stayed slow

Even after correctness issues were fixed, the weight-streaming run still had
high inference time.

This required separating two questions:

1. is the runtime following the plan?
2. is the plan itself any good?

The investigation showed that the executable plan was still very large:

- many cold-start prefetches
- many H2D prefetches
- many VRAM evictions
- many DRAM evictions

This pointed to a schedule-quality problem, not just an injection-placement
bug.

The current evidence indicates:

- exact weight VRAM ops are now mostly issued at intended stages
- the plan itself is still heavy enough to cause substantial slowdown

## What Was Solved

The following correctness properties are now the intended design.

### Solved: launch placement identity

The system now has a real notion of compiled launch identity:

- assigned in wrapper codegen
- emitted into profiler markers
- exported into bundle metadata
- consumed by the scheduler
- used by the injector

This replaced count-based approximation as the primary path.

### Solved: exact VRAM tensor identity

VRAM schedule ops can now target:

- the correct compiled graph input
- the correct runtime tensor object

instead of relying on legacy by-name behavior.

### Solved: compiled-graph scoping

The scheduler now understands that:

- compiled launch IDs are graph-local
- compiled tensor IDs are graph-local
- bundle artifacts may contain unrelated pipeline activity

The result is that the plan is scoped to the compiled graph it actually drives.

### Solved: hard errors instead of silent wrong behavior

This was an important design choice.

The system now prefers:

- fail if `compiled_tensor_id` is missing for exact VRAM ops
- fail if tensor crosswalk is ambiguous
- fail if `compilation_hash` mismatches

instead of silently falling back to semantically wrong execution.

## What Is Correct Today

The current intended semantics are:

### Exact launch placement

For schedule rows that have:

- `before_launch_id`
- `after_launch_id`

the injector attaches them directly to the matching compiled wrapper launch.

### Exact VRAM execution

For VRAM ops that have:

- exact launch ID
- valid `compiled_tensor_id`

the injector emits variable-reference calls and the runtime operates on the
actual model weight tensors.

### SSD/DRAM exact placement

SSD prefetch and DRAM eviction use exact launch placement when launch IDs are
available, but still use string-name runtime calls.

This means their placement can be exact while their storage semantics remain a
separate issue.

### Partial bundle exactness

The scheduler uses exact launch IDs for the resolved compiled subset and skips
GPU nodes outside that scope.

That is the correct behavior when profiling a whole SDXL pipeline run but only
scheduling one compiled graph.

## What Is Still Out Of Scope Or Incomplete

Not every part of weight streaming is fully solved yet.

### 1. SSD/DRAM storage realism is separate

The runtime’s SSD path depends on backing file metadata and `_read_from_file()`.
Exact launch placement alone does not make SSD behavior faithful if file-backed
metadata is incomplete.

In other words:

- SSD/DRAM placement can be exact
- storage behavior can still be incomplete

Those are separate concerns.

### 2. The plan can still be poor

Correct execution does not imply good performance.

A schedule can be:

- exact
- internally consistent
- faithfully executed

and still be too expensive.

That appears to be the remaining issue for the current SDXL plan.

### 3. Some legacy/rescaled support still exists

The code still contains some legacy support for ops without launch IDs.
That compatibility path exists to avoid breaking older artifacts immediately.
It is not the preferred path.

### 4. Partial bundle coverage remains normal

Not every GPU row in a whole-pipeline profiler bundle will map to a compiled
UNet launch.

That is expected unless the profiling/export flow is narrowed to exactly the
compiled graph of interest.

## Practical Workflow

The practical workflow is:

### 1. Profiling pass

Run:

```bash
python3 scripts/profile_sdxl_turbo_gpu.py \
  --model /data/llamasim/models/sdxl-turbo/ \
  --steps 8 \
  --height 512 \
  --width 512 \
  --prompt "a cute dog and a cute cat in a park" \
  --fusion inductor \
  --output-dir sdxl-turbo-profile-new
```

This should produce a bundle under:

- `sdxl-turbo-profile-new/llama_bundle/`

including:

- `runtime_nodes.csv`
- `pytorch_runtime_tensors.csv`
- `compiled_launch_map_graph0.json`
- `compiled_tensor_map_graph0.json`

### 2. Planning pass

Run:

```bash
python3 graph_modifiers/jit_sim_prune/main.py \
  /data/pytorch-source/sdxl-turbo-profile-new/llama_bundle/
```

This should produce:

- `jit_sim_prune_schedule.json`

with:

- launch IDs on executable ops
- `compiled_tensor_id` on VRAM weight ops

### 3. Execution pass

Run:

```bash
python3 ws_scripts/run_weight_streaming_plan.py \
  --model /data/llamasim/models/sdxl-turbo/ \
  --steps 8 \
  --height 512 \
  --width 512 \
  --prompt "a cute dog and a cute cat in a park" \
  --plan-dir sdxl-turbo-profile-new/llama_bundle/jit_sim_prune_output/
```

This compiles again with the schedule injected into the generated wrapper.

## Summary

The original system failed because it relied on two approximations:

- approximate launch placement
- missing tensor identity

The main engineering solution was to introduce two compiler-assigned IDs:

- `compiled_launch_id`
- `compiled_tensor_id`

and then carry those IDs through:

- profiling
- bundle export
- scheduling
- wrapper injection
- runtime execution

The hardest bugs were not random implementation mistakes. They were mostly
scope and contract mismatches:

- bundle contained whole-pipeline activity while the sidecars described one
  compiled graph
- CONTEXT tensors were incorrectly treated like weight graph inputs
- exact launch placement existed before exact tensor identity existed
- the runtime had two different execution contracts:
  - by-name
  - by-object

The final important semantic fix was:

- keep exact VRAM placement
- resolve exact VRAM tensor identity
- emit variable-reference calls so the runtime acts on the real registered
  model weights

That is what made the execution plan-faithful for VRAM behavior.

What remains is mostly schedule quality and SSD/DRAM storage realism, not the
basic correctness of exact launch/tensor attachment.
