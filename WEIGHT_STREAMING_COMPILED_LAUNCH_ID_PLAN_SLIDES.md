# Compiled Launch ID Slides Outline

## Slide 1: The Matching Problem

- The scheduler plan is built from profiled runtime nodes
- The weight-streaming hook runs on final compiled wrapper launches
- Those two views are not the same
- Fusion changes the execution structure
- Many trace nodes collapse into fewer compiled launches
- So direct trace-node to wrapper-line matching is not reliable
- Today this forces snapping and rescaling heuristics

## Slide 2: Why Current Matching Breaks

- Runtime traces describe what actually ran
- Inductor codegen sees the final fused launch structure
- One profiled kernel name may appear many times
- One fused launch may represent many original trace nodes
- CPU trace nodes are richer than wrapper insertion points
- Correlation IDs help reconstruct runtime relations
- But they do not identify a stable compiled launch site

## Slide 3: The High-Level Fix

- Introduce a new shared key: `compiled_launch_id`
- Each compiled wrapper launch gets one exact ID
- That same ID is exposed to profiling
- That same ID is exported into the llama bundle
- The scheduler then builds plans against that ID
- The wrapper injector later attaches IO using that same ID
- This turns matching from heuristic to direct lookup

## Slide 4: Where The ID Should Be Created

- The ID should be created at the compiled-launch boundary
- That is where Inductor still knows both:
  - the final emitted launch
  - the fused source nodes behind it
- This is later than FX graph rewriting
- But earlier than post-hoc trace interpretation
- It preserves real compiled-launch structure
- While still keeping provenance information

## Slide 5: End-To-End Flow

- Step 1: compile and assign `compiled_launch_id`
- Step 2: emit runtime-visible markers carrying that ID
- Step 3: profile execution and capture those IDs
- Step 4: export llama bundle with `compiled_launch_id`
- Step 5: build schedule keyed by `compiled_launch_id`
- Step 6: re-inject prefetch/evict ops by `compiled_launch_id`
- Same ID flows through the whole system

## Slide 6: What Changes In Profiling

- Yes, profiling must change for exact matching
- The profile needs to carry compiler-assigned launch IDs
- Kernel name alone is not enough
- Correlation ID alone is not enough
- The cleanest method is launch markers
  - `record_function`
  - NVTX
  - or another profiler-visible annotation
- The key requirement is one marker per compiled launch site

## Slide 7: What Changes In The Llama Bundle

- The llama bundle should export `compiled_launch_id`
- Add it to:
  - `runtime_nodes.csv`
  - `ggml_profile_node_records.csv`
- GPU runtime nodes should carry exact launch IDs
- Related CPU submit nodes may also carry the same ID
- This makes the bundle schedule-ready without name heuristics
- It also makes debugging easier

## Slide 8: What Changes In Scheduling

- The scheduler should stop targeting raw trace indices
- Instead, it should target compiled launch IDs
- Old fields like:
  - `before_node`
  - `after_node`
- Should evolve toward:
  - `before_launch_id`
  - `after_launch_id`
- Then a plan refers to exact compiled launch sites

## Slide 9: What Changes In Injection

- The injector no longer needs to guess placement
- It no longer needs to:
  - classify trace nodes
  - snap CPU nodes
  - rescale trace indices after fusion
- It can walk wrapper launches directly
- Read each launch's `compiled_launch_id`
- And attach scheduled IO by exact ID match

## Slide 10: Why This Is Better

- Exact 1:1 matching for compiled GPU launches
- No kernel-name guessing
- No launch-order guessing
- No trace-to-wrapper rescaling heuristic
- Works naturally with fused kernels
- Makes schedule behavior easier to explain and debug
- Keeps the design aligned with real compiled execution

## Slide 11: What This Does Not Solve Automatically

- CPU-side trace nodes are still more complex
- Not every CPU event has a durable wrapper counterpart
- Exact GPU launch matching is the primary target
- Exact CPU-side matching would need extra markers
- That can be a second phase if needed
- The first goal should be exact compiled GPU launch identity
- That is the highest-value improvement

## Slide 12: Suggested Closing Summary

- The right fix is not a better heuristic
- The right fix is a shared compiler-assigned launch ID
- That ID should flow through:
  - compile
  - profile
  - export
  - schedule
  - injection
- Once that exists, exact 1:1 matching becomes practical
