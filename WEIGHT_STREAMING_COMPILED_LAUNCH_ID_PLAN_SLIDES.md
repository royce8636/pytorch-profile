# Compiled Launch ID Slides Outline

## Slide 1: The Matching Problem

- The scheduler plan is built from profiled runtime nodes
- The weight-streaming hook runs on final compiled wrapper launches
- Those two views are not the same
- Fusion changes the execution structure
- Many trace nodes collapse into fewer compiled launches
- So direct trace-node to wrapper-line matching is not reliable
- Today this forces snapping and rescaling heuristics

## Slide 2: Scope Of The Fix

- This plan is mainly for schedule-driven SSD/DRAM placement
- That is the part that still depends on trace-node matching
- The current VRAM path already uses a different mechanism
  - compile-time first-use / last-use analysis
- So phase 1 should not replace that VRAM mechanism
- Instead:
  - keep VRAM placement as-is
  - fix SSD/DRAM placement with exact launch IDs

## Slide 3: Why Current Matching Breaks

- Runtime traces describe what actually ran
- Inductor codegen sees the final fused launch structure
- One profiled kernel name may appear many times
- One fused launch may represent many original trace nodes
- CPU trace nodes are richer than wrapper insertion points
- Correlation IDs help reconstruct runtime relations
- But they do not identify a stable compiled launch site

## Slide 4: The High-Level Fix

- Introduce a new shared key: `compiled_launch_id`
- Each compiled wrapper launch gets one exact ID
- That same ID is exposed to profiling
- That same ID is exported into the llama bundle
- The scheduler then builds plans against that ID
- The wrapper injector later attaches SSD/DRAM IO using that same ID
- This turns matching from heuristic to direct lookup

## Slide 5: What The ID Solves

- It solves launch placement
- It does not solve tensor-name mapping
- In other words:
  - it solves where to attach
  - it does not solve what tensor name means
- Tensor-to-launch mapping and tensor naming remain separate concerns
- That separation should stay explicit in the design

## Slide 6: Where The ID Should Be Created

- The ID should be created at the compiled-launch boundary
- That is where Inductor still knows both:
  - the final emitted launch
  - the fused source nodes behind it
- This is later than FX graph rewriting
- But earlier than post-hoc trace interpretation
- It preserves real compiled-launch structure
- While still keeping provenance information

## Slide 7: End-To-End Flow

- Step 1: compile and assign launch identity
- Step 2: emit profiler-visible launch markers
- Step 3: profile execution and capture those IDs
- Step 4: export llama bundle with launch IDs
- Step 5: build schedule keyed by launch IDs
- Step 6: inject SSD/DRAM placement by exact launch ID
- Same launch identity flows through the whole system

## Slide 8: What Changes In Profiling

- Yes, profiling must change for exact matching
- The profile needs compiler-assigned launch IDs
- Kernel name alone is not enough
- Correlation ID alone is not enough
- The cleanest method is launch markers
  - `record_function`
  - NVTX
  - or another profiler-visible annotation
- Markers should be gated behind a config flag because of overhead

## Slide 9: Partitions And Uniqueness

- A model may compile into multiple graph partitions
- So launch IDs must be unique across partitions
- Two good designs are:
  - one global launch counter
  - graph ID + local launch ID
- The second is usually clearer
- It keeps partition boundaries explicit
- And avoids ambiguity in multi-graph runs

## Slide 10: What Changes In The Llama Bundle

- The llama bundle should export launch identity
- Add it to:
  - `runtime_nodes.csv`
  - `ggml_profile_node_records.csv`
- GPU runtime nodes should carry exact launch IDs
- Related CPU submit nodes may also carry the same ID
- This makes the bundle schedule-ready without name heuristics
- It also makes debugging easier

## Slide 11: What Changes In Scheduling

- The scheduler should stop targeting raw trace indices for SSD/DRAM placement
- Instead, it should target compiled launch identity
- Old fields like:
  - `before_node`
  - `after_node`
- Should evolve toward launch-ID based fields
- Then a plan refers to exact compiled launch sites
- No fusion rescaling is needed anymore

## Slide 12: What Changes In Injection

- The injector no longer needs to guess placement
- It no longer needs to:
  - classify trace nodes
  - snap CPU nodes
  - rescale trace indices after fusion
- It can walk wrapper launches directly
- Read each launch's identity
- And attach schedule-driven SSD/DRAM IO by exact ID match

## Slide 13: Invalidation And Safety

- A schedule from compilation A must not silently apply to compilation B
- Recompilation may change fusion and launch ordering
- So schedules should carry a `compilation_hash`
- The compiled artifact should carry the same hash
- At injection time:
  - hash mismatch should fail fast
- This prevents silent wrong placement

## Slide 14: Suggested Closing Summary

- The right fix is not a stronger heuristic
- The right fix is a shared compiler-assigned launch identity
- That identity should flow through:
  - compile
  - profile
  - export
  - schedule
  - injection
- Phase 1 should target SSD/DRAM placement and keep the current VRAM path
- Once that exists, exact 1:1 launch matching becomes practical
