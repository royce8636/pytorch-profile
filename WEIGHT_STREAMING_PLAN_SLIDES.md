# Weight Streaming Slides Outline

## Slide 1: What Exists Today

- Weight streaming is implemented in current code, not just planned
- It is integrated into Inductor wrapper codegen
- It is not an FX graph rewrite
- It is not based on custom ops
- The compile path injects Python runtime calls into generated wrapper code
- The runtime handles SSD -> DRAM, DRAM -> VRAM, and eviction behavior
- The exported wrapper shows the actual injected calls

## Slide 2: Main Code Locations

- `torch/_inductor/weight_streaming/plan.py`
  - schedule dataclasses and JSON / CSV loader
- `torch/_inductor/weight_streaming/codegen.py`
  - trace-index mapping and wrapper call emission helpers
- `torch/_inductor/weight_streaming/runtime.py`
  - runtime singleton and IO / memory movement logic
- `torch/_inductor/codegen/wrapper.py`
  - injection into generated Inductor wrapper
- `scripts/run_weight_streaming.py`
  - end-to-end driver used to compile and run

## Slide 3: Schedule Inputs

- The loader reads scheduler-style JSON
- Main JSON sections are:
  - `nodes`
  - `io_operations`
  - `spill_decisions`
  - `cold_start_prefetches`
- Optional CSVs add:
  - `runtime_nodes.csv` for `resource_kind`
  - `pytorch_runtime_tensors.csv` for tensor metadata

## Slide 4: Parsed Operation Types

- `prefetch`
  - becomes `PrefetchOp`
  - represents SSD -> DRAM prefetch
- `vram_prefetch_h2d`
  - becomes `H2DPrefetchOp`
  - represents DRAM -> VRAM prefetch intent
- `vram_evict_d2h`
  - becomes `EvictVramOp`
- `spill_decisions`
  - become `EvictDramOp`

## Slide 5: How Compile-Time Injection Works

- `run_weight_streaming.py` initializes the runtime first
- It sets Inductor config knobs for plan and CSV paths
- Then it compiles `pipe.unet` with `torch.compile(..., backend="inductor")`
- In `wrapper.py`, `_generate()` calls `_inject_weight_streaming_io()`
- Injection happens after normal wrapper IR passes
- The wrapper is modified before final code emission
- If export is enabled, final wrapper code is written to disk

## Slide 6: What The Wrapper Emits

- Header lines:
  - import `WeightStreamRuntime`
  - `_ws_rt = WeightStreamRuntime.instance()`
- Startup call:
  - `_ws_rt.cold_start_prefetch(...)`
- Scheduled pre-kernel call:
  - `_ws_rt.ssd_prefetch(...)`
- Graph-input first-use / last-use calls:
  - `_ws_rt.h2d_prefetch(tensor)`
  - `_ws_rt.evict_vram(tensor)`
- Scheduled post-kernel call:
  - `_ws_rt.evict_dram(...)`

## Slide 7: Trace Mapping And Fusion

- Scheduler trace contains mixed CPU and GPU nodes
- Wrapper has fewer insertion points than the original trace
- `ScheduleAdapter` classifies GPU nodes by:
  - `resource_kind`
  - or GPU-like name prefixes
- CPU-targeted prefetch ops snap forward to the next GPU node
- CPU-targeted eviction ops snap backward to the previous GPU node
- Fused wrapper indices are produced by rescaling to wrapper kernel count

## Slide 8: Runtime Behavior

- Runtime owns:
  - async SSD read thread pool
  - pinned DRAM buffers
  - dedicated CUDA H2D stream
  - H2D events
  - VRAM tensor tracking
- `ssd_prefetch(...)` starts async SSD -> DRAM reads
- `cold_start_prefetch(...)` blocks until startup weights reach DRAM
- `h2d_prefetch(...)` restores weights to VRAM

## Slide 9: Weight Registration And VRAM Restore

- `run_weight_streaming.py` registers model weights before compile
- `register_weight(...)` stores a pinned CPU backup of each GPU weight
- The current wrapper passes actual tensor objects to:
  - `_ws_rt.h2d_prefetch(tensor)`
  - `_ws_rt.evict_vram(tensor)`
- On restore, the runtime:
  - resizes GPU storage back to original size
  - copies CPU backup to GPU on H2D stream
  - records an event and waits immediately on current stream
- No explicit emitted `wait_h2d(...)` is needed for this path

## Slide 10: Exported Generated Code

- `weight_streaming_generated.py` is the final generated wrapper
- It is written from the final Inductor wrapper buffer
- It is not manually edited after generation
- It is not a simplified pseudocode export
- The visible `_ws_rt.*` calls are the real injected calls
- This makes the file useful for inspection and debugging
- It reflects what the compiled wrapper actually runs

## Slide 11: Current Limitations

- No FX-level weight replacement pass
- No custom-op based implementation
- Runtime initialization is still external to generated code
- Scheduled `H2DPrefetchOp` / `EvictVramOp` are parsed but not directly emitted
- Instead, VRAM restore / evict is driven by graph-input first-use / last-use
- Some parsed fields are not yet used for placement
  - `PrefetchOp.after_node`
  - `PrefetchOp.eager_start`
  - `ColdStartOp.attach_before_node`

## Slide 12: Short Takeaway

- The implementation is real and compile-path integrated
- Schedule loading, wrapper injection, runtime IO, and export all exist
- SSD -> DRAM prefetch is schedule-driven today
- DRAM -> VRAM restore and VRAM eviction are graph-input driven today
- The exported wrapper is trustworthy for understanding actual behavior
- The main remaining gaps are around fuller schedule use and polish
- The system is beyond a prototype, but not yet fully generalized
