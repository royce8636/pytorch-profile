# Weight Streaming Slides Outline

## Slide 1: Problem And Goal

- Large models cannot keep all useful weights in VRAM at once
- The goal is to move weights through a memory hierarchy as needed
- The hierarchy is:
  - SSD
  - DRAM
  - VRAM
- The system tries to fetch earlier and evict later to reduce stalls
- The key question is where to hook this into compiled execution

## Slide 2: High-Level Design

- The design hooks into the compiled wrapper, not the model graph itself
- A scheduler produces a plan using trace-level node indices
- The compiler turns the model into a smaller set of fused compute launches
- The wrapper is the place where these worlds meet
- Weight-streaming calls are inserted around wrapper compute launches
- Runtime then executes SSD / DRAM / VRAM movement at those points

## Slide 3: Main Inputs

- One input is the scheduler plan
- That plan describes:
  - which tensor to move
  - what kind of movement it is
  - which traced node it should be attached to
- Another input is runtime metadata about nodes and tensors
- Node metadata helps decide which traced nodes are GPU-facing
- Tensor metadata helps decide what can be streamed and how

## Slide 4: Core Hook Point

- The hook point is after wrapper structure is built
- It is before final generated code is emitted
- At that point, the compiler already knows its actual compute launches
- The injector can walk those launches and insert runtime calls around them
- This avoids rewriting the model graph itself
- It also means the exported wrapper shows the real final behavior

## Slide 5: Why Node Matching Is Needed

- The scheduler works on a detailed execution trace
- That trace contains both CPU and GPU nodes
- The compiled wrapper does not preserve that exact structure
- Fusion reduces many traced GPU nodes into fewer wrapper launches
- So a direct one-to-one mapping does not exist
- The system needs an adapter to translate trace positions into wrapper positions

## Slide 6: How Nodes Are Matched

- First, traced nodes are split into CPU-like and GPU-like nodes
- GPU identity comes mainly from resource kind
- If that is missing, GPU-like names are used as a fallback heuristic
- Prefetch-style ops attach to the next GPU node
- Eviction-style ops attach to the previous GPU node
- This is conservative:
  - fetch a little earlier
  - evict a little later

## Slide 7: How Fusion Is Handled

- Even after GPU-node matching, wrapper launch count is still smaller
- The adapter rescales trace GPU indices into wrapper kernel indices
- So the schedule is not copied literally
- It is compressed onto the fused wrapper timeline
- This is how a detailed scheduler trace can still drive a fused compiled model
- The result is approximate but structured, not arbitrary

## Slide 8: Two Placement Strategies Used Today

- Strategy 1: schedule-driven placement
  - used for SSD -> DRAM prefetch
  - used for DRAM eviction
- Strategy 2: graph-input use analysis
  - used for DRAM -> VRAM restore
  - used for VRAM eviction
- So not every movement comes directly from the schedule
- Some are attached using first-use / last-use of compiled inputs

## Slide 9: Runtime Execution Model

- At runtime, tensors can be thought of as living in one of three places:
  - SSD
  - DRAM
  - VRAM
- SSD reads are launched asynchronously into pinned DRAM buffers
- VRAM restore uses a dedicated H2D stream
- VRAM eviction frees GPU storage after last use
- DRAM eviction drops the pinned CPU copy when it is no longer needed

## Slide 10: How Restores Work In Practice

- The current wrapper restores registered weight tensors by object identity
- Each registered weight has a CPU backup
- Before first use, the wrapper asks the runtime to restore that weight
- The runtime recreates GPU storage if it was evicted
- Then it copies the CPU backup onto the GPU
- The current stream is synchronized with that restore before compute continues

## Slide 11: What Is Actually Guaranteed Today

- There is real schedule loading
- There is real wrapper injection
- There is real runtime IO and memory movement
- There is real exported generated code showing inserted calls
- SSD -> DRAM behavior is schedule-driven
- DRAM -> VRAM and VRAM eviction are currently use-driven
- So the system is real, but the control sources are mixed

## Slide 12: Main Limitations

- The full schedule is not used uniformly for all movement types
- CPU trace nodes are not first-class wrapper insertion points
- Some schedule fields are parsed but not yet used for placement
- Matching across fusion is approximate by design
- Runtime setup still has to happen outside generated code
- The design is functional today, but not yet the fully generalized end state

## Slide 13: Suggested Closing Summary

- The key idea is to hook weight streaming into the compiled wrapper
- The wrapper is where traced schedule intent meets fused compiled execution
- Node matching is done by:
  - GPU-node identification
  - conservative snapping
  - fusion-aware rescaling
- Runtime then performs the actual SSD / DRAM / VRAM movement
- This is the right mental model for understanding the current implementation
