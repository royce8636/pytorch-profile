# How `demo_profile_time_split.py` Detects "SSD Time" and DRAM->VRAM Time

This note explains what the splitter in
[`demo_profile_time_split.py`](/data/pytorch-source/scripts/demo_profile_time_split.py)
is actually doing.

The short version:

- The script does **not** observe a first-class "SSD read event".
- It reads a llama bundle and infers which profiled node durations are likely:
  - real compute
  - DRAM->VRAM transfer work
  - CPU-side offload overhead that may include hidden SSD page-in time
- "SSD time" in the script is really **offload-related stall/overhead time**,
  not a guaranteed direct measurement of disk I/O only.

## Input Data

The script consumes the same llama bundle artifacts used by `/data/llamasim`:

- `runtime_nodes.csv`
- `runtime_edges.csv`
- `ggml_profile_node_records.csv`

These are loaded in:

- `load_nodes(...)`
- `load_edges(...)`
- `load_bundle_analysis(...)`

The important point is that the bundle contains profiled runtime nodes and
dependencies, but it usually does **not** contain an explicit SSD lane.

## What Counts as a Transfer

Direct transfer-like nodes are detected by name pattern matching in
`_is_transfer_like(...)`.

Current patterns include strings such as:

- `memcpy`
- `htod`
- `dtoh`
- `h2d`
- `d2h`
- `pageable -> device`
- `device -> host`
- `host -> device`
- `prefetch`
- `offload`
- `evict`
- `reload`
- `ssd`
- `disk`
- `nvme`
- `pread`
- `pwrite`
- `io_uring`

In practice this is how the script recognizes things like:

- `cudaMemcpyAsync`
- `Memcpy HtoD (Pageable -> Device)`

Those are the clearest DRAM->VRAM / VRAM->DRAM style transfer signals in the
bundle.

## What Counts as a Wait

Wait/sync nodes are detected in `_is_wait_like(...)`.

Current signals include:

- `runtime_role == "wait"`
- names containing `synchronize`
- names starting with `wait`
- names containing ` wait`
- names containing `barrier`

This catches nodes like:

- `cudaStreamSynchronize`
- `cudaEventSynchronize`
- `cudaDeviceSynchronize`

## Why SSD Time Is Hard

The llama bundle usually tells us:

- a CPU op ran for `X ns`
- a GPU memcpy op ran for `Y ns`
- a wait node blocked for `Z ns`

But it usually does **not** tell us:

- "out of `X ns`, exactly `P ns` was SSD page fault / page-in time"

So the script cannot directly subtract "disk time only".

Instead it uses a heuristic:

- if a node is clearly part of the offload transfer path, count it as
  offload/stall time
- otherwise keep it as compute

That means:

- direct GPU/CPU transfer ops are the best DRAM->VRAM signal
- SSD latency may be folded into CPU-side nodes that touch cold file-backed
  pages
- many CPU-side staging/runtime nodes are real offload overhead even when they
  are not literally disk I/O

## Version Summary

### `v0`

Baseline passthrough:

- every node is counted as compute
- `ssd_time_ns = 0`

This is only useful as a raw imported-timing baseline.

### `v1`

Strict mode:

- keep obvious GPU compute
- keep obvious CPU compute
- count transfers, waits, and CPU runtime/control as non-compute

### `v2`

More inclusive CPU mode:

- like `v1`
- but preserves CPU control/runtime nodes as compute

### `v3`

Immediate wait attribution:

- if a wait node has an incoming `wait` edge from a transfer-like node, count
  it as offload/stall
- if it waits on compute, preserve it as CPU control
- if it immediately follows a transfer-like predecessor on `thread_order`,
  count it as offload/stall

This is the first version that tries to preserve synchronization semantics while
separating transfer-driven waits from compute-driven waits.

### `v4`

Inclusive wait/control mode:

- keep wait/sync as CPU control by default
- only classify it as stall if `v3` already found direct transfer evidence

### `v5`

Recursive sync-lineage mode:

- extends `v4`
- walks upstream through `thread_order` and `wait` edges
- if a later CPU sync/wait node inherits transfer lineage, count it as stall

This was added because offload traces often contain:

- direct transfer wait
- then a few CPU nodes
- then another `cudaStreamSynchronize`

and the later sync is still part of the offload path.

### `v6`

Broader offload-lineage mode:

- extends `v5`
- still strips transfer-linked waits/syncs
- also strips CPU staging/control/runtime nodes when they inherit transfer
  lineage

`v6` is the current default.

## What `v6` Treats as CPU Offload Overhead

`v6` adds three CPU-side buckets.

### 1. CPU staging ops

Detected by `_is_cpu_offload_staging_like(...)`.

Current patterns include:

- `aten::empty`
- `aten::empty_strided`
- `aten::as_strided`
- `aten::view`
- `aten::_unsafe_view`
- `aten::_reshape_alias`
- `aten::to`
- `aten::copy_`
- `aten::clone`
- `aten::resize_`
- `detach`

These are treated as offload overhead only when they are on transfer lineage.

### 2. Compiler / wrapper runtime ops

Detected by `_is_cpu_compiler_runtime_like(...)`.

Current patterns include:

- `TorchDynamo Cache Lookup`
- `AOTDispatcher Runtime Wrapper Prologue`
- `Pregraph bytecode`
- `CompiledFxGraph`

These also count as offload overhead only when they sit on transfer lineage.

### 3. CUDA memory/runtime ops on CPU

Detected by `_is_cpu_cuda_memory_runtime_like(...)`.

Current patterns include:

- `cuMemCreate`
- `cuMemMap`
- `cuMemUnmap`
- `cuMemRelease`
- `cudaStreamIsCapturing`
- `cudaDeviceGetAttribute`
- `cudaMemcpyAsync`
- `cudaStreamSynchronize`

Again, `v6` only strips these when they inherit transfer/offload lineage.

## How Lineage Is Tracked

There are two related ancestry walkers.

### Wait lineage

`_has_transfer_lineage_for_wait(...)`:

- starts from a wait/sync node
- walks upstream through `thread_order` and `wait` edges
- if it finds transfer-like sources, it marks that wait as transfer-linked

This is mainly for `v5`.

### General CPU offload lineage

`_has_transfer_lineage_for_cpu_node(...)`:

- starts from a CPU node
- walks upstream through `thread_order` and `wait` edges
- allows traversal through other offload-lineage carrier nodes
- if it eventually reaches transfer-like sources, it marks the current CPU node
  as offload-linked

This is what lets `v6` strip nodes like:

- `aten::to`
- `aten::empty_strided`
- `TorchDynamo Cache Lookup`
- `AOTDispatcher Runtime Wrapper Prologue`

when they appear after a transfer/wait chain on the same CPU thread.

## What the Script Calls "SSD Time"

The `ssd_time_ns` field in the output should be interpreted as:

- time attributed to the offload path
- transfer-driven waits
- CPU-side offload staging/runtime overhead
- possible hidden SSD page-in latency inside those CPU nodes

It should **not** be interpreted as:

- exact direct disk I/O time only

That distinction matters. In a sequential offload trace, a lot of CPU time is
real host-side orchestration even if the pages are already resident in DRAM.

## What the Script Detects More Reliably

The most reliable direct signals are:

### DRAM->VRAM / VRAM->DRAM transfer work

These are inferred from explicit transfer-like node names, especially:

- `cudaMemcpyAsync`
- `Memcpy HtoD (Pageable -> Device)`
- other `Memcpy*` style nodes

These are much closer to "actual transfer time" than the inferred SSD bucket.

### Transfer-driven waits

These are inferred from:

- `wait` / `thread_order` graph edges
- wait-like node names
- transfer ancestry

This is more reliable than trying to isolate SSD page-in time itself.

## Compare Mode

The script also has a direct bundle diff mode:

```bash
python3 scripts/demo_profile_time_split.py \
  <offload_bundle> \
  --compare-bundle <baseline_bundle> \
  --compare-limit 20
```

This prints:

- raw bundle summaries for both bundles
- top CPU-side raw deltas by `(runtime_role, node_name)`

That mode is useful when the heuristic seems wrong, because it shows which
operations actually grew most between two runs.

If you also want the slower "what is still retained as compute?" view, add:

```bash
--compare-include-retained-compute
```

## What the Compare Mode Revealed for SDXL Offload

On the real `sdxl-offload` vs `sdxl-none` bundles, the largest CPU-side deltas
were not only sync nodes. Large increases showed up in:

- `TorchDynamo Cache Lookup`
- `aten::empty_strided`
- `aten::to`
- `AOTDispatcher Runtime Wrapper Prologue`
- `aten::empty`
- `aten::as_strided`
- `cudaMemcpyAsync`
- `cuMemUnmap`
- `aten::copy_`
- `Pregraph bytecode`
- `detach`
- `cudaStreamSynchronize`
- `aten::view`

That means the discrepancy is broader than "missed SSD wait events". It
includes substantial CPU staging/runtime overhead introduced by the offload path
itself.

## Practical Interpretation

If you are asking:

- "How much time is obviously DRAM->VRAM transfer?"

look first at explicit transfer nodes such as `cudaMemcpyAsync` and
`Memcpy HtoD`.

If you are asking:

- "How much time belongs to the offload path and should not be counted as pure
  compute?"

then `v6` is the current best answer in this script.

If you are asking:

- "How much of that time is literal SSD read latency only?"

the current bundle format does not provide that directly. The best the script
can do is infer offload-related time that may include hidden SSD page-in cost.
