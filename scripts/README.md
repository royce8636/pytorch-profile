This directory contains useful tools.

Compared with upstream PyTorch (`origin/main`), this checkout also carries a
local end-to-end profiling workflow for SDXL Turbo and Qwen-Image.

## How profiling is done in this checkout

1. Start from a model-specific entrypoint:
- `profile_sdxl_turbo_cpu.py`
- `profile_sdxl_turbo_gpu.py`
- `profile_qwen_image_gpu.py`

Use `run_qwen_image.py` when you want the same Qwen runtime path without
profiler instrumentation.

Use `run_accelerate_cpu_offload.py` when you want the same local SDXL Turbo or
Qwen-Image models to run with Diffusers Accelerate CPU offload, so the pipeline
can exceed available GPU VRAM at the cost of extra transfer overhead. Both
offload entrypoints also accept `--fusion inductor` to compile the hot model
path with `torch.compile` before offload hooks are installed.

Use `profile_accelerate_cpu_offload.py` when you want a steady-state profiler
trace for the offloaded run. It starts profiling only after model load and
unprofiled warmup, so checkpoint reads and first-touch storage costs are kept
out of the captured region. It also emits a llama bundle next to the trace by
default.
Pass `--offload-mode none` when you want the same harness and steady-state
profiling flow without installing Accelerate CPU offload hooks. In that mode,
the full pipeline is moved onto the requested accelerator and stays there.
When `--output-dir` is set, the offload profiler writes the bundle under
`<output-dir>/llama_bundle` by default.

Use `demo_profile_time_split.py` when you want to inspect an emitted
`llamasim-runtime` llama bundle directory or its `manifest.json` and compare
several heuristics for splitting the
imported node durations into pure compute time versus offload/stall time.
Its summary output can also break raw, compute, and stall totals into CPU and
GPU components. The newer `v5` and `v6` modes propagate transfer/offload stall
lineage through downstream CPU nodes. `v5` fixes later sync/wait nodes, while
`v6` also strips CPU staging and wrapper work such as `aten::to`,
`aten::empty_strided`, `TorchDynamo Cache Lookup`, and AOT dispatcher prologue
nodes when they are on the offload read path. The CLI now defaults to `v6`.
It only writes annotated CSV / JSON reports when `--write-reports` or
`--out-dir` is passed. Use `--compare-bundle <other_bundle>` to diff two llama
bundles and print the largest raw CPU-side deltas. Add
`--compare-include-retained-compute` when you also want the slower section that
prints the largest CPU nodes still retained as compute under the chosen version.

2. Choose the execution mode and capture options:
- `--fusion none|inductor` compares eager execution with
  `torch.compile(..., backend="inductor")`
- `--warmup-runs` pre-runs compiled graphs before capture; it defaults to `1`
  for Inductor and `0` otherwise
- `--record-shapes`, `--profile-memory`, and `--with-stack` widen the captured
  profiler data
- `--output-dir` or explicit `--trace`, `--trace-csv`, and `--image` paths
  control where artifacts are written

3. Run a single profiled inference:
- the scripts load the pipeline, optionally compile the hot module (`unet` for
  SDXL, `transformer.forward` for Qwen), perform warmup runs, then capture one
  inference inside `torch.profiler.profile(...)`
- the model invocation is wrapped in `torch.autograd.profiler.record_function(...)`
  so the exported trace has a stable top-level run scope
- when a mode needs execution-trace data, the scripts also enable
  `torch.profiler.ExecutionTraceObserver()`

4. Export post-processing artifacts:
- always: Chrome trace JSON, profiler CSV when available, and the decoded
  output PNG
- optional DOT exports via `--dot-level`: `fx`, `execution`, `memory`,
  `kineto`, `hybrid`, `runtime-io`, `llamasim-runtime`, `both`, or `all`
- `runtime-io` combines Kineto runtime timing with exact tensor inputs/outputs
  from Execution Trace Observer; see `runtime-io-dot.md`
- `llamasim-runtime` writes a bundle for `/data/llamasim` with
  `step_0_compute_graph.dot`, runtime/tensor CSVs, and a manifest

5. Split the exported CSV when you want leaf-only summaries:
- `split_trace_csv_leaf_events.py <trace.csv>` writes `*_leaf_cpu.csv`,
  `*_leaf_gpu.csv`, and `*_leaf_activity_summary.svg`
- the CPU output keeps only leaf CPU rows; the GPU output keeps true kernel
  rows and drops user annotations, memcpys, and memsets

## How the llamasim bundle is created

This is the `llamasim-runtime` bundle, sometimes referred to here as the
`llama_bundle`. It is emitted only when `--dot-level llamasim-runtime` or
`--dot-level all` is enabled. The exporter requires a GPU trace and an
Execution Trace Observer JSON file; if `--output-dir` is set, the default
bundle directory is `<output-dir>/<stem>_llamasim_runtime`, otherwise it is
created next to the trace path.

1. Collect the two source views of the run:
- raw Kineto events and `trace_start_ns` from the profiler result
- Execution Trace Observer nodes from `--execution-trace`

2. Match runtime events to execution-trace nodes:
- GPU execution nodes are matched to Kineto GPU kernels by linked `rf_id`, then
  narrowed by kernel name and stream hint when available
- CPU runtime events are matched by `rf_id`
- ambiguous GPU matches are a hard error for the export
- if a GPU kernel has no direct execution-trace node, the exporter tries to
  propagate tensor I/O from the linked parent CPU event

3. Select which runtime nodes go into the bundle:
- CPU nodes are reduced to leaf CPU events only
- GPU nodes keep non-CPU runtime events but drop GPU user annotations
- selected runtime nodes get stable IDs `k0`, `k1`, ... and are connected with
  `thread_order`, `stream_order`, `submit`, `wait`, and `sync_wait` edges
- runtime roles such as `cpu_leaf`, `gpu_runtime`, `submit`, and `wait` are
  recorded in the emitted metadata

4. Materialize tensor records from execution-trace inputs and outputs:
- tensor entries are flattened into canonical tuples
  `(tensor_id, storage_id, offset, numel, itemsize, device)`
- tuples are deduplicated into bundle tensor records and assigned stable IDs
  `t0`, `t1`, ...
- tensors with producers in the selected graph are marked as `CONTEXT`; tensors
  that are only consumed are marked as `WEIGHT`
- data edges are added from tensor nodes to runtime nodes for inputs and from
  runtime nodes to tensor nodes for outputs

5. Write the bundle artifacts:
- `step_0_compute_graph.dot`: the combined runtime+tensors graph
- `ggml_profile_node_records.csv`: per-node timing table used by the downstream
  llamasim tooling
- `runtime_nodes.csv`: node metadata, timings, device/thread/stream IDs,
  `rf_id`, `kernel_file`, and whether tensor I/O was attached
- `runtime_edges.csv`: ordering, submit/wait, and tensor data-flow edges
- `pytorch_runtime_tensors.csv`: tensor metadata, shape/dtype, size, and
  producer/consumer counts
- `manifest.json`: schema, file names, node/tensor/edge counts, tensor-I/O
  coverage, propagation statistics, match-warning count, and peak memory stats

## Example commands

```bash
PYTHONNOUSERSITE=1 /tmp/ptvenv/bin/python scripts/profile_sdxl_turbo_gpu.py \
  --fusion inductor \
  --output-dir /tmp/sdxl_gpu_inductor \
  --dot-level all \
  --disable-progress-bar
```

```bash
PYTHONNOUSERSITE=1 /tmp/ptvenv/bin/python scripts/profile_qwen_image_gpu.py \
  --fusion inductor \
  --output-dir /tmp/qwen_image_gpu_inductor \
  --dot-level runtime-io \
  --disable-progress-bar
```

```bash
PYTHONNOUSERSITE=1 /tmp/ptvenv/bin/python scripts/run_accelerate_cpu_offload.py \
  sdxl-turbo \
  --offload-mode model \
  --disable-progress-bar
```

```bash
PYTHONNOUSERSITE=1 /tmp/ptvenv/bin/python scripts/run_accelerate_cpu_offload.py \
  qwen-image \
  --offload-mode sequential \
  --disable-progress-bar
```

```bash
PYTHONNOUSERSITE=1 /tmp/ptvenv/bin/python scripts/profile_accelerate_cpu_offload.py \
  qwen-image \
  --offload-mode sequential \
  --output-dir /tmp/qwen_image_offload_profile \
  --disable-progress-bar
```

```bash
python3 scripts/split_trace_csv_leaf_events.py \
  /tmp/sdxl_gpu_inductor/sdxl_turbo_gpu_inductor_trace.csv
```

```bash
python3 scripts/demo_profile_time_split.py \
  /data/pytorch-source/llamasim_bundle \
  --version all \
  --out-dir /tmp/llamasim_bundle_time_split
```

## Compared with upstream PyTorch (`origin/main`)

Upstream PyTorch provides the generic profiler building blocks such as
`torch.profiler.profile(...)`, `ExecutionTraceObserver`, Chrome trace export,
and the core profiler docs/tests. This checkout adds a local workflow on top of
those primitives:

- model-specific SDXL Turbo and Qwen-Image harnesses
- eager vs Inductor comparison knobs and warmup handling
- multiple graph exporters: `fx`, `execution`, `memory`, `kineto`, `hybrid`,
  `runtime-io`, and `llamasim-runtime`
- a llamasim bundle exporter for `/data/llamasim`
- a CSV splitter for leaf CPU / true GPU-kernel summaries
- regression coverage in `test/profiler/test_profiler.py` for the local helper
  behavior

## Related docs

- `runtime-io-dot.md`: how the `runtime-io.dot` export works, where its timings
  and tensor I/O come from, and which files were changed to add it
