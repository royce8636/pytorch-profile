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
python3 scripts/split_trace_csv_leaf_events.py \
  /tmp/sdxl_gpu_inductor/sdxl_turbo_gpu_inductor_trace.csv
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
