# SDXL Turbo Script Explainer

This note explains how these three scripts work:

- `scripts/profile_sdxl_turbo_gpu.py`
- `scripts/run_sdxl_turbo_gpu.py`
- `scripts/run_weight_streaming.py` for the current SDXL Turbo path only

## Big Picture

There are two normal SDXL Turbo execution paths and one experimental weight-streaming path.

- `profile_sdxl_turbo_gpu.py` is the profiled path. It captures one inference inside `torch.profiler`, exports trace artifacts, and can also emit several DOT graph formats.
- `run_sdxl_turbo_gpu.py` is the non-profiled path. It runs the same model logic without profiler overhead and prints phase timings.
- `run_weight_streaming.py` is a separate compile-and-run harness for schedule-driven Inductor weight streaming. It is described as generic, but the current implementation hardcodes `StableDiffusionXLPipeline`, so in practice this script is SDXL Turbo-specific right now.

Two files matter a lot for the first two scripts:

- `scripts/profile_sdxl_turbo_gpu.py:15-23` is only a thin GPU entrypoint.
- `scripts/profile_sdxl_turbo_common.py` contains almost all shared SDXL logic: argument parsing, output path resolution, model loading, optional `torch.compile`, profiler capture, latent decoding, and DOT exports.

## 1. How `scripts/profile_sdxl_turbo_gpu.py` works

### What the file itself does

The file is intentionally tiny.

1. It prepends its own directory to `sys.path` so sibling helper modules can be imported from `scripts/`.
2. It imports `main` from `profile_sdxl_turbo_common`.
3. It calls that shared `main(...)` with GPU-specific defaults:
   - `default_device="cuda:0"`
   - `default_dtype="float16"`
   - `default_output_prefix="sdxl_turbo_gpu"`
   - `default_profile_memory=True`
   - `default_record_shapes=True`
   - `default_with_stack=True`

So this file is not the real implementation. It is a GPU-flavored preset for the common SDXL profiling harness.

### What the shared `main(...)` does for this script

The real flow is in `scripts/profile_sdxl_turbo_common.py:3223-3392`.

#### Step 1: Parse and validate arguments

`parse_args(...)` in `scripts/profile_sdxl_turbo_common.py` builds the CLI. It includes:

- model/prompt/inference shape options
- `--fusion none|inductor`
- warmup control
- output paths
- optional DOT exporters via `--dot-level`
- profiler knobs like `--profile-memory`, `--record-shapes`, and `--with-stack`

`validate_dot_args(...)` checks that the selected output options are internally consistent. For example, memory and hybrid DOT exports require memory profiling, recorded shapes, and stack capture.

`validate_device(...)` checks that CUDA is actually available for a CUDA device.

`validate_fusion_runtime(...)` checks the `torch.compile` prerequisites. On CUDA it specifically verifies that Triton is installed and usable before allowing `--fusion=inductor` (`scripts/profile_sdxl_turbo_common.py:3115-3134`).

#### Step 2: Resolve output locations

`resolve_output_paths(...)` computes all artifact paths (`scripts/profile_sdxl_turbo_common.py:417-584`).

It always prepares:

- Chrome trace JSON
- profiler CSV, when supported
- decoded PNG image

It conditionally prepares:

- FX DOT
- execution-trace JSON
- execution DOT
- memory DOT
- Kineto DOT and Kineto map CSV
- hybrid DOT
- runtime-I/O DOT
- llamasim runtime bundle directory

The output stem changes when `--fusion=inductor`, so profiled eager and compiled runs do not overwrite each other.

#### Step 3: Load the pipeline and optionally compile the UNet

`load_pipeline(...)` imports `diffusers.StableDiffusionXLPipeline` and loads it from the local model directory with `use_safetensors=True`, then moves it to the requested device (`scripts/profile_sdxl_turbo_common.py:599-617`).

`maybe_compile(...)` compiles `pipe.unet` with:

```python
torch.compile(
    pipe.unet,
    backend="inductor",
    mode=args.compile_mode,
    fullgraph=args.fullgraph,
)
```

This only happens when `--fusion=inductor` (`scripts/profile_sdxl_turbo_common.py:620-630`).

#### Step 4: Optionally capture example UNet inputs

If FX export is requested, `capture_example_inputs(...)` installs a forward pre-hook on the UNet and snapshots the first real call inputs (`scripts/profile_sdxl_turbo_common.py:645-659`).

That captured example is later used to export an FX graph of the underlying eager UNet, even if the runtime path is compiled.

#### Step 5: Warm up

Warmup defaults to:

- `0` runs for eager mode
- `1` run for compiled mode

Warmup uses `run_pipeline(...)`, which always runs the SDXL pipeline with:

- `guidance_scale=0.0`
- `output_type="latent"`
- the prompt, steps, height, and width from the CLI

The warmup output is discarded after synchronizing the device (`scripts/profile_sdxl_turbo_common.py:3173-3181`, `3261-3264`).

#### Step 6: Run one profiled inference

If execution-trace-derived outputs are requested, the script enables `torch.profiler.ExecutionTraceObserver()` and points it at the resolved execution-trace JSON path (`scripts/profile_sdxl_turbo_common.py:3266-3270`).

The main profiled run is then:

1. entered under `torch.profiler.profile(...)`
2. wrapped in `torch.autograd.profiler.record_function("sdxl_turbo_run", ...)`
3. executed once via `run_pipeline(...)`
4. synchronized before exiting the profiler context

That produces:

- CPU and optionally CUDA profiler events
- shape, memory, and stack metadata when enabled
- optional execution-trace JSON if the observer was active

#### Step 7: Export artifacts

After profiling, the script:

1. exports the Chrome trace JSON
2. exports profiler CSV if available
3. decodes the latent output to PIL images
4. saves the first image as PNG

Latent decoding happens in `decode_latents_to_pil(...)` (`scripts/profile_sdxl_turbo_common.py:3137-3170`). That helper:

- handles VAE upcasting when SDXL requires it
- rescales latents using the VAE config
- decodes through `pipe.vae.decode(...)`
- reapplies watermarking if present
- postprocesses to PIL

#### Step 8: Optionally export graph views

Depending on `--dot-level`, the script may emit:

- FX graph DOT
- execution-trace DOT
- memory/dataflow DOT
- Kineto runtime DOT
- hybrid DOT
- runtime-I/O DOT
- a llamasim runtime bundle

That post-processing is driven from `scripts/profile_sdxl_turbo_common.py:3301-3356`.

#### Step 9: Print a summary

The script ends by printing:

- the profiler key-averages table
- device, dtype, fusion mode, dot level
- capture flags
- artifact paths
- latent shape
- metadata from the `sdxl_turbo_run` scope

### What this script is for

Use this script when you want one SDXL Turbo run plus profiler outputs. It is the instrumentation-heavy harness.

## 2. How `scripts/run_sdxl_turbo_gpu.py` works

This script reuses the same SDXL helper module, but removes the profiler and graph-export machinery. It is the lightweight runtime path.

### Main structure

The important flow is in `scripts/run_sdxl_turbo_gpu.py:159-214`.

#### Step 1: Parse runtime-only arguments

The script accepts:

- model/prompt/steps/height/width
- `--device` and `--dtype`
- `--fusion none|inductor`
- compile flags
- warmup count
- image output path controls
- `--disable-progress-bar`

Compared with the profiled script, there are no trace, CSV, or DOT options.

#### Step 2: Validate the runtime

`validate_run_device(...)` rejects CUDA devices when `torch.cuda.is_available()` is false (`scripts/run_sdxl_turbo_gpu.py:152-156`).

It also calls the shared `validate_fusion_runtime(...)`, so `--fusion=inductor` still checks for a working Triton/CUDA compile environment (`scripts/run_sdxl_turbo_gpu.py:161-163`).

#### Step 3: Resolve the image output path

`resolve_image_path(...)` chooses:

- the explicit `--image` path, or
- `<output-dir>/<stem>_output.png`, or
- `/tmp/<stem>_output.png`

The stem includes the fusion mode so eager and compiled outputs stay separate (`scripts/run_sdxl_turbo_gpu.py:117-126`).

#### Step 4: Load the pipeline and optionally compile the UNet

This script calls the same shared helpers as the profiler path:

- `load_pipeline(...)`
- `maybe_compile(...)`
- `synchronize_device(...)`

So the model loading and optional `torch.compile` behavior are identical to the profiled path.

#### Step 5: Warm up

Warmup behavior is also the same default:

- `0` for eager
- `1` for compiled

Each warmup iteration runs the SDXL pipeline in latent mode and discards the result (`scripts/run_sdxl_turbo_gpu.py:179-184`).

#### Step 6: Do one timed inference

The main inference call is:

```python
output = run_pipeline(pipe, args)
```

That is the same latent-only SDXL invocation used by the profiling harness. The script times this section with `time.perf_counter()` and synchronizes the device before stopping the timer (`scripts/run_sdxl_turbo_gpu.py:186-189`).

#### Step 7: Decode and save the image

Unlike the weight-streaming script, this file does decode the latent result back into an image:

1. `decode_latents_to_pil(pipe, output.images)`
2. `images[0].save(image_path)`

It times decode and save separately (`scripts/run_sdxl_turbo_gpu.py:191-199`).

#### Step 8: Print timing output

The script prints:

- device, dtype, fusion mode
- warmup count
- output path
- latent shape
- `load_seconds`
- `warmup_seconds`
- `inference_seconds`
- `decode_seconds`
- `save_seconds`
- `e2e_seconds`

### What this script is for

Use this script when you want the same SDXL Turbo inference path without profiler overhead. It is the simplest way to compare eager and Inductor runtime behavior while still producing a PNG.

## 3. How `scripts/run_weight_streaming.py` works for SDXL Turbo

This script is different from the other two. It is not just "run SDXL." It is the harness that wires a scheduler-produced IO plan into Inductor-generated wrapper code.

### Important scope note

Even though the CLI and module docstring sound model-generic, the current implementation is SDXL-only in practice because `load_pipeline(...)` hardcodes `diffusers.StableDiffusionXLPipeline` (`scripts/run_weight_streaming.py:214-232`).

Also, the script currently always compiles `pipe.unet`. The `--component` flag exists in the parser, but it is not used anywhere else in the file right now.

### High-level flow

The main control flow is in `scripts/run_weight_streaming.py:276-404`.

#### Step 1: Parse the plan and model arguments

The key inputs are:

- `--plan-dir`: directory containing `jit_sim_prune_schedule.json`
- optional `--nodes-csv`
- optional model/prompt/shape options
- compile flags
- warmup count
- optional `--export-code`
- optional `--dry-run`
- optional `--log-io`

#### Step 2: Resolve and load the scheduler outputs

`resolve_schedule_paths(...)` looks for:

- `jit_sim_prune_schedule.json`
- `runtime_nodes.csv`
- `pytorch_runtime_tensors.csv`

It supports either being pointed at the actual prune-output directory or its parent bundle directory (`scripts/run_weight_streaming.py:155-192`).

Then `load_io_schedule(...)` parses those files into an `IOSchedule` (`torch/_inductor/weight_streaming/plan.py:101-204`).

That loader builds structured operation lists for:

- SSD prefetches
- H2D prefetches
- VRAM evictions
- DRAM evictions
- cold-start prefetches

It also loads tensor metadata and filters VRAM operations so only `WEIGHT` tensors are kept. `CONTEXT` tensors are skipped because those are computed intermediates that should already be produced by execution, not streamed in like weights.

#### Step 3: Optional dry run

If `--dry-run` is set, the script stops after validating the schedule-to-kernel mapping:

1. it constructs a `ScheduleAdapter`
2. it counts mapped pre-kernel ops
3. it counts mapped post-kernel ops
4. it prints startup-op count
5. it exits without loading a model

This is useful for checking that the schedule parses and can be adapted without paying the model-load or compile cost.

### Step 4: Initialize the runtime singleton

If this is a real run, the script creates the global runtime with:

```python
rt = WeightStreamRuntime.initialize(schedule, device)
```

`WeightStreamRuntime` lives in `torch/_inductor/weight_streaming/runtime.py:26-245`.

The runtime owns:

- a thread pool for SSD to DRAM reads
- pinned CPU DRAM buffers
- a dedicated CUDA stream for H2D copies
- CUDA events for copy completion
- dictionaries tracking which tensors are on SSD, in DRAM, or in VRAM
- CPU backup copies of registered GPU weights

If `--log-io` is set, `install_io_logging(...)` monkey-patches the runtime methods so every prefetch or eviction call is printed as it happens (`scripts/run_weight_streaming.py:254-273`).

#### Step 5: Tell Inductor to inject weight-streaming code

Before compiling, the script writes several config flags into `torch._inductor.config`:

- `weight_streaming_plan`
- `weight_streaming_nodes_csv`
- `weight_streaming_tensor_csv`
- `weight_streaming_output_code`

This is the bridge between the script and Inductor code generation (`scripts/run_weight_streaming.py:320-327`).

During wrapper generation, Inductor checks `config.weight_streaming_plan` and calls `_inject_weight_streaming_io()` if it is set (`torch/_inductor/codegen/wrapper.py:1975-1976`, `2061-2186`).

#### Step 6: Load the SDXL pipeline and register UNet weights

The script loads `StableDiffusionXLPipeline` on the target device and disables the progress bar (`scripts/run_weight_streaming.py:329-334`).

Then `register_model_weights(...)` walks every UNet parameter and buffer and calls `rt.register_weight(param.data)` (`scripts/run_weight_streaming.py:235-251`, `336-338`).

That registration is important. In the runtime, `register_weight(...)` creates a pinned CPU backup copy of each live GPU tensor and stores the original GPU storage size (`torch/_inductor/weight_streaming/runtime.py:82-90`).

This is what makes the "tensor-object" form of eviction and restore possible:

- `evict_vram(tensor)` can shrink the tensor storage to zero bytes
- `h2d_prefetch(tensor)` can resize the storage back and copy data from the pinned CPU backup

#### Step 7: Compile the UNet with `torch.compile`

The script compiles only `pipe.unet`:

```python
pipe.unet = torch.compile(
    pipe.unet,
    backend="inductor",
    mode=args.compile_mode,
    fullgraph=args.fullgraph,
)
```

That compile step is where Inductor actually injects the streaming calls into the generated wrapper code.

### How the Inductor injection works

This is split across `torch/_inductor/weight_streaming/codegen.py` and `torch/_inductor/codegen/wrapper.py`.

#### `ScheduleAdapter` maps scheduler nodes to wrapper kernels

The scheduler trace contains interleaved CPU and GPU nodes. The compiled Inductor wrapper only has GPU kernel dispatch sites.

`ScheduleAdapter`:

- classifies scheduler nodes as GPU or CPU
- builds a trace-index to wrapper-kernel-index mapping
- snaps prefetches targeting CPU nodes forward to the next GPU kernel
- snaps evictions targeting CPU nodes backward to the previous GPU kernel
- rescales the schedule positions because fused Inductor code often has fewer wrapper kernels than the original runtime trace had GPU nodes

See `torch/_inductor/weight_streaming/codegen.py:57-174`.

#### The wrapper injects IO calls around kernel launches

`_inject_weight_streaming_io()` in `torch/_inductor/codegen/wrapper.py:2061-2186`:

1. reloads the schedule from config
2. creates a `ScheduleAdapter`
3. imports `WeightStreamRuntime` into the generated wrapper header
4. counts actual wrapper kernel calls
5. rescales SSD and DRAM ops to those kernel indices
6. builds a mapping from scheduler tensor names to actual graph input variables
7. inserts custom `WeightStreamingLine` objects before and after kernel calls

The placement logic is:

- before any kernel: `cold_start_prefetch(...)`
- before a kernel: `ssd_prefetch(...)`
- before the first kernel that uses a weight tensor: `h2d_prefetch(...)`
- after the first kernel that uses a streamed weight: `evict_vram(...)`
- after a kernel: `evict_dram(...)`

#### How weights get matched to graph inputs

For VRAM operations, the wrapper tries to map scheduler tensor names to actual graph input variables by matching unique `numel` values (`torch/_inductor/weight_streaming/codegen.py:212-258`).

Then it groups H2D and VRAM-evict ops per graph input variable (`torch/_inductor/weight_streaming/codegen.py:261-296`).

That matters because H2D/VRAM operations are performed on the live tensor objects themselves, not just on string names.

### Step 8: Warm up and do the timed run

After compile, the script runs SDXL inference in latent mode for each warmup iteration and synchronizes CUDA after each run (`scripts/run_weight_streaming.py:351-364`).

Then it performs one timed inference run, prints:

- elapsed inference time
- latent tensor shape
- current and peak GPU memory

Unlike `run_sdxl_turbo_gpu.py`, this script does not decode the latent to a PNG.

### Step 9: Optionally export generated code

If `--export-code` is set, the wrapper generator writes:

- `<export-code>/weight_streaming_generated.py`

That export happens inside Inductor after wrapper generation when both `weight_streaming_plan` and `weight_streaming_output_code` are set (`torch/_inductor/codegen/wrapper.py:2045-2059`).

This is useful if you want to inspect exactly where `_ws_rt.ssd_prefetch(...)`, `_ws_rt.h2d_prefetch(...)`, `_ws_rt.evict_vram(...)`, and `_ws_rt.evict_dram(...)` ended up in the generated wrapper.

### Step 10: Cleanup

The script clears the Inductor config flags and resets the singleton runtime at the end (`scripts/run_weight_streaming.py:399-404`).

That prevents the next compile in the same Python process from accidentally reusing the previous streaming configuration.

## Relationship Between the Three Files

- `profile_sdxl_turbo_gpu.py` and `run_sdxl_turbo_gpu.py` are normal SDXL Turbo runners built around the same shared helper module.
- `profile_sdxl_turbo_gpu.py` adds profiler capture and optional graph exports.
- `run_sdxl_turbo_gpu.py` strips that down to a simpler timed run plus PNG save.
- `run_weight_streaming.py` is not built on the shared helper. It is a separate experimental path that configures Inductor code generation and a runtime singleton so compiled UNet wrapper code can prefetch and evict weights according to an offline plan.

## Pseudocode Reference

These pseudocode blocks show the main execution path for the actual scripts that run the models.

They are not limited to the generated Inductor wrapper.

- The SDXL, Qwen, and Accelerate offload pseudocode blocks describe the top-level Python control flow in the entry scripts and shared helper modules.
- Only the weight-streaming subsection named "How the Inductor injection works" is specifically about wrapper/codegen insertion.

### SDXL Turbo runtime path

This corresponds to `scripts/run_sdxl_turbo_gpu.py` plus shared helpers from `scripts/profile_sdxl_turbo_common.py`.

```python
args = parse_args()
device = validate_run_device(args.device)
validate_fusion_runtime(args, device)
torch_dtype = DTYPE_BY_NAME[args.dtype]
warmup_runs = args.warmup_runs if args.warmup_runs is not None else (
    1 if args.fusion == "inductor" else 0
)
image_path = resolve_image_path(args)

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model,
    torch_dtype=torch_dtype,
    use_safetensors=True,
).to(device)

if args.disable_progress_bar:
    pipe.set_progress_bar_config(disable=True)

if args.fusion == "inductor":
    pipe.unet = torch.compile(
        pipe.unet,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )

synchronize_device(device)

for _ in range(warmup_runs):
    _ = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        height=args.height,
        width=args.width,
        output_type="latent",
    )
    synchronize_device(device)

output = pipe(
    prompt=args.prompt,
    num_inference_steps=args.steps,
    guidance_scale=0.0,
    height=args.height,
    width=args.width,
    output_type="latent",
)
synchronize_device(device)

with torch.no_grad():
    images = decode_sdxl_latents_with_vae(pipe, output.images)

images[0].save(image_path)
print_timing_summary()
```

### SDXL Turbo profiled path

This corresponds to `scripts/profile_sdxl_turbo_gpu.py`, which is a thin preset over `scripts/profile_sdxl_turbo_common.py`.

```python
args = parse_profile_args()
validate_dot_args(args)
device = validate_device(args)
validate_fusion_runtime(args, device)
output_paths = resolve_output_paths(args)

pipe = load_sdxl_pipeline(args.model, dtype=args.dtype, device=device)
maybe_compile_unet(pipe, args)

if fx_dot_requested:
    captured_inputs = install_forward_pre_hook(pipe.unet)

for _ in range(default_or_requested_warmups):
    _ = run_sdxl_pipeline(pipe, args)  # latent output
    synchronize_device(device)

observer = maybe_create_execution_trace_observer(output_paths.execution_trace_path)

with torch.profiler.profile(..., execution_trace_observer=observer) as prof:
    with record_function("sdxl_turbo_run", metadata_for_scope(args)):
        output = run_sdxl_pipeline(pipe, args)  # latent output
        synchronize_device(device)

export_chrome_trace(prof, output_paths.trace_path)
export_csv_if_available(prof, output_paths.csv_path)

with torch.no_grad():
    images = decode_sdxl_latents_with_vae(pipe, output.images)
images[0].save(output_paths.image_path)

maybe_export_fx_dot(...)
maybe_export_execution_dot(...)
maybe_export_memory_dot(...)
maybe_export_kineto_dot(...)
maybe_export_hybrid_dot(...)
maybe_export_runtime_io_dot(...)
maybe_export_llamasim_bundle(...)

print_profiler_table_and_artifact_summary()
```

### Qwen-Image runtime path

This corresponds to `scripts/run_qwen_image.py` plus shared helpers from `scripts/profile_qwen_image_common.py`.

```python
args = parse_args()
device = validate_run_device(args.device)
validate_fusion_runtime(args, device)
torch_dtype = DTYPE_BY_NAME[args.dtype]
warmup_runs = args.warmup_runs if args.warmup_runs is not None else (
    1 if args.fusion == "inductor" else 0
)
image_path = resolve_image_path(args)

install_qwen25_transformers_compat()
model_path = Path(args.model)

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path / "text_encoder",
    torch_dtype=torch_dtype,
)
tokenizer = Qwen2Tokenizer.from_pretrained(model_path / "tokenizer")
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_path / "transformer",
    torch_dtype=torch_dtype,
)
vae = AutoencoderKLQwenImage.from_pretrained(
    model_path / "vae",
    torch_dtype=torch_dtype,
)
pipe = QwenImagePipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer,
).to(device)

if args.disable_progress_bar:
    pipe.set_progress_bar_config(disable=True)

if args.fusion == "inductor":
    # Compile only forward to preserve transformer instance state.
    pipe.transformer.forward = torch.compile(
        pipe.transformer.forward,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )

synchronize_device(device)

generator = (
    None
    if args.seed is None or args.seed < 0
    else torch.Generator(device=str(device)).manual_seed(args.seed)
)

for _ in range(warmup_runs):
    _ = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=args.guidance_scale,         # only when not None
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        max_sequence_length=args.max_sequence_length,
        generator=generator,                        # only when not None
        output_type="latent",
    )
    synchronize_device(device)

output = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    true_cfg_scale=args.true_cfg_scale,
    guidance_scale=args.guidance_scale,             # only when not None
    num_inference_steps=args.steps,
    height=args.height,
    width=args.width,
    max_sequence_length=args.max_sequence_length,
    generator=generator,                            # only when not None
    output_type="latent",
)
synchronize_device(device)

with torch.no_grad():
    latents = pipe._unpack_latents(
        output.images,
        args.height,
        args.width,
        pipe.vae_scale_factor,
    )
    latents = denormalize_qwen_latents(latents, pipe.vae.config)
    image_tensor = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    images = pipe.image_processor.postprocess(image_tensor, output_type="pil")

images[0].save(image_path)
print_timing_summary()
```

### Qwen-Image profiled path

This corresponds to `scripts/profile_qwen_image_gpu.py`, which is a thin preset over `scripts/profile_qwen_image_common.py`.

```python
args = parse_profile_args()
validate_dot_args(args)
device = validate_device(args)
validate_fusion_runtime(args, device)
output_paths = resolve_output_paths(args)

pipe = load_qwen_pipeline(args.model, dtype=args.dtype, device=device)
maybe_compile_qwen_transformer_forward(pipe, args)
generator = build_generator(args, device)

if fx_dot_requested:
    captured_inputs = install_forward_pre_hook(pipe.transformer)

for _ in range(default_or_requested_warmups):
    _ = run_qwen_pipeline(pipe, args, generator)  # latent output
    synchronize_device(device)

observer = maybe_create_execution_trace_observer(output_paths.execution_trace_path)

with torch.profiler.profile(..., execution_trace_observer=observer) as prof:
    with record_function("qwen_image_run", metadata_for_scope(args)):
        output = run_qwen_pipeline(pipe, args, generator)  # latent output
        synchronize_device(device)

export_chrome_trace(prof, output_paths.trace_path)
export_csv_if_available(prof, output_paths.csv_path)

with torch.no_grad():
    images = decode_qwen_latents_with_vae(
        pipe,
        output.images,
        height=args.height,
        width=args.width,
    )
images[0].save(output_paths.image_path)

maybe_export_transformer_fx_dot(...)
maybe_export_execution_dot(...)
maybe_export_memory_dot(...)
maybe_export_kineto_dot(...)
maybe_export_hybrid_dot(...)
maybe_export_runtime_io_dot(...)
maybe_export_llamasim_bundle(...)

print_profiler_table_and_artifact_summary()
```

### Accelerate CPU offload runtime path

This corresponds to `scripts/run_accelerate_cpu_offload.py`. It is the actual runner for both `sdxl-turbo` and `qwen-image`.

```python
args = parse_args()                     # subcommand: sdxl-turbo or qwen-image
device = validate_run_device(args.device)
validate_fusion_runtime(args, device)
accelerate_version = ensure_accelerate_available() if args.offload_mode != "none" else "not_used"
torch_dtype = DTYPE_BY_NAME[args.dtype]
image_path = resolve_image_path(args)

if args.pipeline == "sdxl-turbo":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
else:
    install_qwen25_transformers_compat()
    pipe = assemble_qwen_pipeline_from_scheduler_text_encoder_tokenizer_transformer_vae(
        args.model,
        torch_dtype=torch_dtype,
    )

if args.disable_progress_bar:
    pipe.set_progress_bar_config(disable=True)

if args.fusion == "inductor":
    if args.pipeline == "sdxl-turbo":
        pipe.unet = torch.compile(pipe.unet, backend="inductor", ...)
    else:
        pipe.transformer.forward = torch.compile(
            pipe.transformer.forward,
            backend="inductor",
            ...,
        )

if args.offload_mode == "none":
    pipe.to(device)
elif args.offload_mode == "model":
    pipe.enable_model_cpu_offload(device=device)
else:
    pipe.enable_sequential_cpu_offload(device=device)

synchronize_device(device)

generator = (
    build_qwen_generator(args, device)
    if args.pipeline == "qwen-image"
    else None
)

for _ in range(args.warmup_runs):
    if args.pipeline == "sdxl-turbo":
        _ = run_sdxl_pipeline(pipe, args)          # latent output
    else:
        _ = run_qwen_pipeline_with_oom_guidance(
            pipe,
            args,
            generator,
        )                                          # latent output
    synchronize_device(device)

if args.pipeline == "sdxl-turbo":
    output = run_sdxl_pipeline(pipe, args)         # latent output
else:
    output = run_qwen_pipeline_with_oom_guidance(
        pipe,
        args,
        generator,
    )                                              # latent output
synchronize_device(device)

with torch.no_grad():
    if args.pipeline == "sdxl-turbo":
        images = decode_sdxl_latents_with_vae(pipe, output.images)
    else:
        images = decode_qwen_latents_with_vae(
            pipe,
            output.images,
            height=args.height,
            width=args.width,
        )
synchronize_device(device)

pipe.maybe_free_model_hooks()
images[0].save(image_path)
print_runtime_summary()
```

### Accelerate CPU offload profiled path

This corresponds to `scripts/profile_accelerate_cpu_offload.py`. It profiles only the steady-state offloaded inference, not model load or checkpoint I/O.

```python
args = parse_args()                     # subcommand: sdxl-turbo or qwen-image
device = validate_run_device(args.device)
validate_fusion_runtime(args, device)
torch_dtype = DTYPE_BY_NAME[args.dtype]
output_paths = resolve_output_paths(args)

pipe = load_pipeline_for_selected_model(args, torch_dtype)

if args.disable_progress_bar:
    pipe.set_progress_bar_config(disable=True)

maybe_compile_hot_path_before_offload(pipe, args)
apply_cpu_offload(pipe, args, device)

generator = (
    build_qwen_generator(args, device)
    if args.pipeline == "qwen-image"
    else None
)

for _ in range(args.warmup_runs):
    if args.pipeline == "sdxl-turbo":
        _ = run_sdxl_pipeline(pipe, args)          # latent output
    else:
        _ = run_qwen_pipeline_with_oom_guidance(
            pipe,
            args,
            generator,
        )                                          # latent output
    synchronize_device(device)

observer = torch.profiler.ExecutionTraceObserver()
observer.register_callback(output_paths.execution_trace_path)

with torch.profiler.profile(..., execution_trace_observer=observer) as prof:
    with record_function(
        f"{args.pipeline.replace('-', '_')}_cpu_offload_run",
        metadata_for_scope(args),
    ):
        if args.pipeline == "sdxl-turbo":
            output = run_sdxl_pipeline(pipe, args)  # latent output
        else:
            output = run_qwen_pipeline_with_oom_guidance(
                pipe,
                args,
                generator,
            )                                       # latent output
        synchronize_device(device)

observer.unregister_callback()
export_chrome_trace(prof, output_paths.trace_path)
export_csv_if_available(prof, output_paths.csv_path)

with torch.no_grad():
    if args.pipeline == "sdxl-turbo":
        images = decode_sdxl_latents_with_vae(pipe, output.images)
    else:
        images = decode_qwen_latents_with_vae(
            pipe,
            output.images,
            height=args.height,
            width=args.width,
        )

images[0].save(output_paths.image_path)
write_llamasim_runtime_bundle(
    prof,
    output_paths.execution_trace_path,
    output_paths.llamasim_output_dir,
)
pipe.maybe_free_model_hooks()
print_profiler_table_and_steady_state_summary()
```

## Bulletpoint Summaries

- `scripts/profile_sdxl_turbo_gpu.py`: a tiny GPU preset that delegates everything to `profile_sdxl_turbo_common.main(...)`, enabling SDXL Turbo profiling with trace export, optional execution observers, and optional DOT/llamasim outputs.
- `scripts/run_sdxl_turbo_gpu.py`: a lightweight SDXL Turbo runner that uses the shared helper functions for model load, optional UNet compilation, latent decoding, PNG saving, and per-phase timing without profiler instrumentation.
- `scripts/run_weight_streaming.py`: an SDXL Turbo-specific weight-streaming harness that loads an offline IO schedule, initializes `WeightStreamRuntime`, configures Inductor to inject prefetch/evict calls into compiled UNet wrapper code, runs latent inference, and can export the generated wrapper for inspection.
