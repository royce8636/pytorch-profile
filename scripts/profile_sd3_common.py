#!/usr/bin/env python3
"""SD 3.5 (medium / large) profiler driver.

Reuses all generic DOT/execution-trace/llamasim-runtime bundle machinery from
profile_sdxl_turbo_common. Overrides only what's specific to the SD3 pipeline:

* load_pipeline         — StableDiffusion3Pipeline (3 text encoders, MMDiT transformer)
* maybe_compile         — compiles pipe.transformer (not pipe.unet) + 3 text encoders + vae.decode
* run_pipeline          — SD3 call signature (guidance_scale default 7.0, max_sequence_length)
* decode_latents_to_pil — SD3 VAE post-processing: (latents / scaling_factor) + shift_factor
* main()                — swaps pipe.unet-based FX capture for pipe.transformer-based
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_sdxl_turbo_common import (  # noqa: E402
    DTYPE_BY_NAME,
    OutputPaths,
    build_module_catalog,
    build_module_hierarchy,
    build_module_id_to_path,
    build_module_id_to_path_from_prof,
    build_pipeline_module_index,
    build_pipeline_module_index_from_prof,
    capture_example_inputs,
    configure_llamasim_inductor_markers,
    dot_level_enabled,
    export_unet_graph_module,
    format_missing_diffusers_runtime_error,
    is_missing_diffusers_error,
    output_stem,
    print_llamasim_runtime_summary,
    profile_activities,
    profiler_synchronize_device,
    resolve_output_paths,
    synchronize_device,
    underlying_unet_module,
    validate_dot_args,
    validate_device,
    validate_fusion_runtime,
    write_execution_trace_dot,
    write_fx_graph_dot,
    write_hybrid_profile_dot,
    write_kineto_profile_dot,
    write_llamasim_runtime_bundle,
    write_memory_profile_dot,
    write_runtime_io_profile_dot,
)

import time  # noqa: E402


def parse_args(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_model: str | None,
    default_component: str,
    default_fusion: str,
    default_dot_level: str,
    default_profile_memory: bool,
    default_record_shapes: bool,
    default_with_stack: bool,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 3.5 profiler driver for the patched PyTorch build."
    )
    parser.add_argument(
        "--model",
        required=default_model is None,
        default=default_model,
        help="Path to the SD3.5 model directory (diffusers format).",
    )
    parser.add_argument(
        "--prompt",
        default="A cute dog and a cat in a park",
        help="Prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale (SD3 default is ~4.5–7.0).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed passed to torch.manual_seed. Set negative to skip seeding.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Max T5 prompt tokens (SD3-specific; default 256, max 512).",
    )
    parser.add_argument(
        "--device",
        default=default_device,
        help="Execution device, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BY_NAME.keys()),
        default=default_dtype,
        help="Torch dtype used to load the pipeline.",
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default=default_fusion,
        help=(
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the transformer "
            "with torch.compile."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode when --fusion=inductor.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=20,
        help=(
            "Warmup iterations run before profiling. Defaults to the SD3.5 "
            "medium denoising step count."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--output-prefix",
        default=default_output_prefix,
        help="Prefix used for trace, image, and optional DOT output filenames.",
    )
    parser.add_argument(
        "--trace",
        default=None,
    )
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument(
        "--dot-level",
        choices=(
            "none", "fx", "execution", "memory", "kineto", "hybrid",
            "runtime-io", "llamasim-runtime", "both", "all",
        ),
        default=default_dot_level,
    )
    parser.add_argument("--fx-dot", default=None)
    parser.add_argument("--execution-dot", default=None)
    parser.add_argument("--execution-trace", default=None)
    parser.add_argument("--memory-dot", default=None)
    parser.add_argument("--kineto-dot", default=None)
    parser.add_argument("--kineto-map", default=None)
    parser.add_argument("--hybrid-dot", default=None)
    parser.add_argument("--runtime-io-dot", default=None)
    parser.add_argument("--llamasim-output-dir", default=None)
    parser.add_argument("--component", default=default_component)
    parser.add_argument("--phase", default="text_to_image")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiled decode (required to avoid OOM at 1024×1024 on 24 GB GPUs).",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help=(
            "Use diffusers' enable_model_cpu_offload (sequential per-submodule CPU↔GPU "
            "swap). Required for SD 3.5 large on 24 GB GPUs. Incompatible with --fusion "
            "because torch.compile doesn't cooperate with offload hooks; incompatible with "
            "--dot-level=llamasim-runtime because kernel order changes per call."
        ),
    )
    parser.add_argument(
        "--profile-memory", dest="profile_memory", action="store_true",
        default=default_profile_memory,
    )
    parser.add_argument(
        "--no-profile-memory", dest="profile_memory", action="store_false",
    )
    parser.add_argument(
        "--record-shapes", dest="record_shapes", action="store_true",
        default=default_record_shapes,
    )
    parser.add_argument(
        "--no-record-shapes", dest="record_shapes", action="store_false",
    )
    parser.add_argument(
        "--with-stack", dest="with_stack", action="store_true",
        default=default_with_stack,
    )
    parser.add_argument(
        "--no-with-stack", dest="with_stack", action="store_false",
    )
    return parser.parse_args()


def load_pipeline(
    model: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    *,
    vae_tiling: bool = False,
    cpu_offload: bool = False,
) -> Any:
    try:
        from diffusers import StableDiffusion3Pipeline
    except ModuleNotFoundError as exc:
        if is_missing_diffusers_error(exc):
            raise RuntimeError(
                format_missing_diffusers_runtime_error(
                    component="the SD3.5 scripts", exc=exc,
                )
            ) from exc
        raise

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model, torch_dtype=torch_dtype, use_safetensors=True,
    )
    if cpu_offload:
        # Keep weights on CPU; diffusers walks submodules onto device around each call.
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)
    if vae_tiling:
        pipe.vae.enable_tiling()
    return pipe


def maybe_compile(
    pipe: Any,
    args: argparse.Namespace,
    output_paths: OutputPaths | None = None,
) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")

    if output_paths is not None and output_paths.llamasim_output_dir is not None:
        configure_llamasim_inductor_markers(output_paths.llamasim_output_dir)

    # SD3's main denoiser is an MMDiT transformer, not a UNet.
    pipe.transformer = torch.compile(
        pipe.transformer,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )
    # SD3 has three text encoders: 2 CLIPs + 1 T5.
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipe, attr, None)
        if module is None:
            continue
        setattr(
            pipe, attr,
            torch.compile(
                module, backend="inductor", mode=args.compile_mode, fullgraph=False,
            ),
        )
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.decode = torch.compile(
            pipe.vae.decode, backend="inductor", mode=args.compile_mode, fullgraph=False,
        )


def run_pipeline(pipe: Any, args: argparse.Namespace) -> Any:
    return pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        max_sequence_length=args.max_sequence_length,
        output_type="latent",
    )


def decode_latents_to_pil(pipe: Any, latents: torch.Tensor) -> list[Any]:
    """SD3-specific VAE post-processing.

    SD3's VAE expects: latents = (model_output / scaling_factor) + shift_factor
    before decode. Mirrors diffusers' StableDiffusion3Pipeline.__call__ tail.
    """
    vae = pipe.vae
    scaling_factor = vae.config.scaling_factor
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    latents = (latents / scaling_factor) + shift_factor
    if latents.dtype != vae.dtype:
        latents = latents.to(vae.dtype)
    image = vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"sample_step=0 phase={args.phase} component={args.component} "
        f"device={args.device} fusion={args.fusion}"
    )


def underlying_transformer_module(module: torch.nn.Module) -> torch.nn.Module:
    # Same convention as profile_sdxl_turbo_common.underlying_unet_module.
    return underlying_unet_module(module)


def main(
    *,
    default_device: str,
    default_dtype: str,
    default_output_prefix: str,
    default_model: str | None,
    default_component: str = "sd3_pipeline",
    default_fusion: str = "none",
    default_dot_level: str = "none",
    default_llamasim_output_dirname: str | None = None,
    default_profile_memory: bool = False,
    default_record_shapes: bool = False,
    default_with_stack: bool = False,
) -> None:
    args = parse_args(
        default_device=default_device,
        default_dtype=default_dtype,
        default_output_prefix=default_output_prefix,
        default_model=default_model,
        default_component=default_component,
        default_fusion=default_fusion,
        default_dot_level=default_dot_level,
        default_profile_memory=default_profile_memory,
        default_record_shapes=default_record_shapes,
        default_with_stack=default_with_stack,
    )
    validate_dot_args(args)
    device = validate_device(args)
    validate_fusion_runtime(args, device)
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    warmup_runs = (
        args.warmup_runs if args.warmup_runs is not None else int(args.fusion != "none")
    )
    output_paths = resolve_output_paths(
        args,
        default_output_prefix,
        default_llamasim_output_dirname=default_llamasim_output_dirname,
    )

    pipe = load_pipeline(
        args.model, torch_dtype, device,
        vae_tiling=args.vae_tiling, cpu_offload=args.cpu_offload,
    )
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    if not args.cpu_offload:
        maybe_compile(pipe, args, output_paths)

    captured_inputs: dict[str, Any] = {}
    capture_handle = None
    if output_paths.fx_dot_path is not None:
        captured_inputs, capture_handle = capture_example_inputs(pipe.transformer)

    for _ in range(warmup_runs):
        warmup_output = run_pipeline(pipe, args)
        with torch.no_grad():
            _ = decode_latents_to_pil(pipe, warmup_output.images)
        synchronize_device(device)
        del warmup_output

    scope_args = metadata_for_scope(args)
    execution_trace_observer = None
    if output_paths.execution_trace_path is not None:
        execution_trace_observer = torch.profiler.ExecutionTraceObserver()
        execution_trace_observer.register_callback(str(output_paths.execution_trace_path))

    with torch.profiler.profile(
        activities=profile_activities(device),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_modules=True,
        execution_trace_observer=execution_trace_observer,
    ) as prof:
        with torch.autograd.profiler.record_function("sd3_run", scope_args):
            output = run_pipeline(pipe, args)
            with torch.no_grad():
                images = decode_latents_to_pil(pipe, output.images)
        profiler_synchronize_device(device)

    if capture_handle is not None:
        capture_handle.remove()
    if execution_trace_observer is not None:
        execution_trace_observer.unregister_callback()

    metadata_json = None
    for event in prof.events():
        if event.name == "sd3_run":
            metadata_json = getattr(event, "metadata_json", None)
            break

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    images[0].save(output_paths.image_path)

    if output_paths.fx_dot_path is not None:
        if "args" not in captured_inputs or "kwargs" not in captured_inputs:
            raise RuntimeError("Failed to capture transformer example inputs for FX DOT export")
        fx_graph_module = export_unet_graph_module(
            underlying_transformer_module(pipe.transformer),
            captured_inputs["args"],
            captured_inputs["kwargs"],
        )
        write_fx_graph_dot(
            fx_graph_module,
            output_paths.fx_dot_path,
            output_paths.fx_dot_path.stem,
        )

    if output_paths.execution_dot_path is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError("Execution DOT export requested without an execution trace path")
        write_execution_trace_dot(
            output_paths.execution_trace_path,
            output_paths.execution_dot_path,
            output_paths.execution_dot_path.stem,
        )
    if output_paths.memory_dot_path is not None:
        write_memory_profile_dot(
            prof, output_paths.memory_dot_path, output_paths.memory_dot_path.stem,
        )
    if output_paths.kineto_dot_path is not None:
        write_kineto_profile_dot(
            prof,
            output_paths.kineto_dot_path,
            output_paths.kineto_dot_path.stem,
            output_paths.kineto_map_path,
        )
    if output_paths.hybrid_dot_path is not None:
        write_hybrid_profile_dot(
            prof, output_paths.hybrid_dot_path, output_paths.hybrid_dot_path.stem,
        )
    if output_paths.runtime_io_dot_path is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError("Runtime-I/O DOT export requested without an execution trace path")
        write_runtime_io_profile_dot(
            prof,
            output_paths.execution_trace_path,
            output_paths.runtime_io_dot_path,
            output_paths.runtime_io_dot_path.stem,
        )
    if output_paths.llamasim_output_dir is not None:
        if output_paths.execution_trace_path is None:
            raise RuntimeError(
                "llamasim-runtime export requested without an execution trace path"
            )
        _trans = underlying_transformer_module(pipe.transformer)
        # Use prof-derived observation-order mapping so the trace's
        # `nn.Module: <cls>_<id>` event names resolve to the correct
        # submodule paths instead of named_modules() walk-order.
        _pipe_catalog, _pipe_id_to_path = build_pipeline_module_index_from_prof(
            prof, pipe,
        )
        write_llamasim_runtime_bundle(
            prof,
            output_paths.execution_trace_path,
            output_paths.llamasim_output_dir,
            trace_json_path=output_paths.trace_path,
            module_catalog=build_module_catalog(_trans),
            module_id_to_path=build_module_id_to_path_from_prof(prof, _trans),
            pipeline_module_catalog=_pipe_catalog,
            pipeline_module_id_to_path=_pipe_id_to_path,
            module_hierarchy=build_module_hierarchy(pipe),
        )

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    if args.fusion != "none":
        from torch._dynamo.utils import counters as _dynamo_counters
        unique_graphs = _dynamo_counters.get("stats", {}).get("unique_graphs", 0)
        print(f"inductor_unique_graphs: {unique_graphs}")
    print("device:", device)
    print("dtype:", args.dtype)
    print("fusion:", args.fusion)
    print("dot_level:", args.dot_level)
    print("seed:", args.seed)
    print("profile_memory:", args.profile_memory)
    print("record_shapes:", args.record_shapes)
    print("with_stack:", args.with_stack)
    print("warmup_runs:", warmup_runs)
    print("trace_path:", output_paths.trace_path)
    if hasattr(prof, "export_csv"):
        print("trace_csv_path:", output_paths.csv_path)
    print("image_path:", output_paths.image_path)
    if output_paths.fx_dot_path is not None:
        print("fx_dot_path:", output_paths.fx_dot_path)
    if output_paths.execution_trace_path is not None:
        print("execution_trace_path:", output_paths.execution_trace_path)
    if output_paths.execution_dot_path is not None:
        print("execution_dot_path:", output_paths.execution_dot_path)
    if output_paths.memory_dot_path is not None:
        print("memory_dot_path:", output_paths.memory_dot_path)
    if output_paths.kineto_dot_path is not None:
        print("kineto_dot_path:", output_paths.kineto_dot_path)
    if output_paths.kineto_map_path is not None:
        print("kineto_map_path:", output_paths.kineto_map_path)
    if output_paths.hybrid_dot_path is not None:
        print("hybrid_dot_path:", output_paths.hybrid_dot_path)
    if output_paths.runtime_io_dot_path is not None:
        print("runtime_io_dot_path:", output_paths.runtime_io_dot_path)
    if output_paths.llamasim_output_dir is not None:
        print("llamasim_output_dir:", output_paths.llamasim_output_dir)
        print_llamasim_runtime_summary(output_paths.llamasim_output_dir)
    print("latent_shape:", tuple(output.images.shape))
    print("metadata_json:", metadata_json)
    if metadata_json:
        print("metadata_parsed:", json.dumps(json.loads(metadata_json), sort_keys=True))


__all__ = [
    "DTYPE_BY_NAME",
    "decode_latents_to_pil",
    "load_pipeline",
    "main",
    "maybe_compile",
    "metadata_for_scope",
    "run_pipeline",
    "underlying_transformer_module",
]
