#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_sd3_common as sd3_common  # noqa: E402
import run_sd3_accelerate_cpu_offload as sd3_offload  # noqa: E402
from run_accelerate_cpu_offload import (  # noqa: E402
    ensure_accelerate_available,
    validate_run_device,
)
from profile_sdxl_turbo_common import (  # noqa: E402
    build_module_catalog,
    build_module_hierarchy,
    build_module_id_to_path,
    build_module_id_to_path_from_prof,
    build_pipeline_module_index,
    build_pipeline_module_index_from_prof,
    configure_llamasim_inductor_markers,
    print_llamasim_runtime_summary,
    profile_activities,
    profiler_synchronize_device,
    underlying_unet_module,
    write_llamasim_runtime_bundle,
)


@dataclass(frozen=True)
class OutputPaths:
    trace_path: Path
    csv_path: Path
    image_path: Path
    execution_trace_path: Path
    llamasim_output_dir: Path


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace", default=None)
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--execution-trace", default=None)
    parser.add_argument("--llamasim-output-dir", default=None)
    parser.add_argument(
        "--record-shapes", dest="record_shapes", action="store_true", default=True,
    )
    parser.add_argument("--no-record-shapes", dest="record_shapes", action="store_false")
    parser.add_argument(
        "--profile-memory", dest="profile_memory", action="store_true", default=True,
    )
    parser.add_argument("--no-profile-memory", dest="profile_memory", action="store_false")
    parser.add_argument(
        "--with-stack", dest="with_stack", action="store_true", default=True,
    )
    parser.add_argument("--no-with-stack", dest="with_stack", action="store_false")
    parser.add_argument(
        "--with-modules", dest="with_modules", action="store_true", default=True,
    )
    parser.add_argument("--no-with-modules", dest="with_modules", action="store_false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile SD 3.5 medium with HF Accelerate CPU offload. Profiling starts "
            "only after model load and warmup, so checkpoint I/O is excluded from the trace."
        )
    )
    sd3_offload.add_common_args(
        parser,
        default_model="/data/llamasim/models/sd-3.5-med",
        default_warmup_runs=4,
        default_output_prefix=None,
    )
    add_profile_args(parser)
    return parser.parse_args()


def output_stem(args: argparse.Namespace) -> str:
    output_prefix = getattr(args, "output_prefix", None)
    if output_prefix is not None:
        return output_prefix
    return f"{sd3_offload.output_stem_base(args)}_profile"


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    stem = output_stem(args)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        trace_path = output_dir / f"{stem}_trace.json"
        csv_path = output_dir / f"{stem}_trace.csv"
        image_path = (
            Path(args.image) if args.image is not None
            else output_dir / f"{stem}_output.png"
        )
        execution_trace_path = (
            Path(args.execution_trace) if args.execution_trace is not None
            else output_dir / f"{stem}_execution_trace.json"
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir) if args.llamasim_output_dir is not None
            else output_dir / "llama_bundle"
        )
    else:
        trace_path = (
            Path(args.trace) if args.trace is not None
            else Path("/tmp") / f"{stem}_trace.json"
        )
        csv_path = (
            Path(args.trace_csv) if args.trace_csv is not None
            else trace_path.with_suffix(".csv")
        )
        image_path = (
            Path(args.image) if args.image is not None
            else trace_path.with_name(f"{stem}_output.png")
        )
        execution_trace_path = (
            Path(args.execution_trace) if args.execution_trace is not None
            else trace_path.with_name(f"{stem}_execution_trace.json")
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir) if args.llamasim_output_dir is not None
            else trace_path.with_name(f"{stem}_llamasim_runtime")
        )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    execution_trace_path.parent.mkdir(parents=True, exist_ok=True)
    llamasim_output_dir.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        trace_path=trace_path,
        csv_path=csv_path,
        image_path=image_path,
        execution_trace_path=execution_trace_path,
        llamasim_output_dir=llamasim_output_dir,
    )


def main() -> None:
    args = parse_args()
    device = validate_run_device(args.device)
    accelerate_version = (
        ensure_accelerate_available()
        if args.offload_mode not in ("none", "group")
        else "not_used"
    )
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
    torch_dtype = sd3_common.DTYPE_BY_NAME[args.dtype]
    output_paths = resolve_output_paths(args)

    pipe = sd3_offload.load_pipeline(args, torch_dtype)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    configure_llamasim_inductor_markers(output_paths.llamasim_output_dir)
    sd3_offload.maybe_compile(pipe, args)
    sd3_offload.apply_cpu_offload(pipe, args, device)

    for _ in range(args.warmup_runs):
        warmup_output = sd3_common.run_pipeline(pipe, args)
        sd3_offload.synchronize_device(device)
        del warmup_output

    execution_trace_observer = torch.profiler.ExecutionTraceObserver()
    execution_trace_observer.register_callback(str(output_paths.execution_trace_path))
    try:
        with torch.profiler.profile(
            activities=profile_activities(device),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            with_modules=args.with_modules,
            execution_trace_observer=execution_trace_observer,
        ) as prof:
            with torch.autograd.profiler.record_function(
                "sd3_cpu_offload_run",
                f"pipeline=sd3_med device={args.device} dtype={args.dtype} "
                f"offload_mode={args.offload_mode} fusion={args.fusion}",
            ):
                output = sd3_common.run_pipeline(pipe, args)
            profiler_synchronize_device(device)
    finally:
        execution_trace_observer.unregister_callback()

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    with torch.no_grad():
        images = sd3_common.decode_latents_to_pil(pipe, output.images)
    images[0].save(output_paths.image_path)

    sd3_offload.free_offload_hooks(pipe)

    transformer = underlying_unet_module(pipe.transformer)
    # Use prof-derived observation-order mapping (see
    # `build_module_id_to_path_from_prof`).
    _pipe_catalog, _pipe_id_to_path = build_pipeline_module_index_from_prof(
        prof, pipe,
    )
    write_llamasim_runtime_bundle(
        prof,
        output_paths.execution_trace_path,
        output_paths.llamasim_output_dir,
        trace_json_path=output_paths.trace_path,
        module_catalog=build_module_catalog(transformer),
        module_id_to_path=build_module_id_to_path_from_prof(prof, transformer),
        pipeline_module_catalog=_pipe_catalog,
        pipeline_module_id_to_path=_pipe_id_to_path,
        module_hierarchy=build_module_hierarchy(pipe),
    )

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("model:", args.model)
    print("requested_device:", device)
    print("offload_mode:", args.offload_mode)
    print("fusion:", args.fusion)
    print("accelerate_version:", accelerate_version)
    print("dtype:", args.dtype)
    print("seed:", args.seed)
    print("steps:", args.steps)
    print("guidance_scale:", args.guidance_scale)
    print("height:", args.height)
    print("width:", args.width)
    print("max_sequence_length:", args.max_sequence_length)
    print("profile_memory:", args.profile_memory)
    print("record_shapes:", args.record_shapes)
    print("with_stack:", args.with_stack)
    print("with_modules:", args.with_modules)
    print("warmup_runs:", args.warmup_runs)
    print("trace_path:", output_paths.trace_path)
    if hasattr(prof, "export_csv"):
        print("trace_csv_path:", output_paths.csv_path)
    print("execution_trace_path:", output_paths.execution_trace_path)
    print("llamasim_output_dir:", output_paths.llamasim_output_dir)
    print_llamasim_runtime_summary(output_paths.llamasim_output_dir)
    print("image_path:", output_paths.image_path)
    print("latent_shape:", tuple(output.images.shape))
    print(
        "profile_scope: steady_state_inference_only "
        "(load, checkpoint reads, and warmup are outside the trace)"
    )


if __name__ == "__main__":
    main()
