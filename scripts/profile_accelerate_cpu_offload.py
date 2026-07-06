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

import profile_sdxl_turbo_common as sdxl_common  # noqa: E402
import run_accelerate_cpu_offload as offload_common  # noqa: E402


@dataclass(frozen=True)
class OutputPaths:
    trace_path: Path
    csv_path: Path
    image_path: Path
    execution_trace_path: Path
    llamasim_output_dir: Path


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional prefix for trace, CSV, image, and execution-trace filenames. "
            "When omitted, filenames include pipeline, offload mode, fusion, and steps."
        ),
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="Path for the exported Chrome trace JSON.",
    )
    parser.add_argument(
        "--trace-csv",
        default=None,
        help="Path for the exported profiler CSV. Defaults next to --trace or under --output-dir.",
    )
    parser.add_argument(
        "--execution-trace",
        default=None,
        help=(
            "Path for the raw execution trace JSON used to build the llama bundle. "
            "Defaults next to --trace or under --output-dir."
        ),
    )
    parser.add_argument(
        "--llamasim-output-dir",
        default=None,
        help=(
            "Directory for the emitted llama bundle. Defaults next to --trace or under "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--record-shapes",
        dest="record_shapes",
        action="store_true",
        help="Capture tensor shapes in the profiler results.",
    )
    parser.add_argument(
        "--no-record-shapes",
        dest="record_shapes",
        action="store_false",
        help="Disable tensor-shape capture in the profiler results.",
    )
    parser.add_argument(
        "--profile-memory",
        dest="profile_memory",
        action="store_true",
        help="Capture memory events in the profiler results.",
    )
    parser.add_argument(
        "--no-profile-memory",
        dest="profile_memory",
        action="store_false",
        help="Disable memory capture in the profiler results.",
    )
    parser.add_argument(
        "--with-stack",
        dest="with_stack",
        action="store_true",
        help="Capture Python source locations in the profiler results.",
    )
    parser.add_argument(
        "--no-with-stack",
        dest="with_stack",
        action="store_false",
        help="Disable Python source-location capture in the profiler results.",
    )
    parser.add_argument(
        "--with-modules",
        dest="with_modules",
        action="store_true",
        help="Record module hierarchy information in profiler events.",
    )
    parser.add_argument(
        "--no-with-modules",
        dest="with_modules",
        action="store_false",
        help="Disable module hierarchy capture in profiler events.",
    )
    parser.set_defaults(
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile SDXL-Turbo with HF Accelerate CPU offload. Profiling starts "
            "only after load and unprofiled warmup, so checkpoint I/O is excluded "
            "from the trace."
        )
    )
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    sdxl_parser = subparsers.add_parser(
        "sdxl-turbo",
        help="Profile the local SDXL-Turbo pipeline with CPU offload.",
    )
    offload_common.add_common_args(
        sdxl_parser,
        default_model="/data/llamasim/models/sdxl-turbo",
        default_dtype="float16",
        default_offload_mode="model",
        default_warmup_runs=4,
    )
    add_profile_args(sdxl_parser)

    return parser.parse_args()


def output_stem(args: argparse.Namespace) -> str:
    output_prefix = getattr(args, "output_prefix", None)
    if output_prefix is not None:
        return sdxl_common.output_stem(output_prefix, args.fusion)
    return f"{offload_common.output_stem_base(args)}_profile"


def resolve_output_paths(args: argparse.Namespace) -> OutputPaths:
    stem = output_stem(args)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        trace_path = output_dir / f"{stem}_trace.json"
        csv_path = output_dir / f"{stem}_trace.csv"
        image_path = Path(args.image) if args.image is not None else output_dir / f"{stem}_output.png"
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace is not None
            else output_dir / f"{stem}_execution_trace.json"
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir is not None
            else output_dir / "llama_bundle"
        )
    else:
        trace_path = Path(args.trace) if args.trace is not None else Path("/tmp") / f"{stem}_trace.json"
        csv_path = (
            Path(args.trace_csv)
            if args.trace_csv is not None
            else trace_path.with_suffix(".csv")
        )
        image_path = (
            Path(args.image)
            if args.image is not None
            else trace_path.with_name(f"{stem}_output.png")
        )
        execution_trace_path = (
            Path(args.execution_trace)
            if args.execution_trace is not None
            else trace_path.with_name(f"{stem}_execution_trace.json")
        )
        llamasim_output_dir = (
            Path(args.llamasim_output_dir)
            if args.llamasim_output_dir is not None
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


def record_function_name(args: argparse.Namespace) -> str:
    return f"{args.pipeline.replace('-', '_')}_cpu_offload_run"


def metadata_for_scope(args: argparse.Namespace) -> str:
    return (
        f"pipeline={args.pipeline} device={args.device} dtype={args.dtype} "
        f"offload_mode={args.offload_mode} fusion={args.fusion}"
    )


def run_profiled_inference(
    pipe: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Any:
    return sdxl_common.run_pipeline(pipe, args)


def decode_images(pipe: Any, args: argparse.Namespace, output: Any) -> list[Any]:
    with torch.no_grad():
        return sdxl_common.decode_latents_to_pil(pipe, output.images)


def main() -> None:
    args = parse_args()
    device = offload_common.validate_run_device(args.device)
    sdxl_common.validate_fusion_runtime(args, device)
    accelerate_version = (
        offload_common.ensure_accelerate_available()
        if args.offload_mode != "none"
        else "not_used"
    )
    torch_dtype = sdxl_common.DTYPE_BY_NAME[args.dtype]
    output_paths = resolve_output_paths(args)
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)

    pipe = offload_common.load_pipeline(args, torch_dtype)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    sdxl_common.configure_llamasim_inductor_markers(
        output_paths.llamasim_output_dir
    )
    offload_common.maybe_compile(pipe, args)
    offload_common.apply_cpu_offload(pipe, args, device)
    for _ in range(args.warmup_runs):
        warmup_output = run_profiled_inference(pipe, args, device)
        sdxl_common.synchronize_device(device)
        del warmup_output

    # The profile starts only after weights are loaded and the offload path has
    # completed warmup, which keeps checkpoint I/O and first-touch storage costs
    # out of the captured steady-state trace.
    execution_trace_observer = torch.profiler.ExecutionTraceObserver()
    execution_trace_observer.register_callback(str(output_paths.execution_trace_path))
    try:
        with torch.profiler.profile(
            activities=sdxl_common.profile_activities(device),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            with_modules=args.with_modules,
            execution_trace_observer=execution_trace_observer,
        ) as prof:
            with torch.autograd.profiler.record_function(
                record_function_name(args),
                metadata_for_scope(args),
            ):
                output = run_profiled_inference(pipe, args, device)
            sdxl_common.profiler_synchronize_device(device)
    finally:
        execution_trace_observer.unregister_callback()

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    images = decode_images(pipe, args, output)
    images[0].save(output_paths.image_path)
    compiled_module = sdxl_common.underlying_unet_module(pipe.unet)
    # Use prof-derived observation-order mapping (see
    # `build_module_id_to_path_from_prof`).
    print("llamasim_runtime_bundle: building pipeline module index", flush=True)
    _pipe_catalog, _pipe_id_to_path = (
        sdxl_common.build_pipeline_module_index_from_prof(prof, pipe)
    )
    print("llamasim_runtime_bundle: building component module index", flush=True)
    _module_id_to_path = sdxl_common.build_module_id_to_path_from_prof(
        prof, compiled_module,
    )
    print("llamasim_runtime_bundle: writing runtime bundle", flush=True)
    sdxl_common.write_llamasim_runtime_bundle(
        prof,
        output_paths.execution_trace_path,
        output_paths.llamasim_output_dir,
        trace_json_path=output_paths.trace_path,
        module_catalog=sdxl_common.build_module_catalog(compiled_module),
        module_id_to_path=_module_id_to_path,
        pipeline_module_catalog=_pipe_catalog,
        pipeline_module_id_to_path=_pipe_id_to_path,
        module_hierarchy=sdxl_common.build_module_hierarchy(pipe),
    )
    offload_common.free_cpu_offload_hooks(pipe)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("pipeline:", args.pipeline)
    print("device:", device)
    print("dtype:", args.dtype)
    print("offload_mode:", args.offload_mode)
    print("fusion:", args.fusion)
    print("accelerate_version:", accelerate_version)
    print("seed:", args.seed)
    print("vae_model:", args.vae_model)
    print("variant:", args.variant)
    print("with_modules:", args.with_modules)
    print("warmup_runs:", args.warmup_runs)
    print("trace_path:", output_paths.trace_path)
    if hasattr(prof, "export_csv"):
        print("trace_csv_path:", output_paths.csv_path)
    print("execution_trace_path:", output_paths.execution_trace_path)
    print("llamasim_output_dir:", output_paths.llamasim_output_dir)
    sdxl_common.print_llamasim_runtime_summary(output_paths.llamasim_output_dir)
    print("image_path:", output_paths.image_path)
    print(
        "profile_scope: steady_state_inference_only "
        "(load, checkpoint reads, and warmup are outside the trace)"
    )


if __name__ == "__main__":
    main()
