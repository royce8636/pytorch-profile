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

import profile_qwen_image_common as qwen_common  # noqa: E402
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
    parser.set_defaults(
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile SDXL-Turbo or Qwen-Image with HF Accelerate CPU offload. "
            "Profiling starts only after load and unprofiled warmup, so checkpoint "
            "I/O is excluded from the trace."
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
        default_warmup_runs=1,
    )
    add_profile_args(sdxl_parser)

    qwen_parser = subparsers.add_parser(
        "qwen-image",
        help="Profile the local Qwen-Image pipeline with CPU offload.",
    )
    offload_common.add_common_args(
        qwen_parser,
        default_model="/data/llamasim/models/qwen-image-2512",
        default_dtype="bfloat16",
        default_offload_mode="sequential",
        default_warmup_runs=1,
    )
    qwen_parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="Negative prompt passed to the pipeline.",
    )
    qwen_parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="Traditional classifier-free guidance scale used by Qwen-Image.",
    )
    qwen_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Optional guidance-distilled scale.",
    )
    qwen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the generation RNG. Set to a negative value to disable.",
    )
    qwen_parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum prompt sequence length passed to the pipeline.",
    )
    add_profile_args(qwen_parser)

    return parser.parse_args()


def output_stem(args: argparse.Namespace) -> str:
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
    generator: torch.Generator | None,
) -> Any:
    if args.pipeline == "sdxl-turbo":
        output = sdxl_common.run_pipeline(pipe, args)
    else:
        output = offload_common.run_qwen_pipeline_with_oom_guidance(
            pipe,
            args,
            generator,
        )
    return output


def decode_images(pipe: Any, args: argparse.Namespace, output: Any) -> list[Any]:
    with torch.no_grad():
        if args.pipeline == "sdxl-turbo":
            return sdxl_common.decode_latents_to_pil(pipe, output.images)
        return qwen_common.decode_latents_to_pil(
            pipe,
            output.images,
            height=args.height,
            width=args.width,
        )


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

    pipe = offload_common.load_pipeline(args, torch_dtype)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    sdxl_common.configure_llamasim_inductor_markers(
        output_paths.llamasim_output_dir
    )
    offload_common.maybe_compile(pipe, args)
    offload_common.apply_cpu_offload(pipe, args, device)
    generator = (
        qwen_common.build_generator(args, device)
        if args.pipeline == "qwen-image"
        else None
    )

    for _ in range(args.warmup_runs):
        warmup_output = run_profiled_inference(pipe, args, device, generator)
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
            with_modules=True,
            execution_trace_observer=execution_trace_observer,
        ) as prof:
            with torch.autograd.profiler.record_function(
                record_function_name(args),
                metadata_for_scope(args),
            ):
                output = run_profiled_inference(pipe, args, device, generator)
            sdxl_common.profiler_synchronize_device(device)
    finally:
        execution_trace_observer.unregister_callback()

    prof.export_chrome_trace(str(output_paths.trace_path))
    if hasattr(prof, "export_csv"):
        prof.export_csv(str(output_paths.csv_path))

    images = decode_images(pipe, args, output)
    images[0].save(output_paths.image_path)
    compiled_module: torch.nn.Module | None = None
    if args.pipeline == "sdxl-turbo":
        compiled_module = sdxl_common.underlying_unet_module(pipe.unet)
    elif hasattr(pipe, "transformer"):
        compiled_module = sdxl_common.underlying_unet_module(pipe.transformer)
    _pipe_catalog, _pipe_id_to_path = sdxl_common.build_pipeline_module_index(pipe)
    sdxl_common.write_llamasim_runtime_bundle(
        prof,
        output_paths.execution_trace_path,
        output_paths.llamasim_output_dir,
        trace_json_path=output_paths.trace_path,
        module_catalog=sdxl_common.build_module_catalog(compiled_module),
        module_id_to_path=sdxl_common.build_module_id_to_path(compiled_module),
        pipeline_module_catalog=_pipe_catalog,
        pipeline_module_id_to_path=_pipe_id_to_path,
    )
    offload_common.free_cpu_offload_hooks(pipe)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("pipeline:", args.pipeline)
    print("device:", device)
    print("dtype:", args.dtype)
    print("offload_mode:", args.offload_mode)
    print("fusion:", args.fusion)
    print("accelerate_version:", accelerate_version)
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
