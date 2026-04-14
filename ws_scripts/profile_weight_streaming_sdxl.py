#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

for path in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch._inductor.config as inductor_config
from profile_sdxl_turbo_common import (
    DTYPE_BY_NAME,
    decode_latents_to_pil,
    load_pipeline,
    run_pipeline,
    synchronize_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the metadata-emission weight-streaming compile pass for "
            "SDXL Turbo. This emits compiled_launch/tensor sidecars and "
            "record_function launch markers, but does not inject a schedule."
        )
    )
    parser.add_argument(
        "--model",
        default="/data/llamasim/models/sdxl-turbo",
        help="Path to the SDXL Turbo model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="a lovely cat",
        help="Prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Output height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Output width.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device, for example cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BY_NAME.keys()),
        default="float16",
        help="Torch dtype used to load the pipeline.",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup iterations before the profiled run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the trace, sidecars, and generated PNG.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path for the generated PNG. Overrides --output-dir when set.",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="Path for the chrome trace JSON. Overrides --output-dir when set.",
    )
    parser.add_argument(
        "--trace-csv",
        default=None,
        help="Path for the profiler CSV. Overrides --output-dir when set.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable torch.profiler memory tracking.",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Enable torch.profiler shape recording.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Enable stack capture in torch.profiler.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.trace is not None:
        output_dir = Path(args.trace).resolve().parent
    elif args.image is not None:
        output_dir = Path(args.image).resolve().parent
    else:
        output_dir = Path("/tmp/ws_profile")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_output_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    output_dir = resolve_output_dir(args)
    stem = "weight_streaming_profile"
    trace_path = (
        Path(args.trace)
        if args.trace is not None
        else output_dir / f"{stem}_trace.json"
    )
    csv_path = (
        Path(args.trace_csv)
        if args.trace_csv is not None
        else output_dir / f"{stem}_trace.csv"
    )
    image_path = (
        Path(args.image)
        if args.image is not None
        else output_dir / f"{stem}_output.png"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return trace_path, csv_path, image_path


def reset_ws_config() -> None:
    inductor_config.weight_streaming_plan = ""
    inductor_config.weight_streaming_nodes_csv = ""
    inductor_config.weight_streaming_tensor_csv = ""
    inductor_config.weight_streaming_output_code = ""
    inductor_config.weight_streaming_emit_ids = False
    inductor_config.weight_streaming_emit_launch_markers = False
    inductor_config.weight_streaming_tensor_crosswalk = ""


def profile_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    if device.type == "cuda":
        return [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    return [torch.profiler.ProfilerActivity.CPU]


def validate_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA execution requested for {device}, but torch.cuda.is_available() is False."
        )
    return device


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    device = validate_device(args.device)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    trace_path, csv_path, image_path = resolve_output_paths(args)
    output_dir = resolve_output_dir(args)

    total_start = time.perf_counter()

    reset_ws_config()
    inductor_config.weight_streaming_emit_ids = True
    inductor_config.weight_streaming_emit_launch_markers = True
    inductor_config.weight_streaming_output_code = str(output_dir)

    try:
        load_start = time.perf_counter()
        pipe = load_pipeline(args.model, torch_dtype, device)
        if args.disable_progress_bar:
            pipe.set_progress_bar_config(disable=True)

        compile_start = time.perf_counter()
        pipe.unet = torch.compile(
            pipe.unet,
            backend="inductor",
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )
        synchronize_device(device)
        compile_seconds = elapsed_seconds(compile_start)

        warmup_start = time.perf_counter()
        for _ in range(args.warmup_runs):
            warmup_output = run_pipeline(pipe, args)
            synchronize_device(device)
            del warmup_output
        warmup_seconds = elapsed_seconds(warmup_start)
        load_seconds = elapsed_seconds(load_start)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        profile_start = time.perf_counter()
        with torch.profiler.profile(
            activities=profile_activities(device),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
        ) as prof:
            with torch.autograd.profiler.record_function(
                "weight_streaming_profile_pass"
            ):
                output = run_pipeline(pipe, args)
                synchronize_device(device)
        inference_seconds = elapsed_seconds(profile_start)

        prof.export_chrome_trace(str(trace_path))
        if hasattr(prof, "export_csv"):
            prof.export_csv(str(csv_path))

        decode_start = time.perf_counter()
        with torch.no_grad():
            images = decode_latents_to_pil(pipe, output.images)
        synchronize_device(device)
        decode_seconds = elapsed_seconds(decode_start)

        save_start = time.perf_counter()
        images[0].save(image_path)
        save_seconds = elapsed_seconds(save_start)
        e2e_seconds = elapsed_seconds(total_start)

        alloc_mb = None
        peak_mb = None
        if device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        print("device:", device)
        print("dtype:", args.dtype)
        print("steps:", args.steps)
        print("warmup_runs:", args.warmup_runs)
        print("trace_path:", trace_path)
        print("trace_csv_path:", csv_path)
        print("image_path:", image_path)
        print(
            "launch_map:",
            output_dir / "compiled_launch_map_graph0.json",
        )
        print(
            "tensor_map:",
            output_dir / "compiled_tensor_map_graph0.json",
        )
        print(
            "index_path:",
            output_dir / "compiled_weight_streaming_index.json",
        )
        print("latent_shape:", tuple(output.images.shape))
        if alloc_mb is not None and peak_mb is not None:
            print(f"GPU memory: {alloc_mb:.1f} MB allocated, {peak_mb:.1f} MB peak")
        print(f"load_seconds: {load_seconds:.6f}")
        print(f"compile_seconds: {compile_seconds:.6f}")
        print(f"warmup_seconds: {warmup_seconds:.6f}")
        print(f"inference_seconds: {inference_seconds:.6f}")
        print(f"decode_seconds: {decode_seconds:.6f}")
        print(f"save_seconds: {save_seconds:.6f}")
        print(f"e2e_seconds: {e2e_seconds:.6f}")
    finally:
        reset_ws_config()


if __name__ == "__main__":
    main()
