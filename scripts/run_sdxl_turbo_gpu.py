#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_sdxl_turbo_common import (  # noqa: E402
    DTYPE_BY_NAME,
    decode_latents_to_pil,
    load_pipeline,
    maybe_compile,
    output_stem,
    run_pipeline,
    synchronize_device,
    validate_fusion_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SDXL Turbo on GPU without profiler instrumentation."
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
        "--fusion",
        choices=("none", "inductor"),
        default="none",
        help=(
            "Execution mode. 'none' keeps eager ops. 'inductor' compiles the UNet "
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
        default=None,
        help=(
            "Warmup iterations run before timing. Defaults to 1 for "
            "--fusion=inductor and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated PNG.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path for the generated PNG. Defaults to /tmp or --output-dir.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output.",
    )
    return parser.parse_args()


def resolve_image_path(args: argparse.Namespace) -> Path:
    stem = f"{output_stem('sdxl_turbo_gpu_run', args.fusion)}_steps{args.steps}"
    if args.image is not None:
        image_path = Path(args.image)
    elif args.output_dir is not None:
        image_path = Path(args.output_dir) / f"{stem}_output.png"
    else:
        image_path = Path("/tmp") / f"{stem}_output.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def cuda_unavailable_error(device: torch.device) -> str:
    imported_from = Path(torch.__file__).resolve()
    lines = [
        f"CUDA execution requested for {device}, but torch.cuda.is_available() is False.",
        f"Imported torch from: {imported_from}",
        f"torch.version.cuda = {torch.version.cuda!r}",
        f"torch.backends.cuda.is_built() = {torch.backends.cuda.is_built()}",
    ]
    if not torch.backends.cuda.is_built():
        lines.append(
            "This Python process is using a CPU-only PyTorch build, so the GPU runner cannot run."
        )
    else:
        lines.append(
            "This PyTorch build has CUDA support, but no CUDA device is available to this process."
        )
    return "\n".join(lines)


def validate_run_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(cuda_unavailable_error(device))
    return device


def main() -> None:
    args = parse_args()
    device = validate_run_device(args.device)
    validate_fusion_runtime(args, device)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    warmup_runs = (
        args.warmup_runs if args.warmup_runs is not None else int(args.fusion != "none")
    )
    image_path = resolve_image_path(args)

    total_start = time.perf_counter()

    load_start = time.perf_counter()
    pipe = load_pipeline(args.model, torch_dtype, device)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    maybe_compile(pipe, args)
    synchronize_device(device)
    load_seconds = elapsed_seconds(load_start)

    warmup_start = time.perf_counter()
    for _ in range(warmup_runs):
        warmup_output = run_pipeline(pipe, args)
        synchronize_device(device)
        del warmup_output
    warmup_seconds = elapsed_seconds(warmup_start)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    inference_start = time.perf_counter()
    output = run_pipeline(pipe, args)
    synchronize_device(device)
    inference_seconds = elapsed_seconds(inference_start)

    decode_start = time.perf_counter()
    with torch.no_grad():
        images = decode_latents_to_pil(pipe, output.images)
    synchronize_device(device)
    decode_seconds = elapsed_seconds(decode_start)

    save_start = time.perf_counter()
    images[0].save(image_path)
    save_seconds = elapsed_seconds(save_start)

    e2e_seconds = elapsed_seconds(total_start)

    print("device:", device)
    print("dtype:", args.dtype)
    print("fusion:", args.fusion)
    print("steps:", args.steps)
    print("warmup_runs:", warmup_runs)
    print("image_path:", image_path)
    print("latent_shape:", tuple(output.images.shape))
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"GPU memory: {alloc_mb:.1f} MB allocated, {peak_mb:.1f} MB peak")
    print(f"load_seconds: {load_seconds:.6f}")
    print(f"warmup_seconds: {warmup_seconds:.6f}")
    print(f"inference_seconds: {inference_seconds:.6f}")
    print(f"decode_seconds: {decode_seconds:.6f}")
    print(f"save_seconds: {save_seconds:.6f}")
    print(f"e2e_seconds: {e2e_seconds:.6f}")


if __name__ == "__main__":
    main()
