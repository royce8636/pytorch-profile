#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_llama_common import (  # noqa: E402
    DTYPE_BY_NAME,
    decode_outputs,
    load_pipeline,
    maybe_compile,
    run_pipeline,
    triton_install_hint,
)


def parse_args(
    *,
    default_model: str,
    default_output_prefix: str,
    description: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=default_model,
        help="HuggingFace hub id or local path to the Llama checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox",
        help="Prompt fed to the causal LM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompt replicas per forward pass.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default is greedy generation).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --do-sample).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BY_NAME.keys()),
        default="bfloat16",
        help="Torch dtype used to load the model.",
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default="none",
        help="Execution mode.",
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
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=True,
        help=(
            "Compile with dynamic shapes (default). Required for meaningful "
            "inference timing because seq_length grows each generation step."
        ),
    )
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="Disable dynamic-shape compilation.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help=(
            "Warmup iterations before timing. Defaults to 1 for --fusion=inductor "
            "and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated text file.",
    )
    parser.add_argument(
        "--text-output",
        default=None,
        help="Path for the generated text. Defaults to /tmp or --output-dir.",
    )
    parser.set_defaults(output_prefix=default_output_prefix)
    return parser.parse_args()


def resolve_text_path(args: argparse.Namespace) -> Path:
    output_prefix = getattr(args, "output_prefix", "llama_gpu_run")
    stem = output_prefix if args.fusion == "none" else f"{output_prefix}_{args.fusion}"
    stem = f"{stem}_tokens{args.max_new_tokens}"
    if args.text_output is not None:
        path = Path(args.text_output)
    elif args.output_dir is not None:
        path = Path(args.output_dir) / f"{stem}_output.txt"
    else:
        path = Path("/tmp") / f"{stem}_output.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def validate_fusion_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if args.fusion == "none":
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this runtime")
    if device.type != "cuda":
        return
    from torch.utils._triton import has_triton, has_triton_package

    if not has_triton_package():
        raise RuntimeError(
            "--fusion=inductor on CUDA requires Triton. "
            f"{triton_install_hint()}"
        )
    if not has_triton():
        raise RuntimeError(
            "Triton is installed but unusable on this device. "
            f"{triton_install_hint()}"
        )


def main(
    *,
    default_model: str = "meta-llama/Llama-3.2-3B",
    default_output_prefix: str = "llama_3b_gpu_run",
    description: str = "Run Llama on GPU without profiler instrumentation.",
) -> None:
    args = parse_args(
        default_model=default_model,
        default_output_prefix=default_output_prefix,
        description=description,
    )
    device = validate_run_device(args.device)
    validate_fusion_runtime(args, device)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    warmup_runs = (
        args.warmup_runs if args.warmup_runs is not None else int(args.fusion != "none")
    )
    text_path = resolve_text_path(args)

    total_start = time.perf_counter()

    load_start = time.perf_counter()
    pipe = load_pipeline(args.model, torch_dtype, device)
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
    decoded = decode_outputs(pipe, output)
    decode_seconds = elapsed_seconds(decode_start)

    save_start = time.perf_counter()
    text_path.write_text("\n---\n".join(decoded), encoding="utf-8")
    save_seconds = elapsed_seconds(save_start)

    e2e_seconds = elapsed_seconds(total_start)

    print("device:", device)
    print("dtype:", args.dtype)
    print("fusion:", args.fusion)
    print("max_new_tokens:", args.max_new_tokens)
    print("batch_size:", args.batch_size)
    print("warmup_runs:", warmup_runs)
    print("text_path:", text_path)
    print("sequence_shape:", tuple(output.sequences.shape))
    print("generated_text_preview:", decoded[0][:200])
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
