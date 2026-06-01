#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_sd3_common as sd3_common  # noqa: E402
from run_accelerate_cpu_offload import (  # noqa: E402
    ensure_accelerate_available,
    validate_run_device,
)


_SD3_OFFLOAD_COMPONENTS = (
    "transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"
)


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_warmup_runs: int,
    default_output_prefix: str | None = None,
) -> None:
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--prompt", default="A cute dog and a cat in a park")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=tuple(sd3_common.DTYPE_BY_NAME.keys()),
        default="bfloat16",
    )
    parser.add_argument("--fusion", choices=("none", "inductor"), default="none")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--offload-mode",
        choices=("none", "model", "sequential", "module", "group"),
        default="model",
    )
    parser.add_argument(
        "--group-offload-type",
        choices=("block_level", "leaf_level"),
        default="block_level",
    )
    parser.add_argument("--group-offload-num-blocks", type=int, default=1)
    parser.add_argument(
        "--group-offload-use-stream",
        dest="group_offload_use_stream",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-group-offload-use-stream",
        dest="group_offload_use_stream",
        action="store_false",
    )
    parser.add_argument("--group-offload-non-blocking", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=default_warmup_runs)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--output-prefix", default=default_output_prefix)
    parser.add_argument("--vae-tiling", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SD 3.5 medium with HF Accelerate CPU offload."
    )
    add_common_args(
        parser,
        default_model="/data/llamasim/models/sd-3.5-med",
        default_warmup_runs=4,
    )
    return parser.parse_args()


def output_stem_base(args: argparse.Namespace) -> str:
    offload_mode = args.offload_mode.replace("-", "_")
    stem = f"sd3med_{offload_mode}_cpu_offload"
    if args.fusion != "none":
        stem = f"{stem}_{args.fusion}"
    return f"{stem}_steps{args.steps}"


def resolve_image_path(args: argparse.Namespace) -> Path:
    output_prefix = getattr(args, "output_prefix", None)
    stem = output_stem_base(args) if output_prefix is None else f"{output_prefix}_steps{args.steps}"
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


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_pipeline(args: argparse.Namespace, torch_dtype: torch.dtype) -> Any:
    try:
        from diffusers import StableDiffusion3Pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError("diffusers is required for the SD3 offload runner.") from exc
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch_dtype, use_safetensors=True,
    )
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    return pipe


def maybe_compile(pipe: Any, args: argparse.Namespace) -> None:
    if args.fusion == "none":
        return
    sd3_common.maybe_compile(pipe, args)


def _apply_sd3_module_cpu_offload(pipe: Any, device: torch.device) -> None:
    from accelerate import cpu_offload
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    pipe.to("cpu")
    for name in _SD3_OFFLOAD_COMPONENTS:
        module = getattr(pipe, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        cpu_offload(module, execution_device=device, offload_buffers=True)


def _apply_sd3_group_cpu_offload(
    pipe: Any, args: argparse.Namespace, device: torch.device
) -> None:
    from diffusers.hooks import apply_group_offloading
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    group_offload_kwargs = dict(
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type=args.group_offload_type,
        num_blocks_per_group=args.group_offload_num_blocks,
        non_blocking=args.group_offload_non_blocking,
        use_stream=args.group_offload_use_stream,
    )
    for name in _SD3_OFFLOAD_COMPONENTS:
        module = getattr(pipe, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        apply_group_offloading(module=module, **group_offload_kwargs)


def apply_cpu_offload(
    pipe: Any, args: argparse.Namespace, device: torch.device
) -> None:
    if args.offload_mode == "none":
        pipe.to(device)
        return
    if args.offload_mode == "model":
        pipe.enable_model_cpu_offload(device=device)
        return
    if args.offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        return
    if args.offload_mode == "module":
        _apply_sd3_module_cpu_offload(pipe, device)
        return
    _apply_sd3_group_cpu_offload(pipe, args, device)


def free_offload_hooks(pipe: Any) -> None:
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()


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
    image_path = resolve_image_path(args)

    total_start = time.perf_counter()

    load_start = time.perf_counter()
    pipe = load_pipeline(args, torch_dtype)
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    maybe_compile(pipe, args)
    apply_cpu_offload(pipe, args, device)
    synchronize_device(device)
    load_seconds = elapsed_seconds(load_start)

    warmup_start = time.perf_counter()
    for _ in range(args.warmup_runs):
        warmup_output = sd3_common.run_pipeline(pipe, args)
        synchronize_device(device)
        del warmup_output
    warmup_seconds = elapsed_seconds(warmup_start)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    inference_start = time.perf_counter()
    output = sd3_common.run_pipeline(pipe, args)
    synchronize_device(device)
    inference_seconds = elapsed_seconds(inference_start)

    decode_start = time.perf_counter()
    with torch.no_grad():
        images = sd3_common.decode_latents_to_pil(pipe, output.images)
    synchronize_device(device)
    decode_seconds = elapsed_seconds(decode_start)

    free_offload_hooks(pipe)

    save_start = time.perf_counter()
    images[0].save(image_path)
    save_seconds = elapsed_seconds(save_start)

    e2e_seconds = elapsed_seconds(total_start)
    alloc_mb = None
    peak_mb = None
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

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
    print("warmup_runs:", args.warmup_runs)
    print("image_path:", image_path)
    print("latent_shape:", tuple(output.images.shape))
    if alloc_mb is not None and peak_mb is not None:
        print(f"GPU memory: {alloc_mb:.1f} MB allocated, {peak_mb:.1f} MB peak")
    print(f"load_seconds: {load_seconds:.6f}")
    print(f"warmup_seconds: {warmup_seconds:.6f}")
    print(f"inference_seconds: {inference_seconds:.6f}")
    print(f"decode_seconds: {decode_seconds:.6f}")
    print(f"save_seconds: {save_seconds:.6f}")
    print(f"e2e_seconds: {e2e_seconds:.6f}")


if __name__ == "__main__":
    main()
