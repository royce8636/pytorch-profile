#!/usr/bin/env python3
"""Run an SDXL model with weight streaming IO calls injected by Inductor.

Given a llamasim scheduler output directory (containing
jit_sim_prune_schedule.json), this script:

1. Loads the SDXL pipeline
2. Initializes the WeightStreamRuntime with the schedule
3. Registers model weights into DRAM (so the runtime has access to them)
4. Compiles the UNet with torch.compile (triggering codegen injection)
5. Runs inference with weight streaming IO calls active
6. Optionally exports the generated wrapper code

Usage:
    python3 scripts/run_weight_streaming.py \\
        --plan-dir /data/pytorch-source/llamasim_bundle/jit_sim_prune_output \\
        --model /data/llamasim/models/sdxl-turbo \\
        --export-code ./ws_output

    # With a separate runtime_nodes.csv (if not embedded in schedule):
    python3 scripts/run_weight_streaming.py \\
        --plan-dir /data/pytorch-source/llamasim_bundle/jit_sim_prune_output \\
        --nodes-csv /data/pytorch-source/llamasim_bundle/runtime_nodes.csv \\
        --model /data/llamasim/models/sdxl-turbo

    # Dry run (no model, just check schedule loading and codegen):
    python3 scripts/run_weight_streaming.py \\
        --plan-dir /data/pytorch-source/llamasim_bundle/jit_sim_prune_output \\
        --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Ensure the source tree is importable (weight_streaming/ may not be in the
# editable-install RECORD if it was created after `pip install -e .`).
# Always insert at position 0 so source tree takes precedence.
_REPO_ROOT = str(THIS_DIR.parent)
sys.path.insert(0, _REPO_ROOT)

# Filter out incompatible torchvision from ~/.local that conflicts with source torch
sys.path = [p for p in sys.path if "torchvision" not in p and not (
    ".local" in p and Path(p).joinpath("torchvision").is_dir()
)]

import torch
from profile_sdxl_turbo_common import (
    configure_llamasim_inductor_markers,
    decode_latents_to_pil,
)
import torch._inductor.config as inductor_config
from torch._inductor.weight_streaming.plan import IOSchedule, load_io_schedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a model with weight streaming IO injected by Inductor."
    )
    parser.add_argument(
        "--plan-dir",
        required=True,
        help=(
            "Path to the llamasim scheduler output directory containing "
            "jit_sim_prune_schedule.json."
        ),
    )
    parser.add_argument(
        "--nodes-csv",
        default=None,
        help=(
            "Path to runtime_nodes.csv for GPU/CPU classification. "
            "If not provided, looks for it in the plan directory and its parent."
        ),
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
        "--height", type=int, default=128, help="Output height."
    )
    parser.add_argument(
        "--width", type=int, default=128, help="Output width."
    )
    parser.add_argument(
        "--device", default="cuda", help="Execution device."
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Torch dtype.",
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
        help="Warmup iterations before the timed run.",
    )
    parser.add_argument(
        "--export-code",
        default=None,
        help="Directory to export the generated wrapper code to.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the generated PNG. Defaults to "
            "/tmp/weight_streaming_run."
        ),
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path for the generated PNG. Overrides --output-dir when set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load and validate the schedule without running a model. "
            "Prints schedule statistics and exits."
        ),
    )
    parser.add_argument(
        "--log-io",
        action="store_true",
        help="Log every runtime IO call (ssd_prefetch, evict_vram, etc.).",
    )
    parser.add_argument(
        "--profile-markers",
        action="store_true",
        help=(
            "Emit ws_launch:K:N / ws_sync:K:N record_function markers in the "
            "generated wrapper. Purely diagnostic (helps correlate chrome trace "
            "events back to compiled launch ids). Off by default because "
            "each marker costs ~5 µs under torch.profiler."
        ),
    )
    parser.add_argument(
        "--component",
        choices=("unet", "full"),
        default="unet",
        help="Which part of the pipeline to compile. 'unet' compiles only the UNet.",
    )
    return parser.parse_args()


DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_image_path(args: argparse.Namespace) -> Path:
    stem = f"weight_streaming_run_steps{args.steps}"
    if args.image is not None:
        image_path = Path(args.image)
    elif args.output_dir is not None:
        image_path = Path(args.output_dir) / f"{stem}_output.png"
    else:
        image_path = Path("/tmp") / stem / f"{stem}_output.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    return image_path


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_schedule_paths(
    args: argparse.Namespace,
) -> tuple[Path, str, str]:
    """Find jit_sim_prune_schedule.json, runtime_nodes.csv, and pytorch_runtime_tensors.csv."""
    plan_dir = Path(args.plan_dir)
    schedule_path = plan_dir / "jit_sim_prune_schedule.json"
    if not schedule_path.exists():
        # Maybe they pointed at the parent bundle directory
        alt = plan_dir / "jit_sim_prune_output" / "jit_sim_prune_schedule.json"
        if alt.exists():
            schedule_path = alt
            plan_dir = alt.parent
        else:
            print(f"ERROR: Cannot find jit_sim_prune_schedule.json in {plan_dir}")
            sys.exit(1)

    nodes_csv = ""
    if args.nodes_csv:
        nodes_csv = args.nodes_csv
    else:
        for candidate in [
            plan_dir / "runtime_nodes.csv",
            plan_dir.parent / "runtime_nodes.csv",
        ]:
            if candidate.exists():
                nodes_csv = str(candidate)
                break

    tensor_csv = ""
    for candidate in [
        plan_dir / "pytorch_runtime_tensors.csv",
        plan_dir.parent / "pytorch_runtime_tensors.csv",
    ]:
        if candidate.exists():
            tensor_csv = str(candidate)
            break

    return schedule_path, nodes_csv, tensor_csv


def print_schedule_stats(schedule: IOSchedule) -> None:
    """Print a summary of the loaded schedule."""
    gpu_nodes = sum(
        1 for n in schedule.nodes
        if n.get("resource_kind", "") in ("gpu_stream", "gpu")
    )
    cpu_nodes = len(schedule.nodes) - gpu_nodes

    print(f"  Total nodes:       {len(schedule.nodes)}")
    print(f"    GPU nodes:       {gpu_nodes}")
    print(f"    CPU nodes:       {cpu_nodes}")
    print(f"  SSD prefetches:    {len(schedule.prefetches)}")
    print(f"  H2D prefetches:    {len(schedule.h2d_prefetches)}")
    print(f"  VRAM evictions:    {len(schedule.evict_vram)}")
    print(f"  DRAM evictions:    {len(schedule.evict_dram)}")
    print(f"  Cold starts:       {len(schedule.cold_starts)}")
    print(f"  Tensor metadata:   {len(schedule.tensors)} entries")


def load_pipeline(
    model: str, torch_dtype: torch.dtype, device: torch.device
) -> Any:
    try:
        from diffusers import StableDiffusionXLPipeline
    except ModuleNotFoundError as exc:
        if exc.name == "diffusers" or "diffusers" in str(exc):
            print(
                "ERROR: `diffusers` is not installed. Install it with:\n"
                "  pip install diffusers transformers accelerate\n"
                "Or use --dry-run to validate the schedule without a model."
            )
            sys.exit(1)
        raise

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model, torch_dtype=torch_dtype, use_safetensors=True,
    ).to(device)
    return pipe


def register_model_weights(pipe: Any, rt: WeightStreamRuntime) -> int:
    """Register GPU weight tensors in the runtime for VRAM eviction/restore.

    Calls rt.register_weight() for each UNet parameter and buffer. The runtime
    identifies tensors by id(), so no name mapping is needed — the generated
    code passes actual tensor objects.

    Returns the number of tensors registered.
    """
    count = 0
    for param in pipe.unet.parameters():
        rt.register_weight(param)
        count += 1
    for buf in pipe.unet.buffers():
        rt.register_weight(buf)
        count += 1
    return count


def install_io_logging(rt: WeightStreamRuntime) -> None:
    """Wrap runtime methods to log every IO call."""
    import functools

    for method_name in (
        "ssd_prefetch", "h2d_prefetch", "wait_h2d",
        "evict_vram", "evict_dram", "cold_start_prefetch",
    ):
        original = getattr(rt, method_name)

        @functools.wraps(original)
        def logged(arg, _name=method_name, _orig=original):
            if isinstance(arg, torch.Tensor):
                desc = f"Tensor(shape={list(arg.shape)}, storage={arg.untyped_storage().nbytes()})"
            else:
                desc = repr(arg)
            print(f"  [ws-io] {_name}({desc})")
            return _orig(arg)

        setattr(rt, method_name, logged)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    image_path = resolve_image_path(args)
    total_start = time.perf_counter()
    load_start = time.perf_counter()

    # ── Resolve and load schedule ──
    schedule_path, nodes_csv, tensor_csv = resolve_schedule_paths(args)
    print(f"Schedule:   {schedule_path}")
    if nodes_csv:
        print(f"Nodes CSV:  {nodes_csv}")
    if tensor_csv:
        print(f"Tensor CSV: {tensor_csv}")

    schedule = load_io_schedule(
        str(schedule_path), nodes_csv=nodes_csv, tensor_csv=tensor_csv,
    )
    print("\nSchedule statistics:")
    print_schedule_stats(schedule)

    if args.dry_run:
        # Validate adapter mapping
        from torch._inductor.weight_streaming.codegen import ScheduleAdapter
        adapter = ScheduleAdapter(schedule)
        total_pre = sum(len(v) for v in adapter.before_kernel.values())
        total_post = sum(len(v) for v in adapter.after_kernel.values())
        print(f"\nAdapter mapping (dry run):")
        print(f"  Pre-kernel ops:    {total_pre}")
        print(f"  Post-kernel ops:   {total_post}")
        print(f"  Startup ops:       {len(adapter.startup_ops)}")
        print("\nSchedule is valid. Use without --dry-run to compile and run.")
        return

    # ── Validate CUDA ──
    if device.type == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but torch.cuda.is_available() is False.")
        sys.exit(1)

    # ── Initialize runtime ──
    print(f"\nInitializing runtime on {device}...")
    rt = WeightStreamRuntime.initialize(schedule, device)

    if args.log_io:
        install_io_logging(rt)

    # ── Configure Inductor ──
    inductor_config.weight_streaming_plan = str(schedule_path)
    if nodes_csv:
        inductor_config.weight_streaming_nodes_csv = nodes_csv
    if tensor_csv:
        inductor_config.weight_streaming_tensor_csv = tensor_csv
    old_marker_config = {
        "weight_streaming_emit_ids": inductor_config.weight_streaming_emit_ids,
        "weight_streaming_emit_launch_markers": (
            inductor_config.weight_streaming_emit_launch_markers
        ),
        "weight_streaming_emit_sync_markers": (
            inductor_config.weight_streaming_emit_sync_markers
        ),
        "weight_streaming_output_code": inductor_config.weight_streaming_output_code,
        "force_disable_caches": inductor_config.force_disable_caches,
    }
    # Disabling caches was the historical default, but it forces a fresh
    # compile that may produce launch-ID/tensor-ID layouts different from the
    # profile's compile — breaking the schedule's structural assumptions.
    # Keep caches enabled so the profile's compile is reused when structurally
    # compatible. Override via --force-recompile if needed.
    if getattr(args, "force_recompile", False):
        inductor_config.force_disable_caches = True
    marker_output_dir = args.export_code or str(schedule_path.parent)
    # Diagnostic launch/sync markers add ~5 µs per record_function call under
    # profiler. This script is for production-like inference runs, so keep
    # them off by default. Enable via --profile-markers for chrome-trace
    # correlation during debugging.
    configure_llamasim_inductor_markers(
        marker_output_dir,
        include_diagnostic_markers=args.profile_markers,
    )

    # ── Load model ──
    print(f"Loading model from {args.model}...")
    t0 = time.perf_counter()
    pipe = load_pipeline(args.model, torch_dtype, device)
    pipe.set_progress_bar_config(disable=True)
    print(f"Model loaded in {elapsed_seconds(t0):.1f}s")

    # Register GPU weight tensors for VRAM eviction/restore
    n_registered = register_model_weights(pipe, rt)
    print(f"Registered {n_registered} GPU weight tensors")

    # ── Compile ──
    print(f"\nCompiling UNet (mode={args.compile_mode}, fullgraph={args.fullgraph})...")
    t0 = time.perf_counter()
    pipe.unet = torch.compile(
        pipe.unet,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
    )
    synchronize_device(device)
    print(f"Compiled in {elapsed_seconds(t0):.1f}s")
    load_seconds = elapsed_seconds(load_start)

    # ── Warmup ──
    warmup_start = time.perf_counter()
    for i in range(args.warmup_runs):
        print(f"Warmup run {i + 1}/{args.warmup_runs}...")
        t0 = time.perf_counter()
        _ = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            height=args.height,
            width=args.width,
            output_type="latent",
        )
        synchronize_device(device)
        print(f"  {elapsed_seconds(t0):.2f}s")
    warmup_seconds = elapsed_seconds(warmup_start)

    # ── Timed run ──
    # Reset peak memory counter so we measure steady-state, not compilation peak
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Drop any stall stats accumulated during warmup so the final report
    # reflects only the timed run.
    rt.drain_h2d_stall_stats()
    print("\nTimed run...")
    synchronize_device(device)
    inference_start = time.perf_counter()
    output = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        height=args.height,
        width=args.width,
        output_type="latent",
    )
    synchronize_device(device)
    inference_seconds = elapsed_seconds(inference_start)

    # ── Decode and save image ──
    decode_start = time.perf_counter()
    with torch.no_grad():
        images = decode_latents_to_pil(pipe, output.images)
    synchronize_device(device)
    decode_seconds = elapsed_seconds(decode_start)

    save_start = time.perf_counter()
    images[0].save(image_path)
    save_seconds = elapsed_seconds(save_start)
    e2e_seconds = elapsed_seconds(total_start)

    # ── GPU memory stats ──
    alloc_mb = None
    peak_mb = None
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # ── Export code ──
    if args.export_code:
        out_file = Path(args.export_code) / "weight_streaming_generated.py"
        if out_file.exists():
            print(f"\nGenerated code exported to: {out_file}")
        else:
            print(f"\nWARNING: Expected export at {out_file} not found.")
            print("  (Code export happens during compilation; check that "
                  "weight_streaming_output_code was set before compile.)")

    print("device:", device)
    print("dtype:", args.dtype)
    print("steps:", args.steps)
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

    stall_stats = rt.drain_h2d_stall_stats()
    if stall_stats.get("enabled"):
        print(
            f"unhidden_h2d_ms: {stall_stats['total_stall_ms']:.3f} "
            f"waits={stall_stats['wait_count']} "
            f"hits={stall_stats['hit_count']} "
            f"misses={stall_stats['miss_count']}"
        )

    # ── Cleanup ──
    inductor_config.weight_streaming_plan = ""
    inductor_config.weight_streaming_nodes_csv = ""
    inductor_config.weight_streaming_tensor_csv = ""
    for name, value in old_marker_config.items():
        setattr(inductor_config, name, value)
    WeightStreamRuntime.reset()


if __name__ == "__main__":
    main()
