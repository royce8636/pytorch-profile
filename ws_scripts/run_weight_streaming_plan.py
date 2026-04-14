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
from run_weight_streaming import (
    DTYPE_BY_NAME,
    elapsed_seconds,
    install_io_logging,
    load_pipeline,
    print_schedule_stats,
    register_model_weights,
    resolve_image_path,
    resolve_schedule_paths,
    synchronize_device,
)
from profile_sdxl_turbo_common import decode_latents_to_pil
from torch._inductor.weight_streaming.codegen import ScheduleAdapter
from torch._inductor.weight_streaming.plan import load_io_schedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the schedule-injection weight-streaming pass for SDXL Turbo. "
            "This is the second pass: it consumes an existing schedule and "
            "optionally validates a tensor crosswalk."
        )
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
        "--tensor-crosswalk",
        default=None,
        help=(
            "Optional crosswalk JSON for compilation-hash validation and "
            "diagnostic annotation."
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
        default="cuda",
        help="Execution device.",
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
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output.",
    )
    parser.add_argument(
        "--log-io",
        action="store_true",
        help="Log every runtime IO call (ssd_prefetch, evict_vram, etc.).",
    )
    return parser.parse_args()


def reset_ws_config() -> None:
    inductor_config.weight_streaming_plan = ""
    inductor_config.weight_streaming_nodes_csv = ""
    inductor_config.weight_streaming_tensor_csv = ""
    inductor_config.weight_streaming_output_code = ""
    inductor_config.weight_streaming_emit_ids = False
    inductor_config.weight_streaming_emit_launch_markers = False
    inductor_config.weight_streaming_tensor_crosswalk = ""
    inductor_config.force_disable_caches = False


def validate_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA execution requested for {device}, but torch.cuda.is_available() is False."
        )
    return device


def main() -> None:
    args = parse_args()

    schedule_path, nodes_csv, tensor_csv = resolve_schedule_paths(args)
    print(f"Schedule:   {schedule_path}")
    if nodes_csv:
        print(f"Nodes CSV:  {nodes_csv}")
    if tensor_csv:
        print(f"Tensor CSV: {tensor_csv}")
    if args.tensor_crosswalk:
        print(f"Crosswalk:  {args.tensor_crosswalk}")

    schedule = load_io_schedule(
        str(schedule_path),
        nodes_csv=nodes_csv,
        tensor_csv=tensor_csv,
    )
    print("\nSchedule statistics:")
    print_schedule_stats(schedule)

    if args.dry_run:
        adapter = ScheduleAdapter(schedule)
        total_pre = sum(len(v) for v in adapter.before_kernel.values())
        total_post = sum(len(v) for v in adapter.after_kernel.values())
        exact_pre = sum(len(v) for v in adapter.before_launch.values())
        exact_post = sum(len(v) for v in adapter.after_launch.values())
        print("\nAdapter mapping (dry run):")
        print(f"  Pre-kernel ops:       {total_pre}")
        print(f"  Post-kernel ops:      {total_post}")
        print(f"  Exact pre-launch ops: {exact_pre}")
        print(f"  Exact post-launch ops:{exact_post}")
        print(f"  Startup ops:          {len(adapter.startup_ops)}")
        print("\nSchedule is valid. Use without --dry-run to compile and run.")
        return

    device = validate_device(args.device)
    image_path = resolve_image_path(args)
    total_start = time.perf_counter()
    load_start = time.perf_counter()

    reset_ws_config()
    try:
        print(f"\nInitializing runtime on {device}...")
        rt = WeightStreamRuntime.initialize(schedule, device)
        if args.log_io:
            install_io_logging(rt)

        # The schedule file can change between runs while the FX graph shape
        # stays identical. Disable Inductor caches for plan-driven runs so we
        # do not accidentally reuse a previously compiled wrapper with stale
        # weight-streaming injections.
        inductor_config.force_disable_caches = True
        inductor_config.weight_streaming_plan = str(schedule_path)
        inductor_config.weight_streaming_nodes_csv = nodes_csv
        inductor_config.weight_streaming_tensor_csv = tensor_csv
        inductor_config.weight_streaming_output_code = args.export_code or ""
        inductor_config.weight_streaming_emit_ids = False
        inductor_config.weight_streaming_emit_launch_markers = False
        inductor_config.weight_streaming_tensor_crosswalk = (
            args.tensor_crosswalk or ""
        )

        torch_dtype = DTYPE_BY_NAME[args.dtype]
        print(f"Loading model from {args.model}...")
        t0 = time.perf_counter()
        pipe = load_pipeline(args.model, torch_dtype, device)
        pipe.set_progress_bar_config(disable=args.disable_progress_bar)
        print(f"Model loaded in {elapsed_seconds(t0):.1f}s")

        n_registered = register_model_weights(pipe, rt)
        print(f"Registered {n_registered} GPU weight tensors")

        print(
            f"\nCompiling UNet (mode={args.compile_mode}, fullgraph={args.fullgraph})..."
        )
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

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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

        if args.export_code:
            out_file = Path(args.export_code) / "weight_streaming_generated.py"
            if out_file.exists():
                print(f"\nGenerated code exported to: {out_file}")
            else:
                print(f"\nWARNING: Expected export at {out_file} not found.")

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
    finally:
        reset_ws_config()
        WeightStreamRuntime.reset()


if __name__ == "__main__":
    main()
