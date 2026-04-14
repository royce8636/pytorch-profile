#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the weight-streaming profiling compile pass and verify that "
            "the expected trace and sidecar artifacts were produced."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/ws_profile_test",
        help="Directory where the profiling pass should write its artifacts.",
    )
    parser.add_argument(
        "--model",
        default="/data/llamasim/models/sdxl-turbo",
        help="Path to the SDXL Turbo model directory.",
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
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Torch dtype used to load the pipeline.",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup iterations before the profiled run.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(THIS_DIR / "profile_weight_streaming_sdxl.py"),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--steps",
        str(args.steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--compile-mode",
        args.compile_mode,
        "--warmup-runs",
        str(args.warmup_runs),
    ]
    if args.fullgraph:
        cmd.append("--fullgraph")
    if args.disable_progress_bar:
        cmd.append("--disable-progress-bar")

    print("Running profiling pass:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    expected = [
        output_dir / "weight_streaming_profile_trace.json",
        output_dir / "weight_streaming_profile_trace.csv",
        output_dir / "weight_streaming_profile_output.png",
        output_dir / "compiled_weight_streaming_index.json",
    ]
    launch_maps = sorted(output_dir.glob("compiled_launch_map_graph*.json"))
    tensor_maps = sorted(output_dir.glob("compiled_tensor_map_graph*.json"))

    missing = [path for path in expected if not path.exists()]
    if not launch_maps:
        missing.append(output_dir / "compiled_launch_map_graph*.json")
    if not tensor_maps:
        missing.append(output_dir / "compiled_tensor_map_graph*.json")

    if missing:
        raise SystemExit(
            "Missing profiling artifacts:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    print("\nVerified profiling artifacts:")
    for path in expected:
        print(" ", path)
    for path in launch_maps:
        print(" ", path)
    for path in tensor_maps:
        print(" ", path)


if __name__ == "__main__":
    main()
