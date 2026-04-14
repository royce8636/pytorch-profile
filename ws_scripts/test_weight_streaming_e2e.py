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
            "Run an end-to-end weight-streaming validation: first the profiling "
            "compile pass, then the schedule-driven execution pass. This verifies "
            "the current two-pass workflow with the local SDXL bundle."
        )
    )
    parser.add_argument(
        "--profile-output-dir",
        default="/tmp/ws_profile_e2e",
        help="Directory for profiling-pass artifacts.",
    )
    parser.add_argument(
        "--run-output-dir",
        default="/tmp/ws_run_e2e",
        help="Directory for schedule-run image output.",
    )
    parser.add_argument(
        "--export-code-dir",
        default="/tmp/ws_run_e2e_code",
        help="Directory for exported generated wrapper code from the run pass.",
    )
    parser.add_argument(
        "--plan-dir",
        default="/data/pytorch-source/llamasim_bundle/jit_sim_prune_output",
        help="Path to the existing scheduler output directory.",
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
        help="Number of inference steps for both passes.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Output height for both passes.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Output width for both passes.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device for both passes.",
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
        help="torch.compile mode for both passes.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup iterations for both passes.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile in both passes.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable diffusers progress output in both passes.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the profiling pass and only run the schedule execution pass.",
    )
    return parser.parse_args()


def run_checked(cmd: list[str]) -> None:
    print("Running:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    profile_output_dir = Path(args.profile_output_dir)
    run_output_dir = Path(args.run_output_dir)
    export_code_dir = Path(args.export_code_dir)

    if not args.skip_profile:
        profile_cmd = [
            sys.executable,
            str(THIS_DIR / "test_weight_streaming_profile_pass.py"),
            "--output-dir",
            str(profile_output_dir),
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
            profile_cmd.append("--fullgraph")
        if args.disable_progress_bar:
            profile_cmd.append("--disable-progress-bar")
        run_checked(profile_cmd)

    run_cmd = [
        sys.executable,
        str(THIS_DIR / "run_weight_streaming_plan.py"),
        "--plan-dir",
        args.plan_dir,
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
        "--export-code",
        str(export_code_dir),
        "--output-dir",
        str(run_output_dir),
    ]
    if args.fullgraph:
        run_cmd.append("--fullgraph")
    if args.disable_progress_bar:
        run_cmd.append("--disable-progress-bar")
    run_checked(run_cmd)

    expected_run = [
        run_output_dir / f"weight_streaming_run_steps{args.steps}_output.png",
        export_code_dir / "weight_streaming_generated.py",
    ]
    missing = [path for path in expected_run if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing run-pass artifacts:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    print("\nVerified run artifacts:")
    for path in expected_run:
        print(" ", path)

    if not args.skip_profile:
        print("\nWorkflow result:")
        print("  profiling pass: OK")
        print("  schedule-driven run pass: OK")
    else:
        print("\nWorkflow result:")
        print("  schedule-driven run pass: OK")


if __name__ == "__main__":
    main()
