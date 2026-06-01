#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent
PROFILE_SCRIPT = THIS_DIR / "profile_accelerate_cpu_offload.py"
DEFAULT_MODEL = "/data/llamasim/models/sdxl-turbo"
DEFAULT_OUTPUT_ROOT = "/data/pytorch-source/exp_results"
DEFAULT_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_VARIANT = "fp16"
OFFLOAD_MODES = ("model", "sequential", "module", "module-hook", "group")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SDXL-Turbo profiling with module-column output and HF Accelerate "
            "CPU offload. This mirrors the sdxl-turbo-modulecols-eager baseline "
            "shape, but records an offloaded execution."
        )
    )
    parser.add_argument(
        "--offload-mode",
        choices=(*OFFLOAD_MODES, "all"),
        default="module",
        help="Offload mode to profile, or 'all' to profile every offload mode.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the SDXL-Turbo model directory.",
    )
    parser.add_argument(
        "--vae-model",
        default=DEFAULT_VAE_MODEL,
        help=(
            "VAE checkpoint loaded into the SDXL pipeline. Use 'none' to keep the "
            "VAE bundled with --model."
        ),
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=(
            "Diffusers checkpoint variant loaded for the SDXL pipeline. Use 'none' "
            "or an empty string to omit the variant argument."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for default exp_results outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory. Only valid when --offload-mode is not all.",
    )
    parser.add_argument(
        "--fusion",
        choices=("none", "inductor"),
        default="none",
        help="Use eager execution with 'none' or torch.compile/Inductor with 'inductor'.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--prompt", default="A cute dog and a cat in a park")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=4)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--group-offload-type",
        choices=("block_level", "leaf_level"),
        default="block_level",
        help="Diffusers group-offload granularity used by --offload-mode group.",
    )
    parser.add_argument(
        "--group-offload-num-blocks",
        type=int,
        default=1,
        help="Number of blocks per group for block-level group offload.",
    )
    parser.add_argument(
        "--group-offload-use-stream",
        dest="group_offload_use_stream",
        action="store_true",
        default=True,
        help="Use a transfer stream for group offload.",
    )
    parser.add_argument(
        "--no-group-offload-use-stream",
        dest="group_offload_use_stream",
        action="store_false",
        help="Disable the transfer stream for group offload.",
    )
    parser.add_argument(
        "--group-offload-non-blocking",
        action="store_true",
        help="Use non-blocking transfers for group offload.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        default=True,
        help="Disable diffusers progress output.",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_false",
        dest="disable_progress_bar",
        help="Show diffusers progress output.",
    )
    parser.add_argument("--profile-memory", dest="profile_memory", action="store_true", default=True)
    parser.add_argument("--no-profile-memory", dest="profile_memory", action="store_false")
    parser.add_argument("--record-shapes", dest="record_shapes", action="store_true", default=True)
    parser.add_argument("--no-record-shapes", dest="record_shapes", action="store_false")
    parser.add_argument("--with-stack", dest="with_stack", action="store_true", default=True)
    parser.add_argument("--no-with-stack", dest="with_stack", action="store_false")
    parser.add_argument("--with-modules", dest="with_modules", action="store_true", default=True)
    parser.add_argument("--no-with-modules", dest="with_modules", action="store_false")
    return parser.parse_args()


def selected_offload_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.offload_mode == "all":
        return OFFLOAD_MODES
    return (args.offload_mode,)


def fusion_dir_label(fusion: str) -> str:
    if fusion == "none":
        return "eager"
    return fusion


def default_output_dir(args: argparse.Namespace, offload_mode: str) -> Path:
    mode_label = offload_mode.replace("_", "-")
    dirname = f"sdxl-turbo-modulecols-{mode_label}-{fusion_dir_label(args.fusion)}"
    return Path(args.output_root) / dirname


def output_dir_for_mode(args: argparse.Namespace, offload_mode: str) -> Path:
    if args.output_dir is not None:
        if args.offload_mode == "all":
            raise RuntimeError("--output-dir cannot be used with --offload-mode all")
        return Path(args.output_dir)
    return default_output_dir(args, offload_mode)


def _append_bool_flag(command: list[str], enabled: bool, true_flag: str, false_flag: str) -> None:
    command.append(true_flag if enabled else false_flag)


def build_profile_command(args: argparse.Namespace, offload_mode: str) -> list[str]:
    command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "sdxl-turbo",
        "--model",
        args.model,
        "--vae-model",
        args.vae_model,
        "--variant",
        args.variant,
        "--offload-mode",
        offload_mode,
        "--fusion",
        args.fusion,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--steps",
        str(args.steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--warmup-runs",
        str(args.warmup_runs),
        "--compile-mode",
        args.compile_mode,
        "--output-dir",
        str(output_dir_for_mode(args, offload_mode)),
        "--output-prefix",
        "sdxl_gpu",
        "--group-offload-type",
        args.group_offload_type,
        "--group-offload-num-blocks",
        str(args.group_offload_num_blocks),
    ]
    if args.fullgraph:
        command.append("--fullgraph")
    _append_bool_flag(
        command,
        args.group_offload_use_stream,
        "--group-offload-use-stream",
        "--no-group-offload-use-stream",
    )
    if args.group_offload_non_blocking:
        command.append("--group-offload-non-blocking")
    if args.disable_progress_bar:
        command.append("--disable-progress-bar")
    _append_bool_flag(
        command,
        args.profile_memory,
        "--profile-memory",
        "--no-profile-memory",
    )
    _append_bool_flag(
        command,
        args.record_shapes,
        "--record-shapes",
        "--no-record-shapes",
    )
    _append_bool_flag(command, args.with_stack, "--with-stack", "--no-with-stack")
    _append_bool_flag(command, args.with_modules, "--with-modules", "--no-with-modules")
    return command


def main() -> None:
    args = parse_args()
    for offload_mode in selected_offload_modes(args):
        output_dir = output_dir_for_mode(args, offload_mode)
        print(f"profiling offload_mode={offload_mode} output_dir={output_dir}", flush=True)
        subprocess.run(build_profile_command(args, offload_mode), check=True)


if __name__ == "__main__":
    main()
