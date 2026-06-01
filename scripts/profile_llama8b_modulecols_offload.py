#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent
PROFILE_SCRIPT = THIS_DIR / "profile_llama_accelerate_cpu_offload.py"
DEFAULT_MODEL = "/data/llamasim/models/Llama-3.1-8b"
DEFAULT_OUTPUT_ROOT = "/data/pytorch-source/exp_results/llama"
OFFLOAD_MODES = ("model", "sequential", "module", "module-hook", "group")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Llama 8B profiling with module-column output and direct CPU "
            "offload modes. Output directories mirror the SDXL modulecols "
            "offload layout under exp_results/llama."
        )
    )
    parser.add_argument(
        "--offload-mode",
        choices=(*OFFLOAD_MODES, "all"),
        default="module",
        help="Offload mode to profile, or 'all' to profile every Llama offload mode.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory. Only valid when --offload-mode is not all.",
    )
    parser.add_argument("--fusion", choices=("none", "inductor"), default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enable-eos-stop", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", default=True)
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.add_argument(
        "--group-offload-type",
        choices=("block_level", "leaf_level"),
        default="block_level",
    )
    parser.add_argument("--group-offload-num-blocks", type=int, default=1)
    parser.add_argument("--group-offload-use-stream", action="store_true")
    parser.add_argument("--group-offload-non-blocking", action="store_true")
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
    dirname = f"llama8b-modulecols-{mode_label}-{fusion_dir_label(args.fusion)}"
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
        "--model",
        args.model,
        "--offload-mode",
        offload_mode,
        "--fusion",
        args.fusion,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--temperature",
        str(args.temperature),
        "--warmup-runs",
        str(args.warmup_runs),
        "--compile-mode",
        args.compile_mode,
        "--output-dir",
        str(output_dir_for_mode(args, offload_mode)),
        "--output-prefix",
        "llama_gpu",
        "--group-offload-type",
        args.group_offload_type,
        "--group-offload-num-blocks",
        str(args.group_offload_num_blocks),
    ]
    if args.do_sample:
        command.append("--do-sample")
    if args.enable_eos_stop:
        command.append("--enable-eos-stop")
    if args.fullgraph:
        command.append("--fullgraph")
    if not args.dynamic:
        command.append("--no-dynamic")
    if args.group_offload_use_stream:
        command.append("--group-offload-use-stream")
    if args.group_offload_non_blocking:
        command.append("--group-offload-non-blocking")
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
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_profile_command(args, offload_mode), check=True)


if __name__ == "__main__":
    main()
