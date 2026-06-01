#!/usr/bin/env python3
"""Run a Llama model with weight streaming IO calls injected by Inductor.

Mirrors scripts/run_weight_streaming.py but for HuggingFace causal-LM
(`AutoModelForCausalLM`) instead of SDXL.

Usage:
    python3 scripts/run_weight_streaming_llama.py \\
        --plan-dir /path/to/jit_sim_prune_output \\
        --model meta-llama/Llama-3.2-3B \\
        --export-code ./ws_output

    # Dry run (no model, just check schedule loading and codegen):
    python3 scripts/run_weight_streaming_llama.py \\
        --plan-dir /path/to/jit_sim_prune_output \\
        --dry-run
"""
from __future__ import annotations

import argparse
import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

_REPO_ROOT = str(THIS_DIR.parent)
sys.path.insert(0, _REPO_ROOT)

# Filter out incompatible torchvision from ~/.local that conflicts with source torch.
sys.path = [p for p in sys.path if "torchvision" not in p and not (
    ".local" in p and Path(p).joinpath("torchvision").is_dir()
)]

import torch  # noqa: E402

import torch._inductor.config as inductor_config  # noqa: E402
from profile_llama_common import (  # noqa: E402
    DTYPE_BY_NAME,
    encode_prompt,
    greedy_generate,
    load_pipeline,
)
from profile_sdxl_turbo_common import configure_llamasim_inductor_markers  # noqa: E402
from torch._inductor.weight_streaming.plan import IOSchedule, load_io_schedule  # noqa: E402
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime  # noqa: E402

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Llama model with weight streaming IO injected by Inductor."
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
        default="meta-llama/Llama-3.2-3B",
        help="HuggingFace hub id or local path to the Llama checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="The quick brown fox",
        help="Prompt fed to the causal LM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed passed to torch.manual_seed. Set negative to skip seeding.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=15,
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
        "--enable-eos-stop",
        action="store_true",
        help="Stop early when every batch item emits EOS. Disabled by default.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
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
        "--dynamic",
        dest="dynamic",
        action="store_true",
        default=True,
        help="Compile with dynamic shapes (default).",
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
        help="Directory for the generated text file.",
    )
    parser.add_argument(
        "--text-output",
        default=None,
        help="Path for the generated text. Overrides --output-dir when set.",
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
            "generated wrapper. Purely diagnostic; adds ~5 µs / call under profiler."
        ),
    )
    parser.add_argument(
        "--force-recompile",
        action="store_true",
        help="Disable Inductor caches to force a fresh compile.",
    )
    return parser.parse_args()


def resolve_text_path(args: argparse.Namespace) -> Path:
    stem = f"weight_streaming_llama_run_tokens{args.max_new_tokens}"
    if args.text_output is not None:
        path = Path(args.text_output)
    elif args.output_dir is not None:
        path = Path(args.output_dir) / f"{stem}_output.txt"
    else:
        path = Path("/tmp") / stem / f"{stem}_output.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_schedule_paths(args: argparse.Namespace) -> tuple[Path, str, str]:
    """Find jit_sim_prune_schedule.json, runtime_nodes.csv, pytorch_runtime_tensors.csv."""
    plan_dir = Path(args.plan_dir)
    schedule_path = plan_dir / "jit_sim_prune_schedule.json"
    if not schedule_path.exists():
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


def register_model_weights(pipe: Any, rt: WeightStreamRuntime) -> int:
    """Register the model's parameters and buffers for VRAM eviction/restore.

    The runtime identifies tensors by `id()`, so the Parameter/Buffer object
    itself must be registered (not `.data`, which may produce a fresh tensor).
    """
    count = 0
    for param in pipe.model.parameters():
        rt.register_weight(param)
        count += 1
    for buf in pipe.model.buffers():
        rt.register_weight(buf)
        count += 1
    return count


def install_io_logging(rt: WeightStreamRuntime) -> None:
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


def _generate(pipe: Any, args: argparse.Namespace) -> Any:
    input_ids = encode_prompt(pipe, args)
    with torch.no_grad():
        sequences = greedy_generate(
            pipe.model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=pipe.tokenizer.eos_token_id if args.enable_eos_stop else None,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
    return sequences


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
    torch_dtype = DTYPE_BY_NAME[args.dtype]
    text_path = resolve_text_path(args)
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
        from torch._inductor.weight_streaming.codegen import ScheduleAdapter
        adapter = ScheduleAdapter(schedule)
        total_pre = sum(len(v) for v in adapter.before_kernel.values())
        total_post = sum(len(v) for v in adapter.after_kernel.values())
        print("\nAdapter mapping (dry run):")
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
    # See run_weight_streaming.py for rationale: caches enabled by default so
    # the profile-time compile is reused when structurally compatible.
    if args.force_recompile:
        inductor_config.force_disable_caches = True
    marker_output_dir = args.export_code or str(schedule_path.parent)
    configure_llamasim_inductor_markers(
        marker_output_dir,
        include_diagnostic_markers=args.profile_markers,
    )

    # ── Load model ──
    print(f"Loading model from {args.model}...")
    t0 = time.perf_counter()
    pipe = load_pipeline(args.model, torch_dtype, device)
    print(f"Model loaded in {elapsed_seconds(t0):.1f}s")

    n_registered = register_model_weights(pipe, rt)
    print(f"Registered {n_registered} GPU weight tensors")

    # ── Compile ──
    print(f"\nCompiling model (mode={args.compile_mode}, fullgraph={args.fullgraph})...")
    t0 = time.perf_counter()
    pipe.model = torch.compile(
        pipe.model,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
    )
    synchronize_device(device)
    print(f"Compiled in {elapsed_seconds(t0):.1f}s")
    load_seconds = elapsed_seconds(load_start)

    # ── Warmup ──
    warmup_start = time.perf_counter()
    for i in range(args.warmup_runs):
        print(f"Warmup run {i + 1}/{args.warmup_runs}...")
        t0 = time.perf_counter()
        _ = _generate(pipe, args)
        synchronize_device(device)
        print(f"  {elapsed_seconds(t0):.2f}s")
    warmup_seconds = elapsed_seconds(warmup_start)

    # ── Timed run ──
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print("\nTimed run...")
    synchronize_device(device)
    inference_start = time.perf_counter()
    sequences = _generate(pipe, args)
    synchronize_device(device)
    inference_seconds = elapsed_seconds(inference_start)

    # ── Decode ──
    decode_start = time.perf_counter()
    decoded = pipe.tokenizer.batch_decode(sequences, skip_special_tokens=True)
    decode_seconds = elapsed_seconds(decode_start)

    save_start = time.perf_counter()
    text_path.write_text("\n---\n".join(decoded), encoding="utf-8")
    save_seconds = elapsed_seconds(save_start)
    e2e_seconds = elapsed_seconds(total_start)

    alloc_mb = peak_mb = None
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
    print("seed:", args.seed)
    print("enable_eos_stop:", args.enable_eos_stop)
    print("max_new_tokens:", args.max_new_tokens)
    print("batch_size:", args.batch_size)
    print("warmup_runs:", args.warmup_runs)
    print("text_path:", text_path)
    print("sequence_shape:", tuple(sequences.shape))
    print("generated_text_preview:", decoded[0][:200])
    if alloc_mb is not None and peak_mb is not None:
        print(f"GPU memory: {alloc_mb:.1f} MB allocated, {peak_mb:.1f} MB peak")
    print(f"load_seconds: {load_seconds:.6f}")
    print(f"warmup_seconds: {warmup_seconds:.6f}")
    print(f"inference_seconds: {inference_seconds:.6f}")
    print(f"decode_seconds: {decode_seconds:.6f}")
    print(f"save_seconds: {save_seconds:.6f}")
    print(f"e2e_seconds: {e2e_seconds:.6f}")

    # ── Cleanup ──
    inductor_config.weight_streaming_plan = ""
    inductor_config.weight_streaming_nodes_csv = ""
    inductor_config.weight_streaming_tensor_csv = ""
    for name, value in old_marker_config.items():
        setattr(inductor_config, name, value)
    WeightStreamRuntime.reset()


if __name__ == "__main__":
    main()
