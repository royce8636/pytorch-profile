#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

for path in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch._inductor.config as inductor_config
from profile_sdxl_turbo_common import configure_llamasim_inductor_markers, decode_latents_to_pil
from run_weight_streaming import (
    DTYPE_BY_NAME,
    elapsed_seconds,
    load_pipeline,
    print_schedule_stats,
    register_model_weights,
    resolve_image_path,
    resolve_schedule_paths,
    synchronize_device,
)
from torch._inductor.weight_streaming.codegen import ScheduleAdapter
from torch._inductor.weight_streaming.plan import load_io_schedule
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime


_EVENT_COLUMNS = [
    "idx",
    "phase",
    "method",
    "duration_us",
    "arg_kind",
    "tensor_name",
    "tensor_id",
    "shape",
    "dtype",
    "storage_before",
    "storage_after",
    "restore_needed",
    "restore_bytes",
    "loc_before",
    "loc_after",
    "has_file",
    "inflight_before",
    "inflight_after",
    "submitted_read",
    "pending_tensor_before",
    "pending_tensor_after",
    "pending_name_before",
    "pending_name_after",
    "queued_eviction",
    "freed_evictions",
    "error",
]


class IOTimingProfiler:
    def __init__(self, rt: WeightStreamRuntime, max_events: int) -> None:
        self.rt = rt
        self.max_events = max_events
        self.phase = "setup"
        self.events: list[dict[str, Any]] = []
        self.stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            self._new_stats
        )
        self._originals: dict[str, Any] = {}
        self._event_idx = 0

    @staticmethod
    def _new_stats() -> dict[str, Any]:
        return {
            "calls": 0,
            "total_us": 0.0,
            "max_us": 0.0,
            "restore_calls": 0,
            "restore_bytes": 0,
            "submitted_reads": 0,
            "no_file_calls": 0,
            "queued_evictions": 0,
            "freed_evictions": 0,
            "errors": 0,
        }

    def install(self) -> None:
        for name in (
            "ssd_prefetch",
            "h2d_prefetch",
            "wait_h2d",
            "evict_vram",
            "evict_dram",
            "cold_start_prefetch",
        ):
            self._wrap_unary(name)
        for name in ("flush_ready_vram_evictions", "flush_all_vram_evictions"):
            self._wrap_nullary(name)

    def restore(self) -> None:
        for name, original in self._originals.items():
            setattr(self.rt, name, original)
        self._originals.clear()

    def _pending_counts(self) -> tuple[int, int]:
        return (
            len(getattr(self.rt, "_pending_vram_evictions", {})),
            len(getattr(self.rt, "_pending_vram_name_evictions", {})),
        )

    @staticmethod
    def _shape(tensor: torch.Tensor) -> str:
        return "x".join(str(x) for x in tensor.shape)

    def _base_fields(self, method: str, arg: Any | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "phase": self.phase,
            "method": method,
            "arg_kind": "none",
            "tensor_name": "",
            "tensor_id": "",
            "shape": "",
            "dtype": "",
            "storage_before": "",
            "storage_after": "",
            "restore_needed": 0,
            "restore_bytes": 0,
            "loc_before": "",
            "loc_after": "",
            "has_file": "",
            "inflight_before": "",
            "inflight_after": "",
            "submitted_read": 0,
            "pending_tensor_before": "",
            "pending_tensor_after": "",
            "pending_name_before": "",
            "pending_name_after": "",
            "queued_eviction": 0,
            "freed_evictions": 0,
            "error": "",
        }
        pending_tensor, pending_name = self._pending_counts()
        fields["pending_tensor_before"] = pending_tensor
        fields["pending_name_before"] = pending_name

        if isinstance(arg, torch.Tensor):
            fields.update(
                {
                    "arg_kind": "tensor",
                    "tensor_id": id(arg),
                    "shape": self._shape(arg),
                    "dtype": str(arg.dtype),
                    "storage_before": arg.untyped_storage().nbytes(),
                }
            )
        elif isinstance(arg, str):
            entry = self.rt._schedule.tensors.get(arg)
            fields.update(
                {
                    "arg_kind": "name",
                    "tensor_name": arg,
                    "loc_before": self.rt._location.get(arg, ""),
                    "has_file": int(bool(entry and entry.file)),
                    "inflight_before": int(arg in self.rt._inflight_reads),
                }
            )
        return fields

    def _finish_fields(self, fields: dict[str, Any], arg: Any | None = None) -> None:
        pending_tensor, pending_name = self._pending_counts()
        fields["pending_tensor_after"] = pending_tensor
        fields["pending_name_after"] = pending_name
        fields["freed_evictions"] = max(
            0,
            int(fields["pending_tensor_before"])
            + int(fields["pending_name_before"])
            - pending_tensor
            - pending_name,
        )

        if isinstance(arg, torch.Tensor):
            storage_after = arg.untyped_storage().nbytes()
            fields["storage_after"] = storage_after
            restore_needed = int(
                fields["method"] == "h2d_prefetch"
                and int(fields["storage_before"]) == 0
                and storage_after > 0
            )
            fields["restore_needed"] = restore_needed
            fields["restore_bytes"] = storage_after if restore_needed else 0
            fields["queued_eviction"] = int(
                fields["method"] == "evict_vram"
                and pending_tensor > int(fields["pending_tensor_before"])
            )
        elif isinstance(arg, str):
            fields["loc_after"] = self.rt._location.get(arg, "")
            fields["inflight_after"] = int(arg in self.rt._inflight_reads)
            fields["submitted_read"] = int(
                fields["method"] == "ssd_prefetch"
                and int(fields["inflight_before"] or 0) == 0
                and int(fields["inflight_after"] or 0) == 1
            )
            fields["queued_eviction"] = int(
                fields["method"] == "evict_vram"
                and pending_name > int(fields["pending_name_before"])
            )

    def _record(self, fields: dict[str, Any], duration_ns: int) -> None:
        duration_us = duration_ns / 1000.0
        fields["idx"] = self._event_idx
        fields["duration_us"] = f"{duration_us:.3f}"
        self._event_idx += 1

        key = (fields["phase"], fields["method"])
        stats = self.stats[key]
        stats["calls"] += 1
        stats["total_us"] += duration_us
        stats["max_us"] = max(stats["max_us"], duration_us)
        stats["restore_calls"] += int(fields.get("restore_needed") or 0)
        stats["restore_bytes"] += int(fields.get("restore_bytes") or 0)
        stats["submitted_reads"] += int(fields.get("submitted_read") or 0)
        if fields["method"] == "ssd_prefetch" and fields.get("has_file") == 0:
            stats["no_file_calls"] += 1
        stats["queued_evictions"] += int(fields.get("queued_eviction") or 0)
        stats["freed_evictions"] += int(fields.get("freed_evictions") or 0)
        stats["errors"] += int(bool(fields.get("error")))

        if len(self.events) < self.max_events:
            self.events.append(fields.copy())

    def _wrap_unary(self, method_name: str) -> None:
        original = getattr(self.rt, method_name)
        self._originals[method_name] = original

        def wrapped(arg: Any) -> Any:
            fields = self._base_fields(method_name, arg)
            start_ns = time.perf_counter_ns()
            try:
                result = original(arg)
                self._finish_fields(fields, arg)
                return result
            except Exception as exc:
                fields["error"] = f"{type(exc).__name__}: {exc}"
                self._finish_fields(fields, arg)
                raise
            finally:
                self._record(fields, time.perf_counter_ns() - start_ns)

        setattr(self.rt, method_name, wrapped)

    def _wrap_nullary(self, method_name: str) -> None:
        original = getattr(self.rt, method_name)
        self._originals[method_name] = original

        def wrapped() -> Any:
            fields = self._base_fields(method_name)
            start_ns = time.perf_counter_ns()
            try:
                result = original()
                self._finish_fields(fields)
                return result
            except Exception as exc:
                fields["error"] = f"{type(exc).__name__}: {exc}"
                self._finish_fields(fields)
                raise
            finally:
                self._record(fields, time.perf_counter_ns() - start_ns)

        setattr(self.rt, method_name, wrapped)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_EVENT_COLUMNS)
            writer.writeheader()
            for event in self.events:
                writer.writerow({k: event.get(k, "") for k in _EVENT_COLUMNS})

    def summary_dict(self) -> dict[str, Any]:
        rows = []
        for (phase, method), stats in sorted(self.stats.items()):
            calls = stats["calls"]
            rows.append(
                {
                    "phase": phase,
                    "method": method,
                    "calls": calls,
                    "total_ms": stats["total_us"] / 1000.0,
                    "avg_us": stats["total_us"] / calls if calls else 0.0,
                    "max_us": stats["max_us"],
                    "restore_calls": stats["restore_calls"],
                    "restore_mb": stats["restore_bytes"] / (1024 * 1024),
                    "submitted_reads": stats["submitted_reads"],
                    "no_file_calls": stats["no_file_calls"],
                    "queued_evictions": stats["queued_evictions"],
                    "freed_evictions": stats["freed_evictions"],
                    "errors": stats["errors"],
                }
            )
        return {"events_recorded": len(self.events), "stats": rows}

    def write_summary(self, path: Path, extra: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary_dict()
        data.update(extra)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        print("\nIO timing summary:")
        for row in self.summary_dict()["stats"]:
            print(
                f"  {row['phase']:>8} {row['method']:<28} "
                f"calls={row['calls']:>6} total={row['total_ms']:>9.3f} ms "
                f"avg={row['avg_us']:>8.3f} us max={row['max_us']:>9.3f} us "
                f"restore={row['restore_mb']:>8.1f} MB reads={row['submitted_reads']} "
                f"no_file={row['no_file_calls']} freed={row['freed_evictions']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SDXL weight streaming and profile runtime IO method timings."
    )
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--nodes-csv", default=None)
    parser.add_argument("--model", default="/data/llamasim/models/sdxl-turbo")
    parser.add_argument("--prompt", default="a lovely cat")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--export-code", default=None)
    parser.add_argument(
        "--output-dir",
        default="/tmp/weight_streaming_io_profile",
        help="Directory for CSV, JSON summary, and output image.",
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument(
        "--max-events",
        type=int,
        default=200000,
        help="Maximum per-call events to retain in the CSV.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip VAE decode and image save after the timed latent run.",
    )
    return parser.parse_args()


def reset_ws_config() -> None:
    inductor_config.weight_streaming_plan = ""
    inductor_config.weight_streaming_nodes_csv = ""
    inductor_config.weight_streaming_tensor_csv = ""
    inductor_config.weight_streaming_output_code = ""
    inductor_config.weight_streaming_emit_ids = False
    inductor_config.weight_streaming_emit_launch_markers = False
    inductor_config.weight_streaming_emit_sync_markers = False


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested for {device}, but CUDA is unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "weight_streaming_io_events.csv"
    summary_path = output_dir / "weight_streaming_io_summary.json"
    image_path = resolve_image_path(args)

    schedule_path, nodes_csv, tensor_csv = resolve_schedule_paths(args)
    print(f"Schedule:   {schedule_path}")
    if nodes_csv:
        print(f"Nodes CSV:  {nodes_csv}")
    if tensor_csv:
        print(f"Tensor CSV: {tensor_csv}")

    schedule = load_io_schedule(str(schedule_path), nodes_csv=nodes_csv, tensor_csv=tensor_csv)
    print("\nSchedule statistics:")
    print_schedule_stats(schedule)
    tensors_with_files = sum(1 for entry in schedule.tensors.values() if entry.file)
    print(f"  Tensor files:      {tensors_with_files} entries with file metadata")

    adapter = ScheduleAdapter(schedule)
    print("\nAdapter mapping:")
    print(f"  Exact pre-launch ops:  {sum(len(v) for v in adapter.before_launch.values())}")
    print(f"  Exact post-launch ops: {sum(len(v) for v in adapter.after_launch.values())}")
    print(f"  Startup ops:           {len(adapter.startup_ops)}")

    torch_dtype = DTYPE_BY_NAME[args.dtype]
    total_start = time.perf_counter()
    load_start = time.perf_counter()

    old_config = {
        "weight_streaming_plan": inductor_config.weight_streaming_plan,
        "weight_streaming_nodes_csv": inductor_config.weight_streaming_nodes_csv,
        "weight_streaming_tensor_csv": inductor_config.weight_streaming_tensor_csv,
        "weight_streaming_output_code": inductor_config.weight_streaming_output_code,
        "weight_streaming_emit_ids": inductor_config.weight_streaming_emit_ids,
        "weight_streaming_emit_launch_markers": inductor_config.weight_streaming_emit_launch_markers,
        "weight_streaming_emit_sync_markers": inductor_config.weight_streaming_emit_sync_markers,
        "force_disable_caches": inductor_config.force_disable_caches,
    }

    profiler: IOTimingProfiler | None = None
    try:
        rt = WeightStreamRuntime.initialize(schedule, device)
        profiler = IOTimingProfiler(rt, max_events=args.max_events)
        profiler.install()

        inductor_config.weight_streaming_plan = str(schedule_path)
        inductor_config.weight_streaming_nodes_csv = nodes_csv
        inductor_config.weight_streaming_tensor_csv = tensor_csv
        inductor_config.force_disable_caches = True
        configure_llamasim_inductor_markers(args.export_code or str(output_dir))

        print(f"\nLoading model from {args.model}...")
        t0 = time.perf_counter()
        pipe = load_pipeline(args.model, torch_dtype, device)
        pipe.set_progress_bar_config(disable=args.disable_progress_bar)
        print(f"Model loaded in {elapsed_seconds(t0):.1f}s")

        n_registered = register_model_weights(pipe, rt)
        print(f"Registered {n_registered} GPU weight tensors")

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

        warmup_start = time.perf_counter()
        for i in range(args.warmup_runs):
            profiler.phase = "warmup"
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
        profiler.phase = "timed"
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

        decode_seconds = 0.0
        save_seconds = 0.0
        if not args.skip_decode:
            decode_start = time.perf_counter()
            with torch.no_grad():
                images = decode_latents_to_pil(pipe, output.images)
            synchronize_device(device)
            decode_seconds = elapsed_seconds(decode_start)

            save_start = time.perf_counter()
            images[0].save(image_path)
            save_seconds = elapsed_seconds(save_start)

        alloc_mb = None
        peak_mb = None
        if device.type == "cuda":
            alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        e2e_seconds = elapsed_seconds(total_start)
        extra = {
            "schedule_path": str(schedule_path),
            "nodes_csv": nodes_csv,
            "tensor_csv": tensor_csv,
            "tensors_with_file_metadata": tensors_with_files,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "warmup_runs": args.warmup_runs,
            "load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds,
            "inference_seconds": inference_seconds,
            "decode_seconds": decode_seconds,
            "save_seconds": save_seconds,
            "e2e_seconds": e2e_seconds,
            "gpu_allocated_mb": alloc_mb,
            "gpu_peak_mb": peak_mb,
            "latent_shape": tuple(output.images.shape),
        }
        profiler.write_csv(csv_path)
        profiler.write_summary(summary_path, extra)
        profiler.print_summary()

        if args.export_code:
            out_file = Path(args.export_code) / "weight_streaming_generated.py"
            if out_file.exists():
                print(f"\nGenerated code exported to: {out_file}")
            else:
                print(f"\nWARNING: Expected export at {out_file} not found.")

        print("\nOutputs:")
        print(f"  io_csv:      {csv_path}")
        print(f"  io_summary:  {summary_path}")
        if not args.skip_decode:
            print(f"  image_path:  {image_path}")
        print("device:", device)
        print("dtype:", args.dtype)
        print("steps:", args.steps)
        print("warmup_runs:", args.warmup_runs)
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
        if profiler is not None:
            profiler.restore()
        WeightStreamRuntime.reset()
        reset_ws_config()
        for name, value in old_config.items():
            setattr(inductor_config, name, value)


if __name__ == "__main__":
    main()
