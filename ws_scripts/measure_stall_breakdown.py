"""Real measurement of weight-streaming stall time breakdown.

Runs SDXL-Turbo inference twice under torch.profiler — once baseline (no weight
streaming), once with the given schedule — and decomposes the excess per-iter
time into measured categories derived from the profile:

  - compute_default:  default-stream GPU kernel time (the "useful" work)
  - h2d_stream:       GPU work on the H2D stream (async copies, typically hidden)
  - ws_rt_cpu:        CPU time in record_function'd ws_rt::* methods
  - stream_sync:      event.synchronize, stream.wait_event calls
  - memcpy:           Memcpy HtoD events (captures both sync + async H2D at GPU)
  - aten_launch_ovh:  CPU time in aten:: op dispatch (compute-kernel prep)
  - other:            residual

Usage:
    python3 ws_scripts/measure_stall_breakdown.py \\
        --model /data/llamasim/models/sdxl-turbo/ \\
        --plan-dir sdxl-turbo-profile-0414/llama_bundle/jit_sim_prune_compiled_tensor_output/ \\
        --steps 8
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch._inductor.config as inductor_config
from torch.profiler import profile, ProfilerActivity


def classify(name: str) -> str:
    """Classify a profiler event by name."""
    if name.startswith("ws_rt::"):
        return "ws_rt_cpu"
    if name.startswith("ws_launch:"):
        return "launch_marker"
    if "Memcpy HtoD" in name:
        return "memcpy_h2d"
    if "Memcpy DtoH" in name:
        return "memcpy_d2h"
    if "Memcpy" in name or "memcpy" in name:
        return "memcpy_other"
    if "Stream::wait_event" in name or name == "cudaStreamWaitEvent":
        return "stream_sync"
    if "Event::synchronize" in name or name == "cudaEventSynchronize":
        return "stream_sync"
    if "cudaStreamSynchronize" in name or "cudaDeviceSynchronize" in name:
        return "stream_sync"
    if "cudaEventRecord" in name or "cudaEventCreate" in name:
        return "stream_sync"
    if name in ("cudaMalloc", "cudaFree", "cudaMallocAsync",
                "cudaFreeAsync", "cudaMemPool") or "Alloc" in name:
        return "allocator"
    if name.startswith("triton_") or name.startswith("void ") or \
       "at::native" in name or "GPU_KERNEL" in name:
        return "compute_kernel"
    if name.startswith("aten::"):
        # Python/C++ side aten dispatcher calls
        return "aten_cpu"
    if name.startswith("extern_kernels"):
        return "compute_kernel"
    return "other"


def run_with_profiler(pipe_fn, device, num_iters: int):
    """Warm up once, then run under torch.profiler."""
    # Warmup
    _ = pipe_fn()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        with_flops=False,
        profile_memory=False,
    ) as prof:
        t0 = time.perf_counter()
        _ = pipe_fn()
        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - t0

    alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return prof, wall_s, alloc_mb, peak_mb


def aggregate_events(prof, top_per_cat: int = 5):
    """Aggregate per-event SELF time (exclusive of children) by category.

    Uses the deduplicated key_averages() so repeated kernel names (e.g. many
    invocations of triton_poi_fused_*) sum into one bucket. self_cpu_time_total
    excludes time spent in nested child ops, so summing by category doesn't
    double-count. Similarly for self_device_time_total.
    """
    events = prof.key_averages()
    cpu_by_cat = defaultdict(int)
    cuda_by_cat = defaultdict(int)
    # (category, name) -> (self_cpu_us, self_cuda_us)
    per_event = defaultdict(lambda: [0, 0])
    for evt in events:
        name = evt.key
        cat = classify(name)
        cpu_by_cat[cat] += int(evt.self_cpu_time_total)
        cuda_by_cat[cat] += int(evt.self_device_time_total)
        per_event[(cat, name)][0] += int(evt.self_cpu_time_total)
        per_event[(cat, name)][1] += int(evt.self_device_time_total)
    # Pick top events per category (by self cpu+cuda)
    top_samples = defaultdict(list)
    for (cat, name), (cpu_us, cuda_us) in per_event.items():
        top_samples[cat].append((name, cpu_us, cuda_us))
    for cat in top_samples:
        top_samples[cat].sort(key=lambda x: -(x[1] + x[2]))
        top_samples[cat] = top_samples[cat][:top_per_cat]
    return dict(cpu_by_cat), dict(cuda_by_cat), dict(top_samples)


def stream_occupancy_from_chrome_trace(trace_path, wall_s):
    """Parse chrome trace JSON to compute per-stream occupancy (non-overlapping).

    Returns: dict stream_id (e.g. 'default', 'h2d', or raw thread id) -> active ns
    """
    import json as _json
    with open(trace_path) as f:
        trace = _json.load(f)
    events = trace.get("traceEvents", [])
    # Group by tid (kineto uses tid for stream id on GPU events).
    # GPU events have "cat" of "kernel" or "gpu_memcpy" typically.
    by_tid_intervals = defaultdict(list)
    for e in events:
        if e.get("ph") != "X":  # complete events only
            continue
        cat = e.get("cat", "")
        if "gpu" not in cat.lower() and cat not in (
            "kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"
        ):
            continue
        ts = float(e.get("ts", 0))
        dur = float(e.get("dur", 0))
        if dur <= 0:
            continue
        tid = e.get("tid", "unknown")
        by_tid_intervals[tid].append((ts, ts + dur))
    # Compute non-overlapping sum per tid
    by_tid_active_us = {}
    for tid, ivs in by_tid_intervals.items():
        ivs.sort()
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        by_tid_active_us[tid] = sum(e - s for s, e in merged)
    return by_tid_active_us


def print_breakdown(tag, prof, wall_s, alloc_mb, peak_mb, num_iters,
                    trace_path=None):
    cpu_by, cuda_by, top_samples = aggregate_events(prof)
    print(f"\n=== {tag} ===")
    print(f"Wall: {wall_s*1000:.1f} ms total, {wall_s*1000/num_iters:.2f} ms/iter "
          f"(over {num_iters} denoising steps)")
    print(f"Memory: allocated={alloc_mb:.0f} MB, peak={peak_mb:.0f} MB")
    print("Per-category CPU/CUDA SELF time (exclusive of nested children):")
    print(f"  {'category':<18} {'CPU ms/iter':>14} {'CUDA ms/iter':>14}")
    all_cats = sorted(set(cpu_by) | set(cuda_by))
    for cat in all_cats:
        cpu_ms = cpu_by.get(cat, 0) / 1000 / num_iters
        cuda_ms = cuda_by.get(cat, 0) / 1000 / num_iters
        if cpu_ms < 0.02 and cuda_ms < 0.02:
            continue
        print(f"  {cat:<18} {cpu_ms:>14.3f} {cuda_ms:>14.3f}")
        # show top 3 event names in this category
        for name, cpu_us, cuda_us in top_samples.get(cat, [])[:3]:
            print(f"      - {name[:70]:<70} "
                  f"cpu={cpu_us/1000/num_iters:>6.2f}  "
                  f"cuda={cuda_us/1000/num_iters:>6.2f}")
    per_stream_active = {}
    if trace_path:
        per_stream_active = stream_occupancy_from_chrome_trace(trace_path, wall_s)
        print("\nPer-stream GPU occupancy (wall-time, non-overlapping):")
        print(f"  {'stream_tid':<15} {'active ms/iter':>16} {'active%':>8}")
        for tid, active_us in sorted(per_stream_active.items(),
                                      key=lambda x: -x[1]):
            ms = active_us / 1000 / num_iters
            pct = 100 * active_us / 1000 / (wall_s * 1000) if wall_s > 0 else 0
            print(f"  {str(tid):<15} {ms:>16.2f} {pct:>7.1f}%")
    return {
        "wall_ms": wall_s * 1000,
        "per_iter_wall_ms": wall_s * 1000 / num_iters,
        "alloc_mb": alloc_mb,
        "peak_mb": peak_mb,
        "cpu_by": cpu_by,
        "cuda_by": cuda_by,
        "per_stream_active_us": per_stream_active,
    }


def print_diff(baseline, ws, num_iters):
    """Print stall = ws - baseline per category."""
    print("\n=== Stall breakdown (ws − baseline, ms/iter) ===")
    stall = ws["per_iter_wall_ms"] - baseline["per_iter_wall_ms"]
    print(f"Total per-iter wall time:")
    print(f"  baseline: {baseline['per_iter_wall_ms']:7.2f} ms/iter")
    print(f"  ws:       {ws['per_iter_wall_ms']:7.2f} ms/iter")
    print(f"  stall:    {stall:+7.2f} ms/iter")
    print(f"\nPer-iter memory:")
    print(f"  baseline: alloc={baseline['alloc_mb']:.0f} peak={baseline['peak_mb']:.0f} MB")
    print(f"  ws:       alloc={ws['alloc_mb']:.0f} peak={ws['peak_mb']:.0f} MB")
    print(f"  Δpeak:    {ws['peak_mb'] - baseline['peak_mb']:+.0f} MB")
    print(f"\n  {'category':<18} {'ΔCPU ms/iter':>14} {'ΔCUDA ms/iter':>14}")
    all_cats = sorted(set(baseline["cpu_by"]) | set(ws["cpu_by"])
                      | set(baseline["cuda_by"]) | set(ws["cuda_by"]))
    totals_cpu = 0.0
    totals_cuda = 0.0
    for cat in all_cats:
        b_cpu = baseline["cpu_by"].get(cat, 0) / 1000 / num_iters
        w_cpu = ws["cpu_by"].get(cat, 0) / 1000 / num_iters
        b_cuda = baseline["cuda_by"].get(cat, 0) / 1000 / num_iters
        w_cuda = ws["cuda_by"].get(cat, 0) / 1000 / num_iters
        d_cpu = w_cpu - b_cpu
        d_cuda = w_cuda - b_cuda
        totals_cpu += d_cpu
        totals_cuda += d_cuda
        if abs(d_cpu) < 0.05 and abs(d_cuda) < 0.05:
            continue
        print(f"  {cat:<18} {d_cpu:>+14.2f} {d_cuda:>+14.2f}")
    print(f"  {'TOTAL Δ':<18} {totals_cpu:>+14.2f} {totals_cuda:>+14.2f}")


def build_baseline_pipe(args, device):
    """Load SDXL-Turbo baseline (no weight streaming)."""
    from profile_sdxl_turbo_common import DTYPE_BY_NAME
    from run_weight_streaming import load_pipeline
    pipe = load_pipeline(args.model, DTYPE_BY_NAME[args.dtype], device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet = torch.compile(
        pipe.unet, backend="inductor", mode="default", fullgraph=False,
    )
    # Compile warmup
    _ = pipe(prompt=args.prompt, num_inference_steps=args.steps,
             guidance_scale=0.0, output_type="latent")
    torch.cuda.synchronize(device)
    return pipe


def build_ws_pipe(args, device):
    """Load SDXL-Turbo with weight streaming enabled."""
    from profile_sdxl_turbo_common import (
        configure_llamasim_inductor_markers, DTYPE_BY_NAME,
    )
    from run_weight_streaming import (
        load_pipeline, register_model_weights, resolve_schedule_paths,
    )
    from torch._inductor.weight_streaming.plan import load_io_schedule
    from torch._inductor.weight_streaming.runtime import WeightStreamRuntime

    schedule_path, nodes_csv, tensor_csv = resolve_schedule_paths(args)
    schedule = load_io_schedule(
        str(schedule_path), nodes_csv=nodes_csv, tensor_csv=tensor_csv,
    )
    WeightStreamRuntime.reset()
    rt = WeightStreamRuntime.initialize(schedule, device)

    inductor_config.weight_streaming_plan = str(schedule_path)
    inductor_config.weight_streaming_nodes_csv = nodes_csv
    inductor_config.weight_streaming_tensor_csv = tensor_csv
    inductor_config.force_disable_caches = True
    configure_llamasim_inductor_markers(
        args.export_code or "/tmp/measure_stall_export",
        include_diagnostic_markers=False,  # avoid inflating the breakdown
    )

    pipe = load_pipeline(args.model, DTYPE_BY_NAME[args.dtype], device)
    pipe.set_progress_bar_config(disable=True)
    register_model_weights(pipe, rt)
    pipe.unet = torch.compile(
        pipe.unet, backend="inductor", mode="default", fullgraph=False,
    )
    _ = pipe(prompt=args.prompt, num_inference_steps=args.steps,
             guidance_scale=0.0, output_type="latent")
    torch.cuda.synchronize(device)
    return pipe


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/llamasim/models/sdxl-turbo")
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--prompt", default="a cute dog and a cute cat in a park")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--export-code", default=None)
    ap.add_argument("--nodes-csv", default=None)
    ap.add_argument("--tensor-csv", default=None)
    ap.add_argument("--only", choices=("baseline", "ws", "both"), default="both")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")

    # Clear inductor caches between baseline and WS runs so compilation_hash
    # mismatch doesn't occur (baseline compiles WITHOUT weight streaming config,
    # WS needs its own compile).
    import shutil
    import os
    baseline_res = None
    ws_res = None

    out_dir = Path("/tmp/stall_breakdown_traces")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("baseline", "both"):
        print("=== Building baseline pipe ===")
        pipe = build_baseline_pipe(args, device)
        def fn():
            _ = pipe(prompt=args.prompt, num_inference_steps=args.steps,
                     guidance_scale=0.0, height=args.height, width=args.width,
                     output_type="latent")
        prof, wall_s, alloc_mb, peak_mb = run_with_profiler(fn, device, args.steps)
        trace_path = out_dir / "baseline_trace.json"
        prof.export_chrome_trace(str(trace_path))
        baseline_res = print_breakdown("BASELINE", prof, wall_s, alloc_mb,
                                        peak_mb, args.steps, trace_path)
        del pipe, prof
        torch.cuda.empty_cache()
        # Clear inductor's torch.compile cache so WS pipe re-compiles.
        torch._dynamo.reset()
        for d in Path("/tmp").glob("torchinductor_*"):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    if args.only in ("ws", "both"):
        print("\n=== Building weight-streaming pipe ===")
        pipe = build_ws_pipe(args, device)
        def fn():
            _ = pipe(prompt=args.prompt, num_inference_steps=args.steps,
                     guidance_scale=0.0, height=args.height, width=args.width,
                     output_type="latent")
        prof, wall_s, alloc_mb, peak_mb = run_with_profiler(fn, device, args.steps)
        trace_path = out_dir / "ws_trace.json"
        prof.export_chrome_trace(str(trace_path))
        ws_res = print_breakdown("WEIGHT STREAMING", prof, wall_s, alloc_mb,
                                  peak_mb, args.steps, trace_path)
        del pipe, prof

    if baseline_res and ws_res:
        print_diff(baseline_res, ws_res, args.steps)


if __name__ == "__main__":
    main()
