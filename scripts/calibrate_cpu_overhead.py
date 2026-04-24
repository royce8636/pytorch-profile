#!/usr/bin/env python3
"""Calibrate ``cpu_per_launch_ns`` for a bundle from its profile + an
unprofiled wall-clock measurement.

Usage::

    python scripts/calibrate_cpu_overhead.py \\
        --bundle /tmp/bench_llama3b_baseline_nows/llama_bundle \\
        --unprofiled-wall-ms 71.2 \\
        --hw-in /data/.../hw_pcie4.json \\
        --hw-out /data/.../hw_llama3b.json

The formula::

    cpu_per_launch_ns = (wall_ns - Σ compute_time_ns) // n_launches

where ``compute_time_ns`` comes from the bundle's ``manifest.json`` /
ggml node records (what cg-sim uses as ``compute_time_micros``), and
``n_launches`` is the count of GPU nodes carrying a valid
``compiled_launch_id``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "/data/cg-sim")

from graph_modifiers.common import (
    load_multi_graph_sidecars, load_trace_from_bundle,
)


def calibrate(bundle_dir: Path, unprofiled_wall_ns: int) -> dict:
    trace = load_trace_from_bundle(str(bundle_dir))
    sidecars = load_multi_graph_sidecars(str(bundle_dir))

    # Sum GPU kernel compute times (nanoseconds).
    gpu_sum_ns = 0
    n_gpu_nodes = 0
    n_launches = 0
    for nid, node in trace.node_map.items():
        rk = str(node.args.get("resource_kind") or "")
        device = str(node.args.get("device_type") or "").upper()
        is_gpu = rk in ("gpu", "gpu_stream") or device in ("CUDA", "GPU")
        if not is_gpu:
            continue
        n_gpu_nodes += 1
        lid = node.args.get("compiled_launch_id")
        if lid is not None and int(lid) >= 0:
            n_launches += 1
        gpu_sum_ns += int(getattr(node, "compute_time_micros", 0) * 1_000)

    cpu_gap_ns = max(unprofiled_wall_ns - gpu_sum_ns, 0)
    per_launch_ns = cpu_gap_ns // max(n_launches, 1)

    return {
        "unprofiled_wall_ns": int(unprofiled_wall_ns),
        "gpu_sum_ns": int(gpu_sum_ns),
        "cpu_gap_ns": int(cpu_gap_ns),
        "n_gpu_nodes": int(n_gpu_nodes),
        "n_launches": int(n_launches),
        "cpu_per_launch_ns": int(per_launch_ns),
        "per_graph_launches": {
            int(gid): len(lm.get("launches", []))
            for gid, lm in sidecars.launch_maps.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--unprofiled-wall-ms", type=float, required=True)
    ap.add_argument("--hw-in", help="Optional source hw JSON to copy knobs from.")
    ap.add_argument("--hw-out", help="Write calibrated hw JSON here.")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    result = calibrate(
        Path(args.bundle),
        int(args.unprofiled_wall_ms * 1_000_000),
    )

    print("=== Calibration report ===")
    print(f"  bundle              : {args.bundle}")
    print(f"  unprofiled wall     : {result['unprofiled_wall_ns']/1e6:.2f} ms")
    print(f"  GPU kernel sum      : {result['gpu_sum_ns']/1e6:.2f} ms")
    print(f"  CPU gap             : {result['cpu_gap_ns']/1e6:.2f} ms")
    print(f"  GPU nodes           : {result['n_gpu_nodes']}")
    print(f"  Launch-annotated    : {result['n_launches']}")
    print(f"  cpu_per_launch_ns   : {result['cpu_per_launch_ns']}  "
          f"({result['cpu_per_launch_ns']/1000:.1f} µs)")
    print(f"  per-graph launches  : {result['per_graph_launches']}")

    if args.hw_out and not args.report_only:
        base: dict = {}
        if args.hw_in:
            with open(args.hw_in) as f:
                base = json.load(f)
        base["cpu_per_launch_ns"] = int(result["cpu_per_launch_ns"])
        base["_calibration"] = {
            k: int(v) if isinstance(v, (int, float)) else v
            for k, v in result.items()
            if k != "per_graph_launches"
        }
        Path(args.hw_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.hw_out, "w") as f:
            json.dump(base, f, indent=2)
        print(f"\n→ wrote {args.hw_out}")


if __name__ == "__main__":
    main()
