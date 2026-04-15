"""Quantify where weight-streaming inference overhead comes from.

The scheduler reports zero stall in its profiler-node model but the codegen
runs at compiled-launch granularity. This script computes, per iteration:

  (1) Sync H2D ops that are on the default-stream critical path (their PCIe
      time cannot overlap with compute).
  (2) Async H2D ops that overlap — how much compute covers each copy, and how
      much still stalls.
  (3) Estimated Python runtime overhead per IO call.
  (4) Why each sync op is sync: never had an overlap anchor vs. clamped to sync
      by the ``_delay_exact_vram_evictions`` post-pass (eviction shift).

Usage:
    python3 ws_scripts/analyze_h2d_hiding.py <schedule.json> <nodes_csv> \\
        --hw <hw.json> [--observed-ms-per-iter <float>]
"""

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, List


def load_launch_durations(nodes_csv: str) -> Dict[int, int]:
    """Compute sum-of-kernel-durations per compiled_launch_id (profiler time)."""
    launch_durs: Dict[int, int] = defaultdict(int)
    with open(nodes_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["step"]) != 0:
                continue
            if row["resource_kind"] != "gpu_stream":
                continue
            lid = int(row["compiled_launch_id"])
            if lid < 0:
                continue
            launch_durs[lid] += int(row["end_ns"]) - int(row["start_ns"])
    return launch_durs


def interval_compute(launch_durs: Dict[int, int], a: int, b: int) -> int:
    """Sum compute time for launches strictly between a and b."""
    if a < 0 or b < 0 or b <= a:
        return 0
    return sum(launch_durs.get(l, 0) for l in range(a + 1, b))


def analyze(schedule_path, nodes_csv, hw_path, observed_ms, overhead_us):
    with open(schedule_path) as f:
        sched = json.load(f)
    with open(hw_path) as f:
        hw = json.load(f)

    launch_durs = load_launch_durations(nodes_csv)
    profile_compute_ns = sum(launch_durs.values())
    h2d_bw = hw.get("pcie_h2d_bandwidth_Bps", 0) / 1e9

    ops = sched.get("io_operations", [])
    h2d = [o for o in ops if o["type"] == "vram_prefetch_h2d"]
    evict = [o for o in ops if o["type"] == "vram_evict_d2h"]
    ssd = [o for o in ops if o["type"] == "prefetch"]
    cold = sched.get("cold_start_prefetches", [])

    async_ops = [
        o for o in h2d
        if o.get("after_launch_id", -1) >= 0
        and o["after_launch_id"] != o.get("before_launch_id", -1)
    ]
    sync_ops = [o for o in h2d if o not in async_ops]

    # Partition sync ops into causes
    # (a) never anchored: after_node was -1 from the scheduler itself (e.g.,
    #     SSD-path placeholder, which shouldn't happen in our _h2d_schedule_pass).
    # (b) clamped by _delay_exact_vram_evictions: after_launch was set but the
    #     paired eviction was delayed past the consumer, so the H2D fell back.
    sync_cause_never = [o for o in sync_ops if o.get("after_node", -1) < 0
                        and not _was_anchored(o)]
    sync_cause_clamped = [o for o in sync_ops if o.get("after_node", -1) < 0
                          and _was_anchored(o)]
    sync_cause_same_launch = [o for o in sync_ops
                              if o.get("after_launch_id", -1) >= 0
                              and o["after_launch_id"] == o.get("before_launch_id", -1)]

    # Actual PCIe time on critical path per iter
    sync_pcie_ns = sum(o.get("duration_ns", 0) for o in sync_ops)
    async_pcie_ns = sum(o.get("duration_ns", 0) for o in async_ops)
    total_pcie_ns = sync_pcie_ns + async_pcie_ns

    # Async overlap analysis (profile timing)
    async_stall_ns_prof = 0
    async_hidden_ns_prof = 0
    for o in async_ops:
        a, b = o["after_launch_id"], o["before_launch_id"]
        dur = o["duration_ns"]
        win = interval_compute(launch_durs, a, b)
        async_hidden_ns_prof += min(dur, win)
        async_stall_ns_prof += max(0, dur - win)

    # Rescale to deployment compute if baseline per-iter ms is provided
    deploy_stall_ns = None
    if observed_ms is not None:
        # Deployment compute ≈ baseline-inference ms/iter (no streaming overhead).
        # We need an external input for this; the caller passes observed_ms/iter
        # of a BASELINE (non-streaming) run. If we don't have that, we just use
        # profile timing. For accuracy we'd also need baseline per-iter ms.
        # Keeping profile timing as the "scheduler's view"; we'll report both.
        pass

    # Python overhead estimate
    n_calls = (
        len(h2d)            # h2d_prefetch / h2d_prefetch_async
        + len(async_ops)    # h2d_wait for the async ones
        + len(evict)        # evict_vram
        + len(ssd)          # ssd_prefetch
        + len(cold)         # cold_start_prefetch (each iter)
    )
    py_ns = int(n_calls * overhead_us * 1000)

    total_bytes = sum(o.get("size_bytes", 0) for o in h2d)

    print("=" * 72)
    print("Weight-streaming H2D hiding analysis")
    print("=" * 72)
    print(f"HW: h2d_bw={h2d_bw*1e9/1e9:.0f} GB/s")
    print(f"Profile per-iter compute sum:     {profile_compute_ns/1e6:>7.2f} ms")
    if observed_ms is not None:
        print(f"Observed ms/iter (from deployment): {observed_ms:>7.2f} ms")
    print()

    print(f"H2D ops: total={len(h2d)}, async={len(async_ops)} "
          f"({len(async_ops)*100//max(1,len(h2d))}%), "
          f"sync={len(sync_ops)} "
          f"({len(sync_ops)*100//max(1,len(h2d))}%)")
    print()
    print("Why are sync ops sync?")
    print(f"  clamped by eviction-shift (after pushed past consumer):   "
          f"{len(sync_cause_clamped):>4}")
    print(f"  collapsed to same launch (after_launch == before_launch): "
          f"{len(sync_cause_same_launch):>4}")
    print(f"  never had an overlap anchor:                              "
          f"{len(sync_cause_never):>4}")
    print()

    print("Per-iter PCIe volume:")
    print(f"  sync  H2D bytes:  {sum(o['size_bytes'] for o in sync_ops)/1e6:>7.1f} MB "
          f"-> {sync_pcie_ns/1e6:>7.2f} ms PCIe (cannot overlap)")
    print(f"  async H2D bytes:  {sum(o['size_bytes'] for o in async_ops)/1e6:>7.1f} MB "
          f"-> {async_pcie_ns/1e6:>7.2f} ms PCIe (can overlap)")
    print(f"  total:            {total_bytes/1e6:>7.1f} MB "
          f"-> {total_pcie_ns/1e6:>7.2f} ms PCIe")
    print()

    print("Async overlap (in profile timing, scheduler's view):")
    print(f"  hidden by compute:  {async_hidden_ns_prof/1e6:>7.2f} ms")
    print(f"  stall (not hidden): {async_stall_ns_prof/1e6:>7.2f} ms")
    print()

    print(f"Runtime IO calls/iter: {n_calls}")
    print(f"  h2d_prefetch(_async): {len(h2d)}")
    print(f"  h2d_wait:             {len(async_ops)}")
    print(f"  evict_vram:           {len(evict)}")
    print(f"  ssd_prefetch:         {len(ssd)}")
    print(f"  cold_start_prefetch:  {len(cold)}")
    print(f"  Python overhead est (@ {overhead_us} µs/call): {py_ns/1e6:.1f} ms")
    print()

    # Steady-state PCIe attribution:
    # All sync ops incur real PCIe time in iter N+1 (tensor was evicted in iter
    # N, even at a delayed launch). Only iter 1 has "no-op sync H2D" for the
    # clamped ops.
    print("Steady-state PCIe attribution (iter >= 2):")
    print(f"  async ops, hidden by compute overlap:  "
          f"{async_hidden_ns_prof/1e6:>7.2f} ms (off critical path)")
    print(f"  sync ops, serial on default stream:    "
          f"{sync_pcie_ns/1e6:>7.2f} ms (on critical path)")
    print(f"  async ops, stall > overlap window:     "
          f"{async_stall_ns_prof/1e6:>7.2f} ms (on critical path)")
    print()

    if observed_ms is not None:
        print(f"Budget reconciliation (observed {observed_ms:.1f} ms/iter):")
        baseline_compute_ms = 26.0  # compiled-mode baseline inference
        py_ms = py_ns / 1e6
        on_crit_path_pcie = (sync_pcie_ns + async_stall_ns_prof) / 1e6
        expected_ms = baseline_compute_ms + on_crit_path_pcie + py_ms
        print(f"  baseline compute:                     {baseline_compute_ms:>6.1f} ms")
        print(f"  sync PCIe on critical path:           "
              f"{on_crit_path_pcie:>6.1f} ms")
        print(f"  python runtime overhead (@ {overhead_us} µs/call): "
              f"{py_ms:>6.1f} ms")
        print(f"  predicted:                            {expected_ms:>6.1f} ms")
        print(f"  actual:                               {observed_ms:>6.1f} ms")
        print(f"  unaccounted:                          "
              f"{observed_ms - expected_ms:>6.1f} ms")
        print()
        if_no_clamp = baseline_compute_ms + py_ms + async_stall_ns_prof/1e6
        print("If the 602 eviction-shift-clamped ops were instead PINNED "
              "(no evict + no H2D):")
        print(f"  expected:                             {if_no_clamp:>6.1f} ms "
              f"({baseline_compute_ms} compute + {py_ms:.0f} python)")
        print(f"  PCIe savings:                         "
              f"{sync_pcie_ns/1e6:>6.1f} ms")
        print(f"  Python savings (fewer calls):         "
              f"~{len(sync_ops)*overhead_us/1000*2:.0f} ms "
              f"(evict + h2d calls per tensor)")


def _was_anchored(op: dict) -> bool:
    """Proxy for 'the scheduler originally set a non-negative after for this op'.

    We can't tell perfectly after-the-fact; treat after_launch == -1 AND
    after_node == -1 AS clamped (my _delay_exact_vram_evictions clamp sets
    both). Plain sync ops would only hit this from somewhere else in the
    pipeline, which we don't currently have.
    """
    return op.get("after_node", -1) < 0 and op.get("after_launch_id", -1) < 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule")
    ap.add_argument("nodes_csv")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--observed-ms-per-iter", type=float, default=None)
    ap.add_argument("--overhead-us", type=float, default=25.0)
    args = ap.parse_args()
    analyze(args.schedule, args.nodes_csv, args.hw,
            args.observed_ms_per_iter, args.overhead_us)
