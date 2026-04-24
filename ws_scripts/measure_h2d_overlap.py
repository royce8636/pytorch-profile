"""Measure actual overlap between the default stream and the h2d stream.

Definitions (all on WS trace; baseline has no h2d stream):
  - tid_default = stream tid with triton kernels
  - tid_h2d     = stream tid with only Memcpy HtoD events
  - default_active_intervals = merged intervals of REAL kernel events on
    tid_default (excluding `gpu_user_annotation` spans like
    `## Call CompiledFxGraph ##` which bracket idle gaps)
  - default_idle_intervals   = within [lo, hi], the complement of the above
  - h2d_active_intervals     = merged intervals of memcpy events on tid_h2d

Reports, per iter:
  hidden_h2d    = time h2d is active AND default is active (overlapped/hidden)
  exposed_h2d   = time h2d is active AND default is idle (GPU waits for h2d)
  cpu_only_idle = time default is idle AND h2d is idle (CPU/sync stall)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


TRACE_PATH = Path("/tmp/stall_breakdown_traces/ws_trace.json")
NUM_ITERS = 8


def load_gpu_events(trace_path):
    with open(trace_path) as f:
        trace = json.load(f)
    evts = []
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        catlow = cat.lower()
        if "gpu" not in catlow and cat not in (
            "kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"
        ):
            continue
        dur = float(e.get("dur", 0))
        if dur <= 0:
            continue
        evts.append({
            "tid": e.get("tid", "?"),
            "name": e.get("name", ""),
            "cat": cat,
            "ts": float(e.get("ts", 0)),
            "dur": dur,
        })
    return evts


def is_annotation(name: str) -> bool:
    return name.startswith("## ") or name.startswith("ws_rt::") or \
        name.startswith("ws_launch:") or name.startswith("ws_ops")


def merge(intervals):
    intervals = sorted(intervals)
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def total(intervals):
    return sum(e - s for s, e in intervals)


def intersect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def complement(intervals, lo, hi):
    out = []
    cur = lo
    for s, e in intervals:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def main():
    evts = load_gpu_events(TRACE_PATH)

    by_tid = defaultdict(list)
    for e in evts:
        by_tid[e["tid"]].append(e)

    tid_default = None
    tid_h2d = None
    for tid, items in by_tid.items():
        has_triton = any(x["name"].startswith("triton_") for x in items)
        has_h2d_copy = any("Memcpy HtoD" in x["name"] for x in items)
        if has_triton:
            tid_default = tid
        elif has_h2d_copy and not has_triton:
            tid_h2d = tid
    assert tid_default is not None and tid_h2d is not None, \
        f"could not identify streams: default={tid_default} h2d={tid_h2d}"

    default_kernel_ivs = merge([
        (x["ts"], x["ts"] + x["dur"])
        for x in by_tid[tid_default] if not is_annotation(x["name"])
    ])
    h2d_kernel_ivs = merge([
        (x["ts"], x["ts"] + x["dur"])
        for x in by_tid[tid_h2d] if not is_annotation(x["name"])
    ])

    all_starts = [iv[0] for iv in default_kernel_ivs + h2d_kernel_ivs]
    all_ends = [iv[1] for iv in default_kernel_ivs + h2d_kernel_ivs]
    lo = min(all_starts)
    hi = max(all_ends)

    default_idle_ivs = complement(default_kernel_ivs, lo, hi)

    both_active = intersect(default_kernel_ivs, h2d_kernel_ivs)
    exposed_h2d = intersect(default_idle_ivs, h2d_kernel_ivs)
    both_idle = complement(
        merge(default_kernel_ivs + h2d_kernel_ivs), lo, hi
    )

    wall_us = hi - lo
    default_active_us = total(default_kernel_ivs)
    default_idle_us = total(default_idle_ivs)
    h2d_active_us = total(h2d_kernel_ivs)
    both_active_us = total(both_active)
    exposed_h2d_us = total(exposed_h2d)
    both_idle_us = total(both_idle)

    def p(label, us):
        pct = 100 * us / wall_us
        print(f"  {label:<52} {us/1000/NUM_ITERS:>8.2f} ms/iter  "
              f"({pct:>5.1f}% of wall)")

    print(f"=== WS stream overlap (8 iters, {TRACE_PATH.name}) ===")
    print(f"  window [{lo:.0f}, {hi:.0f}] µs, "
          f"wall {wall_us/1000/NUM_ITERS:.2f} ms/iter")
    print()
    print("Per-stream totals:")
    p("default — real compute (no annotations)", default_active_us)
    p("default — idle", default_idle_us)
    p("h2d — copy active", h2d_active_us)
    print()
    print("Overlap (what pays the stall?):")
    p("h2d active & default active (HIDDEN)", both_active_us)
    p("h2d active & default idle   (EXPOSED — GPU waits for h2d)",
      exposed_h2d_us)
    p("default idle & h2d idle     (CPU stall, no h2d activity)",
      both_idle_us)
    print()
    hidden_pct = 100 * both_active_us / max(h2d_active_us, 1)
    exposed_pct = 100 * exposed_h2d_us / max(h2d_active_us, 1)
    print("H2D copy breakdown (of the 32.6 ms of h2d activity):")
    print(f"  hidden   {both_active_us/1000/NUM_ITERS:>6.2f} ms/iter  "
          f"({hidden_pct:.1f}% of h2d)")
    print(f"  exposed  {exposed_h2d_us/1000/NUM_ITERS:>6.2f} ms/iter  "
          f"({exposed_pct:.1f}% of h2d)")
    print()
    print(f"Default-stream idle breakdown "
          f"({default_idle_us/1000/NUM_ITERS:.2f} ms/iter total):")
    print(f"  waiting for h2d (h2d active, default idle)  "
          f"{exposed_h2d_us/1000/NUM_ITERS:>6.2f} ms/iter  "
          f"({100*exposed_h2d_us/max(default_idle_us,1):>5.1f}% of idle)")
    print(f"  waiting for CPU (both streams idle)         "
          f"{both_idle_us/1000/NUM_ITERS:>6.2f} ms/iter  "
          f"({100*both_idle_us/max(default_idle_us,1):>5.1f}% of idle)")


if __name__ == "__main__":
    main()
