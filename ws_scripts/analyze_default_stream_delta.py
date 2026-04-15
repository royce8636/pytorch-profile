"""Analyze per-stream GPU events in baseline vs WS chrome traces.

Goal: identify what is eating the +53.8 ms of default-stream GPU time in WS.

Reads /tmp/stall_breakdown_traces/{baseline,ws}_trace.json, groups GPU events
(ph == "X", kernel/memcpy categories) by their thread id (= kineto's stream id),
and prints:
  - which tid corresponds to which stream (default vs h2d), inferred from whether
    Memcpy HtoD events are present
  - total GPU active time per stream per run
  - top kernel/memcpy names by duration on each stream, with a diff table
    highlighting what is NEW on default stream in WS
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


TRACE_DIR = Path("/tmp/stall_breakdown_traces")


def load_gpu_events(trace_path: Path):
    """Return list of (tid, name, cat, ts_us, dur_us) for GPU 'X' events."""
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
        evts.append((
            e.get("tid", "?"),
            e.get("name", ""),
            cat,
            float(e.get("ts", 0)),
            dur,
        ))
    return evts


def classify_event(name: str) -> str:
    if "Memcpy HtoD" in name:
        return "memcpy_h2d"
    if "Memcpy DtoH" in name:
        return "memcpy_d2h"
    if "Memcpy" in name or "memcpy" in name:
        return "memcpy_other"
    if "Memset" in name:
        return "memset"
    if name.startswith("triton_"):
        return "triton"
    if name.startswith("void "):
        return "cuda_kernel"
    if "cutlass" in name.lower():
        return "cutlass"
    if "cudnn" in name.lower() or "cudnn" in name:
        return "cudnn"
    return "other_kernel"


def per_stream_summary(trace_path: Path, num_iters: int):
    evts = load_gpu_events(trace_path)
    # tid -> list of (name, dur)
    by_tid = defaultdict(list)
    for tid, name, cat, ts, dur in evts:
        by_tid[tid].append((name, cat, ts, dur))

    # Identify each tid: is it default stream (has triton kernels),
    # h2d stream (has Memcpy HtoD and few/no kernels), or something else?
    tid_labels = {}
    tid_active_us = {}
    for tid, items in by_tid.items():
        # Non-overlapping active sum
        ivs = sorted((ts, ts + dur) for _name, _cat, ts, dur in items)
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        active = sum(e - s for s, e in merged)
        tid_active_us[tid] = active

        has_triton = any(n.startswith("triton_") for n, *_ in items)
        has_h2d = any("Memcpy HtoD" in n for n, *_ in items)
        has_d2h = any("Memcpy DtoH" in n for n, *_ in items)
        if has_triton:
            tid_labels[tid] = "default(compute)"
        elif has_h2d and not has_triton:
            tid_labels[tid] = "h2d(copy)"
        elif has_d2h and not has_h2d:
            tid_labels[tid] = "d2h(evict)"
        else:
            tid_labels[tid] = "other"

    return by_tid, tid_labels, tid_active_us


def breakdown_by_category(items):
    """Given a list of (name, cat, ts, dur) on a single tid, bucket durations."""
    by_cat_dur = defaultdict(float)
    by_cat_count = defaultdict(int)
    by_name_dur = defaultdict(float)
    by_name_count = defaultdict(int)
    for name, _cat, _ts, dur in items:
        c = classify_event(name)
        by_cat_dur[c] += dur
        by_cat_count[c] += 1
        by_name_dur[name] += dur
        by_name_count[name] += 1
    return by_cat_dur, by_cat_count, by_name_dur, by_name_count


def format_row(label: str, us: float, count: int, num_iters: int):
    ms_iter = us / 1000.0 / num_iters
    per_iter_count = count / num_iters
    return f"    {label[:60]:<60} {ms_iter:>10.3f} ms/iter  {per_iter_count:>7.1f} calls/iter"


def dump_stream(by_tid, tid_labels, tid_active_us, num_iters, tag):
    print(f"\n=== {tag} — per-stream GPU breakdown ===")
    # Sort streams by active time
    for tid, active_us in sorted(tid_active_us.items(), key=lambda x: -x[1]):
        items = by_tid[tid]
        label = tid_labels[tid]
        print(f"\n  tid={tid} ({label})  active={active_us/1000/num_iters:.2f} ms/iter  "
              f"events={len(items)}  ({len(items)/num_iters:.0f}/iter)")
        by_cat_dur, by_cat_count, by_name_dur, by_name_count = breakdown_by_category(items)
        print(f"    {'category':<60} {'dur':>10}           {'calls':>7}")
        for c, d in sorted(by_cat_dur.items(), key=lambda x: -x[1]):
            print(format_row(c, d, by_cat_count[c], num_iters))
        # Top individual events on this tid
        print(f"    -- top event names on this tid --")
        for name, dur in sorted(by_name_dur.items(), key=lambda x: -x[1])[:10]:
            print(format_row(name, dur, by_name_count[name], num_iters))


def diff_default_stream(b_by_tid, b_labels, w_by_tid, w_labels, num_iters):
    """Find default-stream tids in baseline and ws, diff per-event durations."""
    b_tid = next((t for t, l in b_labels.items() if l == "default(compute)"), None)
    w_tid = next((t for t, l in w_labels.items() if l == "default(compute)"), None)
    if b_tid is None or w_tid is None:
        print("\n[warn] could not identify default-stream tid in one of the traces")
        return
    _, _, b_by_name_dur, b_by_name_count = breakdown_by_category(b_by_tid[b_tid])
    _, _, w_by_name_dur, w_by_name_count = breakdown_by_category(w_by_tid[w_tid])
    names = set(b_by_name_dur) | set(w_by_name_dur)
    rows = []
    for n in names:
        bd = b_by_name_dur.get(n, 0)
        wd = w_by_name_dur.get(n, 0)
        bc = b_by_name_count.get(n, 0)
        wc = w_by_name_count.get(n, 0)
        rows.append((n, bd, wd, wd - bd, bc, wc))
    rows.sort(key=lambda r: -abs(r[3]))

    print(f"\n=== Default-stream event diff (WS − baseline) ===")
    print(f"  baseline tid={b_tid}, ws tid={w_tid}")
    print(f"  {'event name':<60} {'Δdur ms/iter':>14} {'base ms/iter':>14} {'ws ms/iter':>14}  {'Δcalls/iter':>12}")
    total_delta = 0.0
    shown = 0
    for n, bd, wd, dd, bc, wc in rows:
        if abs(dd) / 1000 / num_iters < 0.05 and shown > 15:
            continue
        shown += 1
        total_delta += dd
        print(f"  {n[:60]:<60} "
              f"{dd/1000/num_iters:>+14.3f} "
              f"{bd/1000/num_iters:>14.3f} "
              f"{wd/1000/num_iters:>14.3f}  "
              f"{(wc-bc)/num_iters:>+12.1f}")
    print(f"  -- total Δ across ALL events (sanity check): {total_delta/1000/num_iters:+.2f} ms/iter")

    # Category rollup on default stream
    print(f"\n=== Default-stream category rollup ===")
    b_cat_dur = defaultdict(float); b_cat_count = defaultdict(int)
    w_cat_dur = defaultdict(float); w_cat_count = defaultdict(int)
    for n, d in b_by_name_dur.items():
        c = classify_event(n)
        b_cat_dur[c] += d
        b_cat_count[c] += b_by_name_count[n]
    for n, d in w_by_name_dur.items():
        c = classify_event(n)
        w_cat_dur[c] += d
        w_cat_count[c] += w_by_name_count[n]
    cats = set(b_cat_dur) | set(w_cat_dur)
    print(f"  {'category':<20} {'base ms/iter':>14} {'ws ms/iter':>14} {'Δms/iter':>12} "
          f"{'base/iter calls':>16} {'ws/iter calls':>16}")
    for c in sorted(cats, key=lambda c: -(w_cat_dur.get(c, 0) - b_cat_dur.get(c, 0))):
        bd, wd = b_cat_dur.get(c, 0), w_cat_dur.get(c, 0)
        bc, wc = b_cat_count.get(c, 0), w_cat_count.get(c, 0)
        print(f"  {c:<20} "
              f"{bd/1000/num_iters:>14.3f} "
              f"{wd/1000/num_iters:>14.3f} "
              f"{(wd-bd)/1000/num_iters:>+12.3f} "
              f"{bc/num_iters:>16.1f} "
              f"{wc/num_iters:>16.1f}")


def main():
    num_iters = 8
    baseline = TRACE_DIR / "baseline_trace.json"
    ws = TRACE_DIR / "ws_trace.json"
    if not baseline.exists() or not ws.exists():
        print(f"[fatal] missing traces in {TRACE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {baseline} ({baseline.stat().st_size/1e6:.0f} MB)...")
    b_by_tid, b_labels, b_active = per_stream_summary(baseline, num_iters)
    print(f"Loading {ws} ({ws.stat().st_size/1e6:.0f} MB)...")
    w_by_tid, w_labels, w_active = per_stream_summary(ws, num_iters)

    dump_stream(b_by_tid, b_labels, b_active, num_iters, "BASELINE")
    dump_stream(w_by_tid, w_labels, w_active, num_iters, "WS")

    diff_default_stream(b_by_tid, b_labels, w_by_tid, w_labels, num_iters)


if __name__ == "__main__":
    main()
