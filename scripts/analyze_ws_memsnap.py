#!/usr/bin/env python3
"""Attribute the real-run allocator peak from a memory-history snapshot.

Replays device_traces to find the peak instant, then classifies the live
blocks allocated SINCE recording started (= since the parked initial
state) against the schedule's tensor-size multisets: streamed weights vs
everything else (activations/workspaces). Blocks resident before
recording (cold weights, static floor) appear as the base offset.

  python3 scripts/analyze_ws_memsnap.py <snapshot.pickle> <neutral_schedule.json>
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter


def main() -> None:
    snap_path, sched_path = sys.argv[1], sys.argv[2]
    snap = pickle.load(open(snap_path, "rb"))
    sched = json.load(open(sched_path))

    uid2t = {t["uid"]: t for t in sched["tensors"]}
    cold_uids = {c["tensor_uid"] for c in sched["cold_starts"]}
    pf_uids = {p["tensor_uid"] for p in sched["prefetches"]}
    streamed_sizes = Counter(
        uid2t[u]["size_bytes"] for u in pf_uids - cold_uids
    )

    traces = snap["device_traces"][0]
    live: dict[int, int] = {}
    cum = 0
    peak = 0
    peak_idx = -1
    peak_live: dict[int, int] = {}
    for i, ev in enumerate(traces):
        act = ev["action"]
        if act == "alloc":
            live[ev["addr"]] = ev["size"]
            cum += ev["size"]
            if cum > peak:
                peak, peak_idx = cum, i
                peak_live = dict(live)
        elif act in ("free_completed", "free_requested"):
            if act == "free_completed" and ev["addr"] in live:
                cum -= live.pop(ev["addr"])

    # Classify the live-at-peak delta blocks.
    remaining = Counter(streamed_sizes)
    cls = Counter()
    for size in peak_live.values():
        if remaining.get(size, 0) > 0:
            remaining[size] -= 1
            cls["streamed_weights"] += size
        else:
            cls["other(activations/ws)"] += size

    n_events = len(traces)
    print(f"events={n_events} peak_delta={peak / 1e6:.0f}MB "
          f"at event {peak_idx} ({peak_idx / max(1, n_events) * 100:.0f}% "
          f"through the run)")
    for k, v in cls.most_common():
        print(f"  {k:24s}: {v / 1e6:7.0f} MB")
    # Largest unclassified blocks — what dominates 'other'.
    others = sorted(
        (s for s in peak_live.values()
         if streamed_sizes.get(s, 0) == 0),
        reverse=True,
    )[:10]
    print("  top other blocks:", [f"{s / 1e6:.0f}MB" for s in others])


if __name__ == "__main__":
    main()
