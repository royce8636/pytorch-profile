#!/usr/bin/env python3
"""Replay a streaming_e2e-produced orderax schedule in cg-sim's
DeviceAwareVanillaAsync executor, plus a no-streaming vanilla baseline.

Gives the simulator-expected makespan/peak for the SAME bundle+schedule
that the real PyTorch run executed, so real-vs-sim extension isolates
framework overhead from scheduler-inherent stall.

  python3 scripts/sim_replay_orderax.py <work_dir> --cap-gib 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CG = "/data/cg-sim"
sys.path.insert(0, CG)
sys.path = [p for p in sys.path if ".local" not in p]

TEMPLATE = (
    f"{CG}/exp_results/0521_sweep/scheduler_sweep/local/sdxl-turbo/"
    f"baseline_vanilla/sim/input.yaml"
)
PAGE_KB = 4

# Faithful-executor knobs the orderax model assumes (DESIGN.md): bounded
# in-flight pool, claim-miss waits for planned evicts.
os.environ.setdefault("DAV_PACED_PREFETCH_MB", "130")
os.environ.setdefault("DAV_PF_WAIT_ON_FULL", "1")


def make_input(work_dir: Path, schedule: Path | None, cap_gib: float,
               out_dir: Path, tag: str) -> Path:
    import yaml
    c = yaml.safe_load(open(TEMPLATE))
    c["trace"]["args"]["profile_dir"] = str(work_dir)
    c["trace"]["args"]["bundle_manifest"] = "llama_bundle/manifest.json"
    if schedule is not None:
        c["trace"]["args"]["inject_schedule_path"] = str(schedule)
        c["trace"]["args"]["xfer_h2d_streams"] = 1
    for m in c["hardware"]["memory"]:
        if m["name"] == "vram0":
            m["args"]["memory_size_KB"] = int(round(cap_gib * 1024 * 1024))
    c["logger"]["args"]["result_path"] = str(out_dir / f"sim_result_{tag}.json")
    c["logger"]["args"]["log_level"] = 0
    out = out_dir / f"input_{tag}.yaml"
    yaml.safe_dump(c, open(out, "w"))
    return out


def run_sim(input_yaml: Path) -> dict:
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        from sim.core import Simulator
        sim = Simulator(str(input_yaml))
        sim.log.on = False
        so = sim.engine.sys
        vram = so.hw.get("vram0")
        t = {"pages": 0, "count": 0}
        orig = so.transfer

        def patched(b, a=None):
            for _s, d in b:
                if getattr(d, "hw", None) is vram:
                    t["pages"] += d.num_pages
                    t["count"] += 1
            return orig(b, a)

        so.transfer = patched
        sim.run()
        res = dict(
            makespan_s=float(sim.engine.timestamp_now) / 1e6,
            h2d_mb=t["pages"] * PAGE_KB / 1024,
            peak_mb=int(vram.space.peak_num_used_pages) * PAGE_KB / 1024,
            abort=bool(getattr(sim.engine, "signal_abort", False)),
            xfers=t["count"],
        )
        sim.log.stop()
        return res
    finally:
        sys.argv = saved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("work_dir", help="streaming_e2e --work-dir (contains llama_bundle/)")
    p.add_argument("--cap-gib", type=float, default=7.0)
    p.add_argument("--baseline-vram-gib", type=float, default=23.0)
    args = p.parse_args()

    work = Path(args.work_dir)
    bundle = work / "llama_bundle"
    schedule = bundle / "neutral_schedule.json"
    assert schedule.exists(), f"missing {schedule}"
    out_dir = work / "sim_replay"
    out_dir.mkdir(exist_ok=True)

    print("→ sim vanilla baseline (no streaming) ...", flush=True)
    base = run_sim(make_input(work, None, args.baseline_vram_gib, out_dir, "vanilla"))
    print(f"  makespan={base['makespan_s']:.4f}s peak={base['peak_mb']:.0f}MB "
          f"abort={base['abort']}")

    print(f"→ sim DAV + orderax schedule @ {args.cap_gib}GiB ...", flush=True)
    ws = run_sim(make_input(work, schedule, args.cap_gib, out_dir, "ws"))
    print(f"  makespan={ws['makespan_s']:.4f}s peak={ws['peak_mb']:.0f}MB "
          f"h2d={ws['h2d_mb']:.0f}MB abort={ws['abort']}")

    ext = ws["makespan_s"] - base["makespan_s"]
    print(f"\nsim extension: {ext * 1000:.1f} ms "
          f"({ext / base['makespan_s'] * 100:.1f}% of baseline)")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"baseline": base, "ws": ws, "extension_s": ext}, f, indent=2)


if __name__ == "__main__":
    main()
