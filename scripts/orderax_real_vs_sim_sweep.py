#!/usr/bin/env python3
"""Reproduce the orderax sim sweep's SDXL-Turbo cap points on the REAL
PyTorch executor and pair each with a sim replay on the identical
schedule. Compares against the published sim sweep curve.

For each (cap, margin): run streaming_e2e (real), then sim_replay on the
produced bundle. Appends one row per cap to a results JSON.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PY = "/home/royce/anaconda3/envs/ptvenv/bin/python"
REPO = "/data/pytorch-source"
MODEL = "/data/llamasim/models/sdxl-turbo"
HW = f"{REPO}/hw_pcie4.json"
OUTROOT = Path(f"{REPO}/exp_results/0618_real_vs_sim")
RESULTS = OUTROOT / "results.json"

# Published sim sweep points (from orderax_full_sweep.json) + per-cap margins.
# cap_gib -> (margin, sim_makespan_s, sim_peak_mb)
SWEEP = {
    8.0: (0.02, 0.1669, 7535),
    6.0: (0.02, 0.1966, 5779),
    4.0: (0.02, 0.3767, 3913),
    3.0: (0.02, 0.5301, 2962),
    2.0: (0.05, 0.9153, 1937),
    1.6: (0.08, 0.9776, 1516),
}
SIM_BASELINE_S = 0.1669  # sdxl-turbo sim vanilla makespan


def run(cmd, env=None, log=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    with open(log, "w") if log else sys.stdout as f:
        p = subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def parse_real(log_path):
    txt = Path(log_path).read_text()
    out = {}
    for key, pat in (("peak_mb", r"peak_mb\s*:\s*([\d.]+)"),
                     ("inference_s", r"inference_s\s*:\s*([\d.]+)"),
                     ("model_peak_mb", r"peak:\s*([\d.]+)\s*MB"),
                     ("model_L_ms", r"lateness:\s*([\d.]+)\s*ms")):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"miss_synced=(\d+)", txt)
    if m:
        out["miss_synced"] = int(m.group(1))
    return out


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {r["cap"] for r in rows}
    caps = [float(x) for x in sys.argv[1:]] or sorted(SWEEP, reverse=True)

    for cap in caps:
        if cap in done:
            print(f"skip {cap} (done)", flush=True)
            continue
        margin, sim_mk, sim_peak = SWEEP[cap]
        work = OUTROOT / f"ws{cap}g"
        real_log = OUTROOT / f"real_{cap}g.log"
        print(f"\n##### SDXL-Turbo @ {cap} GiB (margin {margin}) #####",
              flush=True)

        # Real run. Match the published sweep schedule methodology
        # (cold-release credit ON -> MILP_COLD_NO_RELEASE=0).
        rc = run(
            [PY, f"{REPO}/scripts/streaming_e2e.py",
             "--variant", "cgsim_milp_orderax", "--model", MODEL, "--hw", HW,
             "--steps", "4", "--height", "512", "--width", "512",
             "--cap-gib", str(cap), "--cap-margin", str(margin),
             "--work-dir", str(work)],
            env={"TORCH_WS_LOG_DISPATCH": "1", "MILP_COLD_NO_RELEASE": "0"},
            log=real_log,
        )
        real = parse_real(real_log)
        real["exit"] = rc

        # Sim replay on the identical produced schedule.
        sim_rep = {}
        sched = work / "llama_bundle" / "neutral_schedule.json"
        if sched.exists():
            sim_log = OUTROOT / f"simreplay_{cap}g.log"
            run([PY, f"{REPO}/scripts/sim_replay_orderax.py", str(work),
                 "--cap-gib", str(cap)],
                env={"PYTHONPATH": "/data/cg-sim"}, log=sim_log)
            sj = work / "sim_replay" / "summary.json"
            if sj.exists():
                d = json.loads(sj.read_text())
                sim_rep = {"sim_mk_s": d["ws"]["makespan_s"],
                           "sim_peak_mb": d["ws"]["peak_mb"],
                           "sim_base_s": d["baseline"]["makespan_s"]}

        rows.append({
            "model": "sdxl-turbo", "cap": cap, "margin": margin,
            "published_sim_mk_s": sim_mk, "published_sim_peak_mb": sim_peak,
            "real": real, "sim_replay": sim_rep,
        })
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"RESULT {cap}G: real_mk={real.get('inference_s')} "
              f"real_peak={real.get('peak_mb')} "
              f"sim_replay_mk={sim_rep.get('sim_mk_s')} "
              f"published_sim_mk={sim_mk}", flush=True)

    # Final table.
    print("\n=== SDXL-Turbo: real executor vs simulator (orderax) ===")
    print(f"{'cap':>5} {'real_mk':>9} {'simrep_mk':>10} {'pub_sim_mk':>11} "
          f"{'real_pk':>8} {'simrep_pk':>10} {'pub_pk':>7} {'miss':>5}")
    for r in sorted(rows, key=lambda x: -x["cap"]):
        re_ = r["real"]
        sr = r["sim_replay"]
        print(f"{r['cap']:>5} {re_.get('inference_s', 0):>9.4f} "
              f"{sr.get('sim_mk_s', 0):>10.4f} {r['published_sim_mk_s']:>11.4f} "
              f"{re_.get('peak_mb', 0):>8.0f} {sr.get('sim_peak_mb', 0):>10.0f} "
              f"{r['published_sim_peak_mb']:>7} {re_.get('miss_synced', -1):>5}")


if __name__ == "__main__":
    main()
