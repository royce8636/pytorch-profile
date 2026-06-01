#!/usr/bin/env python3
"""Run noprofile offload timing for all 4 models × 4 HF offload modes."""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

PYTHON = sys.executable
SCRIPTS = Path(__file__).resolve().parent
BASE = Path("exp_results/0521_sweep")

SDXL = "/data/llamasim/models/sdxl-turbo"
SD3  = "/data/llamasim/models/sd-3.5-med"
L3B  = "/data/llamasim/models/Llama-3.2-3B-Instruct"
L8B  = "/data/llamasim/models/Llama-3.1-8b"
VAE  = "madebyollin/sdxl-vae-fp16-fix"

JOBS = []

for mode in ("model", "module", "sequential", "group"):
    JOBS.append((
        f"sdxl_offload_{mode}/noprofile",
        [PYTHON, str(SCRIPTS / "run_accelerate_cpu_offload.py"), "sdxl-turbo",
         "--model", SDXL, "--offload-mode", mode, "--fusion", "none",
         "--device", "cuda:0", "--dtype", "float16",
         "--vae-model", VAE, "--variant", "fp16",
         "--steps", "4", "--warmup-runs", "4",
         "--seed", "0", "--height", "512", "--width", "512",
         "--group-offload-type", "block_level", "--group-offload-num-blocks", "1",
         "--group-offload-use-stream", "--disable-progress-bar"],
    ))

for mode in ("sequential", "group"):
    JOBS.append((
        f"sd3_offload_{mode}/noprofile",
        [PYTHON, str(SCRIPTS / "run_sd3_accelerate_cpu_offload.py"),
         "--model", SD3, "--offload-mode", mode, "--fusion", "none",
         "--device", "cuda:0", "--dtype", "bfloat16",
         "--steps", "20", "--warmup-runs", "4",
         "--seed", "0", "--height", "512", "--width", "512",
         "--guidance-scale", "4.5", "--max-sequence-length", "256",
         "--group-offload-type", "block_level", "--group-offload-num-blocks", "1",
         "--group-offload-use-stream", "--disable-progress-bar"],
    ))

for mode in ("model", "module", "sequential", "group"):
    JOBS.append((
        f"llama3b_offload_{mode}/noprofile",
        [PYTHON, str(SCRIPTS / "run_llama_accelerate_cpu_offload.py"),
         "--model", L3B, "--offload-mode", mode, "--fusion", "none",
         "--device", "cuda:0", "--dtype", "bfloat16",
         "--warmup-runs", "1", "--seed", "0", "--max-new-tokens", "15",
         "--group-offload-type", "block_level", "--group-offload-num-blocks", "1"],
    ))
    JOBS.append((
        f"llama8b_offload_{mode}/noprofile",
        [PYTHON, str(SCRIPTS / "run_llama_accelerate_cpu_offload.py"),
         "--model", L8B, "--offload-mode", mode, "--fusion", "none",
         "--device", "cuda:0", "--dtype", "bfloat16",
         "--warmup-runs", "1", "--seed", "0", "--max-new-tokens", "15",
         "--group-offload-type", "block_level", "--group-offload-num-blocks", "1"],
    ))

for label, cmd in JOBS:
    out_dir = BASE / label
    log = out_dir / "terminal_output.log"
    if log.exists() and b"exit_code: 0" in log.read_bytes():
        print(f"skip (done): {label}", flush=True)
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {label}", flush=True)
    start = time.perf_counter()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as proc:
        lines = []
        for line in proc.stdout:
            print(f"  {line}", end="", flush=True)
            lines.append(line)
        proc.wait()
    elapsed = time.perf_counter() - start
    status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
    print(f"[{status} in {elapsed:.1f}s]", flush=True)
    log.write_text("".join(lines) + f"\n# exit_code: {proc.returncode}\n# elapsed_seconds: {elapsed:.3f}\n")
