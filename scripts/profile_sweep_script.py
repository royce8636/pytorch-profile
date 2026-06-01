#!/usr/bin/env python3
"""Full profiling sweep: SDXL-Turbo, SD3.5-Med, Llama-8B, Llama-3B × eager/inductor/4 offload modes.

Each job writes:
  <output_root>/<label>/terminal_output.log  — captured stdout+stderr
  <output_root>/<label>/args.txt             — cmdline + all effective args
  <output_root>/<label>/noprofile/terminal_output.log
  <output_root>/<label>/noprofile/args.txt

Jobs regenerate by default. Use --skip-existing to reuse successful existing logs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

SDXL_MODEL = "/data/llamasim/models/sdxl-turbo"
SD3_MODEL = "/data/llamasim/models/sd-3.5-med"
LLAMA8B_MODEL = "/data/llamasim/models/Llama-3.1-8b"
LLAMA3B_MODEL = "/data/llamasim/models/Llama-3.2-3B-Instruct"

SDXL_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
SDXL_VARIANT = "fp16"
DIFFUSION_PROMPT = "A cute dog and a cat in a park"
LLAMA_PROMPT = "The quick brown fox"

HF_OFFLOAD_MODES = ("model", "module", "sequential", "group")
LLAMA_OFFLOAD_MODES = ("module",)


@dataclass
class Job:
    label: str
    profile_cmd: list[str]
    noprofile_cmd: list[str]
    profile_effective: dict[str, Any]
    noprofile_effective: dict[str, Any]


def _args_text(cmd: list[str], effective: dict[str, Any]) -> str:
    lines = ["cmdline:", "  " + " ".join(str(c) for c in cmd), "", "effective_args:"]
    for k, v in effective.items():
        lines.append(f"  {k}: {v!r}")
    return "\n".join(lines) + "\n"


def _log_succeeded(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return "# exit_code: 0" in log_path.read_text(errors="replace")


def run_job_dir(
    cmd: list[str],
    output_dir: Path,
    effective: dict[str, Any],
    *,
    skip_existing: bool,
) -> None:
    log_path = output_dir / "terminal_output.log"
    args_path = output_dir / "args.txt"

    if skip_existing and _log_succeeded(log_path):
        print(f"  skip (successful log exists): {output_dir.name}", flush=True)
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*cmd, "--output-dir", str(output_dir)]
    effective_with_output = {**effective, "output_dir": str(output_dir)}
    args_path.write_text(_args_text(cmd, effective_with_output))

    start = time.perf_counter()
    print(f"  run: {output_dir.name}", flush=True)
    try:
        with log_path.open("w") as log_file:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()
                proc.wait()
        elapsed = time.perf_counter() - start
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  {status} in {elapsed:.1f}s: {output_dir.name}", flush=True)
        with log_path.open("a") as log_file:
            log_file.write(f"\n# exit_code: {proc.returncode}\n")
            log_file.write(f"# elapsed_seconds: {elapsed:.3f}\n")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except Exception as exc:
        print(f"  ERROR {exc}: {output_dir.name}", flush=True)
        with log_path.open("a") as log_file:
            log_file.write(f"\n# exception: {exc}\n")
        raise


def run_job(job: Job, output_root: Path, *, skip_existing: bool) -> None:
    print(f"\n=== {job.label} ===", flush=True)
    profile_dir = output_root / job.label
    noprofile_dir = profile_dir / "noprofile"
    run_job_dir(
        job.profile_cmd,
        profile_dir,
        job.profile_effective,
        skip_existing=skip_existing,
    )
    run_job_dir(
        job.noprofile_cmd,
        noprofile_dir,
        job.noprofile_effective,
        skip_existing=skip_existing,
    )


# ---------------------------------------------------------------------------
# SDXL-Turbo jobs
# ---------------------------------------------------------------------------

def _sdxl_base_effective(*, fusion: str, steps: int = 4, warmup_runs: int = 4) -> dict[str, Any]:
    return {
        "model": SDXL_MODEL,
        "prompt": DIFFUSION_PROMPT,
        "steps": steps,
        "height": 512,
        "width": 512,
        "seed": 0,
        "device": "cuda:0",
        "dtype": "float16",
        "vae_model": SDXL_VAE_MODEL,
        "variant": SDXL_VARIANT,
        "fusion": fusion,
        "compile_mode": "default",
        "fullgraph": False,
        "warmup_runs": warmup_runs,
        "disable_progress_bar": True,
    }


def sdxl_eager_job() -> Job:
    profile_eff = {
        **_sdxl_base_effective(fusion="none"),
        "output_prefix": "sdxl_eager_gpu",
    }
    noprofile_eff = {
        **_sdxl_base_effective(fusion="none"),
        "output_prefix": "sdxl_eager_gpu",
    }
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_sdxl_gpu.py"),
        "--model", SDXL_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "none",
        "--output-prefix", "sdxl_eager_gpu",
        "--steps", "4", "--warmup-runs", "4",
        "--seed", "0", "--device", "cuda:0", "--dtype", "float16",
        "--vae-model", SDXL_VAE_MODEL, "--variant", SDXL_VARIANT,
        "--height", "512", "--width", "512",
        "--disable-progress-bar",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_sdxl_gpu.py"),
        "--model", SDXL_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "none",
        "--output-prefix", "sdxl_eager_gpu",
        "--steps", "4", "--warmup-runs", "4",
        "--seed", "0", "--device", "cuda:0", "--dtype", "float16",
        "--vae-model", SDXL_VAE_MODEL, "--variant", SDXL_VARIANT,
        "--height", "512", "--width", "512",
        "--disable-progress-bar",
    ]
    return Job(
        label="sdxl_eager",
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


def sdxl_inductor_job() -> Job:
    profile_eff = {
        **_sdxl_base_effective(fusion="inductor"),
        "output_prefix": "sdxl_inductor_gpu",
    }
    noprofile_eff = {
        **_sdxl_base_effective(fusion="inductor"),
        "output_prefix": "sdxl_inductor_gpu",
    }
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_sdxl_gpu.py"),
        "--model", SDXL_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "inductor",
        "--output-prefix", "sdxl_inductor_gpu",
        "--steps", "4", "--warmup-runs", "4",
        "--seed", "0", "--device", "cuda:0", "--dtype", "float16",
        "--vae-model", SDXL_VAE_MODEL, "--variant", SDXL_VARIANT,
        "--height", "512", "--width", "512",
        "--disable-progress-bar",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_sdxl_gpu.py"),
        "--model", SDXL_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "inductor",
        "--output-prefix", "sdxl_inductor_gpu",
        "--steps", "4", "--warmup-runs", "4",
        "--seed", "0", "--device", "cuda:0", "--dtype", "float16",
        "--vae-model", SDXL_VAE_MODEL, "--variant", SDXL_VARIANT,
        "--height", "512", "--width", "512",
        "--disable-progress-bar",
    ]
    return Job(
        label="sdxl_inductor",
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


def sdxl_offload_job(offload_mode: str) -> Job:
    label = f"sdxl_offload_{offload_mode}"
    prefix = f"sdxl_offload_{offload_mode}_gpu"
    common_eff = {
        **_sdxl_base_effective(fusion="none"),
        "offload_mode": offload_mode,
        "group_offload_type": "block_level",
        "group_offload_num_blocks": 1,
        "group_offload_use_stream": True,
        "group_offload_non_blocking": False,
        "output_prefix": prefix,
    }
    profile_eff = {
        **common_eff,
        "profile_memory": True,
        "record_shapes": True,
        "with_stack": True,
        "with_modules": True,
    }
    noprofile_eff = dict(common_eff)
    base_args = [
        "--model", SDXL_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--offload-mode", offload_mode,
        "--fusion", "none",
        "--device", "cuda:0", "--dtype", "float16",
        "--vae-model", SDXL_VAE_MODEL, "--variant", SDXL_VARIANT,
        "--steps", "4", "--warmup-runs", "4",
        "--seed", "0", "--height", "512", "--width", "512",
        "--group-offload-type", "block_level",
        "--group-offload-num-blocks", "1",
        "--group-offload-use-stream",
        "--disable-progress-bar",
    ]
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_accelerate_cpu_offload.py"), "sdxl-turbo",
        *base_args,
        "--output-prefix", prefix,
        "--profile-memory", "--record-shapes", "--with-stack", "--with-modules",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_accelerate_cpu_offload.py"), "sdxl-turbo",
        *base_args,
    ]
    return Job(
        label=label,
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


# ---------------------------------------------------------------------------
# SD3.5-Medium jobs
# ---------------------------------------------------------------------------

def _sd3_base_effective(*, fusion: str, steps: int = 20, warmup_runs: int = 20) -> dict[str, Any]:
    return {
        "model": SD3_MODEL,
        "prompt": DIFFUSION_PROMPT,
        "steps": steps,
        "guidance_scale": 4.5,
        "height": 512,
        "width": 512,
        "seed": 0,
        "max_sequence_length": 256,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "fusion": fusion,
        "compile_mode": "default",
        "fullgraph": False,
        "warmup_runs": warmup_runs,
        "disable_progress_bar": True,
    }


def sd3_eager_job() -> Job:
    profile_eff = {
        **_sd3_base_effective(fusion="none"),
        "output_prefix": "sd3_eager_gpu",
    }
    noprofile_eff = {
        **_sd3_base_effective(fusion="none"),
        "output_prefix": "sd3_eager_gpu",
    }
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_sd3_gpu.py"),
        "--model", SD3_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "none",
        "--output-prefix", "sd3_eager_gpu",
        "--steps", "20", "--warmup-runs", "20",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--guidance-scale", "4.5", "--height", "512", "--width", "512",
        "--max-sequence-length", "256",
        "--disable-progress-bar",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_sd3_gpu.py"),
        "--model", SD3_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "none",
        "--output-prefix", "sd3_eager_gpu",
        "--steps", "20", "--warmup-runs", "20",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--guidance-scale", "4.5", "--height", "512", "--width", "512",
        "--max-sequence-length", "256",
        "--disable-progress-bar",
    ]
    return Job(
        label="sd3_eager",
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


def sd3_inductor_job() -> Job:
    profile_eff = {
        **_sd3_base_effective(fusion="inductor"),
        "output_prefix": "sd3_inductor_gpu",
    }
    noprofile_eff = {
        **_sd3_base_effective(fusion="inductor"),
        "output_prefix": "sd3_inductor_gpu",
    }
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_sd3_gpu.py"),
        "--model", SD3_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "inductor",
        "--output-prefix", "sd3_inductor_gpu",
        "--steps", "20", "--warmup-runs", "20",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--guidance-scale", "4.5", "--height", "512", "--width", "512",
        "--max-sequence-length", "256",
        "--disable-progress-bar",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_sd3_gpu.py"),
        "--model", SD3_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--fusion", "inductor",
        "--output-prefix", "sd3_inductor_gpu",
        "--steps", "20", "--warmup-runs", "20",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--guidance-scale", "4.5", "--height", "512", "--width", "512",
        "--max-sequence-length", "256",
        "--disable-progress-bar",
    ]
    return Job(
        label="sd3_inductor",
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


def sd3_offload_job(offload_mode: str) -> Job:
    label = f"sd3_offload_{offload_mode}"
    prefix = f"sd3_offload_{offload_mode}_gpu"
    common_eff = {
        **_sd3_base_effective(fusion="none"),
        "offload_mode": offload_mode,
        "group_offload_type": "block_level",
        "group_offload_num_blocks": 1,
        "group_offload_use_stream": True,
        "group_offload_non_blocking": False,
        "output_prefix": prefix,
    }
    profile_eff = {
        **common_eff,
        "profile_memory": True,
        "record_shapes": True,
        "with_stack": True,
        "with_modules": True,
    }
    noprofile_eff = dict(common_eff)
    base_offload_args = [
        "--model", SD3_MODEL,
        "--prompt", DIFFUSION_PROMPT,
        "--offload-mode", offload_mode,
        "--fusion", "none",
        "--device", "cuda:0", "--dtype", "bfloat16",
        "--steps", "20", "--warmup-runs", "20",
        "--seed", "0", "--height", "512", "--width", "512",
        "--guidance-scale", "4.5", "--max-sequence-length", "256",
        "--group-offload-type", "block_level",
        "--group-offload-num-blocks", "1",
        "--group-offload-use-stream",
        "--output-prefix", prefix,
        "--disable-progress-bar",
    ]
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_sd3_accelerate_cpu_offload.py"),
        *base_offload_args,
        "--profile-memory", "--record-shapes", "--with-stack", "--with-modules",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_sd3_accelerate_cpu_offload.py"),
        *base_offload_args,
    ]
    return Job(
        label=label,
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


# ---------------------------------------------------------------------------
# Llama jobs (shared helper for 8B and 3B)
# ---------------------------------------------------------------------------

def _llama_base_effective(
    *, model: str, fusion: str, warmup_runs: int = 1, max_new_tokens: int = 15
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": LLAMA_PROMPT,
        "seed": 0,
        "max_new_tokens": max_new_tokens,
        "batch_size": 1,
        "do_sample": False,
        "temperature": 1.0,
        "enable_eos_stop": False,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "fusion": fusion,
        "compile_mode": "default",
        "fullgraph": False,
        "dynamic": True,
        "warmup_runs": warmup_runs,
    }


def llama_eager_job(model_name: str, model_path: str) -> Job:
    profile_eff = {
        **_llama_base_effective(model=model_path, fusion="none"),
        "output_prefix": f"{model_name}_eager_gpu",
    }
    noprofile_eff = {
        **_llama_base_effective(model=model_path, fusion="none"),
        "output_prefix": f"{model_name}_eager_gpu",
    }
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_llama_gpu.py"),
        "--model", model_path,
        "--prompt", LLAMA_PROMPT,
        "--fusion", "none",
        "--output-prefix", f"{model_name}_eager_gpu",
        "--warmup-runs", "1",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--max-new-tokens", "15",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_llama_gpu.py"),
        "--model", model_path,
        "--prompt", LLAMA_PROMPT,
        "--fusion", "none",
        "--output-prefix", f"{model_name}_eager_gpu",
        "--warmup-runs", "1",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--max-new-tokens", "15",
    ]
    return Job(
        label=f"{model_name}_eager",
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


def llama_inductor_job(model_name: str, model_path: str) -> Job:
    profile_eff = {
        **_llama_base_effective(model=model_path, fusion="inductor"),
        "output_prefix": f"{model_name}_inductor_gpu",
    }
    noprofile_eff = {
        **_llama_base_effective(model=model_path, fusion="inductor"),
        "output_prefix": f"{model_name}_inductor_gpu",
    }
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_llama_gpu.py"),
        "--model", model_path,
        "--prompt", LLAMA_PROMPT,
        "--fusion", "inductor",
        "--output-prefix", f"{model_name}_inductor_gpu",
        "--warmup-runs", "1",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--max-new-tokens", "15",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_llama_gpu.py"),
        "--model", model_path,
        "--prompt", LLAMA_PROMPT,
        "--fusion", "inductor",
        "--output-prefix", f"{model_name}_inductor_gpu",
        "--warmup-runs", "1",
        "--seed", "0", "--device", "cuda:0", "--dtype", "bfloat16",
        "--max-new-tokens", "15",
    ]
    return Job(
        label=f"{model_name}_inductor",
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


def llama_offload_job(model_name: str, model_path: str, offload_mode: str) -> Job:
    label = f"{model_name}_offload_{offload_mode}"
    prefix = f"{model_name}_offload_{offload_mode}"
    common_eff = {
        **_llama_base_effective(model=model_path, fusion="none"),
        "offload_mode": offload_mode,
        "group_offload_type": "block_level",
        "group_offload_num_blocks": 1,
        "group_offload_use_stream": False,
        "group_offload_non_blocking": False,
        "output_prefix": prefix,
    }
    profile_eff = {
        **common_eff,
        "profile_memory": True,
        "record_shapes": True,
        "with_stack": True,
        "with_modules": True,
    }
    noprofile_eff = dict(common_eff)
    base_offload_args = [
        "--model", model_path,
        "--prompt", LLAMA_PROMPT,
        "--offload-mode", offload_mode,
        "--fusion", "none",
        "--device", "cuda:0", "--dtype", "bfloat16",
        "--warmup-runs", "1",
        "--seed", "0",
        "--max-new-tokens", "15",
        "--output-prefix", prefix,
    ]
    profile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "profile_llama_accelerate_cpu_offload.py"),
        *base_offload_args,
        "--profile-memory", "--record-shapes", "--with-stack", "--with-modules",
    ]
    noprofile_cmd = [
        PYTHON, str(SCRIPTS_DIR / "run_llama_accelerate_cpu_offload.py"),
        *base_offload_args,
    ]
    return Job(
        label=label,
        profile_cmd=profile_cmd,
        noprofile_cmd=noprofile_cmd,
        profile_effective=profile_eff,
        noprofile_effective=noprofile_eff,
    )


# ---------------------------------------------------------------------------
# Build full job list
# ---------------------------------------------------------------------------

def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    # SDXL-Turbo
    jobs.append(sdxl_eager_job())
    jobs.append(sdxl_inductor_job())
    for mode in HF_OFFLOAD_MODES:
        jobs.append(sdxl_offload_job(mode))

    # SD3.5-Medium
    jobs.append(sd3_eager_job())
    jobs.append(sd3_inductor_job())
    for mode in HF_OFFLOAD_MODES:
        jobs.append(sd3_offload_job(mode))

    # Llama 8B
    jobs.append(llama_eager_job("llama8b", LLAMA8B_MODEL))
    jobs.append(llama_inductor_job("llama8b", LLAMA8B_MODEL))
    for mode in LLAMA_OFFLOAD_MODES:
        jobs.append(llama_offload_job("llama8b", LLAMA8B_MODEL, mode))

    # Llama 3B
    jobs.append(llama_eager_job("llama3b", LLAMA3B_MODEL))
    jobs.append(llama_inductor_job("llama3b", LLAMA3B_MODEL))
    for mode in LLAMA_OFFLOAD_MODES:
        jobs.append(llama_offload_job("llama3b", LLAMA3B_MODEL, mode))

    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full profiling sweep: SDXL-Turbo, SD3.5-Med, Llama-8B, Llama-3B "
            "× eager / inductor / 4 HF offload modes."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="exp_results/0521_sweep",
        help="Root output directory (default: exp_results/0521_sweep)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs with an existing terminal_output.log that has exit_code 0.",
    )
    parser.add_argument(
        "--force",
        dest="skip_existing",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated list of job labels to run (substring match).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()

    if args.only is not None:
        filters = [f.strip() for f in args.only.split(",")]
        jobs = [j for j in jobs if any(f in j.label for f in filters)]

    print(f"Output root: {output_root}", flush=True)
    print(f"Total jobs: {len(jobs)} × 2 (profile + noprofile) = {len(jobs) * 2} runs", flush=True)
    if args.skip_existing:
        print("--skip-existing: reusing successful existing logs", flush=True)
    else:
        print("Regenerating all selected profile and noprofile outputs", flush=True)

    sweep_start = time.perf_counter()
    for job in jobs:
        run_job(job, output_root, skip_existing=args.skip_existing)

    elapsed = time.perf_counter() - sweep_start
    print(f"\nSweep complete in {elapsed / 3600:.2f}h ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
