#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_sdxl_turbo_gpu import main


if __name__ == "__main__":
    main(
        default_model="/data/llamasim/models/sdxl-1.0-base",
        default_vae_model=None,
        default_variant=None,
        default_output_prefix="sdxl_base_gpu_run",
        description="Run SDXL 1.0 base on GPU without profiler instrumentation.",
    )
