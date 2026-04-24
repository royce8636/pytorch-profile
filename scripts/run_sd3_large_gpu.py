#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_sd3_med_gpu import main


if __name__ == "__main__":
    main(
        default_model="/data/llamasim/models/sd-3.5-large",
        default_output_prefix="sd3_large_gpu_run",
        description="Run SD 3.5 large on GPU without profiler instrumentation.",
    )
