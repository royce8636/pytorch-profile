#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_llama_3b_gpu import main


if __name__ == "__main__":
    main(
        default_model="meta-llama/Meta-Llama-3-8B",
        default_output_prefix="llama_8b_gpu_run",
        description="Run Llama 3 8B on GPU without profiler instrumentation.",
    )
