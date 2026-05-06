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
        default_model=None,
        default_output_prefix="llama_gpu_run",
        description=(
            "Run a HuggingFace-format Llama-family checkpoint on GPU without "
            "profiler instrumentation."
        ),
    )
