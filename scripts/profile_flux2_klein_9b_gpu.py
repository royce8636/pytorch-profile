#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_flux2_common import main


if __name__ == "__main__":
    main(
        default_model="/data/llamasim/models/flux2-klein-9b",
        default_device="cuda:0",
        default_dtype="bfloat16",
        default_output_prefix="flux2_klein_9b_gpu",
        default_component="flux2_klein_9b_pipeline",
        default_fusion="inductor",
        default_offload_mode="model",
    )
