#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_sd3_common import main


if __name__ == "__main__":
    main(
        default_model="/data/llamasim/models/sd-3.5-large",
        default_device="cuda:0",
        default_dtype="bfloat16",
        default_output_prefix="sd3_large_gpu",
        default_component="sd3_large_pipeline",
        default_fusion="inductor",
        default_dot_level="llamasim-runtime",
        default_llamasim_output_dirname="llama_bundle",
        default_profile_memory=True,
        default_record_shapes=True,
        default_with_stack=True,
    )
