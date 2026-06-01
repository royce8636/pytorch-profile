#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_llama_common import main


if __name__ == "__main__":
    main(
        default_device="cuda:0",
        default_dtype="bfloat16",
        default_output_prefix="llama_gpu",
        default_output_dir=(
            "/data/pytorch-source/exp_results/llama/llama8b-modulecols-eager"
        ),
        default_model="/data/llamasim/models/Llama-3.1-8b",
        default_component="llama8b",
        default_fusion="none",
        default_dot_level="llamasim-runtime",
        default_llamasim_output_dirname="llama_bundle",
        default_max_new_tokens=15,
        default_profile_memory=True,
        default_record_shapes=True,
        default_with_stack=True,
    )
