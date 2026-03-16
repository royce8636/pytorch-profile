#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_sdxl_turbo_common import main


if __name__ == "__main__":
    main(
        default_device="cuda:0",
        default_dtype="float16",
        default_output_prefix="sdxl_turbo_gpu",
        default_profile_memory=True,
        default_record_shapes=True,
        default_with_stack=True,
    )
