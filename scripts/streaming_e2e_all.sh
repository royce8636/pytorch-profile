#!/bin/bash
# Benchmark all scheduler variants through the unified same-process pipeline.
set +e
PY=/home/royce/anaconda3/envs/ptvenv/bin/python
HW=/data/pytorch-source/sdxl-turbo-profile-0414/llama_bundle/hw_pcie4.json

run_one() {
  local variant=$1
  local workdir=/tmp/streaming_e2e_${variant}
  echo "================= $variant ================="
  $PY /data/pytorch-source/scripts/streaming_e2e.py \
    --hw "$HW" \
    --variant "$variant" \
    --work-dir "$workdir" \
    --steps 1 --warmup-runs 1 --height 128 --width 128 2>&1 | \
    grep -E "H2D scheduled|H2D bytes|H2D ops|H2D dropped|Pinned|Peak VRAM|Chosen cap|RESULTS|variant|allocated_mb|peak_mb|inference_s" | head -20
  echo ""
}

run_one baseline
run_one ct_maxbytes
run_one ct_largest
run_one ct_belady_pcie
