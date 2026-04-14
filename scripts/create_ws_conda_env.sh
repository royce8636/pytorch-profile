#!/usr/bin/env bash
set -euo pipefail

# Create a conda environment for weight streaming + profiling.
#
# This reuses the existing PyTorch C++ build artifacts — no rebuild needed.
# The env gets: Python 3.10, diffusers, transformers, accelerate, triton,
# and the source tree added to the Python path via a .pth file.
#
# Usage:
#   bash scripts/create_ws_conda_env.sh
#
# Then activate and run:
#   conda activate ptvenv
#   python scripts/run_weight_streaming.py --plan-dir ... --model ...
#   python scripts/profile_sdxl_turbo_gpu.py --fusion inductor ...

ENV_NAME="${WS_ENV_NAME:-ptvenv}"
PT_SOURCE="${PT_SOURCE:-/data/pytorch-source}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"

if [[ -z "$CONDA_BIN" ]]; then
    echo "error: conda not found. Set CONDA_BIN or add conda to PATH." >&2
    exit 1
fi

echo "=== Creating conda env '$ENV_NAME' ==="

# Remove existing env if present
if "$CONDA_BIN" info --envs | grep -q "^${ENV_NAME} "; then
    echo "Removing existing env '$ENV_NAME'"
    "$CONDA_BIN" env remove -n "$ENV_NAME" -y
fi

# Create env with Python 3.10 to match the existing build
"$CONDA_BIN" create -n "$ENV_NAME" python=3.10 numpy -y

# Locate the env directory
ENV_DIR="$("$CONDA_BIN" info --base)/envs/$ENV_NAME"
PYTHON_BIN="$ENV_DIR/bin/python"
PIP_BIN="$ENV_DIR/bin/pip"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "error: python not found at $PYTHON_BIN" >&2
    exit 1
fi

echo "=== Setting PYTHONNOUSERSITE in activation ==="

# Prevent user site-packages (~/.local/lib/...) from shadowing conda packages.
# This is critical because the system has a separate torch install there.
mkdir -p "$ENV_DIR/etc/conda/activate.d"
cat > "$ENV_DIR/etc/conda/activate.d/no_user_site.sh" << 'EOF'
export PYTHONNOUSERSITE=1
EOF

mkdir -p "$ENV_DIR/etc/conda/deactivate.d"
cat > "$ENV_DIR/etc/conda/deactivate.d/no_user_site.sh" << 'EOF'
unset PYTHONNOUSERSITE
EOF

echo "=== Installing dependencies ==="

# Use PYTHONNOUSERSITE so pip installs into the conda env, not user site.
# Use --no-deps for packages whose transitive deps would pull in a separate torch.
PYTHONNOUSERSITE=1 "$PIP_BIN" install \
    expecttest \
    diffusers \
    jinja2 \
    pyyaml \
    filelock \
    psutil

# These must be --no-deps to avoid pulling in torch (and its nvidia deps)
PYTHONNOUSERSITE=1 "$PIP_BIN" install --no-deps \
    accelerate \
    networkx \
    sympy \
    fsspec \
    safetensors \
    Pillow \
    regex \
    tqdm \
    rich \
    anyio \
    httpx \
    httpcore \
    certifi \
    sniffio \
    h11 \
    pygments \
    markdown-it-py \
    mdurl \
    mpmath \
    markupsafe \
    idna

# Pin transformers/tokenizers/huggingface-hub to known-good versions
PYTHONNOUSERSITE=1 "$PIP_BIN" install --no-deps \
    transformers==4.48.3 \
    tokenizers==0.21.4 \
    huggingface-hub==0.36.0

echo "=== Installing Triton ==="
(
    cd "$PT_SOURCE"
    PATH="$ENV_DIR/bin:$PATH" PYTHONNOUSERSITE=1 \
        bash "$PT_SOURCE/scripts/install_triton_wheel.sh"
) || echo "WARNING: Triton install failed; GPU kernel compilation may not work."

echo "=== Linking PyTorch source tree (no rebuild) ==="

# Add the source tree to the conda env's Python path via a .pth file.
# This avoids `pip install -e .` which triggers a full CMake reconfigure.
SITE_DIR="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
echo "$PT_SOURCE" > "$SITE_DIR/pytorch-source.pth"
echo "Created $SITE_DIR/pytorch-source.pth -> $PT_SOURCE"

# Create minimal dist-info so importlib.metadata.version('torch') works.
# Libraries like diffusers check for torch via metadata, not import.
TORCH_VERSION="$(PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c 'import torch; print(torch.__version__)')"
DIST_DIR="$SITE_DIR/torch-${TORCH_VERSION}.dist-info"
mkdir -p "$DIST_DIR"
cat > "$DIST_DIR/METADATA" << METAEOF
Metadata-Version: 2.1
Name: torch
Version: ${TORCH_VERSION}
Summary: PyTorch (source tree at ${PT_SOURCE})
METAEOF
echo "pip" > "$DIST_DIR/INSTALLER"
touch "$DIST_DIR/RECORD"
echo "Created $DIST_DIR"

echo "=== Verifying ==="
PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c "
import torch
print(f'torch:       {torch.__version__}')
print(f'torch file:  {torch.__file__}')
print(f'CUDA built:  {torch.backends.cuda.is_built()}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
print(f'CUDA ver:    {torch.version.cuda}')
try:
    import diffusers
    print(f'diffusers:   {diffusers.__version__}')
except ImportError:
    print('diffusers:   NOT INSTALLED')
try:
    import triton
    print(f'triton:      {triton.__version__}')
except ImportError:
    print('triton:      NOT INSTALLED')
from torch._inductor.weight_streaming import IOSchedule
print('weight_streaming: OK')
"

cat <<EOF

=== Done ===
Conda env:  $ENV_NAME ($ENV_DIR)
Activate:   conda activate $ENV_NAME

The env sets PYTHONNOUSERSITE=1 on activation to avoid conflicts
with packages in ~/.local/lib/python3.10/site-packages.

Run weight streaming:
  python scripts/run_weight_streaming.py \\
      --plan-dir llamasim_bundle/jit_sim_prune_output \\
      --model /data/llamasim/models/sdxl-turbo \\
      --export-code ./ws_output --log-io

Run profiling:
  python scripts/profile_sdxl_turbo_gpu.py \\
      --fusion inductor --output-dir /tmp/profile_out
EOF
