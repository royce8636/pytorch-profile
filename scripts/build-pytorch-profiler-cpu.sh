#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  script/build-pytorch-profiler-cpu.sh [--skip-deps] [--tests]

Build the patched /data/pytorch-source as a CPU-only editable install.

Environment overrides:
  PT_SOURCE      PyTorch source tree (default: /data/pytorch-source)
  PT_VENV        Virtualenv path (default: /tmp/ptvenv)
  PT_PYTHON      Python executable used to create the venv (default: python3)
  PT_JOBS        Parallel build jobs for ninja (default: nproc)
  PT_SKIP_DEPS   Skip pip dependency installation when set to 1
  PT_RUN_TESTS   Run profiler regression tests when set to 1
EOF
}

PT_SOURCE="${PT_SOURCE:-/data/pytorch-source}"
PT_VENV="${PT_VENV:-/tmp/ptvenv}"
PT_PYTHON="${PT_PYTHON:-python3}"
PT_JOBS="${PT_JOBS:-$(nproc)}"
PT_SKIP_DEPS="${PT_SKIP_DEPS:-0}"
PT_RUN_TESTS="${PT_RUN_TESTS:-0}"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --skip-deps)
            PT_SKIP_DEPS=1
            shift
            ;;
        --tests)
            PT_RUN_TESTS=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$PT_SOURCE" ]]; then
    echo "error: PT_SOURCE does not exist: $PT_SOURCE" >&2
    exit 1
fi

BUILD_DIR="$PT_SOURCE/build"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    if grep -Eq '^USE_(CUDA|ROCM|XPU):BOOL=1$' "$BUILD_DIR/CMakeCache.txt"; then
        BACKUP_DIR="${BUILD_DIR}-noncpu-backup-$(date +%Y%m%d-%H%M%S)"
        echo "Moving existing non-CPU build cache to $BACKUP_DIR"
        mv "$BUILD_DIR" "$BACKUP_DIR"
    fi
fi

echo "Creating virtualenv at $PT_VENV"
"$PT_PYTHON" -m venv "$PT_VENV"

PYTHON_BIN="$PT_VENV/bin/python"
PIP_BIN="$PT_VENV/bin/pip"
PATH_PREFIX="$PT_VENV/bin:/usr/bin:/bin"

if [[ "$PT_SKIP_DEPS" != "1" ]]; then
    echo "Installing build and runtime dependencies"
    "$PIP_BIN" install -r "$PT_SOURCE/requirements-build.txt"
    "$PIP_BIN" install expecttest diffusers jinja2 networkx sympy filelock fsspec psutil
    "$PIP_BIN" install --no-deps \
        transformers==4.48.3 \
        tokenizers==0.21.4 \
        huggingface-hub==0.36.0
fi

echo "Configuring CPU-only PyTorch build"
env \
    PATH="$PATH_PREFIX" \
    PYTHONNOUSERSITE=1 \
    CMAKE_FRESH=1 \
    CMAKE_ONLY=1 \
    USE_CUDA=0 \
    USE_ROCM=0 \
    USE_XPU=0 \
    BUILD_TEST=0 \
    USE_DISTRIBUTED=0 \
    CMAKE_GENERATOR=Ninja \
    CMAKE_PREFIX_PATH="$PT_VENV" \
    "$PYTHON_BIN" "$PT_SOURCE/setup.py" build

echo "Building torch_python"
env PATH="$PATH_PREFIX" ninja -C "$BUILD_DIR" -j "$PT_JOBS" torch_python

echo "Installing editable PyTorch package"
env \
    PATH="$PATH_PREFIX" \
    PYTHONNOUSERSITE=1 \
    USE_CUDA=0 \
    USE_ROCM=0 \
    USE_XPU=0 \
    BUILD_TEST=0 \
    USE_DISTRIBUTED=0 \
    CMAKE_GENERATOR=Ninja \
    CMAKE_PREFIX_PATH="$PT_VENV" \
    "$PIP_BIN" install --no-build-isolation -v -e "$PT_SOURCE"

echo "Verifying import"
env PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
    "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"

if [[ "$PT_RUN_TESTS" == "1" ]]; then
    echo "Running profiler regression tests"
    env PYTHONNOUSERSITE=1 "$PYTHON_BIN" \
        "$PT_SOURCE/test/profiler/test_profiler.py" \
        TestProfiler.test_user_annotation_args_metadata \
        TestProfiler.test_user_annotation_fast_kwargs_without_record_shapes \
        -v
fi

cat <<EOF
Build completed.
Venv: $PT_VENV
PyTorch source: $PT_SOURCE

Example:
  PYTHONNOUSERSITE=1 $PYTHON_BIN -c "import torch; print(torch.__version__)"
EOF
