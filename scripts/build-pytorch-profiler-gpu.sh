#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  scripts/build-pytorch-profiler-gpu.sh [--skip-deps] [--skip-triton] [--tests]

Build the patched /data/pytorch-source as a CUDA-enabled editable install.

Environment overrides:
  PT_SOURCE                PyTorch source tree (default: /data/pytorch-source)
  PT_VENV                  Virtualenv path (default: /tmp/ptvenv)
  PT_PYTHON                Python executable used to create the venv (default: python3)
  PT_JOBS                  Parallel build jobs for ninja (default: nproc)
  PT_SKIP_DEPS             Skip pip dependency installation when set to 1
  PT_INSTALL_TRITON        Install the pinned Triton wheel when set to 1 (default: 1)
  PT_RUN_TESTS             Run profiler regression tests when set to 1
  PT_CUDA_HOME             CUDA toolkit root. Auto-detected from nvcc when unset.
  PT_CUDA_NVCC_EXECUTABLE  Explicit nvcc path. Auto-detected when unset.
  PT_CUDNN_ROOT            cuDNN root passed through to the build when set.
  PT_TORCH_CUDA_ARCH_LIST  TORCH_CUDA_ARCH_LIST override, for example 8.9
  PT_CMAKE_CUDA_COMPILER   Explicit nvcc path passed as CMAKE_CUDA_COMPILER.
EOF
}

detect_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" ]]; then
        printf '%s\n' "$CUDA_HOME"
        return
    fi
    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(readlink -f "$(command -v nvcc)")"
        dirname "$(dirname "$nvcc_path")"
    fi
}

detect_nvcc() {
    if [[ -n "${CUDA_NVCC_EXECUTABLE:-}" ]]; then
        printf '%s\n' "$CUDA_NVCC_EXECUTABLE"
        return
    fi
    if [[ -n "${PT_CMAKE_CUDA_COMPILER:-}" ]]; then
        printf '%s\n' "$PT_CMAKE_CUDA_COMPILER"
        return
    fi
    if [[ -n "${PT_CUDA_HOME:-}" && -x "${PT_CUDA_HOME}/bin/nvcc" ]]; then
        printf '%s\n' "${PT_CUDA_HOME}/bin/nvcc"
        return
    fi
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
    fi
}

PT_SOURCE="${PT_SOURCE:-/data/pytorch-source}"
PT_VENV="${PT_VENV:-/tmp/ptvenv}"
PT_PYTHON="${PT_PYTHON:-python3}"
PT_JOBS="${PT_JOBS:-$(nproc)}"
PT_SKIP_DEPS="${PT_SKIP_DEPS:-0}"
PT_INSTALL_TRITON="${PT_INSTALL_TRITON:-1}"
PT_RUN_TESTS="${PT_RUN_TESTS:-0}"
PT_CUDA_HOME="${PT_CUDA_HOME:-$(detect_cuda_home)}"
PT_CMAKE_CUDA_COMPILER="${PT_CMAKE_CUDA_COMPILER:-}"
PT_CUDA_NVCC_EXECUTABLE="${PT_CUDA_NVCC_EXECUTABLE:-$(detect_nvcc)}"
PT_CUDNN_ROOT="${PT_CUDNN_ROOT:-${CUDNN_ROOT:-}}"
PT_TORCH_CUDA_ARCH_LIST="${PT_TORCH_CUDA_ARCH_LIST:-${TORCH_CUDA_ARCH_LIST:-}}"

if [[ -z "$PT_CMAKE_CUDA_COMPILER" && -n "$PT_CUDA_NVCC_EXECUTABLE" ]]; then
    PT_CMAKE_CUDA_COMPILER="$PT_CUDA_NVCC_EXECUTABLE"
fi

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
        --skip-triton)
            PT_INSTALL_TRITON=0
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

if [[ -z "$PT_CUDA_HOME" ]]; then
    echo "error: failed to detect CUDA_HOME; set PT_CUDA_HOME explicitly" >&2
    exit 1
fi

if [[ -z "$PT_CUDA_NVCC_EXECUTABLE" || ! -x "$PT_CUDA_NVCC_EXECUTABLE" ]]; then
    echo "error: failed to detect nvcc; set PT_CUDA_NVCC_EXECUTABLE explicitly" >&2
    exit 1
fi

BUILD_DIR="$PT_SOURCE/build"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    if ! grep -Eq '^USE_CUDA:BOOL=1$' "$BUILD_DIR/CMakeCache.txt"; then
        BACKUP_DIR="${BUILD_DIR}-nongpu-backup-$(date +%Y%m%d-%H%M%S)"
        echo "Moving existing non-GPU build cache to $BACKUP_DIR"
        mv "$BUILD_DIR" "$BACKUP_DIR"
    fi
fi

echo "Creating virtualenv at $PT_VENV"
"$PT_PYTHON" -m venv "$PT_VENV"

PYTHON_BIN="$PT_VENV/bin/python"
PIP_BIN="$PT_VENV/bin/pip"
NVCC_BIN_DIR="$(dirname "$PT_CUDA_NVCC_EXECUTABLE")"
PATH_PREFIX="$PT_VENV/bin:$NVCC_BIN_DIR:/usr/bin:/bin"

if [[ "$PT_SKIP_DEPS" != "1" ]]; then
    echo "Installing build and runtime dependencies"
    "$PIP_BIN" install -r "$PT_SOURCE/requirements-build.txt"
    "$PIP_BIN" install expecttest diffusers jinja2 networkx sympy filelock fsspec psutil
    "$PIP_BIN" install --no-deps \
        transformers==4.48.3 \
        tokenizers==0.21.4 \
        huggingface-hub==0.36.0
fi

if [[ "$PT_INSTALL_TRITON" == "1" ]]; then
    echo "Installing pinned Triton wheel"
    (
        cd "$PT_SOURCE"
        env PATH="$PATH_PREFIX" PYTHONNOUSERSITE=1 \
            bash "$PT_SOURCE/scripts/install_triton_wheel.sh"
    )
fi

BUILD_ENV=(
    env
    PATH="$PATH_PREFIX"
    PYTHONNOUSERSITE=1
    CMAKE_FRESH=1
    CMAKE_ONLY=1
    USE_CUDA=1
    USE_CUDNN=1
    USE_ROCM=0
    USE_XPU=0
    BUILD_TEST=0
    USE_DISTRIBUTED=0
    CMAKE_GENERATOR=Ninja
    CMAKE_PREFIX_PATH="$PT_VENV"
    CUDA_HOME="$PT_CUDA_HOME"
    CUDA_NVCC_EXECUTABLE="$PT_CUDA_NVCC_EXECUTABLE"
    CMAKE_CUDA_COMPILER="$PT_CMAKE_CUDA_COMPILER"
)

INSTALL_ENV=(
    env
    PATH="$PATH_PREFIX"
    PYTHONNOUSERSITE=1
    USE_CUDA=1
    USE_CUDNN=1
    USE_ROCM=0
    USE_XPU=0
    BUILD_TEST=0
    USE_DISTRIBUTED=0
    CMAKE_GENERATOR=Ninja
    CMAKE_PREFIX_PATH="$PT_VENV"
    CUDA_HOME="$PT_CUDA_HOME"
    CUDA_NVCC_EXECUTABLE="$PT_CUDA_NVCC_EXECUTABLE"
    CMAKE_CUDA_COMPILER="$PT_CMAKE_CUDA_COMPILER"
)

if [[ -n "$PT_CUDNN_ROOT" ]]; then
    BUILD_ENV+=(CUDNN_ROOT="$PT_CUDNN_ROOT")
    INSTALL_ENV+=(CUDNN_ROOT="$PT_CUDNN_ROOT")
fi

if [[ -n "$PT_TORCH_CUDA_ARCH_LIST" ]]; then
    BUILD_ENV+=(TORCH_CUDA_ARCH_LIST="$PT_TORCH_CUDA_ARCH_LIST")
    INSTALL_ENV+=(TORCH_CUDA_ARCH_LIST="$PT_TORCH_CUDA_ARCH_LIST")
fi

echo "Configuring CUDA-enabled PyTorch build"
"${BUILD_ENV[@]}" "$PYTHON_BIN" "$PT_SOURCE/setup.py" build

echo "Building torch_python"
env PATH="$PATH_PREFIX" ninja -C "$BUILD_DIR" -j "$PT_JOBS" torch_python

echo "Installing editable PyTorch package"
"${INSTALL_ENV[@]}" "$PIP_BIN" install --no-build-isolation -v -e "$PT_SOURCE"

echo "Verifying import"
env PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
    "import torch; print('torch', torch.__version__); print('torch_file', torch.__file__); print('torch.version.cuda', torch.version.cuda); print('cuda_built', torch.backends.cuda.is_built()); print('cuda_available', torch.cuda.is_available())"

if [[ "$PT_RUN_TESTS" == "1" ]]; then
    echo "Running profiler regression tests"
    env PYTHONNOUSERSITE=1 "$PYTHON_BIN" \
        "$PT_SOURCE/test/profiler/test_profiler.py" \
        TestProfiler.test_user_annotation_args_metadata \
        TestProfiler.test_user_annotation_fast_kwargs_without_record_shapes \
        TestProfiler.test_export_csv \
        -v
fi

cat <<EOF
Build completed.
Venv: $PT_VENV
PyTorch source: $PT_SOURCE
CUDA_HOME: $PT_CUDA_HOME
CUDA_NVCC_EXECUTABLE: $PT_CUDA_NVCC_EXECUTABLE

Example:
  PYTHONNOUSERSITE=1 $PYTHON_BIN -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
EOF
