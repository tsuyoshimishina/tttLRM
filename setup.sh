#!/usr/bin/env bash
# tttLRM Environment Setup Script
# Sets up a conda environment for test-time training 3D reconstruction:
#   tttlrm (Python 3.10, PyTorch 2.5.1, CUDA 11.8)
#
# The conda env is fully self-contained: CUDA toolkit 11.8 is installed
# inside the env so that native extensions (flash-attn, diff-gaussian-
# rasterization, gsplat) are built against the same CUDA version as
# PyTorch — regardless of the system CUDA version.
#
# Prerequisites:
#   - mamba or conda
#
# Usage:
#   ./setup.sh          # full setup
#   ./setup.sh --check  # verify installation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ──────────────────────────────────────────────
# 1. Utility functions
# ──────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

ENV_NAME="tttlrm"
BUILD_LOG="${SCRIPT_DIR}/build.log"

CONDA_CMD=""
CONDA_ENVS_DIR=""

detect_conda() {
    if command -v mamba &>/dev/null; then
        CONDA_CMD="mamba"
    elif command -v conda &>/dev/null; then
        CONDA_CMD="conda"
    else
        log_error "Neither mamba nor conda found. Please install miniforge or miniconda."
        exit 1
    fi
    CONDA_ENVS_DIR="$(conda info --base 2>/dev/null | tr -d '[:space:]')/envs"
    log_info "Using $CONDA_CMD (envs: $CONDA_ENVS_DIR)"
}

env_exists() {
    local name="$1"
    [ -d "${CONDA_ENVS_DIR}/${name}" ]
}

create_env() {
    local name="$1"
    local python_version="$2"
    if env_exists "$name"; then
        log_warn "Environment '$name' already exists — skipping creation"
    else
        log_info "Creating environment '$name' (Python $python_version)..."
        $CONDA_CMD create -y -n "$name" python="$python_version" --quiet
        log_ok "Created '$name'"
    fi
}

pip_in_env() {
    local env="$1"; shift
    (cd "$SCRIPT_DIR" && "${CONDA_ENVS_DIR}/${env}/bin/pip" "$@")
}

python_in_env() {
    local env="$1"; shift
    "${CONDA_ENVS_DIR}/${env}/bin/python" "$@"
}

conda_install() {
    local env="$1"; shift
    $CONDA_CMD install -y -n "$env" "$@"
}

# ──────────────────────────────────────────────
# 2. Prerequisite checks
# ──────────────────────────────────────────────

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "${SCRIPT_DIR}/requirements.txt" ]; then
        log_error "requirements.txt not found — is this the tttLRM root?"
        exit 1
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi not found — GPU-dependent builds may fail"
    fi

    detect_conda
    log_ok "Prerequisites OK"
}

# ──────────────────────────────────────────────
# 3. Environment setup
# ──────────────────────────────────────────────

setup_tttlrm() {
    local ENV="$ENV_NAME"
    log_info "Setting up $ENV..."
    create_env "$ENV" 3.10 || return 1

    # 1. PyTorch + xformers (cu118 — matches upstream README)
    log_info "[$ENV] Installing PyTorch (cu118)..."
    pip_in_env "$ENV" install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers \
        --index-url https://download.pytorch.org/whl/cu118 || return 1

    # 2. CUDA toolkit 11.8 + C++ toolchain inside conda env
    #    Ensures native extension builds (flash-attn, gsplat, diff-gaussian-rasterization)
    #    use the same CUDA version as PyTorch, independent of system CUDA
    log_info "[$ENV] Installing CUDA toolkit 11.8, GCC toolchain..."
    conda_install "$ENV" \
        -c "nvidia/label/cuda-11.8.0" -c conda-forge \
        cuda-toolkit cuda-version=11.8 \
        gxx_linux-64=11 sysroot_linux-64=2.17 || return 1
    log_ok "[$ENV] CUDA toolkit 11.8 and GCC installed"

    # 3. Build tools
    log_info "[$ENV] Installing build tools..."
    pip_in_env "$ENV" install -U setuptools wheel packaging ninja || return 1

    # 4. Prepare environment for building native CUDA extensions
    : > "$BUILD_LOG"  # truncate log file
    local ENV_PREFIX="${CONDA_ENVS_DIR}/${ENV}"
    export CUDA_HOME="${ENV_PREFIX}"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
    export CPATH="${ENV_PREFIX}/include:${ENV_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
    export LIBRARY_PATH="${ENV_PREFIX}/lib:${LIBRARY_PATH:-}"

    # Suppress C++ warnings
    export CFLAGS="-w ${CFLAGS:-}"
    export CXXFLAGS="-w ${CXXFLAGS:-}"
    export NVCC_PREPEND_FLAGS="-Xcompiler -w"

    # 5. flash-attn (requires --no-build-isolation to link against installed torch)
    #    psutil is a build dependency of flash-attn but won't be auto-installed
    #    due to --no-build-isolation
    log_info "[$ENV] Installing flash-attn build dependencies..."
    pip_in_env "$ENV" install psutil || return 1
    log_info "[$ENV] Building flash-attn... (log: $BUILD_LOG)"
    pip_in_env "$ENV" install flash_attn==2.5.9.post1 --no-build-isolation \
        >> "$BUILD_LOG" 2>&1 || { log_error "flash-attn build failed — see $BUILD_LOG"; return 1; }
    log_ok "[$ENV] flash-attn installed"

    # 6. CUDA extensions that import torch at build time → --no-build-isolation
    log_info "[$ENV] Building diff-gaussian-rasterization... (log: $BUILD_LOG)"
    pip_in_env "$ENV" install --no-build-isolation \
        "diff_gaussian_rasterization @ git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d" \
        >> "$BUILD_LOG" 2>&1 || { log_error "diff-gaussian-rasterization build failed — see $BUILD_LOG"; return 1; }
    log_ok "[$ENV] diff-gaussian-rasterization installed"

    log_info "[$ENV] Building gsplat... (log: $BUILD_LOG)"
    pip_in_env "$ENV" install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat@v1.5.0" \
        >> "$BUILD_LOG" 2>&1 || { log_error "gsplat build failed — see $BUILD_LOG"; return 1; }
    log_ok "[$ENV] gsplat installed"

    # 7. Remaining Python dependencies
    #    --no-build-isolation is needed because requirements.txt still references
    #    the git-based CUDA packages; pip may re-resolve them and their setup.py
    #    imports torch, which is unavailable in an isolated build environment
    log_info "[$ENV] Installing requirements.txt... (log: $BUILD_LOG)"
    pip_in_env "$ENV" install --no-build-isolation -r "${SCRIPT_DIR}/requirements.txt" \
        >> "$BUILD_LOG" 2>&1 || { log_error "requirements.txt install failed — see $BUILD_LOG"; return 1; }
    log_ok "[$ENV] requirements.txt installed"

    # 7. Download model checkpoints
    log_info "[$ENV] Downloading model checkpoints..."
    if [ -d "${SCRIPT_DIR}/checkpoints" ] && [ "$(ls -A "${SCRIPT_DIR}/checkpoints" 2>/dev/null)" ]; then
        log_info "[$ENV] Checkpoints directory already exists — skipping download"
    else
        pip_in_env "$ENV" install -q huggingface_hub 2>/dev/null || true
        "${CONDA_ENVS_DIR}/${ENV}/bin/huggingface-cli" download chenwang/tttLRM \
            --local-dir "${SCRIPT_DIR}/checkpoints" --local-dir-use-symlinks False \
            || log_warn "[$ENV] Checkpoint download failed — you can run 'bash script/download_ckpts.sh' manually"
    fi

    # Depth Anything V2 weights
    local DA_WEIGHTS="${SCRIPT_DIR}/depth_anything_v2/depth_anything_vits.pth"
    if [ -f "$DA_WEIGHTS" ]; then
        log_info "[$ENV] Depth Anything V2 weights already present — skipping"
    else
        log_info "[$ENV] Downloading Depth Anything V2 weights..."
        wget -q --show-progress -O "$DA_WEIGHTS" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" \
            || log_warn "[$ENV] Depth Anything download failed — you can download manually"
    fi

    log_ok "$ENV setup complete"
}

# ──────────────────────────────────────────────
# 4. Environment check (--check)
# ──────────────────────────────────────────────

try_import() {
    local env="$1"; shift
    "${CONDA_ENVS_DIR}/${env}/bin/python" -c "$*" 2>/dev/null
}

check_tttlrm() {
    local ENV="$ENV_NAME"
    try_import "$ENV" "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
import flash_attn
import diff_gaussian_rasterization
import gsplat
print(f'OK (torch {torch.__version__}, CUDA {torch.version.cuda}, flash_attn {flash_attn.__version__})')
"
}

run_checks() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║      tttLRM Environment Check        ║"
    echo "╚══════════════════════════════════════╝"
    echo ""

    if ! env_exists "$ENV_NAME"; then
        log_error "$ENV_NAME — not found"
        return 1
    fi

    local result
    result=$( check_tttlrm 2>&1 ) && {
        log_ok "$ENV_NAME — $result"
    } || {
        log_error "$ENV_NAME — FAIL"
        if [ -n "$result" ]; then
            echo "       $result"
        fi
        return 1
    }

    # Model checkpoints
    if [ -d "${SCRIPT_DIR}/checkpoints" ] && [ "$(ls -A "${SCRIPT_DIR}/checkpoints" 2>/dev/null)" ]; then
        log_ok "Model checkpoints — present"
    else
        log_warn "Model checkpoints — not found (run: bash script/download_ckpts.sh)"
    fi

    local DA_WEIGHTS="${SCRIPT_DIR}/depth_anything_v2/depth_anything_vits.pth"
    if [ -f "$DA_WEIGHTS" ]; then
        log_ok "Depth Anything V2 weights — present"
    else
        log_warn "Depth Anything V2 weights — not found"
    fi

    echo ""
    log_ok "All checks passed"
}

# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

main() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║       tttLRM Environment Setup       ║"
    echo "╚══════════════════════════════════════╝"
    echo ""

    check_prerequisites

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if setup_tttlrm; then
        echo ""
        echo "╔══════════════════════════════════════╗"
        echo "║           Setup Summary              ║"
        echo "╚══════════════════════════════════════╝"
        echo ""
        log_ok "Succeeded: $ENV_NAME"
        echo ""
        echo "Next steps:"
        echo "  1. Activate the environment:"
        echo "       conda activate $ENV_NAME"
        echo "  2. Verify the installation:"
        echo "       ./setup.sh --check"
        echo "  3. Download checkpoints (if not done):"
        echo "       bash script/download_ckpts.sh"
        echo "  4. Download & convert DL3DV sample data:"
        echo "       python data/dl3dv_eval_download.py --odir ./data_example/dl3dv_benchmark \\"
        echo "           --subset hash --only_level4 \\"
        echo "           --hash 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
        echo "       python data/dl3dv_format_convert.py"
        echo "  5. Run inference:"
        echo "       bash script/inference_dl3dv.sh"
        echo ""
    else
        echo ""
        echo "╔══════════════════════════════════════╗"
        echo "║           Setup Summary              ║"
        echo "╚══════════════════════════════════════╝"
        echo ""
        log_error "Failed: $ENV_NAME"
        echo ""
        return 1
    fi
}

# Entry point: dispatch --check or run main setup
if [ "${1:-}" = "--check" ]; then
    detect_conda
    run_checks
else
    main "$@"
fi
