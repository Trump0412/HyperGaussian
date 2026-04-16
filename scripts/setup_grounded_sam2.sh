#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_conda_bin

GSAM2_ROOT="${GS_ROOT}/external/Grounded-SAM-2"
PYTHON_VERSION="${1:-3.10}"
TORCH_VERSION="${GSAM2_TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${GSAM2_TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${GSAM2_TORCHAUDIO_VERSION:-2.5.1}"
PIP_MIRROR="${GSAM2_PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
SAM2_MODEL_ID="${GSAM2_SAM2_MODEL_ID:-facebook/sam2-hiera-large}"
GDINO_MODEL_ID="${GSAM2_GDINO_MODEL_ID:-IDEA-Research/grounding-dino-base}"
INSTALL_EDITABLE="${GSAM2_INSTALL_EDITABLE:-0}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_MIRROR}"
export PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}"
export XDG_CACHE_HOME="${GS_CACHE_ROOT}"
export TORCH_HOME="${GS_TORCH_HOME}"
export MPLCONFIGDIR="${GS_MPLCONFIGDIR}"
export CUDA_HOME="${GS4D_CUDA_HOME:-/usr/local/cuda-12.1}"
export SAM2_BUILD_ALLOW_ERRORS=1

mkdir -p "$(dirname "${GSAM2_ENV_PATH}")"

if [[ ! -d "${GSAM2_ENV_PATH}" ]]; then
  "${GS_CONDA_BIN}" create -y -p "${GSAM2_ENV_PATH}" "python=${PYTHON_VERSION}" pip
fi

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install --upgrade pip setuptools wheel

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu121

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install -i "${PIP_MIRROR}" \
    "numpy<2" "transformers>=4.46" "huggingface_hub>=0.27" "pillow>=10" "tqdm>=4.66" \
    "hydra-core>=1.3.2" "iopath>=0.1.10" "opencv-python>=4.8" "supervision>=0.25" \
    "pyyaml>=6.0" "matplotlib>=3.8" "accelerate>=0.34" "sentencepiece>=0.2"

if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  pushd "${GSAM2_ROOT}" >/dev/null
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_HOME="${CUDA_HOME}" SAM2_BUILD_ALLOW_ERRORS=1 \
    "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip install --no-build-isolation -e .
  popd >/dev/null
fi

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_ENDPOINT="${HF_MIRROR}" \
  "${GS_CONDA_BIN}" run --no-capture-output -p "${GSAM2_ENV_PATH}" python - <<PY
import sys
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
sys.path.insert(0, "${GSAM2_ROOT}")
from sam2.sam2_video_predictor import SAM2VideoPredictor

gdino_model_id = "${GDINO_MODEL_ID}"
sam2_model_id = "${SAM2_MODEL_ID}"

processor = AutoProcessor.from_pretrained(gdino_model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id)
predictor = SAM2VideoPredictor.from_pretrained(sam2_model_id)

print("grounding processor", type(processor).__name__)
print("grounding model", type(grounding_model).__name__)
print("sam2 predictor", type(predictor).__name__)
print("gsam2 env ready:", "${GSAM2_ENV_PATH}")
PY
