#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/data/hypernerf"
DOWNLOAD_ROOT="${ROOT_DIR}/data/downloads"
GROUP="${HYPERNERF_GROUP:-virg}"
SCENE="${HYPERNERF_SCENE:-broom2}"
ASSET_NAME="${HYPERNERF_ASSET:-vrig_broom.zip}"
TARGET_DIR="${DATA_ROOT}/${GROUP}/${SCENE}"
ZIP_PATH="${DOWNLOAD_ROOT}/${ASSET_NAME}"
URL="https://github.com/google/hypernerf/releases/download/v0.1/${ASSET_NAME}"

mkdir -p "${DATA_ROOT}" "${DOWNLOAD_ROOT}" "$(dirname "${TARGET_DIR}")"
if [[ ! -f "${ZIP_PATH}" ]]; then
  wget -O "${ZIP_PATH}" "${URL}"
fi

python - <<'PY' "${ZIP_PATH}" "${TARGET_DIR}"
import os
import shutil
import sys
import tempfile
import zipfile

zip_path, target_dir = sys.argv[1], sys.argv[2]
temp_dir = tempfile.mkdtemp(prefix="hypernerf_extract_", dir=os.path.dirname(target_dir))
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(temp_dir)

scene_root = None
for root, _dirs, files in os.walk(temp_dir):
    if {"dataset.json", "metadata.json", "scene.json"}.issubset(set(files)):
        scene_root = root
        break
if scene_root is None:
    raise SystemExit(f"Unable to locate HyperNeRF scene root in {zip_path}")

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.move(scene_root, target_dir)
shutil.rmtree(temp_dir)
PY

if [[ ! -f "${TARGET_DIR}/points3D_downsample2.ply" ]]; then
  if command -v colmap >/dev/null 2>&1; then
    bash "${ROOT_DIR}/external/4DGaussians/colmap.sh" "${TARGET_DIR}" hypernerf
    PY_CMD=(python)
    if [[ "${CONDA_PREFIX:-}" != "${GS_ENV_PATH}" ]]; then
      PY_CMD=(env CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" conda run --no-capture-output -p "${GS_ENV_PATH}" python)
    fi
    "${PY_CMD[@]}" "${ROOT_DIR}/external/4DGaussians/scripts/downsample_point.py" \
      "${TARGET_DIR}/colmap/dense/workspace/fused.ply" \
      "${TARGET_DIR}/points3D_downsample2.ply"
  else
    echo "points3D_downsample2.ply is missing and 'colmap' is not available." >&2
    echo "Install COLMAP or place a pregenerated point cloud at ${TARGET_DIR}/points3D_downsample2.ply." >&2
    exit 2
  fi
fi

echo "Prepared HyperNeRF scene at ${TARGET_DIR}"
