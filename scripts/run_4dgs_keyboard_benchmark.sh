#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_4dgaussians

PY="${GS4D_KEYBOARD_PY:-${GS_ENV_PATH}/bin/python}"
RUN_DIR="${GS_ROOT}/runs/baseline_4dgs_keyboard_benchmark_20260408"
SOURCE_PATH="${GS_ROOT}/data/hypernerf/misc/keyboard"
CONFIG_PATH="${GS_ROOT}/external/4DGaussians/arguments/hypernerf/default.py"
LOG_PATH="${RUN_DIR}/train.log"

# Set PYTHONPATH exactly as common.sh does
export PYTHONPATH="${GS_ROOT}:${GS_ROOT}/external/4DGaussians:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}"

echo "[$(date)] Starting 4DGS keyboard benchmark run" | tee -a "${LOG_PATH}"
echo "[$(date)] PYTHONPATH=${PYTHONPATH}" | tee -a "${LOG_PATH}"
echo "[$(date)] Using python: ${PY}" | tee -a "${LOG_PATH}"

START_TS=$(date +%s)

# Monitor GPU memory in background
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "${RUN_DIR}/gpu_mem_log.txt" &
GPU_MON_PID=$!

# Train
"${PY}" "${GS_ROOT}/external/4DGaussians/train.py" \
    -s "${SOURCE_PATH}" \
    -m "${RUN_DIR}" \
    --expname "baseline_4dgs_benchmark/hypernerf/keyboard" \
    --configs "${CONFIG_PATH}" \
    --port 6019 \
    2>&1 | tee -a "${LOG_PATH}"

TRAIN_END_TS=$(date +%s)
kill ${GPU_MON_PID} 2>/dev/null || true
TRAIN_SECONDS=$((TRAIN_END_TS - START_TS))
GPU_PEAK_MB=$(sort -rn "${RUN_DIR}/gpu_mem_log.txt" 2>/dev/null | head -1 || echo 0)
echo "[$(date)] Training done in ${TRAIN_SECONDS}s, GPU peak: ${GPU_PEAK_MB} MiB" | tee -a "${LOG_PATH}"

# Render (measure FPS)
echo "[$(date)] Starting render..." | tee -a "${LOG_PATH}"
RENDER_START=$(date +%s)

"${PY}" "${GS_ROOT}/external/4DGaussians/render.py" \
    -m "${RUN_DIR}" \
    --configs "${CONFIG_PATH}" \
    2>&1 | tee -a "${LOG_PATH}"

RENDER_END=$(date +%s)
RENDER_SECONDS=$((RENDER_END - RENDER_START))

# Count test frames
RENDERS_DIR="${RUN_DIR}/test/ours_14000/renders"
N_FRAMES=$(ls "${RENDERS_DIR}"/*.png 2>/dev/null | wc -l || echo 0)
RENDER_FPS_STR=$("${PY}" -c "print(round(${N_FRAMES}/max(${RENDER_SECONDS},1), 3))" 2>/dev/null || echo "N/A")
echo "[$(date)] Render: ${N_FRAMES} frames in ${RENDER_SECONDS}s = ${RENDER_FPS_STR} FPS" | tee -a "${LOG_PATH}"

# Compute quality metrics via collect_metrics
echo "[$(date)] Computing metrics..." | tee -a "${LOG_PATH}"
"${PY}" "${GS_ROOT}/scripts/collect_metrics.py" \
    --run-dir "${RUN_DIR}" \
    --write-summary 2>&1 | tee -a "${LOG_PATH}" || true

# Storage: point_cloud dir
PC_SIZE_BYTES=$(du -sb "${RUN_DIR}/point_cloud" 2>/dev/null | cut -f1 || echo 0)
PC_SIZE_MB=$(("${PC_SIZE_BYTES}" / 1024 / 1024))
echo "[$(date)] Storage (point_cloud): ${PC_SIZE_MB} MB" | tee -a "${LOG_PATH}"

# Write summary JSON
"${PY}" -c "
import json
summary = {
    'scene': 'keyboard',
    'method': '4DGS_baseline',
    'iterations': 14000,
    'train_seconds': ${TRAIN_SECONDS},
    'render_seconds': ${RENDER_SECONDS},
    'n_test_frames': ${N_FRAMES},
    'render_fps': ${RENDER_FPS_STR},
    'gpu_peak_mb': ${GPU_PEAK_MB},
    'point_cloud_mb': ${PC_SIZE_BYTES} / 1e6,
}
path = '${RUN_DIR}/benchmark_summary.json'
with open(path, 'w') as f:
    json.dump(summary, f, indent=2)
print('Written to', path)
print(json.dumps(summary, indent=2))
" 2>&1 | tee -a "${LOG_PATH}"

echo "[$(date)] All done." | tee -a "${LOG_PATH}"
