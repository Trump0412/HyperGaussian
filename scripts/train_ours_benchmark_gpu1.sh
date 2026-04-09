#!/usr/bin/env bash
# train_ours_benchmark_gpu1.sh
# GPU1 串行训练队列：torchocolate → flame_steak
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPORT_DIR="${GS_ROOT}/reports/ours_benchmark_eval"
mkdir -p "${REPORT_DIR}"
LOG="${REPORT_DIR}/train_gpu1.log"
: > "${LOG}"

log_msg() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

export TEMPORAL_TUBE_SAMPLES=3
export TEMPORAL_TUBE_SPAN=0.40
export TEMPORAL_TUBE_SIGMA=0.32
export TEMPORAL_TUBE_WEIGHT_POWER=1.0
export TEMPORAL_TUBE_COVARIANCE_MIX=0.05
export TEMPORAL_DRIFT_SCALE=1.0
export TEMPORAL_GATE_MIX=1.0
export TEMPORAL_DRIFT_MIX=1.0
export TEMPORAL_ACCELERATION_ENABLED=0
export TEMPORAL_VELOCITY_REG_WEIGHT=0.0
export TEMPORAL_ACCELERATION_REG_WEIGHT=0.0
export CUDA_VISIBLE_DEVICES=1

COMMON_TRAIN_ARGS=(
  "--iterations" "14000"
  "--coarse_iterations" "3000"
  "--test_iterations" "3000" "7000" "14000"
  "--save_iterations" "7000" "14000"
  "--checkpoint_iterations" "7000" "14000"
)

# ===========================================================
# 1. torchocolate (HyperNeRF/interp)
# ===========================================================
TORCH_NAMESPACE="stellar_tube_ours_benchmark_torchocolate"
TORCH_RUN_DIR="${GS_ROOT}/runs/${TORCH_NAMESPACE}/hypernerf/torchocolate"

if [[ -d "${TORCH_RUN_DIR}/point_cloud" ]] && ls "${TORCH_RUN_DIR}/point_cloud/iteration_"* &>/dev/null; then
  log_msg "torchocolate 已有训练结果，跳过"
else
  log_msg "=== 开始训练 torchocolate (GPU1) ==="
  GS_RUN_NAMESPACE="${TORCH_NAMESPACE}" GS_PORT=6402 \
    bash "${GS_ROOT}/scripts/train_stellar_tube.sh" \
    hypernerf interp/torchocolate "${COMMON_TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG}"
  log_msg "torchocolate 训练完成"
fi

if [[ ! -f "${TORCH_RUN_DIR}/entitybank/entities.json" ]]; then
  log_msg "导出 torchocolate entitybank..."
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${TORCH_RUN_DIR}" 2>&1 | tee -a "${LOG}"
fi

# ===========================================================
# 2. flame_steak (dynerf) - 重训 weaktube span040_sigma032
# ===========================================================
FS_NAMESPACE="stellar_tube_ours_benchmark_flame_steak"
FS_RUN_DIR="${GS_ROOT}/runs/${FS_NAMESPACE}/dynerf/flame_steak"

if [[ -d "${FS_RUN_DIR}/point_cloud" ]] && ls "${FS_RUN_DIR}/point_cloud/iteration_"* &>/dev/null; then
  log_msg "flame_steak (weaktube) 已有训练结果，跳过"
else
  log_msg "=== 开始训练 flame_steak weaktube (GPU1) ==="
  GS_RUN_NAMESPACE="${FS_NAMESPACE}" GS_PORT=6404 \
    bash "${GS_ROOT}/scripts/train_stellar_tube.sh" \
    dynerf flame_steak "${COMMON_TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG}"
  log_msg "flame_steak 训练完成"
fi

if [[ ! -f "${FS_RUN_DIR}/entitybank/entities.json" ]]; then
  log_msg "导出 flame_steak entitybank..."
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${FS_RUN_DIR}" 2>&1 | tee -a "${LOG}"
fi

log_msg "GPU1 训练队列全部完成"
