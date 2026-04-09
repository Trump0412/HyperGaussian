#!/usr/bin/env bash
# train_ours_benchmark_gpu0.sh
# GPU0 串行训练队列：keyboard → coffee_martini
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPORT_DIR="${GS_ROOT}/reports/ours_benchmark_eval"
mkdir -p "${REPORT_DIR}"
LOG="${REPORT_DIR}/train_gpu0.log"
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
export CUDA_VISIBLE_DEVICES=0

COMMON_TRAIN_ARGS=(
  "--iterations" "14000"
  "--coarse_iterations" "3000"
  "--test_iterations" "3000" "7000" "14000"
  "--save_iterations" "7000" "14000"
  "--checkpoint_iterations" "7000" "14000"
)

# ===========================================================
# 1. keyboard (HyperNeRF)
# ===========================================================
KEYBOARD_NAMESPACE="stellar_tube_ours_benchmark_keyboard"
KEYBOARD_RUN_DIR="${GS_ROOT}/runs/${KEYBOARD_NAMESPACE}/hypernerf/keyboard"

if [[ -d "${KEYBOARD_RUN_DIR}/point_cloud" ]] && ls "${KEYBOARD_RUN_DIR}/point_cloud/iteration_"* &>/dev/null; then
  log_msg "keyboard 已有训练结果，跳过"
else
  log_msg "=== 开始训练 keyboard (GPU0) ==="
  GS_RUN_NAMESPACE="${KEYBOARD_NAMESPACE}" GS_PORT=6401 \
    bash "${GS_ROOT}/scripts/train_stellar_tube.sh" \
    hypernerf misc/keyboard "${COMMON_TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG}"
  log_msg "keyboard 训练完成"
fi

# 导出 entitybank
if [[ ! -f "${KEYBOARD_RUN_DIR}/entitybank/entities.json" ]]; then
  log_msg "导出 keyboard entitybank..."
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${KEYBOARD_RUN_DIR}" 2>&1 | tee -a "${LOG}"
fi

# ===========================================================
# 2. coffee_martini (dynerf) - 完整 14000 iter weaktube
# ===========================================================
CM_NAMESPACE="stellar_tube_ours_benchmark_coffee_martini"
CM_RUN_DIR="${GS_ROOT}/runs/${CM_NAMESPACE}/dynerf/coffee_martini"

if [[ -d "${CM_RUN_DIR}/point_cloud" ]] && ls "${CM_RUN_DIR}/point_cloud/iteration_"* &>/dev/null; then
  log_msg "coffee_martini 已有训练结果，跳过"
else
  log_msg "=== 开始训练 coffee_martini (GPU0) ==="
  GS_PORT=6403 GS_RUN_NAMESPACE="${CM_NAMESPACE}" \
    bash "${GS_ROOT}/scripts/train_stellar_tube.sh" \
    dynerf coffee_martini "${COMMON_TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG}"
  log_msg "coffee_martini 训练完成"
fi

if [[ ! -f "${CM_RUN_DIR}/entitybank/entities.json" ]]; then
  log_msg "导出 coffee_martini entitybank..."
  gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
    --run-dir "${CM_RUN_DIR}" 2>&1 | tee -a "${LOG}"
fi

log_msg "GPU0 训练队列全部完成"
