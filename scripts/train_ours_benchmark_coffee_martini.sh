#!/usr/bin/env bash
# 使用最优weaktube配置训练 coffee_martini (14000 iter)
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export GS_RUN_NAMESPACE="stellar_tube_ours_benchmark_coffee_martini"
export GS_PORT="${GS_PORT:-6404}"

# weaktube 最优配置
export TEMPORAL_TUBE_SPAN="0.40"
export TEMPORAL_TUBE_SIGMA="0.32"
export TEMPORAL_TUBE_COVARIANCE_MIX="0.05"

LOG="${GS_ROOT}/reports/ours_benchmark_eval/train_coffee_martini.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date '+%H:%M:%S')] 开始训练 coffee_martini (weaktube, 14000 iter)" | tee -a "${LOG}"

bash "${GS_ROOT}/scripts/train_stellar_tube.sh" \
  dynerf \
  coffee_martini \
  --iterations 14000 \
  --coarse_iterations 3000 \
  --test_iterations 3000 7000 14000 \
  --save_iterations 7000 14000 \
  --checkpoint_iterations 7000 14000 2>&1 | tee -a "${LOG}"

echo "[$(date '+%H:%M:%S')] coffee_martini 训练完成" | tee -a "${LOG}"
