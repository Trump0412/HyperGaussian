#!/usr/bin/env bash
# run_missing_queries_and_train.sh
# Comprehensive script to:
# 1. Re-run negative queries with improved Qwen prompt (select + render only)
# 2. Complete coffee_martini queries (from Qwen export step)
# 3. Run flame_steak queries (full pipeline)
# 4. Run espresso_q4 and split_cookie_q7
# 5. Start Neu3D scene training

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

GS="${GS_ROOT}"
REPORT_DIR="${GS}/reports/ours_benchmark_eval"
LOG_DIR="${REPORT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# Helper: Re-run select_qwen + render_query for a query
# ============================================================
rerun_select_and_render() {
  local GPU_ID="$1"
  local RUN_DIR="$2"
  local DATASET_DIR="$3"
  local QUERY_TEXT="$4"
  local QUERY_NAME="$5"
  local LOG_FILE="$6"

  local OUTPUT_ROOT="${RUN_DIR}/entitybank/query_guided/${QUERY_NAME}"
  local QUERY_RUN_DIR="${OUTPUT_ROOT}/query_worldtube_run"
  local ASSIGNMENTS_PATH="${QUERY_RUN_DIR}/entitybank/semantic_assignments_qwen.json"
  local NATIVE_ASSIGNMENTS_PATH="${QUERY_RUN_DIR}/entitybank/native_semantic_assignments.json"
  local QWEN_SELECTION_PATH="${QUERY_RUN_DIR}/entitybank/selected_query_qwen.json"
  local FINAL_RENDER_DIR="${OUTPUT_ROOT}/final_query_render_sourcebg"

  log "Re-running select+render for ${QUERY_NAME} on GPU${GPU_ID}"

  # Determine assignments path (use qwen if available, else native)
  if [[ -f "${ASSIGNMENTS_PATH}" ]]; then
    local USE_ASSIGNMENTS="${ASSIGNMENTS_PATH}"
  elif [[ -f "${NATIVE_ASSIGNMENTS_PATH}" ]]; then
    local USE_ASSIGNMENTS="${NATIVE_ASSIGNMENTS_PATH}"
  else
    log "ERROR: No assignments found for ${QUERY_NAME}"
    return 1
  fi

  # Re-run select_qwen_query_entities.py
  CUDA_VISIBLE_DEVICES="${GPU_ID}" gsam2_python "${GS}/scripts/select_qwen_query_entities.py" \
    --assignments-path "${USE_ASSIGNMENTS}" \
    --query "${QUERY_TEXT}" \
    --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
    --output-path "${QWEN_SELECTION_PATH}"

  # Re-run render_query_video.py
  CUDA_VISIBLE_DEVICES="${GPU_ID}" gs_python "${GS}/scripts/render_query_video.py" \
    --run-dir "${QUERY_RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --selection-path "${QWEN_SELECTION_PATH}" \
    --output-dir "${FINAL_RENDER_DIR}" \
    --background-mode source \
    --fps 6 \
    --stride 1

  log "Done: ${QUERY_NAME}"
}

# ============================================================
# Re-run negative queries with improved Qwen prompt
# (These already have query_worldtube_run set up)
# ============================================================
run_negative_query_fixes() {
  local GPU_ID="${1:-0}"
  log "=== Re-running negative queries with improved Qwen prompt (GPU${GPU_ID}) ==="

  # espresso_q5
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_4dlangsplat_refresh_20260328_espresso" \
    "${GS}/data/hypernerf/misc/espresso" \
    "桌子上的蓝色玻璃杯" \
    "espresso_q5" \
    "${LOG_DIR}/espresso_q5_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/espresso_q5_rerun.log"

  # americano_q7
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_4dlangsplat_refresh_20260328_americano" \
    "${GS}/data/hypernerf/misc/americano" \
    "托盘上的蓝色玻璃杯" \
    "americano_q7" \
    "${LOG_DIR}/americano_q7_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/americano_q7_rerun.log"

  # keyboard_q4
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_ours_benchmark_keyboard" \
    "${GS}/data/hypernerf/interp/keyboard" \
    "红色机械键盘" \
    "keyboard_q4" \
    "${LOG_DIR}/keyboard_q4_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/keyboard_q4_rerun.log"

  # torchchocolate_q5
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_ours_benchmark_torchocolate" \
    "${GS}/data/hypernerf/misc/torchocolate" \
    "完全凝固且坚硬的白色巧克力" \
    "torchchocolate_q5" \
    "${LOG_DIR}/torchchocolate_q5_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/torchchocolate_q5_rerun.log"

  # split_cookie_q5
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032" \
    "${GS}/data/hypernerf/misc/split-cookie" \
    "木板上的黑色曲奇饼干" \
    "split_cookie_q5" \
    "${LOG_DIR}/split_cookie_q5_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/split_cookie_q5_rerun.log"

  log "=== Negative query re-runs complete ==="
}

# ============================================================
# Re-run low-scoring positive queries
# ============================================================
run_low_score_fixes() {
  local GPU_ID="${1:-1}"
  log "=== Re-running low-scoring positive queries (GPU${GPU_ID}) ==="

  # cut_lemon_q1 - Acc=12.53% - need to check why
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_cutlemon_refresh_20260329" \
    "${GS}/data/hypernerf/interp/cut-lemon1" \
    "正在切割柠檬的刀" \
    "cut_lemon_q1" \
    "${LOG_DIR}/cut_lemon_q1_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/cut_lemon_q1_rerun.log"

  # americano_q5 - Acc=7.63% - "always stationary objects"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" rerun_select_and_render "${GPU_ID}" \
    "${GS}/runs/stellar_tube_4dlangsplat_refresh_20260328_americano" \
    "${GS}/data/hypernerf/misc/americano" \
    "在整个视频中始终保持静止的所有物体" \
    "americano_q5" \
    "${LOG_DIR}/americano_q5_rerun.log" \
    2>&1 | tee -a "${LOG_DIR}/americano_q5_rerun.log"

  log "=== Low-score re-runs complete ==="
}

# ============================================================
# Run espresso_q4 and split_cookie_q7 full pipeline
# ============================================================
run_incomplete_queries() {
  local GPU_ID="${1:-0}"
  log "=== Running espresso_q4 and split_cookie_q7 (GPU${GPU_ID}) ==="

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"

  # espresso_q4: "在整个视频中物理位置始终保持静止的所有物体"
  QUERY_FORCE_RERUN=1 bash "${GS}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${GS}/runs/stellar_tube_4dlangsplat_refresh_20260328_espresso" \
    "${GS}/data/hypernerf/misc/espresso" \
    "在整个视频中物理位置始终保持静止的所有物体" \
    "espresso_q4" \
    2>&1 | tee -a "${LOG_DIR}/espresso_q4.log"

  # split_cookie_q7: "除了正在施力的手之外的所有物体"
  QUERY_FORCE_RERUN=1 bash "${GS}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${GS}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032" \
    "${GS}/data/hypernerf/misc/split-cookie" \
    "除了正在施力的手之外的所有物体" \
    "split_cookie_q7" \
    2>&1 | tee -a "${LOG_DIR}/split_cookie_q7.log"

  log "=== Incomplete queries complete ==="
}

# ============================================================
# Run coffee_martini queries (continue from Qwen export)
# ============================================================
run_coffee_martini_queries() {
  local GPU_ID="${1:-1}"
  log "=== Running coffee_martini queries (GPU${GPU_ID}) ==="

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  CM_RUN="${GS}/runs/stellar_tube_ours_benchmark_coffee_martini"
  CM_DATA="${GS}/data/dynerf/coffee_martini"

  # coffee_martini_q7: "除了黑色水壶和木碗之外的所有物体" - need full pipeline (no query_worldtube_run)
  QUERY_FORCE_RERUN=1 bash "${GS}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${CM_RUN}" "${CM_DATA}" \
    "除了黑色水壶和木碗之外的所有物体" \
    "coffee_martini_q7" \
    2>&1 | tee -a "${LOG_DIR}/coffee_martini_q7.log"

  # coffee_martini_q2: "倾倒任何咖啡之前的金属杯" - has query_worldtube_run, re-run from qwen
  rerun_select_and_render "${GPU_ID}" \
    "${CM_RUN}" "${CM_DATA}" \
    "倾倒任何咖啡之前的金属杯" \
    "coffee_martini_q2" \
    "${LOG_DIR}/coffee_martini_q2.log" \
    2>&1 | tee -a "${LOG_DIR}/coffee_martini_q2.log"

  # coffee_martini_q6: "倾倒过程中握持马提尼酒杯的手" - has query_worldtube_run, re-run from qwen
  rerun_select_and_render "${GPU_ID}" \
    "${CM_RUN}" "${CM_DATA}" \
    "倾倒过程中握持马提尼酒杯的手" \
    "coffee_martini_q6" \
    "${LOG_DIR}/coffee_martini_q6.log" \
    2>&1 | tee -a "${LOG_DIR}/coffee_martini_q6.log"

  log "=== coffee_martini queries complete ==="
}

# ============================================================
# Run flame_steak queries (full pipeline)
# ============================================================
run_flame_steak_queries() {
  local GPU_ID="${1:-0}"
  log "=== Running flame_steak queries (GPU${GPU_ID}) ==="

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  FS_RUN="${GS}/runs/stellar_tube_ours_benchmark_flame_steak"
  FS_DATA="${GS}/data/dynerf/flame_steak"

  # flame_steak_q7: "除了菠菜碗和绿色瓶子之外的所有物体"
  QUERY_FORCE_RERUN=1 bash "${GS}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${FS_RUN}" "${FS_DATA}" \
    "除了菠菜碗和绿色瓶子之外的所有物体" \
    "flame_steak_q7" \
    2>&1 | tee -a "${LOG_DIR}/flame_steak_q7.log"

  # flame_steak_q1: "正在自由移动以均匀炙烤牛排的喷枪"
  QUERY_FORCE_RERUN=1 bash "${GS}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${FS_RUN}" "${FS_DATA}" \
    "正在自由移动以均匀炙烤牛排的喷枪" \
    "flame_steak_q1" \
    2>&1 | tee -a "${LOG_DIR}/flame_steak_q1.log"

  # flame_steak_q6: "在喷枪炙烤过程中翻动牛排的手"
  QUERY_FORCE_RERUN=1 bash "${GS}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${FS_RUN}" "${FS_DATA}" \
    "在喷枪炙烤过程中翻动牛排的手" \
    "flame_steak_q6" \
    2>&1 | tee -a "${LOG_DIR}/flame_steak_q6.log"

  log "=== flame_steak queries complete ==="
}

# ============================================================
# Start Neu3D scene training
# ============================================================
start_neu3d_training() {
  log "=== Starting Neu3D scene training ==="

  # GPU0: cook_spinach then cut_roasted_beef
  (
    log "GPU0: Training cook_spinach..."
    CUDA_VISIBLE_DEVICES=0 GS_PORT=6030 GS_RUN_NAMESPACE=stellar_tube_ours_benchmark \
      bash "${GS}/scripts/train_stellar_tube.sh" dynerf cook_spinach \
      2>&1 | tee -a "${LOG_DIR}/train_cook_spinach.log"

    log "GPU0: Training cut_roasted_beef..."
    CUDA_VISIBLE_DEVICES=0 GS_PORT=6031 GS_RUN_NAMESPACE=stellar_tube_ours_benchmark \
      bash "${GS}/scripts/train_stellar_tube.sh" dynerf cut_roasted_beef \
      2>&1 | tee -a "${LOG_DIR}/train_cut_roasted_beef.log"

    log "GPU0: Training sear_steak..."
    CUDA_VISIBLE_DEVICES=0 GS_PORT=6032 GS_RUN_NAMESPACE=stellar_tube_ours_benchmark \
      bash "${GS}/scripts/train_stellar_tube.sh" dynerf sear_steak \
      2>&1 | tee -a "${LOG_DIR}/train_sear_steak.log"
  ) &
  log "GPU0: Neu3D training queue started (PID $!)"

  # GPU1: flame_salmon_1
  (
    log "GPU1: Training flame_salmon_1..."
    CUDA_VISIBLE_DEVICES=1 GS_PORT=6033 GS_RUN_NAMESPACE=stellar_tube_ours_benchmark \
      bash "${GS}/scripts/train_stellar_tube.sh" dynerf flame_salmon_1 \
      2>&1 | tee -a "${LOG_DIR}/train_flame_salmon_1.log"
  ) &
  log "GPU1: flame_salmon_1 training started (PID $!)"
}

# ============================================================
# Main execution
# ============================================================
CMD="${1:-all}"

case "${CMD}" in
  negatives)
    run_negative_query_fixes 0
    ;;
  low_score)
    run_low_score_fixes 1
    ;;
  espresso_split)
    run_incomplete_queries 0
    ;;
  coffee)
    run_coffee_martini_queries 1
    ;;
  flame_steak)
    run_flame_steak_queries 0
    ;;
  train_neu3d)
    start_neu3d_training
    ;;
  all)
    # Run all in parallel on both GPUs
    # GPU0: negatives + espresso_split + flame_steak
    # GPU1: low_score + coffee
    log "=== Starting all jobs in parallel ==="

    (
      run_negative_query_fixes 0
      run_incomplete_queries 0
      run_flame_steak_queries 0
    ) >> "${LOG_DIR}/gpu0_all.log" 2>&1 &
    GPU0_PID=$!
    log "GPU0 batch started (PID ${GPU0_PID})"

    (
      run_low_score_fixes 1
      run_coffee_martini_queries 1
    ) >> "${LOG_DIR}/gpu1_all.log" 2>&1 &
    GPU1_PID=$!
    log "GPU1 batch started (PID ${GPU1_PID})"

    log "Both GPU batches running. Monitor:"
    log "  tail -f ${LOG_DIR}/gpu0_all.log"
    log "  tail -f ${LOG_DIR}/gpu1_all.log"
    ;;
  *)
    echo "Usage: $0 [all|negatives|low_score|espresso_split|coffee|flame_steak|train_neu3d]"
    exit 1
    ;;
esac
