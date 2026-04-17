#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-}"
BATCH_NAME="${2:-}"

if [[ -z "${GPU_ID}" || -z "${BATCH_NAME}" ]]; then
  echo "Usage: $0 <gpu_id> <batch_name>"
  echo "batch_name: gpu0 | gpu1"
  exit 1
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ROOT="${GS_ROOT}"
LOG_DIR="${ROOT}/reports/ours_benchmark_eval/run_logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/tmux_wait_${BATCH_NAME}_${TS}.log"

# Conservative thresholds for 32GB cards.
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-12000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-30}"
POLL_SEC="${POLL_SEC:-20}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

gpu_uuid() {
  nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i "${GPU_ID}" | tr -d ' '
}

wait_gpu_idle() {
  local uuid
  uuid="$(gpu_uuid)"
  log "Waiting GPU${GPU_ID} idle (min_free=${MIN_FREE_MEM_MB}MB, max_util=${MAX_GPU_UTIL}%)"
  while true; do
    local free util
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')"
    if [[ -z "${free}" || -z "${util}" ]]; then
      log "nvidia-smi parse failed, retry in ${POLL_SEC}s"
      sleep "${POLL_SEC}"
      continue
    fi
    if (( free >= MIN_FREE_MEM_MB && util <= MAX_GPU_UTIL )); then
      log "GPU${GPU_ID} is idle enough: free=${free}MB util=${util}%"
      break
    fi
    log "GPU${GPU_ID} busy: free=${free}MB util=${util}%, sleep ${POLL_SEC}s"
    sleep "${POLL_SEC}"
  done
}

run_query() {
  local qid="$1"
  local qtext="$2"
  local run_dir="$3"
  local data_dir="$4"
  log "[RUN][GPU${GPU_ID}] ${qid}"
  (
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    QUERY_FORCE_RERUN=1 \
    QUERY_SKIP_QWEN_SELECTION=1 \
    QUERY_SKIP_QWEN_EXPORT=1 \
    QUERY_SKIP_ENTITY_LIBRARY=1 \
    bash scripts/run_query_specific_worldtube_pipeline.sh \
      "${run_dir}" "${data_dir}" "${qtext}" "${qid}"
  ) 2>&1 | tee -a "${LOG_FILE}"
}

run_batch_gpu0() {
  run_query "cook-spinach_q1" "正在翻炒菠菜的铲子。" \
    "runs/stellar_tube_ours_benchmark/dynerf/cook_spinach" "data/dynerf/cook_spinach"
  run_query "cook-spinach_q4" "炒锅中正在被翻动的菠菜。" \
    "runs/stellar_tube_ours_benchmark/dynerf/cook_spinach" "data/dynerf/cook_spinach"
  run_query "cook-spinach_q5" "木板上的红色生牛排。" \
    "runs/stellar_tube_ours_benchmark/dynerf/cook_spinach" "data/dynerf/cook_spinach"
  run_query "cut_roasted_beef_q1" "切烤牛肉的刀。" \
    "runs/stellar_tube_ours_benchmark/dynerf/cut_roasted_beef" "data/dynerf/cut_roasted_beef"
  run_query "cut_roasted_beef_q4" "所有参与切肉动作的物体。" \
    "runs/stellar_tube_ours_benchmark/dynerf/cut_roasted_beef" "data/dynerf/cut_roasted_beef"
  run_query "cut_roasted_beef_q5" "砧板上的整条未切牛里脊。" \
    "runs/stellar_tube_ours_benchmark/dynerf/cut_roasted_beef" "data/dynerf/cut_roasted_beef"
  run_query "coffee_martini_q7" "除了黑色水壶和木碗之外的所有物体。" \
    "runs/stellar_tube_ours_benchmark_coffee_martini/dynerf/coffee_martini" "data/dynerf/coffee_martini"
  run_query "coffee_martini_q2" "倾倒任何咖啡之前的金属杯。" \
    "runs/stellar_tube_ours_benchmark_coffee_martini/dynerf/coffee_martini" "data/dynerf/coffee_martini"
}

run_batch_gpu1() {
  run_query "flame_salmon_q1" "正在炙烤三文鱼的喷枪喷嘴。" \
    "runs/stellar_tube_ours_benchmark/dynerf/flame_salmon_1" "data/dynerf/flame_salmon_1"
  run_query "flame_salmon_q4" "参与炙烤动作的所有物体。" \
    "runs/stellar_tube_ours_benchmark/dynerf/flame_salmon_1" "data/dynerf/flame_salmon_1"
  run_query "flame_salmon_q5" "桌面上未被加热的整块牛排。" \
    "runs/stellar_tube_ours_benchmark/dynerf/flame_salmon_1" "data/dynerf/flame_salmon_1"
  run_query "sear_steak_q2" "正在被火焰炙烤的牛排区域。" \
    "runs/stellar_tube_ours_benchmark/dynerf/sear_steak" "data/dynerf/sear_steak"
  run_query "sear_steak_q5" "未参与炙烤的白色瓷盘。" \
    "runs/stellar_tube_ours_benchmark/dynerf/sear_steak" "data/dynerf/sear_steak"
  run_query "sear_steak_q6" "除了喷枪之外所有正在运动的物体。" \
    "runs/stellar_tube_ours_benchmark/dynerf/sear_steak" "data/dynerf/sear_steak"
  run_query "flame_steak_q1" "正在自由移动以均匀炙烤牛排的喷枪。" \
    "runs/stellar_tube_ours_benchmark_flame_steak/dynerf/flame_steak" "data/dynerf/flame_steak"
  run_query "flame_steak_q6" "在喷枪炙烤过程中翻动牛排的手。" \
    "runs/stellar_tube_ours_benchmark_flame_steak/dynerf/flame_steak" "data/dynerf/flame_steak"
  run_query "flame_steak_q7" "除了菠菜碗和绿色瓶子之外的所有物体。" \
    "runs/stellar_tube_ours_benchmark_flame_steak/dynerf/flame_steak" "data/dynerf/flame_steak"
  run_query "coffee_martini_q6" "倾倒过程中握持马提尼酒杯的手。" \
    "runs/stellar_tube_ours_benchmark_coffee_martini/dynerf/coffee_martini" "data/dynerf/coffee_martini"
}

wait_gpu_idle

case "${BATCH_NAME}" in
  gpu0) run_batch_gpu0 ;;
  gpu1) run_batch_gpu1 ;;
  *)
    log "Unknown batch_name: ${BATCH_NAME}"
    exit 1
    ;;
esac

log "Batch ${BATCH_NAME} finished."
