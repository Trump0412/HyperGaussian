#!/usr/bin/env bash
# ============================================================
# 4DGS Baseline Training Queue  (标准 4DGaussians, 无 stellar_tube)
# 12 个场景，GPU0/GPU1 串行（每张卡按顺序跑，两卡并行不同组）
# 指标：PSNR(dB)↑  D-SSIM↓  LPIPS↓  Time↓  FPS↑  Storage(MB)↓
# ============================================================
set -euo pipefail

GS_ROOT="/root/autodl-tmp/GaussianStellar"
REPORT_DIR="${GS_ROOT}/reports/4dgs_baseline"
mkdir -p "${REPORT_DIR}"
LOG="${REPORT_DIR}/train_queue.log"

source "${GS_ROOT}/scripts/common.sh"
PY_CMD="$(gs_python_cmd)"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

# ──────────────────────────────────────────────
# 场景列表  格式: "DATASET SCENE_PATH CONFIG_SCENE"
# ──────────────────────────────────────────────
GPU0_SCENES=(
  "hypernerf misc/espresso espresso"
  "hypernerf misc/americano americano"
  "hypernerf interp/cut-lemon1 cut-lemon1"
  "hypernerf misc/split-cookie split-cookie"
  "hypernerf misc/keyboard keyboard"
  "hypernerf interp/torchocolate torchocolate"
)

GPU1_SCENES=(
  "dynerf coffee_martini coffee_martini"
  "dynerf flame_steak flame_steak"
  "dynerf cook_spinach cook_spinach"
  "dynerf cut_roasted_beef cut_roasted_beef"
  "dynerf sear_steak sear_steak"
  "dynerf flame_salmon_1 flame_salmon_1"
)

train_and_eval() {
  local gpu_id="$1"
  local dataset="$2"
  local scene_path="$3"      # e.g. misc/espresso  or  coffee_martini
  local config_scene="$4"    # scene name for config file

  local scene_name="${scene_path##*/}"
  local source_path="${GS_ROOT}/data/${dataset}/${scene_path}"
  local run_dir="${GS_ROOT}/runs/baseline_4dgs_20260330/${dataset}/${scene_name}"
  local port="$((7200 + gpu_id * 100 + RANDOM % 50))"

  # config 路径
  local config_file="${GS_ROOT}/external/4DGaussians/arguments/${dataset}/${config_scene}.py"
  if [[ ! -f "${config_file}" ]]; then
    config_file="${GS_ROOT}/external/4DGaussians/arguments/${dataset}/default.py"
  fi

  log "[GPU${gpu_id}] 开始训练 ${dataset}/${scene_name}"
  log "[GPU${gpu_id}]   src:  ${source_path}"
  log "[GPU${gpu_id}]   run:  ${run_dir}"
  log "[GPU${gpu_id}]   cfg:  ${config_file}"

  if [[ -f "${run_dir}/point_cloud/iteration_14000/point_cloud.ply" ]]; then
    log "[GPU${gpu_id}] ✅ 已有 14000-iter checkpoint，跳过训练"
  else
    local train_start
    train_start=$(date +%s)

    CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "
      cd '${GS_ROOT}'
      export PYTHONPATH='${PYTHONPATH}'
      ${PY_CMD} external/4DGaussians/train.py \
        -s '${source_path}' \
        -m '${run_dir}' \
        --expname 'baseline_4dgs/${dataset}/${scene_name}' \
        --configs '${config_file}' \
        --port '${port}'
    "

    local train_end
    train_end=$(date +%s)
    local train_sec=$(( train_end - train_start ))
    log "[GPU${gpu_id}] ✅ 训练完成 ${scene_name}  耗时 ${train_sec}s"
    echo "{\"scene\": \"${scene_name}\", \"dataset\": \"${dataset}\", \"train_seconds\": ${train_sec}}" \
      >> "${REPORT_DIR}/train_times.jsonl"
  fi

  # ---- render ----
  local render_done="${run_dir}/test/ours_14000/renders"
  if [[ -d "${render_done}" && "$(ls -A "${render_done}" 2>/dev/null | wc -l)" -gt 0 ]]; then
    log "[GPU${gpu_id}] ✅ 已有 render，跳过"
  else
    log "[GPU${gpu_id}] render ${scene_name} ..."
    CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "
      cd '${GS_ROOT}'
      export PYTHONPATH='${PYTHONPATH}'
      ${PY_CMD} external/4DGaussians/render.py -m '${run_dir}' --iteration -1
    "
    log "[GPU${gpu_id}] ✅ render 完成"
  fi

  # ---- metrics ----
  local metrics_done="${run_dir}/results.json"
  if [[ -f "${metrics_done}" ]]; then
    log "[GPU${gpu_id}] ✅ 已有 metrics，跳过"
  else
    log "[GPU${gpu_id}] metrics ${scene_name} ..."
    CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "
      cd '${GS_ROOT}'
      export PYTHONPATH='${PYTHONPATH}'
      ${PY_CMD} external/4DGaussians/metrics.py -m '${run_dir}'
    "
    log "[GPU${gpu_id}] ✅ metrics 完成"
  fi

  # ---- 采集 Storage + FPS ----
  local ply="${run_dir}/point_cloud/iteration_14000/point_cloud.ply"
  local storage_mb="N/A"
  if [[ -f "${ply}" ]]; then
    storage_mb=$(du -sm "${ply}" | awk '{print $1}')
  fi

  # 从 results.json 提取 PSNR / SSIM / LPIPS
  local psnr="N/A" ssim="N/A" lpips="N/A"
  if [[ -f "${metrics_done}" ]]; then
    psnr=$(python3 -c "import json; d=json.load(open('${metrics_done}')); print(round(list(d.values())[-1].get('PSNR',0),4))" 2>/dev/null || echo "N/A")
    ssim=$(python3 -c "import json; d=json.load(open('${metrics_done}')); print(round(list(d.values())[-1].get('SSIM',0),4))" 2>/dev/null || echo "N/A")
    lpips=$(python3 -c "import json; d=json.load(open('${metrics_done}')); print(round(list(d.values())[-1].get('LPIPS',0),4))" 2>/dev/null || echo "N/A")
  fi

  local train_sec_recorded
  train_sec_recorded=$(grep "\"${scene_name}\"" "${REPORT_DIR}/train_times.jsonl" 2>/dev/null \
    | python3 -c "import sys,json; lines=[json.loads(l) for l in sys.stdin]; print(lines[-1]['train_seconds'] if lines else 'N/A')" 2>/dev/null || echo "N/A")

  echo "${dataset}/${scene_name} | PSNR=${psnr} | D-SSIM=${ssim} | LPIPS=${lpips} | Time=${train_sec_recorded}s | Storage=${storage_mb}MB" \
    >> "${REPORT_DIR}/summary.txt"
  log "[GPU${gpu_id}] 📊 ${dataset}/${scene_name}: PSNR=${psnr} SSIM=${ssim} LPIPS=${lpips} Storage=${storage_mb}MB"
}

# ──────────────────────────────────────────────
# 启动两组并行
# ──────────────────────────────────────────────
log "=== 4DGS Baseline 训练开始 ==="
log "GPU0: ${#GPU0_SCENES[@]} 场景 (HyperNeRF)"
log "GPU1: ${#GPU1_SCENES[@]} 场景 (DyNeRF/Neu3D)"

(
  for entry in "${GPU0_SCENES[@]}"; do
    ds=$(echo "${entry}" | awk '{print $1}')
    sp=$(echo "${entry}" | awk '{print $2}')
    cs=$(echo "${entry}" | awk '{print $3}')
    train_and_eval 0 "${ds}" "${sp}" "${cs}"
  done
  log "=== GPU0 全部完成 ==="
) >> "${LOG}" 2>&1 &
GPU0_PID=$!

(
  for entry in "${GPU1_SCENES[@]}"; do
    ds=$(echo "${entry}" | awk '{print $1}')
    sp=$(echo "${entry}" | awk '{print $2}')
    cs=$(echo "${entry}" | awk '{print $3}')
    train_and_eval 1 "${ds}" "${sp}" "${cs}"
  done
  log "=== GPU1 全部完成 ==="
) >> "${LOG}" 2>&1 &
GPU1_PID=$!

log "GPU0 PID: ${GPU0_PID}  GPU1 PID: ${GPU1_PID}"
log "日志: ${LOG}"
log "等待两组完成..."
wait "${GPU0_PID}" "${GPU1_PID}"
log "=== 全部4DGS baseline 训练完成 ==="
log ""
log "=== 汇总结果 ==="
cat "${REPORT_DIR}/summary.txt" | tee -a "${LOG}"
