#!/usr/bin/env bash
# download_neu3d_scenes.sh
# 从 GitHub facebookresearch/Neural_3D_Video 下载 Neu3D 场景
# 下载 → 解压 → 抽帧 → 生成 camera JSON
set -euo pipefail

# !!!  取消所有代理，使用直连或镜像  !!!
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY no_proxy NO_PROXY

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATA_ROOT="${GS_ROOT}/data/dynerf"
TMP_DIR="${GS_ROOT}/data/downloads/neu3d_tmp"
REPORT_DIR="${GS_ROOT}/reports/ours_benchmark_eval"
LOG="${REPORT_DIR}/download_neu3d.log"

mkdir -p "${DATA_ROOT}" "${TMP_DIR}" "${REPORT_DIR}"

log_msg() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

# 使用 GitHub 镜像加速（gh-proxy.com 支持 aria2c 多线程）
MIRROR_PREFIX="https://gh-proxy.com/https://github.com"
BASE_URL="${MIRROR_PREFIX}/facebookresearch/Neural_3D_Video/releases/download/v1.0"

# aria2c 下载函数（16连接并行）
aria_download() {
  local url="$1"
  local out_dir="$2"
  local out_file="$3"
  aria2c \
    --no-conf \
    --continue=true \
    --split=16 \
    --max-connection-per-server=16 \
    --min-split-size=1M \
    --max-tries=5 \
    --retry-wait=3 \
    --timeout=60 \
    --all-proxy="" \
    --no-proxy="*" \
    "${url}" \
    --dir="${out_dir}" \
    --out="${out_file}" 2>&1 | tee -a "${LOG}"
}

# ============================================================
# 只下载 4 个缺失场景（coffee_martini 和 flame_steak 已有）
# ============================================================
declare -A SCENE_URLS=(
  ["cook_spinach"]="${BASE_URL}/cook_spinach.zip"
  ["cut_roasted_beef"]="${BASE_URL}/cut_roasted_beef.zip"
  ["sear_steak"]="${BASE_URL}/sear_steak.zip"
)
# flame_salmon_1 是分割压缩包，单独处理

# ============================================================
# 通用：下载单文件场景
# ============================================================
download_and_extract() {
  local scene="$1"
  local url="$2"
  local zip_path="${TMP_DIR}/${scene}.zip"
  local scene_dir="${DATA_ROOT}/${scene}"

  if [[ -d "${scene_dir}/cam00/images" ]] && \
     [[ $(ls "${scene_dir}/cam00/images/"*.png 2>/dev/null | wc -l) -gt 0 ]]; then
    log_msg "[skip] ${scene} 帧已存在 ($(ls "${scene_dir}/cam00/images/"*.png | wc -l) 帧)"
    return 0
  fi

  log_msg "[download] 开始下载 ${scene} (${url})"
  aria_download "${url}" "${TMP_DIR}" "${scene}.zip" || {
      log_msg "[error] 下载失败: ${scene}"
      return 1
    }
  log_msg "[download] 完成: ${zip_path} ($(du -sh "${zip_path}" | awk '{print $1}'))"

  log_msg "[extract] 解压 ${scene}..."
  unzip -o "${zip_path}" -d "${DATA_ROOT}" 2>&1 | tail -5 | tee -a "${LOG}"

  # 解压后可能在 DATA_ROOT/scene/ 或 DATA_ROOT/scene/scene/ 目录下
  if [[ -d "${DATA_ROOT}/${scene}/${scene}" ]]; then
    mv "${DATA_ROOT}/${scene}/${scene}" "${DATA_ROOT}/${scene}_inner"
    rm -rf "${DATA_ROOT}/${scene}"
    mv "${DATA_ROOT}/${scene}_inner" "${DATA_ROOT}/${scene}"
  fi

  log_msg "[extract] 完成: ${scene_dir}"
  extract_frames "${scene}"
  rm -f "${zip_path}"
}

# ============================================================
# 抽帧：从 mp4 提取 PNG 帧
# ============================================================
extract_frames() {
  local scene="$1"
  local scene_dir="${DATA_ROOT}/${scene}"

  # 找所有 camXX.mp4
  local cams_done=0
  for mp4 in "${scene_dir}"/cam*.mp4; do
    [[ -f "${mp4}" ]] || continue
    local cam_name
    cam_name=$(basename "${mp4}" .mp4)
    local img_dir="${scene_dir}/${cam_name}/images"

    if [[ -d "${img_dir}" ]] && [[ $(ls "${img_dir}"/*.png 2>/dev/null | wc -l) -gt 0 ]]; then
      log_msg "[skip] ${scene}/${cam_name} 帧已存在"
      ((cams_done++)) || true
      continue
    fi

    mkdir -p "${img_dir}"
    log_msg "[ffmpeg] 抽帧 ${scene}/${cam_name}..."
    ffmpeg -y -i "${mp4}" \
      -vf "fps=fps" \
      -q:v 1 \
      -start_number 0 \
      "${img_dir}/%04d.png" \
      2>&1 | tail -3 | tee -a "${LOG}" || {
        log_msg "[warn] ffmpeg 失败: ${scene}/${cam_name}"
        continue
      }
    local n_frames
    n_frames=$(ls "${img_dir}"/*.png 2>/dev/null | wc -l)
    log_msg "[ffmpeg] ${scene}/${cam_name}: ${n_frames} 帧"
    ((cams_done++)) || true
  done

  log_msg "[frames] ${scene}: ${cams_done} 个摄像头抽帧完成"

  # 生成 camera JSON
  log_msg "[camera] 为 ${scene} 生成 camera JSON..."
  gs_python "${GS_ROOT}/scripts/generate_dynerf_camera_jsons.py" \
    --dataset-dir "${scene_dir}" \
    --cam-index 0 2>&1 | tee -a "${LOG}" || log_msg "[warn] camera JSON 生成失败: ${scene}"
}

# ============================================================
# flame_salmon_1：分割压缩包，需要先合并再解压
# ============================================================
download_flame_salmon() {
  local scene="flame_salmon_1"
  local scene_dir="${DATA_ROOT}/${scene}"

  if [[ -d "${scene_dir}/cam00/images" ]] && \
     [[ $(ls "${scene_dir}/cam00/images/"*.png 2>/dev/null | wc -l) -gt 0 ]]; then
    log_msg "[skip] flame_salmon_1 帧已存在"
    return 0
  fi

  log_msg "[download] flame_salmon_1 (分割压缩包，4个部分共~4.7GB)"

  # 下载所有分割部分
  local parts=("flame_salmon_1_split.z01" "flame_salmon_1_split.z02" "flame_salmon_1_split.z03" "flame_salmon_1_split.zip")
  for part in "${parts[@]}"; do
    local part_path="${TMP_DIR}/${part}"
    if [[ -f "${part_path}" ]] && [[ $(stat -c%s "${part_path}") -gt 1000000 ]]; then
      log_msg "[skip] ${part} 已下载"
      continue
    fi
    log_msg "[download] 下载 ${part}..."
    aria_download "${BASE_URL}/${part}" "${TMP_DIR}" "${part}" || {
        log_msg "[error] 下载失败: ${part}"
        return 1
      }
    log_msg "[download] 完成: ${part}"
  done

  log_msg "[extract] 解压 flame_salmon_1 分割包..."
  # 合并并解压
  cd "${TMP_DIR}"
  zip -s 0 "flame_salmon_1_split.zip" --out "flame_salmon_1_combined.zip" 2>&1 | tee -a "${LOG}" || {
    # 如果 zip -s 不支持，用 cat 合并 + unzip
    log_msg "[warn] zip -s 失败，尝试 cat 合并..."
    cat flame_salmon_1_split.z01 flame_salmon_1_split.z02 flame_salmon_1_split.z03 flame_salmon_1_split.zip > flame_salmon_1_combined.zip || {
      log_msg "[error] 合并失败"
      cd -
      return 1
    }
  }
  unzip -o "flame_salmon_1_combined.zip" -d "${DATA_ROOT}" 2>&1 | tail -5 | tee -a "${LOG}"
  cd -

  # 整理目录
  if [[ -d "${DATA_ROOT}/flame_salmon_1/flame_salmon_1" ]]; then
    mv "${DATA_ROOT}/flame_salmon_1/flame_salmon_1" "${DATA_ROOT}/flame_salmon_1_inner"
    rm -rf "${DATA_ROOT}/flame_salmon_1"
    mv "${DATA_ROOT}/flame_salmon_1_inner" "${DATA_ROOT}/flame_salmon_1"
  fi

  extract_frames "flame_salmon_1"

  # 清理临时文件
  rm -f "${TMP_DIR}"/flame_salmon_1_*
}

# ============================================================
# 主流程
# ============================================================
log_msg "=== 开始下载缺失 Neu3D 场景 ==="
log_msg "目标目录: ${DATA_ROOT}"
log_msg "磁盘可用: $(df -h "${DATA_ROOT}" | awk 'NR==2 {print $4}')"

# 并行下载 3 个单文件场景
for scene in cook_spinach cut_roasted_beef sear_steak; do
  url="${SCENE_URLS[${scene}]}"
  download_and_extract "${scene}" "${url}" &
done

# flame_salmon_1 串行（因为需要顺序下载部分文件）
download_flame_salmon &

wait
log_msg "=== 全部下载+抽帧完成 ==="
log_msg "磁盘剩余: $(df -h "${DATA_ROOT}" | awk 'NR==2 {print $4}')"

# 验证
for scene in cook_spinach cut_roasted_beef flame_salmon_1 sear_steak; do
  n=$(ls "${DATA_ROOT}/${scene}/cam00/images/"*.png 2>/dev/null | wc -l)
  cam_jsons=$(ls "${DATA_ROOT}/${scene}/camera/"*.json 2>/dev/null | wc -l)
  log_msg "  ${scene}: ${n} 帧, ${cam_jsons} camera JSON"
done
