#!/usr/bin/env bash
# download_neu3d_missing.sh
# 下载缺失的Neu3D/Neural-3D-Video场景数据
# 目标场景: cook_spinach, cut_roasted_beef, flame_salmon_1, sear_steak
# 数据来源: Neural 3D Video Synthesis (Facebook Research)
#
# 注意：该数据集较大，每个场景约10-30GB
# 需要先完成下载再进行COLMAP预处理

set -uo pipefail

DYNERF_ROOT="/root/autodl-tmp/HyperGaussian/data/dynerf"
REPORT_DIR="/root/autodl-tmp/HyperGaussian/reports/ours_benchmark_eval"
LOG="${REPORT_DIR}/download_neu3d.log"
mkdir -p "${DYNERF_ROOT}" "${REPORT_DIR}"

log_msg() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

log_msg "=== 下载 Neu3D 缺失场景 ==="

# 可用的数据源（按优先级排列）
# 1. HuggingFace mirror（可能有镜像）
# 2. 官方 GitHub 页面数据
# 3. 其他可访问的学术镜像

SCENES=(
  "cook_spinach"
  "cut_roasted_beef"
  "flame_salmon_1"
  "sear_steak"
)

# 检查磁盘空间
FREE_GB=$(df -BG /root/autodl-tmp | awk 'NR==2{print int($4)}')
log_msg "可用磁盘空间: ${FREE_GB} GB"

if [[ "${FREE_GB}" -lt 50 ]]; then
  log_msg "警告：磁盘空间不足50GB，下载可能失败"
fi

# 尝试从各个数据源下载
download_scene() {
  local scene="$1"
  local scene_dir="${DYNERF_ROOT}/${scene}"

  if [[ -d "${scene_dir}" ]] && [[ -f "${scene_dir}/poses_bounds.npy" ]]; then
    log_msg "${scene}: 已存在，跳过下载"
    return 0
  fi

  mkdir -p "${scene_dir}"
  log_msg "${scene}: 开始下载..."

  # 尝试从HuggingFace下载预处理好的数据
  # 注：neural-3d-video 数据集有HuggingFace镜像
  local hf_base="https://hf-mirror.com/datasets/facebook/neural-3d-video/resolve/main"
  local alt_url="https://huggingface.co/datasets/neural3d/video/resolve/main/${scene}.tar.gz"

  # 首先尝试检查是否有可用的处理后数据
  # 方式1: 尝试从公共数据集仓库下载
  local urls=(
    "https://hf-mirror.com/datasets/nyu-video/neural3d/resolve/main/${scene}.tar.gz"
    "https://hf-mirror.com/datasets/lirong-li/neural3dvideo/resolve/main/${scene}_processed.tar.gz"
  )

  for url in "${urls[@]}"; do
    log_msg "${scene}: 尝试 ${url}"
    if wget -q --spider "${url}" 2>/dev/null; then
      log_msg "${scene}: 找到可用URL，开始下载..."
      if wget -c -O "${scene_dir}/${scene}.tar.gz" "${url}" 2>&1 | tee -a "${LOG}"; then
        log_msg "${scene}: 下载完成，解压..."
        tar -xzf "${scene_dir}/${scene}.tar.gz" -C "${scene_dir}" --strip-components=1 || true
        rm -f "${scene_dir}/${scene}.tar.gz"
        log_msg "${scene}: 解压完成"
        return 0
      fi
    fi
  done

  log_msg "${scene}: 自动下载失败"
  log_msg "${scene}: 请手动从以下来源下载:"
  log_msg "  官方: https://github.com/facebookresearch/Neural_3D_Video"
  log_msg "  目标路径: ${scene_dir}"
  log_msg "  需要文件: cam*/images/*.jpg, poses_bounds.npy, points3D_downsample2.ply"
  return 1
}

# 对每个缺失场景尝试下载
for scene in "${SCENES[@]}"; do
  download_scene "${scene}" || log_msg "${scene}: 跳过（下载失败）"
done

# 检查结果
log_msg "=== 下载结果检查 ==="
for scene in "${SCENES[@]}"; do
  scene_dir="${DYNERF_ROOT}/${scene}"
  if [[ -f "${scene_dir}/poses_bounds.npy" ]]; then
    log_msg "${scene}: ✓ 数据就绪"
  else
    log_msg "${scene}: ✗ 数据缺失（需要手动下载后运行preprocess）"
  fi
done

log_msg "下载脚本完成"
