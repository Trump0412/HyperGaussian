#!/usr/bin/env bash
set -euo pipefail

GS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_ENV_ROOT="${GS4D_ENV_ROOT:-/root/autodl-tmp/.conda-envs}"
GS_CACHE_ROOT="${GS4D_CACHE_ROOT:-/root/autodl-tmp/.cache}"
GS_CONDA_PKGS_DIRS="${GS4D_CONDA_PKGS_DIRS:-/root/autodl-tmp/.conda-pkgs}"
GS_PIP_CACHE_DIR="${GS4D_PIP_CACHE_DIR:-${GS_CACHE_ROOT}/pip}"
GS_TORCH_HOME="${GS4D_TORCH_HOME:-${GS_CACHE_ROOT}/torch}"
GS_MPLCONFIGDIR="${GS4D_MPLCONFIGDIR:-${GS_CACHE_ROOT}/matplotlib}"
GSAM2_ENV_PATH="${GS4D_GSAM2_ENV_PATH:-${GS_ENV_ROOT}/grounded-sam2-py310}"

detect_default_env_path() {
  if [[ -n "${GS4D_ENV_PATH:-}" ]]; then
    printf '%s' "${GS4D_ENV_PATH}"
    return
  fi
  if [[ -d "${GS_ENV_ROOT}/gs4d-cuda121-py310" ]]; then
    printf '%s' "${GS_ENV_ROOT}/gs4d-cuda121-py310"
    return
  fi
  if [[ -d "${GS_ENV_ROOT}/gs4d-baseline-py37" ]]; then
    printf '%s' "${GS_ENV_ROOT}/gs4d-baseline-py37"
    return
  fi
  if [[ -d "/root/miniconda3/envs/gs4d-cuda121-py310" ]]; then
    printf '%s' "/root/miniconda3/envs/gs4d-cuda121-py310"
    return
  fi
  if [[ -d "/root/miniconda3/envs/gs4d-baseline-py37" ]]; then
    printf '%s' "/root/miniconda3/envs/gs4d-baseline-py37"
    return
  fi
  printf '%s' "${GS_ENV_ROOT}/gs4d-cuda121-py310"
}

mkdir -p "${GS_ENV_ROOT}" "${GS_CONDA_PKGS_DIRS}" "${GS_PIP_CACHE_DIR}" "${GS_TORCH_HOME}" "${GS_MPLCONFIGDIR}"
GS_ENV_PATH="$(detect_default_env_path)"
GS_ENV_NAME="${GS4D_ENV_NAME:-$(basename "${GS_ENV_PATH}")}"
export GS_ROOT
export GS_ENV_ROOT
export GS_CACHE_ROOT
export GS_ENV_PATH
export GS_ENV_NAME
export GS_CONDA_PKGS_DIRS
export GS_PIP_CACHE_DIR
export GS_TORCH_HOME
export GS_MPLCONFIGDIR
export GSAM2_ENV_PATH
export XDG_CACHE_HOME="${GS_CACHE_ROOT}"
export TORCH_HOME="${GS_TORCH_HOME}"
export MPLCONFIGDIR="${GS_MPLCONFIGDIR}"
export PYTHONPATH="${GS_ROOT}:${GS_ROOT}/external/4DGaussians:${PYTHONPATH:-}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

gs_python() {
  if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
    python "$@"
  else
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      conda run --no-capture-output -p "${GS_ENV_PATH}" python "$@"
  fi
}

gs_pip() {
  if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
    python -m pip "$@"
  else
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      conda run --no-capture-output -p "${GS_ENV_PATH}" python -m pip "$@"
  fi
}

gsam2_python() {
  if [[ "${CONDA_PREFIX:-}" == "${GSAM2_ENV_PATH}" ]]; then
    python "$@"
  else
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      conda run --no-capture-output -p "${GSAM2_ENV_PATH}" python "$@"
  fi
}

gsam2_pip() {
  if [[ "${CONDA_PREFIX:-}" == "${GSAM2_ENV_PATH}" ]]; then
    python -m pip "$@"
  else
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME="${GS_CACHE_ROOT}" TORCH_HOME="${GS_TORCH_HOME}" MPLCONFIGDIR="${GS_MPLCONFIGDIR}" CONDA_PKGS_DIRS="${GS_CONDA_PKGS_DIRS}" PIP_CACHE_DIR="${GS_PIP_CACHE_DIR}" \
      conda run --no-capture-output -p "${GSAM2_ENV_PATH}" python -m pip "$@"
  fi
}

gs_python_cmd() {
  if [[ "${CONDA_PREFIX:-}" == "${GS_ENV_PATH}" ]]; then
    printf 'python'
  else
    printf 'env OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q XDG_CACHE_HOME=%q TORCH_HOME=%q MPLCONFIGDIR=%q CONDA_PKGS_DIRS=%q PIP_CACHE_DIR=%q conda run --no-capture-output -p %q python' "1" "1" "${GS_CACHE_ROOT}" "${GS_TORCH_HOME}" "${GS_MPLCONFIGDIR}" "${GS_CONDA_PKGS_DIRS}" "${GS_PIP_CACHE_DIR}" "${GS_ENV_PATH}"
  fi
}

shell_join() {
  local quoted=""
  local arg
  for arg in "$@"; do
    printf -v quoted '%s%q ' "${quoted}" "${arg}"
  done
  printf '%s' "${quoted% }"
}

dataset_source_path() {
  local dataset="$1"
  local scene="$2"
  case "${dataset}" in
    dnerf)
      printf '%s/data/dnerf/%s' "${GS_ROOT}" "${scene}"
      ;;
    dynerf)
      printf '%s/data/dynerf/%s' "${GS_ROOT}" "${scene}"
      ;;
    hypernerf)
      if [[ "${scene}" == */* ]]; then
        printf '%s/data/hypernerf/%s' "${GS_ROOT}" "${scene}"
      else
        printf '%s/data/hypernerf/virg/%s' "${GS_ROOT}" "${scene}"
      fi
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

dataset_config_path() {
  local dataset="$1"
  local scene="$2"
  local config_scene="${scene##*/}"
  case "${dataset}" in
    dnerf)
      printf '%s/external/4DGaussians/arguments/dnerf/%s.py' "${GS_ROOT}" "${config_scene}"
      ;;
    dynerf)
      local candidate="${GS_ROOT}/external/4DGaussians/arguments/dynerf/${config_scene}.py"
      if [[ -f "${candidate}" ]]; then
        printf '%s' "${candidate}"
      else
        printf '%s/external/4DGaussians/arguments/dynerf/default.py' "${GS_ROOT}"
      fi
      ;;
    hypernerf)
      case "${config_scene}" in
        slice-banana)
          config_scene="banana"
          ;;
        chickchicken)
          config_scene="chicken"
          ;;
      esac
      local candidate="${GS_ROOT}/external/4DGaussians/arguments/hypernerf/${config_scene}.py"
      if [[ -f "${candidate}" ]]; then
        printf '%s' "${candidate}"
      else
        printf '%s/external/4DGaussians/arguments/hypernerf/default.py' "${GS_ROOT}"
      fi
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

run_with_gpu_monitor() {
  local log_path="$1"
  local meta_path="$2"
  shift 2

  mkdir -p "$(dirname "${log_path}")"
  mkdir -p "$(dirname "${meta_path}")"
  : > "${log_path}"

  local start_ts
  start_ts="$(date +%s)"

  ("$@" > >(tee -a "${log_path}") 2> >(tee -a "${log_path}" >&2)) &
  local cmd_pid=$!
  local peak_file
  peak_file="$(mktemp)"
  printf '0' > "${peak_file}"

  (
    while kill -0 "${cmd_pid}" 2>/dev/null; do
      local used
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{print int($1)}')"
      local current_peak
      current_peak="$(cat "${peak_file}")"
      if [[ -n "${used}" && "${used}" =~ ^[0-9]+$ && "${used}" -gt "${current_peak}" ]]; then
        printf '%s' "${used}" > "${peak_file}"
      fi
      sleep 1
    done
  ) &
  local monitor_pid=$!

  local status=0
  wait "${cmd_pid}" || status=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  local end_ts
  end_ts="$(date +%s)"
  local peak_mb
  peak_mb="$(cat "${peak_file}")"
  rm -f "${peak_file}"
  cat > "${meta_path}" <<EOF
{
  "status": ${status},
  "elapsed_seconds": $((end_ts - start_ts)),
  "gpu_peak_mb": ${peak_mb}
}
EOF
  return "${status}"
}
