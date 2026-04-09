#!/usr/bin/env bash
# ============================================================
# Recompute Ours reconstruction metrics under the SAME protocol
# as baseline_4dgs_20260330:
#   render.py -m <run_dir> --iteration -1  (with correct stellar_tube args)
#   metrics.py -m <run_dir>
#
# Key detail:
#   Do NOT rely on env defaults (TEMPORAL_TUBE_SPAN/SIGMA/etc).
#   Instead, read per-run hyper-params from <run_dir>/config.yaml
#   to ensure evaluation matches the trained run.
# ============================================================
set -euo pipefail

GS_ROOT="${GS_ROOT:-/root/autodl-tmp/GaussianStellar}"
REPORT_DIR="${GS_ROOT}/reports/ours_recon_same_protocol"
mkdir -p "${REPORT_DIR}"
LOG="${REPORT_DIR}/eval_queue.log"

source "${GS_ROOT}/scripts/common.sh"
PY_CMD="$(gs_python_cmd)"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

python3 - <<'PY' > "${REPORT_DIR}/ours_runs_manifest.tsv"
import os
from pathlib import Path

GS_ROOT = Path(os.environ.get("GS_ROOT", "/root/autodl-tmp/GaussianStellar"))

# run_dir, dataset, scene_name
RUNS = [
    # ready weaktube (hypernerf)
    (GS_ROOT / "runs/stellar_tube_4dlangsplat_refresh_20260328_espresso/hypernerf/espresso", "hypernerf", "espresso"),
    (GS_ROOT / "runs/stellar_tube_4dlangsplat_refresh_20260328_americano/hypernerf/americano", "hypernerf", "americano"),
    (GS_ROOT / "runs/stellar_tube_cutlemon_refresh_20260329/hypernerf/cut-lemon1", "hypernerf", "cut-lemon1"),
    (GS_ROOT / "runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie", "hypernerf", "split-cookie"),
    # benchmark training (hypernerf)
    (GS_ROOT / "runs/stellar_tube_ours_benchmark_keyboard/hypernerf/keyboard", "hypernerf", "keyboard"),
    (GS_ROOT / "runs/stellar_tube_ours_benchmark_torchocolate/hypernerf/torchocolate", "hypernerf", "torchocolate"),
    # benchmark training (dynerf)
    (GS_ROOT / "runs/stellar_tube_ours_benchmark_coffee_martini/dynerf/coffee_martini", "dynerf", "coffee_martini"),
    (GS_ROOT / "runs/stellar_tube_ours_benchmark_flame_steak/dynerf/flame_steak", "dynerf", "flame_steak"),
    # Neu3D (dynerf layout in this repo)
    (GS_ROOT / "runs/stellar_tube_ours_benchmark/dynerf/cook_spinach", "dynerf", "cook_spinach"),
    (GS_ROOT / "runs/stellar_tube_ours_benchmark/dynerf/cut_roasted_beef", "dynerf", "cut_roasted_beef"),
    (GS_ROOT / "runs/stellar_tube_ours_benchmark/dynerf/sear_steak", "dynerf", "sear_steak"),
    (GS_ROOT / "runs/stellar_tube_ours_benchmark/dynerf/flame_salmon_1", "dynerf", "flame_salmon_1"),
]

for run_dir, ds, scene in RUNS:
    print(f"{run_dir}\t{ds}\t{scene}")
PY

eval_one() {
  local run_dir="$1"
  local dataset="$2"
  local scene="$3"

  if [[ ! -d "${run_dir}" ]]; then
    log "[skip] missing run_dir: ${run_dir}"
    return 0
  fi
  if [[ ! -d "${run_dir}/point_cloud" ]]; then
    log "[skip] no point_cloud (not trained yet): ${dataset}/${scene} (${run_dir})"
    return 0
  fi
  if [[ ! -f "${run_dir}/config.yaml" ]]; then
    log "[skip] missing config.yaml: ${run_dir}"
    return 0
  fi

  log "[eval] ${dataset}/${scene} @ ${run_dir}"

  # Extract params from config.yaml
  local params_json
  params_json="$(python3 - <<PY
import json, yaml
cfg=yaml.safe_load(open('${run_dir}/config.yaml','r'))
keys=[
  'temporal_gate_sharpness','temporal_drift_scale','temporal_gate_mix','temporal_drift_mix',
  'temporal_tube_enabled','temporal_tube_samples','temporal_tube_span','temporal_tube_sigma',
  'temporal_tube_weight_power','temporal_tube_covariance_mix','temporal_acceleration_enabled'
]
out={k:cfg.get(k) for k in keys}
print(json.dumps(out))
PY
)"

  # Convert JSON -> bash vars via python (robust against None)
  local gate_sharpness drift_scale gate_mix drift_mix tube_samples tube_span tube_sigma tube_weight_power tube_cov_mix accel_enabled
  read -r gate_sharpness drift_scale gate_mix drift_mix tube_samples tube_span tube_sigma tube_weight_power tube_cov_mix accel_enabled < <(python3 - <<PY
import json
d=json.loads('''${params_json}''')
def f(x, default):
  return default if x is None else x
print(
  f(d.get('temporal_gate_sharpness'),1.0),
  f(d.get('temporal_drift_scale'),1.0),
  f(d.get('temporal_gate_mix'),1.0),
  f(d.get('temporal_drift_mix'),1.0),
  f(d.get('temporal_tube_samples'),5),
  f(d.get('temporal_tube_span'),1.0),
  f(d.get('temporal_tube_sigma'),0.75),
  f(d.get('temporal_tube_weight_power'),1.0),
  f(d.get('temporal_tube_covariance_mix'),1.0),
  f(d.get('temporal_acceleration_enabled'),0),
)
PY
)

  local accel_args=""
  if [[ "${accel_enabled}" == "1" ]]; then
    accel_args="--temporal_acceleration_enabled"
  fi

  # Render + metrics (same entry as baseline, but with stellar tube flags)
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash -lc "
    cd '${GS_ROOT}'
    export PYTHONPATH='${PYTHONPATH}'
    ${PY_CMD} external/4DGaussians/render.py \
      -m '${run_dir}' --iteration -1 \
      --warp_enabled --temporal_warp_type 'stellar' \
      --temporal_extent_enabled \
      --temporal_gate_sharpness '${gate_sharpness}' \
      --temporal_drift_scale '${drift_scale}' \
      --temporal_gate_mix '${gate_mix}' \
      --temporal_drift_mix '${drift_mix}' \
      --temporal_tube_enabled \
      --temporal_tube_samples '${tube_samples}' \
      --temporal_tube_span '${tube_span}' \
      --temporal_tube_sigma '${tube_sigma}' \
      --temporal_tube_weight_power '${tube_weight_power}' \
      --temporal_tube_covariance_mix '${tube_cov_mix}' \
      ${accel_args}
  " | tee -a "${run_dir}/render_same_protocol.log"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash -lc "
    cd '${GS_ROOT}'
    export PYTHONPATH='${PYTHONPATH}'
    ${PY_CMD} external/4DGaussians/metrics.py -m '${run_dir}'
  " | tee -a "${run_dir}/metrics_same_protocol.log"

  log "[done] ${dataset}/${scene}"
}

log "=== Ours recon eval (same protocol as baseline_4dgs_20260330) ==="
log "manifest: ${REPORT_DIR}/ours_runs_manifest.tsv"

while IFS=$'\t' read -r run_dir dataset scene; do
  [[ -z "${run_dir}" ]] && continue
  eval_one "${run_dir}" "${dataset}" "${scene}"
done < "${REPORT_DIR}/ours_runs_manifest.tsv"

log "=== All done ==="

