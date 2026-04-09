#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

RUN_DIR="${GS_ROOT}/runs/stellar_tube_split-cookie_compare5k/hypernerf/split-cookie"
FINAL_PLY="${RUN_DIR}/point_cloud/iteration_5000/point_cloud.ply"
PY_CMD="$(gs_python_cmd)"

echo "[wait] monitoring ${FINAL_PLY}"
while [[ ! -f "${FINAL_PLY}" ]]; do
  sleep 60
done

echo "[wait] final checkpoint found, starting test-only render"
bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration -1 --warp_enabled --temporal_warp_type 'stellar' --temporal_extent_enabled --temporal_gate_sharpness '1.0' --temporal_drift_scale '1.0' --temporal_gate_mix '1.0' --temporal_drift_mix '1.0' --temporal_tube_enabled --temporal_tube_samples '5' --temporal_tube_span '1.0' --temporal_tube_sigma '0.75' --temporal_tube_covariance_mix '1.0' --temporal_acceleration_enabled --skip_train --skip_video"

echo "[wait] computing fullframe metrics with LPIPS"
gs_python "${GS_ROOT}/scripts/fullframe_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --with-lpips \
  --out-name "full_metrics_with_lpips_rerun_20260326.json"

echo "[wait] post-eval finished"
