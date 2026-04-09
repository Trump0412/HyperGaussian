#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

RUN_DIR="${GS_ROOT}/runs/baseline_split-cookie_compare5k_14000/hypernerf/split-cookie"
FINAL_PLY="${RUN_DIR}/point_cloud/iteration_14000/point_cloud.ply"
PY_CMD="$(gs_python_cmd)"

echo "[wait] monitoring ${FINAL_PLY}"
while [[ ! -f "${FINAL_PLY}" ]]; do
  sleep 60
done

echo "[wait] final checkpoint found, starting test-only render"
bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration -1 --skip_train --skip_video"

echo "[wait] computing fullframe metrics with LPIPS"
gs_python "${GS_ROOT}/scripts/fullframe_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --with-lpips \
  --out-name "full_metrics_with_lpips_rerun_20260326.json"

echo "[wait] post-eval finished"
