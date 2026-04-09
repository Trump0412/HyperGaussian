#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

WARP_MODE="${1:-all}"
if [[ "${WARP_MODE}" == "mlp" || "${WARP_MODE}" == "density" || "${WARP_MODE}" == "identity" || "${WARP_MODE}" == "all" ]]; then
  shift
else
  WARP_MODE="all"
fi

DATASET="${1:-dnerf}"
SCENE="${2:-mutant}"
shift $(( $# > 1 ? 2 : $# ))
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-chronometric_4dgs}"

WARP_TYPES=(mlp density)
if [[ "${WARP_MODE}" != "all" ]]; then
  WARP_TYPES=("${WARP_MODE}")
fi

for WARP_TYPE in "${WARP_TYPES[@]}"; do
RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${WARP_TYPE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/render.log"
META_PATH="${RUN_DIR}/render_meta.json"

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration -1 --warp_enabled --temporal_warp_type '${WARP_TYPE}' ${EXTRA_ARGS}"

if [[ "${GS_SKIP_FULL_METRICS:-0}" == "1" ]]; then
  gs_python "${GS_ROOT}/scripts/quick_subset_metrics.py" \
    --run-dir "${RUN_DIR}" \
    --max-frames "${GS_QUICK_METRIC_FRAMES:-32}" \
    --with-lpips
else
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/metrics.py -m '${RUN_DIR}'" | tee -a "${RUN_DIR}/metrics.log"
fi
gs_python "${GS_ROOT}/scripts/plot_time_warp.py" --run-dir "${RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_segmentation_bootstrap.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary
done
