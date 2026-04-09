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
SOURCE_PATH="$(dataset_source_path "${DATASET}" "${SCENE}")"
CONFIG_PATH="$(dataset_config_path "${DATASET}" "${SCENE}")"
RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${WARP_TYPE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/train.log"
META_PATH="${RUN_DIR}/train_meta.json"

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/config.yaml" <<EOF
phase: chronometric
dataset: ${DATASET}
scene: ${SCENE}
source_path: ${SOURCE_PATH}
config_path: ${CONFIG_PATH}
warp_enabled: true
temporal_warp_type: ${WARP_TYPE}
warp_hidden_dim: ${WARP_HIDDEN_DIM:-32}
warp_num_layers: ${WARP_NUM_LAYERS:-2}
warp_num_bins: ${WARP_NUM_BINS:-128}
warp_mono_weight: ${WARP_MONO_WEIGHT:-0.05}
warp_smooth_weight: ${WARP_SMOOTH_WEIGHT:-0.01}
warp_budget_weight: ${WARP_BUDGET_WEIGHT:-0.01}
warp_sample_count: ${WARP_SAMPLE_COUNT:-128}
EOF

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/train.py -s '${SOURCE_PATH}' -m '${RUN_DIR}' --expname '${WARP_TYPE}/${DATASET}/${SCENE##*/}' --configs '${CONFIG_PATH}' --port 6017 --warp_enabled --temporal_warp_type '${WARP_TYPE}' --warp_hidden_dim '${WARP_HIDDEN_DIM:-32}' --warp_num_layers '${WARP_NUM_LAYERS:-2}' --warp_num_bins '${WARP_NUM_BINS:-128}' --warp_mono_weight '${WARP_MONO_WEIGHT:-0.05}' --warp_smooth_weight '${WARP_SMOOTH_WEIGHT:-0.01}' --warp_budget_weight '${WARP_BUDGET_WEIGHT:-0.01}' --warp_sample_count '${WARP_SAMPLE_COUNT:-128}' ${EXTRA_ARGS}"

gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary || true
done
