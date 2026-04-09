#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-dnerf}"
SCENE="${2:-mutant}"
shift $(( $# > 1 ? 2 : $# ))
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-stellar_spacetime}"
SPACETIME_AWARE_OPTIMIZATION="${SPACETIME_AWARE_OPTIMIZATION:-0}"
TEMPORAL_ACTIVITY_WEIGHT_VALUE="${TEMPORAL_ACTIVITY_WEIGHT:-0.35}"
TEMPORAL_PRUNE_PROTECT_QUANTILE_VALUE="${TEMPORAL_PRUNE_PROTECT_QUANTILE:-0.85}"
TEMPORAL_VELOCITY_REG_WEIGHT_VALUE="${TEMPORAL_VELOCITY_REG_WEIGHT:-0.0}"
TEMPORAL_ACCELERATION_ENABLED_VALUE="${TEMPORAL_ACCELERATION_ENABLED:-0}"
TEMPORAL_ACCELERATION_REG_WEIGHT_VALUE="${TEMPORAL_ACCELERATION_REG_WEIGHT:-0.0}"
TEMPORAL_GATE_MIX_VALUE="${TEMPORAL_GATE_MIX:-1.0}"
TEMPORAL_DRIFT_MIX_VALUE="${TEMPORAL_DRIFT_MIX:-1.0}"

SOURCE_PATH="$(dataset_source_path "${DATASET}" "${SCENE}")"
CONFIG_PATH="$(dataset_config_path "${DATASET}" "${SCENE}")"
RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/train.log"
META_PATH="${RUN_DIR}/train_meta.json"

mkdir -p "${RUN_DIR}"
SPACETIME_OPT_BOOL=false
SPACETIME_OPT_ARGS=""
SPACETIME_PHASE="stellar_spacetime"
if [[ "${SPACETIME_AWARE_OPTIMIZATION}" == "1" ]]; then
  SPACETIME_OPT_BOOL=true
  SPACETIME_PHASE="stellar_spacetime_opt"
  SPACETIME_OPT_ARGS="--spacetime_aware_optimization --temporal_activity_weight '${TEMPORAL_ACTIVITY_WEIGHT_VALUE}' --temporal_prune_protect_quantile '${TEMPORAL_PRUNE_PROTECT_QUANTILE_VALUE}'"
fi
cat > "${RUN_DIR}/config.yaml" <<EOF
phase: ${SPACETIME_PHASE}
dataset: ${DATASET}
scene: ${SCENE}
source_path: ${SOURCE_PATH}
config_path: ${CONFIG_PATH}
warp_enabled: true
temporal_warp_type: stellar
warp_hidden_dim: ${WARP_HIDDEN_DIM:-32}
warp_num_layers: ${WARP_NUM_LAYERS:-2}
warp_num_bins: ${WARP_NUM_BINS:-128}
warp_mono_weight: ${WARP_MONO_WEIGHT:-0.05}
warp_smooth_weight: ${WARP_SMOOTH_WEIGHT:-0.01}
warp_budget_weight: ${WARP_BUDGET_WEIGHT:-0.01}
warp_sample_count: ${WARP_SAMPLE_COUNT:-128}
temporal_lr_init: ${TEMPORAL_LR_INIT:-0.00016}
temporal_lr_final: ${TEMPORAL_LR_FINAL:-0.000016}
temporal_lr_delay_mult: ${TEMPORAL_LR_DELAY_MULT:-0.01}
temporal_extent_enabled: true
temporal_gate_sharpness: ${TEMPORAL_GATE_SHARPNESS:-1.0}
temporal_drift_scale: ${TEMPORAL_DRIFT_SCALE:-1.0}
temporal_gate_mix: ${TEMPORAL_GATE_MIX_VALUE}
temporal_drift_mix: ${TEMPORAL_DRIFT_MIX_VALUE}
temporal_acceleration_enabled: ${TEMPORAL_ACCELERATION_ENABLED_VALUE}
temporal_velocity_reg_weight: ${TEMPORAL_VELOCITY_REG_WEIGHT_VALUE}
temporal_acceleration_reg_weight: ${TEMPORAL_ACCELERATION_REG_WEIGHT_VALUE}
spacetime_aware_optimization: ${SPACETIME_OPT_BOOL}
temporal_activity_weight: ${TEMPORAL_ACTIVITY_WEIGHT_VALUE}
temporal_prune_protect_quantile: ${TEMPORAL_PRUNE_PROTECT_QUANTILE_VALUE}
EOF

ACCEL_ARGS=""
if [[ "${TEMPORAL_ACCELERATION_ENABLED_VALUE}" == "1" ]]; then
  ACCEL_ARGS="--temporal_acceleration_enabled --temporal_acceleration_reg_weight '${TEMPORAL_ACCELERATION_REG_WEIGHT_VALUE}'"
fi

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/train.py -s '${SOURCE_PATH}' -m '${RUN_DIR}' --expname '${SPACETIME_PHASE}/${DATASET}/${SCENE##*/}' --configs '${CONFIG_PATH}' --port 6019 --warp_enabled --temporal_warp_type 'stellar' --warp_hidden_dim '${WARP_HIDDEN_DIM:-32}' --warp_num_layers '${WARP_NUM_LAYERS:-2}' --warp_num_bins '${WARP_NUM_BINS:-128}' --warp_mono_weight '${WARP_MONO_WEIGHT:-0.05}' --warp_smooth_weight '${WARP_SMOOTH_WEIGHT:-0.01}' --warp_budget_weight '${WARP_BUDGET_WEIGHT:-0.01}' --warp_sample_count '${WARP_SAMPLE_COUNT:-128}' --temporal_lr_init '${TEMPORAL_LR_INIT:-0.00016}' --temporal_lr_final '${TEMPORAL_LR_FINAL:-0.000016}' --temporal_lr_delay_mult '${TEMPORAL_LR_DELAY_MULT:-0.01}' --temporal_extent_enabled --temporal_gate_sharpness '${TEMPORAL_GATE_SHARPNESS:-1.0}' --temporal_drift_scale '${TEMPORAL_DRIFT_SCALE:-1.0}' --temporal_gate_mix '${TEMPORAL_GATE_MIX_VALUE}' --temporal_drift_mix '${TEMPORAL_DRIFT_MIX_VALUE}' --temporal_velocity_reg_weight '${TEMPORAL_VELOCITY_REG_WEIGHT_VALUE}' ${ACCEL_ARGS} ${SPACETIME_OPT_ARGS} ${EXTRA_ARGS}"

gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary || true
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
