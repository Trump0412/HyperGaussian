#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-dnerf}"
SCENE="${2:-mutant}"
shift $(( $# > 1 ? 2 : $# ))
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-stellar_worldtube}"
ITERATIONS_VALUE="${GS_ITERATIONS:-14000}"
COARSE_ITERATIONS_VALUE="${GS_COARSE_ITERATIONS:-3000}"
TEST_ITERATIONS_VALUE="${GS_TEST_ITERATIONS:-3000 7000 14000}"
SAVE_ITERATIONS_VALUE="${GS_SAVE_ITERATIONS:-14000 20000 30000 45000 60000}"
CHECKPOINT_ITERATIONS_VALUE="${GS_CHECKPOINT_ITERATIONS:-}"
START_CHECKPOINT_VALUE="${GS_START_CHECKPOINT:-}"
TEMPORAL_GATE_MIX_VALUE="${TEMPORAL_GATE_MIX:-1.0}"
TEMPORAL_DRIFT_MIX_VALUE="${TEMPORAL_DRIFT_MIX:-1.0}"
TEMPORAL_ACCELERATION_ENABLED_VALUE="${TEMPORAL_ACCELERATION_ENABLED:-1}"
TEMPORAL_WORLDTUBE_ADAPTIVE_SUPPORT_VALUE="${TEMPORAL_WORLDTUBE_ADAPTIVE_SUPPORT:-1}"
SPACETIME_AWARE_OPTIMIZATION_VALUE="${SPACETIME_AWARE_OPTIMIZATION:-1}"
TEMPORAL_WORLDTUBE_ENERGY_PRESERVING_VALUE="${TEMPORAL_WORLDTUBE_ENERGY_PRESERVING:-0}"
TEMPORAL_WORLDTUBE_LITE_VALUE="${TEMPORAL_WORLDTUBE_LITE:-0}"
TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT_VALUE="${TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT:-0}"

SOURCE_PATH="$(dataset_source_path "${DATASET}" "${SCENE}")"
CONFIG_PATH="$(dataset_config_path "${DATASET}" "${SCENE}")"
RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/train.log"
META_PATH="${RUN_DIR}/train_meta.json"

mkdir -p "${RUN_DIR}"
cat > "${RUN_DIR}/config.yaml" <<EOF
phase: stellar_worldtube
dataset: ${DATASET}
scene: ${SCENE}
source_path: ${SOURCE_PATH}
config_path: ${CONFIG_PATH}
iterations: ${ITERATIONS_VALUE}
coarse_iterations: ${COARSE_ITERATIONS_VALUE}
test_iterations: ${TEST_ITERATIONS_VALUE}
save_iterations: ${SAVE_ITERATIONS_VALUE}
checkpoint_iterations: ${CHECKPOINT_ITERATIONS_VALUE}
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
temporal_velocity_reg_weight: ${TEMPORAL_VELOCITY_REG_WEIGHT:-0.0}
temporal_acceleration_reg_weight: ${TEMPORAL_ACCELERATION_REG_WEIGHT:-0.0}
temporal_worldtube_enabled: true
temporal_worldtube_lite: ${TEMPORAL_WORLDTUBE_LITE_VALUE}
temporal_worldtube_transmittance_split: ${TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT_VALUE}
temporal_worldtube_samples: ${TEMPORAL_WORLDTUBE_SAMPLES:-5}
temporal_worldtube_span: ${TEMPORAL_WORLDTUBE_SPAN:-1.0}
temporal_worldtube_sigma: ${TEMPORAL_WORLDTUBE_SIGMA:-0.5}
temporal_worldtube_render_weight_power: ${TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER:-1.0}
temporal_worldtube_opacity_mix: ${TEMPORAL_WORLDTUBE_OPACITY_MIX:-1.0}
temporal_worldtube_scale_mix: ${TEMPORAL_WORLDTUBE_SCALE_MIX:-0.15}
temporal_worldtube_support_scale_mix: ${TEMPORAL_WORLDTUBE_SUPPORT_SCALE_MIX:-0.15}
temporal_worldtube_child_scale_shrink: ${TEMPORAL_WORLDTUBE_CHILD_SCALE_SHRINK:-1.0}
temporal_worldtube_energy_preserving: ${TEMPORAL_WORLDTUBE_ENERGY_PRESERVING_VALUE}
temporal_worldtube_energy_gate_mix: ${TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX:-1.0}
temporal_worldtube_reg_weight: ${TEMPORAL_WORLDTUBE_REG_WEIGHT:-0.02}
temporal_worldtube_ratio_weight: ${TEMPORAL_WORLDTUBE_RATIO_WEIGHT:-0.02}
temporal_worldtube_support_min: ${TEMPORAL_WORLDTUBE_SUPPORT_MIN:-0.08}
temporal_worldtube_support_max: ${TEMPORAL_WORLDTUBE_SUPPORT_MAX:-0.45}
temporal_worldtube_ratio_target: ${TEMPORAL_WORLDTUBE_RATIO_TARGET:-1.0}
temporal_worldtube_ratio_tolerance: ${TEMPORAL_WORLDTUBE_RATIO_TOLERANCE:-0.5}
temporal_worldtube_densify_weight: ${TEMPORAL_WORLDTUBE_DENSIFY_WEIGHT:-0.5}
temporal_worldtube_split_shrink: ${TEMPORAL_WORLDTUBE_SPLIT_SHRINK:-0.65}
temporal_worldtube_support_gain: ${TEMPORAL_WORLDTUBE_SUPPORT_GAIN:-1.0}
temporal_worldtube_support_min_factor: ${TEMPORAL_WORLDTUBE_SUPPORT_MIN_FACTOR:-0.75}
temporal_worldtube_support_max_factor: ${TEMPORAL_WORLDTUBE_SUPPORT_MAX_FACTOR:-2.5}
temporal_worldtube_opacity_floor: ${TEMPORAL_WORLDTUBE_OPACITY_FLOOR:-0.25}
temporal_worldtube_visibility_mix: ${TEMPORAL_WORLDTUBE_VISIBILITY_MIX:-0.35}
temporal_worldtube_integral_mix: ${TEMPORAL_WORLDTUBE_INTEGRAL_MIX:-0.65}
temporal_worldtube_prune_keep_quantile: ${TEMPORAL_WORLDTUBE_PRUNE_KEEP_QUANTILE:-0.75}
temporal_worldtube_densify_grad_normalize: ${TEMPORAL_WORLDTUBE_DENSIFY_GRAD_NORMALIZE:-0}
temporal_worldtube_densify_grad_power: ${TEMPORAL_WORLDTUBE_DENSIFY_GRAD_POWER:-0.5}
temporal_worldtube_adaptive_support: ${TEMPORAL_WORLDTUBE_ADAPTIVE_SUPPORT_VALUE}
spacetime_aware_optimization: ${SPACETIME_AWARE_OPTIMIZATION_VALUE}
temporal_activity_weight: ${TEMPORAL_ACTIVITY_WEIGHT:-0.35}
temporal_prune_protect_quantile: ${TEMPORAL_PRUNE_PROTECT_QUANTILE:-0.85}
EOF

ACCEL_ARGS=""
if [[ "${TEMPORAL_ACCELERATION_ENABLED_VALUE}" == "1" ]]; then
  ACCEL_ARGS="--temporal_acceleration_enabled --temporal_acceleration_reg_weight '${TEMPORAL_ACCELERATION_REG_WEIGHT:-0.0}'"
fi

ADAPTIVE_SUPPORT_ARGS=""
if [[ "${TEMPORAL_WORLDTUBE_ADAPTIVE_SUPPORT_VALUE}" == "1" ]]; then
  ADAPTIVE_SUPPORT_ARGS="--temporal_worldtube_adaptive_support"
fi

SPACETIME_ARGS=""
if [[ "${SPACETIME_AWARE_OPTIMIZATION_VALUE}" == "1" ]]; then
  SPACETIME_ARGS="--spacetime_aware_optimization --temporal_activity_weight '${TEMPORAL_ACTIVITY_WEIGHT:-0.35}' --temporal_prune_protect_quantile '${TEMPORAL_PRUNE_PROTECT_QUANTILE:-0.85}'"
fi

ENERGY_PRESERVING_ARGS=""
if [[ "${TEMPORAL_WORLDTUBE_ENERGY_PRESERVING_VALUE}" == "1" ]]; then
  ENERGY_PRESERVING_ARGS="--temporal_worldtube_energy_preserving"
fi

LITE_ARGS=""
if [[ "${TEMPORAL_WORLDTUBE_LITE_VALUE}" == "1" ]]; then
  LITE_ARGS="--temporal_worldtube_lite"
fi

TRANSMITTANCE_SPLIT_ARGS=""
if [[ "${TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT_VALUE}" == "1" ]]; then
  TRANSMITTANCE_SPLIT_ARGS="--temporal_worldtube_transmittance_split"
fi

TEST_ITERATION_ARGS="--test_iterations ${TEST_ITERATIONS_VALUE}"
SAVE_ITERATION_ARGS="--save_iterations ${SAVE_ITERATIONS_VALUE}"

CHECKPOINT_ARGS=""
if [[ -n "${CHECKPOINT_ITERATIONS_VALUE}" ]]; then
  CHECKPOINT_ARGS="--checkpoint_iterations ${CHECKPOINT_ITERATIONS_VALUE}"
fi

START_CHECKPOINT_ARGS=""
if [[ -n "${START_CHECKPOINT_VALUE}" ]]; then
  START_CHECKPOINT_ARGS="--start_checkpoint '${START_CHECKPOINT_VALUE}'"
fi

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/train.py -s '${SOURCE_PATH}' -m '${RUN_DIR}' --expname 'stellar_worldtube/${DATASET}/${SCENE##*/}' --configs '${CONFIG_PATH}' --port 6021 --iterations '${ITERATIONS_VALUE}' --coarse_iterations '${COARSE_ITERATIONS_VALUE}' ${TEST_ITERATION_ARGS} ${SAVE_ITERATION_ARGS} ${CHECKPOINT_ARGS} ${START_CHECKPOINT_ARGS} --warp_enabled --temporal_warp_type 'stellar' --warp_hidden_dim '${WARP_HIDDEN_DIM:-32}' --warp_num_layers '${WARP_NUM_LAYERS:-2}' --warp_num_bins '${WARP_NUM_BINS:-128}' --warp_mono_weight '${WARP_MONO_WEIGHT:-0.05}' --warp_smooth_weight '${WARP_SMOOTH_WEIGHT:-0.01}' --warp_budget_weight '${WARP_BUDGET_WEIGHT:-0.01}' --warp_sample_count '${WARP_SAMPLE_COUNT:-128}' --temporal_lr_init '${TEMPORAL_LR_INIT:-0.00016}' --temporal_lr_final '${TEMPORAL_LR_FINAL:-0.000016}' --temporal_lr_delay_mult '${TEMPORAL_LR_DELAY_MULT:-0.01}' --temporal_extent_enabled --temporal_gate_sharpness '${TEMPORAL_GATE_SHARPNESS:-1.0}' --temporal_drift_scale '${TEMPORAL_DRIFT_SCALE:-1.0}' --temporal_gate_mix '${TEMPORAL_GATE_MIX_VALUE}' --temporal_drift_mix '${TEMPORAL_DRIFT_MIX_VALUE}' --temporal_velocity_reg_weight '${TEMPORAL_VELOCITY_REG_WEIGHT:-0.0}' --temporal_worldtube_enabled ${LITE_ARGS} ${TRANSMITTANCE_SPLIT_ARGS} --temporal_worldtube_samples '${TEMPORAL_WORLDTUBE_SAMPLES:-5}' --temporal_worldtube_span '${TEMPORAL_WORLDTUBE_SPAN:-1.0}' --temporal_worldtube_sigma '${TEMPORAL_WORLDTUBE_SIGMA:-0.5}' --temporal_worldtube_render_weight_power '${TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER:-1.0}' --temporal_worldtube_opacity_mix '${TEMPORAL_WORLDTUBE_OPACITY_MIX:-1.0}' --temporal_worldtube_scale_mix '${TEMPORAL_WORLDTUBE_SCALE_MIX:-0.15}' --temporal_worldtube_support_scale_mix '${TEMPORAL_WORLDTUBE_SUPPORT_SCALE_MIX:-0.15}' --temporal_worldtube_child_scale_shrink '${TEMPORAL_WORLDTUBE_CHILD_SCALE_SHRINK:-1.0}' --temporal_worldtube_energy_gate_mix '${TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX:-1.0}' --temporal_worldtube_reg_weight '${TEMPORAL_WORLDTUBE_REG_WEIGHT:-0.02}' --temporal_worldtube_ratio_weight '${TEMPORAL_WORLDTUBE_RATIO_WEIGHT:-0.02}' --temporal_worldtube_support_min '${TEMPORAL_WORLDTUBE_SUPPORT_MIN:-0.08}' --temporal_worldtube_support_max '${TEMPORAL_WORLDTUBE_SUPPORT_MAX:-0.45}' --temporal_worldtube_ratio_target '${TEMPORAL_WORLDTUBE_RATIO_TARGET:-1.0}' --temporal_worldtube_ratio_tolerance '${TEMPORAL_WORLDTUBE_RATIO_TOLERANCE:-0.5}' --temporal_worldtube_densify_weight '${TEMPORAL_WORLDTUBE_DENSIFY_WEIGHT:-0.5}' --temporal_worldtube_split_shrink '${TEMPORAL_WORLDTUBE_SPLIT_SHRINK:-0.65}' --temporal_worldtube_support_gain '${TEMPORAL_WORLDTUBE_SUPPORT_GAIN:-1.0}' --temporal_worldtube_support_min_factor '${TEMPORAL_WORLDTUBE_SUPPORT_MIN_FACTOR:-0.75}' --temporal_worldtube_support_max_factor '${TEMPORAL_WORLDTUBE_SUPPORT_MAX_FACTOR:-2.5}' --temporal_worldtube_opacity_floor '${TEMPORAL_WORLDTUBE_OPACITY_FLOOR:-0.25}' --temporal_worldtube_visibility_mix '${TEMPORAL_WORLDTUBE_VISIBILITY_MIX:-0.35}' --temporal_worldtube_integral_mix '${TEMPORAL_WORLDTUBE_INTEGRAL_MIX:-0.65}' --temporal_worldtube_prune_keep_quantile '${TEMPORAL_WORLDTUBE_PRUNE_KEEP_QUANTILE:-0.75}' --temporal_worldtube_densify_grad_power '${TEMPORAL_WORLDTUBE_DENSIFY_GRAD_POWER:-0.5}' ${ENERGY_PRESERVING_ARGS} ${ADAPTIVE_SUPPORT_ARGS} ${SPACETIME_ARGS} ${ACCEL_ARGS} $( [[ "${TEMPORAL_WORLDTUBE_DENSIFY_GRAD_NORMALIZE:-0}" == "1" ]] && printf '%s' '--temporal_worldtube_densify_grad_normalize' ) ${EXTRA_ARGS}"

gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary || true
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
