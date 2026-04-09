#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-dnerf}"
SCENE="${2:-mutant}"
shift $(( $# > 1 ? 2 : $# ))
PY_CMD="$(gs_python_cmd)"
EXTRA_ARGS="$(shell_join "$@")"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-stellar_worldtube}"
TEMPORAL_GATE_MIX_VALUE="${TEMPORAL_GATE_MIX:-1.0}"
TEMPORAL_DRIFT_MIX_VALUE="${TEMPORAL_DRIFT_MIX:-1.0}"
TEMPORAL_ACCELERATION_ENABLED_VALUE="${TEMPORAL_ACCELERATION_ENABLED:-1}"
TEMPORAL_WORLDTUBE_ENERGY_PRESERVING_VALUE="${TEMPORAL_WORLDTUBE_ENERGY_PRESERVING:-0}"
TEMPORAL_WORLDTUBE_LITE_VALUE="${TEMPORAL_WORLDTUBE_LITE:-0}"
TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT_VALUE="${TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT:-0}"

RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
LOG_PATH="${RUN_DIR}/render.log"
META_PATH="${RUN_DIR}/render_meta.json"
RENDER_ITERATION_VALUE="${GS_RENDER_ITERATION:--1}"

ACCEL_ARGS=""
if [[ "${TEMPORAL_ACCELERATION_ENABLED_VALUE}" == "1" ]]; then
  ACCEL_ARGS="--temporal_acceleration_enabled"
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

run_with_gpu_monitor "${LOG_PATH}" "${META_PATH}" \
  bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration '${RENDER_ITERATION_VALUE}' --warp_enabled --temporal_warp_type 'stellar' --temporal_extent_enabled --temporal_gate_sharpness '${TEMPORAL_GATE_SHARPNESS:-1.0}' --temporal_drift_scale '${TEMPORAL_DRIFT_SCALE:-1.0}' --temporal_gate_mix '${TEMPORAL_GATE_MIX_VALUE}' --temporal_drift_mix '${TEMPORAL_DRIFT_MIX_VALUE}' --temporal_worldtube_enabled ${LITE_ARGS} ${TRANSMITTANCE_SPLIT_ARGS} --temporal_worldtube_samples '${TEMPORAL_WORLDTUBE_SAMPLES:-5}' --temporal_worldtube_span '${TEMPORAL_WORLDTUBE_SPAN:-1.0}' --temporal_worldtube_sigma '${TEMPORAL_WORLDTUBE_SIGMA:-0.5}' --temporal_worldtube_render_weight_power '${TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER:-1.0}' --temporal_worldtube_opacity_mix '${TEMPORAL_WORLDTUBE_OPACITY_MIX:-1.0}' --temporal_worldtube_scale_mix '${TEMPORAL_WORLDTUBE_SCALE_MIX:-0.15}' --temporal_worldtube_support_scale_mix '${TEMPORAL_WORLDTUBE_SUPPORT_SCALE_MIX:-0.15}' --temporal_worldtube_child_scale_shrink '${TEMPORAL_WORLDTUBE_CHILD_SCALE_SHRINK:-1.0}' --temporal_worldtube_energy_gate_mix '${TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX:-1.0}' --temporal_worldtube_opacity_floor '${TEMPORAL_WORLDTUBE_OPACITY_FLOOR:-0.25}' --temporal_worldtube_visibility_mix '${TEMPORAL_WORLDTUBE_VISIBILITY_MIX:-0.35}' --temporal_worldtube_integral_mix '${TEMPORAL_WORLDTUBE_INTEGRAL_MIX:-0.65}' ${ENERGY_PRESERVING_ARGS} ${ACCEL_ARGS} ${EXTRA_ARGS}"

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
gs_python "${GS_ROOT}/scripts/export_native_semantics.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_segmentation_bootstrap.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary
