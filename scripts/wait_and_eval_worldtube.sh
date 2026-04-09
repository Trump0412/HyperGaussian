#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

DATASET="${1:-dnerf}"
SCENE="${2:-mutant}"
RUN_NAMESPACE="${3:-stellar_worldtube}"

RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
FINAL_ITERATION="${GS_FINAL_ITERATION:-14000}"
FINAL_PLY="${RUN_DIR}/point_cloud/iteration_${FINAL_ITERATION}/point_cloud.ply"
PY_CMD="$(gs_python_cmd)"

TEMPORAL_GATE_MIX_VALUE="${TEMPORAL_GATE_MIX:-1.0}"
TEMPORAL_DRIFT_MIX_VALUE="${TEMPORAL_DRIFT_MIX:-1.0}"
TEMPORAL_ACCELERATION_ENABLED_VALUE="${TEMPORAL_ACCELERATION_ENABLED:-1}"
TEMPORAL_WORLDTUBE_ENERGY_PRESERVING_VALUE="${TEMPORAL_WORLDTUBE_ENERGY_PRESERVING:-0}"
TEMPORAL_WORLDTUBE_LITE_VALUE="${TEMPORAL_WORLDTUBE_LITE:-0}"
TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT_VALUE="${TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT:-0}"

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

echo "[wait] monitoring ${FINAL_PLY}"
while [[ ! -f "${FINAL_PLY}" ]]; do
  sleep 60
done

echo "[wait] final checkpoint found, starting worldtube render"
bash -lc "cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && ${PY_CMD} external/4DGaussians/render.py -m '${RUN_DIR}' --iteration '${FINAL_ITERATION}' --warp_enabled --temporal_warp_type 'stellar' --temporal_extent_enabled --temporal_gate_sharpness '${TEMPORAL_GATE_SHARPNESS:-1.0}' --temporal_drift_scale '${TEMPORAL_DRIFT_SCALE:-1.0}' --temporal_gate_mix '${TEMPORAL_GATE_MIX_VALUE}' --temporal_drift_mix '${TEMPORAL_DRIFT_MIX_VALUE}' --temporal_worldtube_enabled ${LITE_ARGS} ${TRANSMITTANCE_SPLIT_ARGS} --temporal_worldtube_samples '${TEMPORAL_WORLDTUBE_SAMPLES:-5}' --temporal_worldtube_span '${TEMPORAL_WORLDTUBE_SPAN:-1.0}' --temporal_worldtube_sigma '${TEMPORAL_WORLDTUBE_SIGMA:-0.5}' --temporal_worldtube_render_weight_power '${TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER:-1.0}' --temporal_worldtube_opacity_mix '${TEMPORAL_WORLDTUBE_OPACITY_MIX:-1.0}' --temporal_worldtube_scale_mix '${TEMPORAL_WORLDTUBE_SCALE_MIX:-0.15}' --temporal_worldtube_support_scale_mix '${TEMPORAL_WORLDTUBE_SUPPORT_SCALE_MIX:-0.15}' --temporal_worldtube_child_scale_shrink '${TEMPORAL_WORLDTUBE_CHILD_SCALE_SHRINK:-1.0}' --temporal_worldtube_energy_gate_mix '${TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX:-1.0}' --temporal_worldtube_opacity_floor '${TEMPORAL_WORLDTUBE_OPACITY_FLOOR:-0.25}' --temporal_worldtube_visibility_mix '${TEMPORAL_WORLDTUBE_VISIBILITY_MIX:-0.35}' --temporal_worldtube_integral_mix '${TEMPORAL_WORLDTUBE_INTEGRAL_MIX:-0.65}' ${ENERGY_PRESERVING_ARGS} ${ACCEL_ARGS} --skip_train --skip_video"

echo "[wait] computing fullframe metrics with LPIPS"
gs_python "${GS_ROOT}/scripts/fullframe_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --with-lpips \
  --out-name "full_metrics_with_lpips_wait_eval.json"

echo "[wait] exporting worldtube semantics"
gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_native_semantics.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_segmentation_bootstrap.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary

echo "[wait] post-eval finished"
