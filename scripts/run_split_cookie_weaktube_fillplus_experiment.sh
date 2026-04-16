#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

EXPERIMENT_ROOT="${GS_EXPERIMENT_ROOT:-/root/autodl-tmp/hypergaussian_experiments}"
STAMP="${GS_EXPERIMENT_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_TAG="${WEAKTUBE_EXPERIMENT_TAG:-splitcookie_weaktube_ellipsoid_fillplus}"
OUT_ROOT="${EXPERIMENT_ROOT}/${STAMP}_${EXPERIMENT_TAG}"
PROPOSAL_DIR="${OUT_ROOT}/proposal_dir"
REMOVAL_DIR="${OUT_ROOT}/removal_bundle"

SOURCE_RUN_DIR="${WEAKTUBE_SOURCE_RUN_DIR:-/root/autodl-tmp/HyperGaussian/runs/stellar_tube_split-cookie_compare5k_weak/hypernerf/split-cookie}"
QUERY_ROOT="${WEAKTUBE_QUERY_ROOT:-${SOURCE_RUN_DIR}/entitybank/query_guided/split-cookie__the_complete_cookie_phaseaware_weak_tube}"
SOURCE_PROPOSAL_DIR="${WEAKTUBE_SOURCE_PROPOSAL_DIR:-${QUERY_ROOT}/proposal_dir}"
PROPOSAL_ALIAS="${WEAKTUBE_PROPOSAL_ALIAS:-cookie__pre_split}"
QUERY_TEXT="${WEAKTUBE_QUERY_TEXT:-the complete cookie}"
SEG_START="${WEAKTUBE_SEG_START:-0}"
SEG_END="${WEAKTUBE_SEG_END:-67}"

SAMPLE_END_INDEX="${WEAKTUBE_SAMPLE_END_INDEX:-18}"
COLOR_THRESHOLD="${WEAKTUBE_COLOR_THRESHOLD:-2.8}"
DISTANCE_MARGIN="${WEAKTUBE_DISTANCE_MARGIN:-0.22}"
PATH_MULTIPLIER="${WEAKTUBE_PATH_MULTIPLIER:-2.6}"
GATE_QUANTILE="${WEAKTUBE_GATE_QUANTILE:-0.03}"
OPACITY_MIN="${WEAKTUBE_OPACITY_MIN:-0.02}"
MODE="${WEAKTUBE_MODE:-ellipsoid}"
ELLIPSOID_MARGIN="${WEAKTUBE_ELLIPSOID_MARGIN:-0.85}"
TWO_STAGE="${WEAKTUBE_TWO_STAGE:-0}"
INTERIOR_BACKFILL="${WEAKTUBE_INTERIOR_BACKFILL:-0}"
INTERIOR_KNN_FACTOR="${WEAKTUBE_INTERIOR_KNN_FACTOR:-2.0}"
INTERIOR_ELLIPSOID_FACTOR="${WEAKTUBE_INTERIOR_ELLIPSOID_FACTOR:-1.15}"
INTERIOR_COLOR_SCALE="${WEAKTUBE_INTERIOR_COLOR_SCALE:-1.1}"
INTERIOR_PATH_SCALE="${WEAKTUBE_INTERIOR_PATH_SCALE:-1.08}"
INTERIOR_GATE_QUANTILE="${WEAKTUBE_INTERIOR_GATE_QUANTILE:-0.05}"
INTERIOR_OPACITY_MIN="${WEAKTUBE_INTERIOR_OPACITY_MIN:-0.01}"

MAX_POINTS="${WEAKTUBE_MAX_POINTS:-40000}"
FRAME_STRIDE="${WEAKTUBE_FRAME_STRIDE:-2}"
FPS="${WEAKTUBE_FPS:-12}"

SELECTION_NOTES="${WEAKTUBE_SELECTION_NOTES:-Weak-tube ellipsoid interior fill plus (color_threshold=${COLOR_THRESHOLD}, path_multiplier=${PATH_MULTIPLIER}, gate_quantile=${GATE_QUANTILE}, opacity_min=${OPACITY_MIN}, ellipsoid_margin=${ELLIPSOID_MARGIN}, two_stage=${TWO_STAGE}, interior_backfill=${INTERIOR_BACKFILL}).}"

mkdir -p "${OUT_ROOT}"

echo "Experiment root: ${OUT_ROOT}"
echo "Source run: ${SOURCE_RUN_DIR}"
echo "Source proposal dir: ${SOURCE_PROPOSAL_DIR}"

EXPAND_ARGS=(
  --run-dir "${SOURCE_RUN_DIR}"
  --source-proposal-dir "${SOURCE_PROPOSAL_DIR}"
  --output-dir "${PROPOSAL_DIR}"
  --proposal-alias "${PROPOSAL_ALIAS}"
  --sample-end-index "${SAMPLE_END_INDEX}"
  --color-threshold "${COLOR_THRESHOLD}"
  --distance-margin "${DISTANCE_MARGIN}"
  --path-multiplier "${PATH_MULTIPLIER}"
  --gate-quantile "${GATE_QUANTILE}"
  --opacity-min "${OPACITY_MIN}"
  --mode "${MODE}"
  --ellipsoid-margin "${ELLIPSOID_MARGIN}"
)
if [[ "${TWO_STAGE}" == "1" ]]; then
  EXPAND_ARGS+=(--two-stage)
fi
if [[ "${INTERIOR_BACKFILL}" == "1" ]]; then
  EXPAND_ARGS+=(
    --interior-backfill
    --interior-knn-factor "${INTERIOR_KNN_FACTOR}"
    --interior-ellipsoid-factor "${INTERIOR_ELLIPSOID_FACTOR}"
    --interior-color-scale "${INTERIOR_COLOR_SCALE}"
    --interior-path-scale "${INTERIOR_PATH_SCALE}"
    --interior-gate-quantile "${INTERIOR_GATE_QUANTILE}"
    --interior-opacity-min "${INTERIOR_OPACITY_MIN}"
  )
fi

gs_python "${GS_ROOT}/scripts/expand_seed_entity.py" "${EXPAND_ARGS[@]}"

gs_python "${GS_ROOT}/scripts/build_query_removal_bundle.py" \
  --source-run-dir "${SOURCE_RUN_DIR}" \
  --query-root "${QUERY_ROOT}" \
  --proposal-entities-json "${PROPOSAL_DIR}/entities.json" \
  --proposal-alias "${PROPOSAL_ALIAS}" \
  --query-text "${QUERY_TEXT}" \
  --segment-start "${SEG_START}" \
  --segment-end "${SEG_END}" \
  --selection-notes "${SELECTION_NOTES}" \
  --output-root "${REMOVAL_DIR}" \
  --max-points "${MAX_POINTS}" \
  --frame-stride "${FRAME_STRIDE}" \
  --fps "${FPS}"

ln -sfn "${REMOVAL_DIR}/visuals/render_triptych.png" "${OUT_ROOT}/render_triptych.png"
ln -sfn "${REMOVAL_DIR}/visuals/full_scene_pointcloud.mp4" "${OUT_ROOT}/full_scene_pointcloud.mp4"
ln -sfn "${REMOVAL_DIR}/visuals/complete_cookie_pointcloud.mp4" "${OUT_ROOT}/complete_cookie_pointcloud.mp4"
ln -sfn "${REMOVAL_DIR}/visuals/scene_without_complete_cookie_pointcloud.mp4" "${OUT_ROOT}/scene_without_complete_cookie_pointcloud.mp4"

echo "${OUT_ROOT}"
