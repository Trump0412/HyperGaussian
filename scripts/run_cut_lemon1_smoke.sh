#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

PHASE="${1:-stellar_tube}"
shift $(( $# > 0 ? 1 : 0 ))

SCENE="interp/cut-lemon1"
RUN_NAMESPACE="${GS_RUN_NAMESPACE:-${PHASE}_cut_lemon1_smoke1200}"
export GS_RUN_NAMESPACE="${RUN_NAMESPACE}"
COARSE_ITERS="${CUT_LEMON1_COARSE_ITERATIONS:-100}"
TRAIN_ITERS="${CUT_LEMON1_ITERATIONS:-1200}"
TEST_ITERS="${CUT_LEMON1_TEST_ITERATIONS:-${TRAIN_ITERS}}"
SAVE_ITERS="${CUT_LEMON1_SAVE_ITERATIONS:-${TRAIN_ITERS}}"
LOCAL_TEST_ITERS="${TEST_ITERS}"
export GS_SKIP_FULL_METRICS="${CUT_LEMON1_SKIP_FULL_METRICS:-1}"

bash "${GS_ROOT}/scripts/prepare_cut_lemon1.sh"

case "${PHASE}" in
  baseline)
    bash "${GS_ROOT}/scripts/train_baseline.sh" hypernerf "${SCENE}" \
      --coarse_iterations "${COARSE_ITERS}" \
      --iterations "${TRAIN_ITERS}" \
      --test_iterations "${TEST_ITERS}" \
      --save_iterations "${SAVE_ITERS}" \
      "$@"
    bash "${GS_ROOT}/scripts/eval_baseline.sh" hypernerf "${SCENE}" "$@"
    ;;
  stellar_core)
    bash "${GS_ROOT}/scripts/train_stellar.sh" hypernerf "${SCENE}" \
      --coarse_iterations "${COARSE_ITERS}" \
      --iterations "${TRAIN_ITERS}" \
      --test_iterations "${TEST_ITERS}" \
      --save_iterations "${SAVE_ITERS}" \
      "$@"
    bash "${GS_ROOT}/scripts/eval_stellar.sh" hypernerf "${SCENE}" "$@"
    ;;
  stellar_tube)
    export TEMPORAL_TUBE_SAMPLES="${TEMPORAL_TUBE_SAMPLES:-3}"
    export TEMPORAL_TUBE_SPAN="${TEMPORAL_TUBE_SPAN:-0.5}"
    export TEMPORAL_TUBE_SIGMA="${TEMPORAL_TUBE_SIGMA:-0.35}"
    export TEMPORAL_TUBE_COVARIANCE_MIX="${TEMPORAL_TUBE_COVARIANCE_MIX:-0.05}"
    export TEMPORAL_ACCELERATION_ENABLED="${TEMPORAL_ACCELERATION_ENABLED:-0}"
    bash "${GS_ROOT}/scripts/train_stellar_tube.sh" hypernerf "${SCENE}" \
      --coarse_iterations "${COARSE_ITERS}" \
      --iterations "${TRAIN_ITERS}" \
      --test_iterations "${TEST_ITERS}" \
      --save_iterations "${SAVE_ITERS}" \
      "$@"
    bash "${GS_ROOT}/scripts/eval_stellar_tube.sh" hypernerf "${SCENE}" "$@"
    ;;
  stellar_worldtube)
    if [[ -z "${CUT_LEMON1_TEST_ITERATIONS:-}" ]]; then
      LOCAL_TEST_ITERS=999999
    fi
    export TEMPORAL_WORLDTUBE_SAMPLES="${TEMPORAL_WORLDTUBE_SAMPLES:-5}"
    export TEMPORAL_WORLDTUBE_SPAN="${TEMPORAL_WORLDTUBE_SPAN:-0.75}"
    export TEMPORAL_WORLDTUBE_SIGMA="${TEMPORAL_WORLDTUBE_SIGMA:-0.45}"
    export TEMPORAL_WORLDTUBE_OPACITY_MIX="${TEMPORAL_WORLDTUBE_OPACITY_MIX:-1.0}"
    export TEMPORAL_WORLDTUBE_SCALE_MIX="${TEMPORAL_WORLDTUBE_SCALE_MIX:-0.12}"
    export TEMPORAL_ACCELERATION_ENABLED="${TEMPORAL_ACCELERATION_ENABLED:-1}"
    bash "${GS_ROOT}/scripts/train_stellar_worldtube.sh" hypernerf "${SCENE}" \
      --coarse_iterations "${COARSE_ITERS}" \
      --iterations "${TRAIN_ITERS}" \
      --test_iterations "${LOCAL_TEST_ITERS}" \
      --save_iterations "${SAVE_ITERS}" \
      "$@"
    bash "${GS_ROOT}/scripts/eval_stellar_worldtube.sh" hypernerf "${SCENE}" "$@"
    ;;
  *)
    echo "Unsupported phase: ${PHASE}" >&2
    exit 2
    ;;
esac

RUN_DIR="${GS_ROOT}/runs/${RUN_NAMESPACE}/hypernerf/cut-lemon1"
gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
  --run-dir "${RUN_DIR}" \
  --num-frames "${ENTITYBANK_NUM_FRAMES:-96}" \
  --sample-ratio "${ENTITYBANK_SAMPLE_RATIO:-0.03}" \
  --min-cluster-size "${ENTITYBANK_MIN_CLUSTER_SIZE:-12}" \
  --min-gaussians-per-entity "${ENTITYBANK_MIN_GAUSSIANS_PER_ENTITY:-48}"
gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/export_segmentation_bootstrap.py" --run-dir "${RUN_DIR}" || true
gs_python "${GS_ROOT}/scripts/quick_subset_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --max-frames "${GS_QUICK_METRIC_FRAMES:-32}" \
  --with-lpips || true
gs_python "${GS_ROOT}/scripts/collect_metrics.py" --run-dir "${RUN_DIR}" --write-summary

echo "Completed ${PHASE} smoke run at ${RUN_DIR}"
