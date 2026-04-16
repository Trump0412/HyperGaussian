#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

SCENE_KEY="${1:?scene key required, e.g. split-cookie or cut-lemon1}"
QUERY_NAME="${2:?query name required, e.g. complete_cookie or lemon}"

EXPERIMENT_ROOT="${GS_EXPERIMENT_ROOT:-/root/autodl-tmp/hypergaussian_experiments}"
STAMP="${GS_EXPERIMENT_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

case "${SCENE_KEY}" in
  split-cookie)
    RUN_DIR="/root/autodl-tmp/HyperGaussian/runs/stellar_worldtube_split-cookie_compare5k/hypernerf/split-cookie"
    DATASET_DIR="/root/autodl-tmp/HyperGaussian/data/hypernerf/misc/split-cookie"
    TRACKS_PATH="${RUN_DIR}/entitybank/query_guided/split-cookie__the_complete_cookie_phaseaware/grounded_sam2/grounded_sam2_query_tracks.json"
    QUERY_ROOT="${RUN_DIR}/entitybank/query_guided/split-cookie__the_complete_cookie_phaseaware"
    case "${QUERY_NAME}" in
      complete_cookie)
        PROPOSAL_ALIAS="cookie__pre_split"
        QUERY_TEXT="the complete cookie"
        SEG_START=0
        SEG_END=67
        ;;
      *)
        echo "Unsupported split-cookie query: ${QUERY_NAME}" >&2
        exit 1
        ;;
    esac
    ;;
  cut-lemon1)
    RUN_DIR="/root/autodl-tmp/HyperGaussian/runs/stellar_worldtube_cut-lemon1_quality5k/hypernerf/cut-lemon1"
    DATASET_DIR="/root/autodl-tmp/HyperGaussian/data/hypernerf/interp/cut-lemon1"
    TRACKS_PATH="${RUN_DIR}/entitybank/query_guided/cut_the_lemon_final/grounded_sam2/grounded_sam2_query_tracks.json"
    QUERY_ROOT="${RUN_DIR}/entitybank/query_guided/cut_the_lemon_final"
    case "${QUERY_NAME}" in
      lemon)
        PROPOSAL_ALIAS="lemon"
        QUERY_TEXT="the lemon"
        SEG_START=36
        SEG_END=41
        ;;
      *)
        echo "Unsupported cut-lemon1 query: ${QUERY_NAME}" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported scene key: ${SCENE_KEY}" >&2
    exit 1
    ;;
esac

OUT_ROOT="${EXPERIMENT_ROOT}/${STAMP}_${SCENE_KEY}_${QUERY_NAME}_joint_cluster"
PROPOSAL_DIR="${OUT_ROOT}/proposal_dir_jointembed_appearance"
REMOVAL_DIR="${OUT_ROOT}/removal_bundle"

mkdir -p "${OUT_ROOT}"

echo "Experiment root: ${OUT_ROOT}"

gs_python "${GS_ROOT}/scripts/build_joint_query_proposal_dir.py" \
  --run-dir "${RUN_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --tracks-path "${TRACKS_PATH}" \
  --output-dir "${PROPOSAL_DIR}" \
  --max-track-frames "${JOINT_CLUSTER_MAX_TRACK_FRAMES:-16}" \
  --proposal-keep-ratio "${JOINT_CLUSTER_KEEP_RATIO:-0.10}" \
  --min-gaussians "${JOINT_CLUSTER_MIN_GAUSSIANS:-2048}" \
  --max-gaussians "${JOINT_CLUSTER_MAX_GAUSSIANS:-4096}" \
  --embed-dim "${JOINT_CLUSTER_EMBED_DIM:-16}" \
  --num-steps "${JOINT_CLUSTER_NUM_STEPS:-500}" \
  --lr "${JOINT_CLUSTER_LR:-0.01}"

gs_python "${GS_ROOT}/scripts/build_query_removal_bundle.py" \
  --source-run-dir "${RUN_DIR}" \
  --query-root "${QUERY_ROOT}" \
  --proposal-entities-json "${PROPOSAL_DIR}/entities.json" \
  --proposal-alias "${PROPOSAL_ALIAS}" \
  --query-text "${QUERY_TEXT}" \
  --segment-start "${SEG_START}" \
  --segment-end "${SEG_END}" \
  --selection-notes "Joint worldtube appearance clustering experiment." \
  --output-root "${REMOVAL_DIR}" \
  --max-points "${JOINT_CLUSTER_MAX_POINTS:-40000}" \
  --frame-stride "${JOINT_CLUSTER_FRAME_STRIDE:-2}" \
  --fps "${JOINT_CLUSTER_FPS:-12}"

echo "${OUT_ROOT}"
