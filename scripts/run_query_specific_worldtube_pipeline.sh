#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_4dgaussians
require_grounded_sam2

RUN_DIR="$1"
DATASET_DIR="$2"
QUERY_TEXT="$3"
QUERY_NAME="${4:-$(echo "${QUERY_TEXT}" | tr ' ' '_' | tr -cd '[:alnum:]_-' | cut -c1-80)}"

OUTPUT_ROOT="${RUN_DIR}/entitybank/query_guided/${QUERY_NAME}"
TRACK_DIR="${OUTPUT_ROOT}/grounded_sam2"
TRACKS_PATH="${TRACK_DIR}/grounded_sam2_query_tracks.json"
PROPOSAL_DIR="${OUTPUT_ROOT}/proposal_dir"
QUERY_ENTITYBANK_DIR="${OUTPUT_ROOT}/query_entitybank"
QUERY_RUN_DIR="${OUTPUT_ROOT}/query_worldtube_run"
ENTITY_LIBRARY_DIR="${OUTPUT_ROOT}/entity_library_qwen_sourcebg"
QWEN_ASSIGNMENTS_PATH="${QUERY_RUN_DIR}/entitybank/semantic_assignments_qwen.json"
QWEN_SELECTION_PATH="${QUERY_RUN_DIR}/entitybank/selected_query_qwen.json"
FINAL_RENDER_DIR="${OUTPUT_ROOT}/final_query_render_sourcebg"
DIAGNOSTIC_DIR="${OUTPUT_ROOT}/diagnostics"
FINAL_VALIDATION_PATH="${FINAL_RENDER_DIR}/validation.json"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${QUERY_FORCE_RERUN:-0}" != "1" && -f "${FINAL_VALIDATION_PATH}" ]]; then
  echo "[skip] existing final validation for ${QUERY_NAME}: ${FINAL_VALIDATION_PATH}"
  echo "${QUERY_RUN_DIR}"
  exit 0
fi

bash "${GS_ROOT}/scripts/run_query_guided_grounded_sam2.sh" \
  "${RUN_DIR}" \
  "${DATASET_DIR}" \
  "${QUERY_TEXT}" \
  "${QUERY_NAME}"

gs_python "${GS_ROOT}/scripts/build_query_proposal_dir.py" \
  --run-dir "${RUN_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --tracks-path "${TRACKS_PATH}" \
  --output-dir "${PROPOSAL_DIR}" \
  --max-track-frames "${QUERY_MAX_TRACK_FRAMES:-16}" \
  --proposal-keep-ratio "${QUERY_PROPOSAL_KEEP_RATIO:-0.03}" \
  --min-gaussians "${QUERY_MIN_GAUSSIANS:-256}" \
  --max-gaussians "${QUERY_MAX_GAUSSIANS:-4096}" \
  --opacity-power "${QUERY_OPACITY_POWER:-0.0}" \
  --cluster-mode "${QUERY_CLUSTER_MODE:-support_only}" \
  --seed-ratio "${QUERY_SEED_RATIO:-0.05}" \
  --expansion-factor "${QUERY_EXPANSION_FACTOR:-4.0}"

gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
  --run-dir "${RUN_DIR}" \
  --proposal-dir "${PROPOSAL_DIR}" \
  --proposal-strict \
  --output-dir "${QUERY_ENTITYBANK_DIR}" \
  --max-entities "${QUERY_MAX_ENTITIES:-12}" \
  --min-gaussians-per-entity "${QUERY_MIN_GAUSSIANS_PER_ENTITY:-32}"

mkdir -p "${QUERY_RUN_DIR}"
ln -sfn "${RUN_DIR}/config.yaml" "${QUERY_RUN_DIR}/config.yaml"
ln -sfn "${RUN_DIR}/point_cloud" "${QUERY_RUN_DIR}/point_cloud"
ln -sfn "${RUN_DIR}/test" "${QUERY_RUN_DIR}/test"
ln -sfn "${QUERY_ENTITYBANK_DIR}" "${QUERY_RUN_DIR}/entitybank"

gs_python "${GS_ROOT}/scripts/export_semantic_slots.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_semantic_tracks.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_semantic_priors.py" --run-dir "${QUERY_RUN_DIR}"
gs_python "${GS_ROOT}/scripts/export_native_semantics.py" --run-dir "${QUERY_RUN_DIR}"
ASSIGNMENTS_PATH="${QWEN_ASSIGNMENTS_PATH}"
if [[ "${QUERY_SKIP_QWEN_EXPORT:-0}" == "1" ]]; then
  ASSIGNMENTS_PATH="${QUERY_RUN_DIR}/entitybank/native_semantic_assignments.json"
elif [[ "${QUERY_REUSE_QWEN_EXPORT:-0}" == "1" && -f "${QWEN_ASSIGNMENTS_PATH}" ]]; then
  echo "Reusing existing Qwen assignments: ${QWEN_ASSIGNMENTS_PATH}"
else
  gsam2_python "${GS_ROOT}/scripts/export_qwen_semantics.py" \
    --run-dir "${QUERY_RUN_DIR}" \
    --query "${QUERY_TEXT}" \
    --max-entities "${QUERY_QWEN_MAX_ENTITIES:-12}"
fi

if [[ "${QUERY_SKIP_ENTITY_LIBRARY:-0}" != "1" ]]; then
  gs_python "${GS_ROOT}/scripts/export_entity_library.py" \
    --run-dir "${QUERY_RUN_DIR}" \
    --dataset-dir "${DATASET_DIR}" \
    --assignments-path "${ASSIGNMENTS_PATH}" \
    --output-root "${ENTITY_LIBRARY_DIR}" \
    --background-mode source \
    --fps "${QUERY_RENDER_FPS:-6}" \
    --stride "${QUERY_RENDER_STRIDE:-1}"
fi

gsam2_python "${GS_ROOT}/scripts/select_qwen_query_entities.py" \
  --assignments-path "${ASSIGNMENTS_PATH}" \
  --query "${QUERY_TEXT}" \
  --query-plan-path "${OUTPUT_ROOT}/query_plan.json" \
  --output-path "${QWEN_SELECTION_PATH}"

gs_python "${GS_ROOT}/scripts/render_query_video.py" \
  --run-dir "${QUERY_RUN_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --selection-path "${QWEN_SELECTION_PATH}" \
  --output-dir "${FINAL_RENDER_DIR}" \
  --background-mode source \
  --fps "${QUERY_RENDER_FPS:-6}" \
  --stride "${QUERY_RENDER_STRIDE:-1}"

if [[ -n "${QUERY_ANNOTATION_DIR:-}" && -d "${QUERY_ANNOTATION_DIR}" ]]; then
  gs_python "${GS_ROOT}/scripts/export_query_diagnostics.py" \
    --query-root "${OUTPUT_ROOT}" \
    --dataset-dir "${DATASET_DIR}" \
    --annotation-dir "${QUERY_ANNOTATION_DIR}" \
    --output-dir "${DIAGNOSTIC_DIR}"
fi

echo "${QUERY_RUN_DIR}"
