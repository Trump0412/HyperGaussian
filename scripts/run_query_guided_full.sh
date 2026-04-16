#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR="${1:-}"
DATASET_DIR="${2:-}"
QUERY_TEXT="${3:-}"
QUERY_NAME="${4:-}"

if [[ -z "${RUN_DIR}" || -z "${DATASET_DIR}" || -z "${QUERY_TEXT}" ]]; then
  echo "Usage: $0 <run_dir> <dataset_dir> <query_text> [query_name]" >&2
  exit 2
fi

if [[ -n "${QUERY_NAME}" ]]; then
  exec bash "${SCRIPT_DIR}/run_query_specific_worldtube_pipeline.sh" \
    "${RUN_DIR}" "${DATASET_DIR}" "${QUERY_TEXT}" "${QUERY_NAME}"
else
  exec bash "${SCRIPT_DIR}/run_query_specific_worldtube_pipeline.sh" \
    "${RUN_DIR}" "${DATASET_DIR}" "${QUERY_TEXT}"
fi
