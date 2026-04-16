#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 <run_dir> [extra args ...]" >&2
  exit 2
fi
shift

gs_python "${GS_ROOT}/scripts/export_entitybank.py" --run-dir "${RUN_DIR}" "$@"
