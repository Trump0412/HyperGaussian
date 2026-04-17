#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

QUERY_ROOT="${GS_ROOT}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie/entitybank/query_guided/split-cookie__the_complete_cookie_phaseaware"
while [[ ! -f "${QUERY_ROOT}/final_query_render_sourcebg/validation.json" ]]; do
  sleep 20
done

export DEEPFILL_SOURCE_RUN_DIR="${GS_ROOT}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie"
export DEEPFILL_QUERY_ROOT="${QUERY_ROOT}"
export GS_EXPERIMENT_STAMP=20260329_refresh_splitcookie_base040_sigma032
export DEEPFILL_EXPERIMENT_TAG=splitcookie_phaseaware_refresh_base040_sigma032

bash "${GS_ROOT}/scripts/run_scene_deepfill_removal_experiment.sh" split-cookie
