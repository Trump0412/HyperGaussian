#!/usr/bin/env bash
set -euo pipefail
QUERY_ROOT="/root/autodl-tmp/GaussianStellar/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie/entitybank/query_guided/split-cookie__the_complete_cookie_phaseaware"
while [[ ! -f "${QUERY_ROOT}/final_query_render_sourcebg/validation.json" ]]; do
  sleep 20
done
source /root/autodl-tmp/GaussianStellar/scripts/common.sh
export DEEPFILL_SOURCE_RUN_DIR=/root/autodl-tmp/GaussianStellar/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie
export DEEPFILL_QUERY_ROOT="${QUERY_ROOT}"
export GS_EXPERIMENT_STAMP=20260329_refresh_splitcookie_base040_sigma032
export DEEPFILL_EXPERIMENT_TAG=splitcookie_phaseaware_refresh_base040_sigma032
bash /root/autodl-tmp/GaussianStellar/scripts/run_scene_deepfill_removal_experiment.sh split-cookie
