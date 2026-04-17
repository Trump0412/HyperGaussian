#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR="${GS_ROOT}/runs/stellar_tube_cutlemon_refresh_20260329/hypernerf/cut-lemon1"
while [[ ! -f "${RUN_DIR}/point_cloud/iteration_14000/point_cloud.ply" ]]; do
  sleep 60
done

bash "${GS_ROOT}/scripts/run_query_specific_worldtube_pipeline.sh" \
  "${RUN_DIR}" \
  "${GS_ROOT}/data/hypernerf/interp/cut-lemon1" \
  "the lemon" \
  cut_the_lemon_final

export DEEPFILL_SOURCE_RUN_DIR="${RUN_DIR}"
export DEEPFILL_QUERY_ROOT="${RUN_DIR}/entitybank/query_guided/cut_the_lemon_final"
export GS_EXPERIMENT_STAMP=20260329_refresh_cutlemon_base040_sigma032
export DEEPFILL_EXPERIMENT_TAG=cutlemon_queryguided_refresh_base040_sigma032

bash "${GS_ROOT}/scripts/run_scene_deepfill_removal_experiment.sh" cut-lemon1
