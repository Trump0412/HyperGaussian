#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${GS_ROOT}"
export QUERY_ANNOTATION_DIR="${GS_ROOT}/data/benchmarks/4dlangsplat/HyperNeRF-Annotation/split-cookie"

bash "${GS_ROOT}/scripts/run_query_specific_worldtube_pipeline.sh" \
  "${GS_ROOT}/runs/stellar_tube_split-cookie_compare5k_weak/hypernerf/split-cookie" \
  "${GS_ROOT}/data/hypernerf/misc/split-cookie" \
  "the complete cookie" \
  split-cookie__the_complete_cookie_phaseaware_weak_tube
