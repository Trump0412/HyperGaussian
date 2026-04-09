#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GaussianStellar
export QUERY_ANNOTATION_DIR=/root/autodl-tmp/GaussianStellar/data/benchmarks/4dlangsplat/HyperNeRF-Annotation/split-cookie

bash scripts/run_query_specific_worldtube_pipeline.sh \
  /root/autodl-tmp/GaussianStellar/runs/stellar_tube_split-cookie_compare5k_weak/hypernerf/split-cookie \
  /root/autodl-tmp/GaussianStellar/data/hypernerf/misc/split-cookie \
  "the complete cookie" \
  split-cookie__the_complete_cookie_phaseaware_weak_tube
