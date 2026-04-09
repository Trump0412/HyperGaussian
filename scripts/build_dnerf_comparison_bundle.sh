#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

OUTPUT_ROOT="${GS_ROOT}/reports/dnerf_comparisons"
mkdir -p "${OUTPUT_ROOT}"

gs_python "${GS_ROOT}/scripts/build_benchmark_report.py" \
  --title "D-NeRF Full Benchmark" \
  --subtitle "Full-budget comparison on representative D-NeRF scenes." \
  --entry "baseline_mutant=${GS_ROOT}/runs/baseline_4dgs_full/dnerf/mutant" \
  --entry "chrono_density_mutant=${GS_ROOT}/runs/chronometric_4dgs_full/density/dnerf/mutant" \
  --entry "stellar_core_mutant=${GS_ROOT}/runs/stellar_core_full/dnerf/mutant" \
  --entry "stellar_spacetime_mutant=${GS_ROOT}/runs/stellar_spacetime_full/dnerf/mutant" \
  --entry "baseline_standup=${GS_ROOT}/runs/baseline_4dgs_full/dnerf/standup" \
  --entry "chrono_density_standup=${GS_ROOT}/runs/chronometric_4dgs_full/density/dnerf/standup" \
  --entry "stellar_core_standup=${GS_ROOT}/runs/stellar_core_full/dnerf/standup" \
  --output "${OUTPUT_ROOT}/dnerf_full_bundle_metrics.md"

gs_python "${GS_ROOT}/scripts/export_comparison_frames.py" \
  --title "Mutant Full Frame 00010" \
  --frame-name 00010.png \
  --columns 2 \
  --entry "baseline 4DGS=${GS_ROOT}/runs/baseline_4dgs_full/dnerf/mutant" \
  --entry "chrono density=${GS_ROOT}/runs/chronometric_4dgs_full/density/dnerf/mutant" \
  --entry "stellar core=${GS_ROOT}/runs/stellar_core_full/dnerf/mutant" \
  --entry "stellar spacetime=${GS_ROOT}/runs/stellar_spacetime_full/dnerf/mutant" \
  --output "${OUTPUT_ROOT}/mutant_full_frame_00010.png"

gs_python "${GS_ROOT}/scripts/export_comparison_frames.py" \
  --title "Standup Full Frame 00010" \
  --frame-name 00010.png \
  --columns 2 \
  --entry "baseline 4DGS=${GS_ROOT}/runs/baseline_4dgs_full/dnerf/standup" \
  --entry "chrono density=${GS_ROOT}/runs/chronometric_4dgs_full/density/dnerf/standup" \
  --entry "stellar core=${GS_ROOT}/runs/stellar_core_full/dnerf/standup" \
  --entry "spacetime quad=${GS_ROOT}/runs/stellar_spacetime_quad_pilot/dnerf/standup" \
  --output "${OUTPUT_ROOT}/standup_frame_00010.png"

gs_python "${GS_ROOT}/scripts/export_comparison_frames.py" \
  --title "Bouncingballs Smoke Frame 00010" \
  --frame-name 00010.png \
  --columns 2 \
  --entry "baseline 4DGS=${GS_ROOT}/runs/baseline_4dgs/dnerf/bouncingballs" \
  --entry "chrono density=${GS_ROOT}/runs/chronometric_4dgs/density/dnerf/bouncingballs" \
  --entry "stellar tube weak=${GS_ROOT}/runs/stellar_tube_weak_da3_smoke/dnerf/bouncingballs" \
  --entry "worldtube smoke=${GS_ROOT}/runs/stellar_worldtube_segment_smoke40/dnerf/bouncingballs" \
  --output "${OUTPUT_ROOT}/bouncingballs_frame_00010.png"

gs_python "${GS_ROOT}/scripts/export_comparison_gif.py" \
  --title "Mutant Full" \
  --frame-step 4 \
  --max-frames 24 \
  --entry "baseline 4DGS=${GS_ROOT}/runs/baseline_4dgs_full/dnerf/mutant" \
  --entry "chrono density=${GS_ROOT}/runs/chronometric_4dgs_full/density/dnerf/mutant" \
  --entry "stellar core=${GS_ROOT}/runs/stellar_core_full/dnerf/mutant" \
  --entry "stellar spacetime=${GS_ROOT}/runs/stellar_spacetime_full/dnerf/mutant" \
  --output "${OUTPUT_ROOT}/mutant_full_compare.gif"

gs_python "${GS_ROOT}/scripts/export_comparison_gif.py" \
  --title "Standup Compare" \
  --frame-step 4 \
  --max-frames 24 \
  --entry "baseline 4DGS=${GS_ROOT}/runs/baseline_4dgs_full/dnerf/standup" \
  --entry "chrono density=${GS_ROOT}/runs/chronometric_4dgs_full/density/dnerf/standup" \
  --entry "stellar core=${GS_ROOT}/runs/stellar_core_full/dnerf/standup" \
  --entry "worldtube v6a=${GS_ROOT}/runs/stellar_worldtube_standup_pilot_v6a/dnerf/standup" \
  --output "${OUTPUT_ROOT}/standup_compare.gif"

echo "Wrote D-NeRF comparison bundle to ${OUTPUT_ROOT}"
