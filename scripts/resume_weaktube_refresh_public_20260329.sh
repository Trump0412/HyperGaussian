#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPO_ROOT="${GS_ROOT}"
REPORT_DIR="${REPO_ROOT}/reports/weaktube_refresh_suite_20260328"
PUBLIC_EVAL_DIR="${REPORT_DIR}/public_eval"
LOG_PATH="${REPORT_DIR}/resume_public_20260329.log"
SUMMARY_MD="${REPORT_DIR}/summary_public_refresh_20260329.md"

mkdir -p "${PUBLIC_EVAL_DIR}"

run_scene_protocol() {
  local scene="$1"
  local protocol_json="$2"
  local run_dir="$3"
  local dataset_dir="$4"
  local annotation_dir="$5"
  local query_root="$6"
  local output_json="$7"
  local output_md="$8"

  echo "[scene] ${scene}" | tee -a "${LOG_PATH}"
  QUERY_ANNOTATION_DIR="${annotation_dir}" \
    bash "${REPO_ROOT}/scripts/run_public_query_protocol.sh" \
      "${protocol_json}" \
      "${run_dir}" \
      "${dataset_dir}" | tee -a "${LOG_PATH}"

  gs_python "${REPO_ROOT}/scripts/evaluate_public_query_protocol.py" \
    --protocol-json "${protocol_json}" \
    --annotation-dir "${annotation_dir}" \
    --dataset-dir "${dataset_dir}" \
    --query-root "${query_root}" \
    --output-json "${output_json}" \
    --output-md "${output_md}" | tee -a "${LOG_PATH}"
}

write_summary() {
  python - <<'PY' "${SUMMARY_MD}" "${PUBLIC_EVAL_DIR}"
import json
import sys
from pathlib import Path

summary_md = Path(sys.argv[1])
public_eval_dir = Path(sys.argv[2])

files = {
    "americano": public_eval_dir / "americano_public_query_eval_refresh_20260328.json",
    "chickchicken": public_eval_dir / "chickchicken_public_query_eval_refresh_20260328.json",
    "espresso": public_eval_dir / "espresso_public_query_eval_refresh_20260328.json",
    "split-cookie": public_eval_dir / "split-cookie_public_query_eval_phaseaware_refresh_20260328.json",
}

lines = [
    "# WeakTube Refresh Public Summary 20260329",
    "",
    "| Scene | Acc | vIoU | tIoU | File |",
    "| --- | ---: | ---: | ---: | --- |",
]
for scene, path in files.items():
    if not path.exists():
        lines.append(f"| `{scene}` | n/a | n/a | n/a | `{path}` |")
        continue
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    acc = float(summary.get("Acc", 0.0)) * 100.0
    viou = float(summary.get("vIoU", 0.0)) * 100.0
    tiou = float(summary.get("temporal_tIoU", 0.0)) * 100.0
    lines.append(f"| `{scene}` | {acc:.2f} | {viou:.2f} | {tiou:.2f} | `{path}` |")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY
}

: > "${LOG_PATH}"
echo "[start] $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_PATH}"

run_scene_protocol \
  "americano" \
  "${REPO_ROOT}/reports/4dlangsplat_compare/protocol_splits/americano.json" \
  "${REPO_ROOT}/runs/stellar_tube_4dlangsplat_refresh_20260328_americano/hypernerf/americano" \
  "${REPO_ROOT}/data/hypernerf/misc/americano" \
  "${REPO_ROOT}/data/benchmarks/4dlangsplat/HyperNeRF-Annotation/americano" \
  "${REPO_ROOT}/runs/stellar_tube_4dlangsplat_refresh_20260328_americano/hypernerf/americano/entitybank/query_guided" \
  "${PUBLIC_EVAL_DIR}/americano_public_query_eval_refresh_20260328.json" \
  "${PUBLIC_EVAL_DIR}/americano_public_query_eval_refresh_20260328.md"

run_scene_protocol \
  "split-cookie" \
  "${REPO_ROOT}/reports/4dlangsplat_compare/split-cookie_query_protocol_phaseaware.json" \
  "${REPO_ROOT}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie" \
  "${REPO_ROOT}/data/hypernerf/misc/split-cookie" \
  "${REPO_ROOT}/data/benchmarks/4dlangsplat/HyperNeRF-Annotation/split-cookie" \
  "${REPO_ROOT}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie/entitybank/query_guided" \
  "${PUBLIC_EVAL_DIR}/split-cookie_public_query_eval_phaseaware_refresh_20260328.json" \
  "${PUBLIC_EVAL_DIR}/split-cookie_public_query_eval_phaseaware_refresh_20260328.md"

write_summary | tee -a "${LOG_PATH}"
echo "[done] $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_PATH}"
