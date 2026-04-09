#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PROTOCOL_JSON="${1:-${GS_ROOT}/reports/4dlangsplat_compare/public_query_protocol.json}"
RUN_SUFFIX="${2:-compare5k}"

if [[ ! -f "${PROTOCOL_JSON}" ]]; then
  echo "Missing protocol json: ${PROTOCOL_JSON}" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python - <<'PY' "${PROTOCOL_JSON}" | while IFS=$'\t' read -r group scene source_root; do
import json
import sys
from pathlib import Path

protocol_path = Path(sys.argv[1])
payload = json.loads(protocol_path.read_text())
seen = set()

source_candidates = {
    "misc": [
        Path("/root/autodl-tmp/data/HyperNeRF/misc"),
        Path("/root/autodl-tmp/data/HyperNeRF"),
        Path("/root/autodl-tmp/GaussianStellar/data/hypernerf/misc"),
    ],
    "interp": [
        Path("/root/autodl-tmp/data/HyperNeRF/interp"),
        Path("/root/autodl-tmp/data/HyperNeRF"),
        Path("/root/autodl-tmp/GaussianStellar/data/hypernerf/interp"),
    ],
}

for item in payload.get("queries", []):
    scene_path = str(item.get("scene", "")).strip()
    if not scene_path:
        continue
    parts = scene_path.split("/")
    if len(parts) < 3:
        continue
    group = parts[-2]
    scene = parts[-1]
    key = (group, scene)
    if key in seen:
        continue
    seen.add(key)
    resolved = None
    for candidate_root in source_candidates.get(group, []):
        candidate = candidate_root / scene
        if candidate.is_dir() and (candidate / "dataset.json").exists():
            resolved = candidate
            break
    if resolved is None:
        print(f"[missing]\t{group}\t{scene}\t", flush=True)
        continue
    print(f"{group}\t{scene}\t{resolved}", flush=True)
PY
  if [[ "${group}" == "[missing]" ]]; then
    echo "[skip] missing source for ${scene}" >&2
    continue
  fi

  baseline_run="${GS_ROOT}/runs/baseline_${scene}_${RUN_SUFFIX}/hypernerf/${scene}"
  worldtube_run="${GS_ROOT}/runs/stellar_worldtube_${scene}_${RUN_SUFFIX}/hypernerf/${scene}"
  if [[ "${COMPARE_FORCE_RERUN:-0}" != "1" && -f "${baseline_run}/metrics.json" && -f "${worldtube_run}/metrics.json" ]]; then
    echo "[skip] ${scene} already has baseline + worldtube metrics"
    continue
  fi

  echo "[queue] ${group}/${scene} from ${source_root}"
  bash "${GS_ROOT}/scripts/run_4dlangsplat_compare_entry.sh" "${scene}" "${source_root}" "${group}" "${RUN_SUFFIX}"
done
