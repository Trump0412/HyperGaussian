#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

ANNOT_ROOT="${GS_ROOT}/data/benchmarks/4dlangsplat/HyperNeRF-Annotation"
RAW_ROOT="${HYPERNERF_RAW_ROOT:-${GS_ROOT}/data/raw/HyperNeRF}"
RUN_SUFFIX="${1:-compare5k}"
shift || true

resolve_group() {
  local scene="$1"
  case "${scene}" in
    chickchicken|torchocolate)
      printf 'interp'
      ;;
    americano|espresso|keyboard|split-cookie)
      printf 'misc'
      ;;
    *)
      return 1
      ;;
  esac
}

collect_scenes() {
  if [[ "$#" -gt 0 ]]; then
    printf '%s\n' "$@"
    return
  fi
  find "${ANNOT_ROOT}" -mindepth 2 -maxdepth 2 -name "video_annotations.json" -printf '%h\n' \
    | xargs -r -n1 basename \
    | sort
}

main() {
  local scenes
  mapfile -t scenes < <(collect_scenes "$@")
  if [[ "${#scenes[@]}" -eq 0 ]]; then
    echo "No public 4DLangSplat scenes with video_annotations.json found under ${ANNOT_ROOT}" >&2
    exit 2
  fi

  for scene in "${scenes[@]}"; do
    local group
    group="$(resolve_group "${scene}")" || {
      echo "Skipping unsupported scene-group mapping: ${scene}" >&2
      continue
    }
    local source_root="${RAW_ROOT}/${group}/${scene}"
    if [[ ! -d "${source_root}" ]]; then
      echo "Skipping ${scene}: missing raw HyperNeRF source ${source_root}" >&2
      continue
    fi
    bash "${GS_ROOT}/scripts/run_4dlangsplat_compare_entry.sh" "${scene}" "${source_root}" "${group}" "${RUN_SUFFIX}"
  done
}

main "$@"
