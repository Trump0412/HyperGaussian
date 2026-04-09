#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/data/dnerf"
DOWNLOAD_ROOT="${ROOT_DIR}/data/downloads"
ZIP_PATH="${DOWNLOAD_ROOT}/dnerf_data.zip"
URL="https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1"

ALL_SCENES=0
SCENE="bouncingballs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      ALL_SCENES=1
      shift
      ;;
    --scene)
      SCENE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${DATA_ROOT}" "${DOWNLOAD_ROOT}"
if [[ ! -f "${ZIP_PATH}" ]]; then
  wget -O "${ZIP_PATH}" "${URL}"
fi

python - <<'PY' "${ZIP_PATH}" "${DATA_ROOT}" "${SCENE}" "${ALL_SCENES}"
import os
import shutil
import sys
import tempfile
import zipfile

zip_path, data_root, scene, all_scenes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
scene_names = {
    "bouncingballs",
    "hellwarrior",
    "hook",
    "jumpingjacks",
    "lego",
    "mutant",
    "standup",
    "trex",
}

with zipfile.ZipFile(zip_path) as archive:
    members = archive.namelist()
    temp_dir = tempfile.mkdtemp(prefix="dnerf_extract_", dir=os.path.dirname(data_root))
    archive.extractall(temp_dir)
    source_root = temp_dir
    nested = [os.path.join(temp_dir, name) for name in os.listdir(temp_dir)]
    if len(nested) == 1 and os.path.isdir(nested[0]):
        source_root = nested[0]

    targets = scene_names if all_scenes else {scene}
    for target in targets:
        found = None
        for root, dirs, _files in os.walk(source_root):
            if target in dirs:
                found = os.path.join(root, target)
                break
        if found is None:
            raise SystemExit(f"Unable to locate D-NeRF scene '{target}' inside {zip_path}")
        destination = os.path.join(data_root, target)
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.move(found, destination)
    shutil.rmtree(temp_dir)
PY

echo "Prepared D-NeRF dataset at ${DATA_ROOT}"

