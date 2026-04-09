#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_ID="${1:-IDEA-Research/grounding-dino-base}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Prefetching detector model: ${MODEL_ID}"
gs_python - <<PY
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

model_id = "${MODEL_ID}"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
print("processor", type(processor).__name__)
print("model", type(model).__name__)
print("cached", model_id)
PY
