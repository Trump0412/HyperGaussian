#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_grounded_sam2.sh" "$@"
