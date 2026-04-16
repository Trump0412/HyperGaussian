#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias: render + metrics + exports.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_stellar_tube.sh" "$@"
