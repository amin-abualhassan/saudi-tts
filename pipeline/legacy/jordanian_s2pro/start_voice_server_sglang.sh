#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /opt/dlami/nvme/miniconda3/envs/fish-speech/bin/python \
  "$SUITE_DIR/sglang_launch.py" "$@"

