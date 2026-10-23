#!/usr/bin/env bash
set -euo pipefail
SUITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$EUID" -ne 0 ]]; then
  exec sudo -- bash "$SUITE_DIR/setup_sglang_docker_nvme.sh" "$@"
fi
exec /usr/bin/python3 "$SUITE_DIR/sglang_docker_storage.py" --install "$@"
