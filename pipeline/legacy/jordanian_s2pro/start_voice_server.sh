#!/usr/bin/env bash
set -euo pipefail
voice_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /opt/dlami/nvme/miniconda3/envs/fish-speech/bin/python \
    "$voice_directory/voice_server.py" --port 7860 --replace-dashboard "$@"
