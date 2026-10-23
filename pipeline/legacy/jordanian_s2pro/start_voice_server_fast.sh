#!/usr/bin/env bash
set -euo pipefail
voice_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Compatibility entrypoint after rollback: use the original decoder and server.
echo "Fast serving was rolled back; starting the original voice server."
exec bash "$voice_directory/start_voice_server.sh" --replace-voice-server "$@"
