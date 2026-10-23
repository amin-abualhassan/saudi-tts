#!/usr/bin/env bash
set -euo pipefail
cd /opt/dlami/nvme/speech-agent
exec .venv/bin/uvicorn app.server:app --host ::1 --port 8002 --no-access-log --no-server-header --log-level warning
