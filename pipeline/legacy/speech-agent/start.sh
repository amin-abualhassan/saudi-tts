#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT_DIR/scripts/ctl.sh" start
"$ROOT_DIR/.venv/bin/python" - <<'PY'
import time
import httpx
for attempt in range(30):
    try:
        response=httpx.get('http://[::1]:8002/health',trust_env=False,timeout=2)
        response.raise_for_status()
        break
    except httpx.HTTPError:
        if attempt==29:raise SystemExit('Voice agent did not become healthy; inspect logs/app.log')
        time.sleep(1)
PY
echo 'Voice agent started in screen session speech-agent. Attach: screen -r speech-agent'
bash "$ROOT_DIR/scripts/ctl.sh" status
