#!/usr/bin/env bash
# Start independently of the training screen; use the existing Fish environment.
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_PYTHON="$APP_DIR/../../miniconda3/envs/fish-speech/bin/python"
mkdir -p "$APP_DIR/testing"
if [[ -f "$APP_DIR/testing/app.pid" ]]; then
    app_pid="$(cat "$APP_DIR/testing/app.pid")"
    if [[ "$app_pid" =~ ^[0-9]+$ ]] && kill -0 "$app_pid" 2>/dev/null; then
        if [[ -r "/proc/$app_pid/cmdline" ]] && tr '\0' ' ' < "/proc/$app_pid/cmdline" | grep -Fq "$APP_DIR/testing_app.py"; then
            echo "Testing app is already running (PID $app_pid). Open http://<server-ip>:8002"
            exit 0
        fi
    fi
fi
nohup "$TEST_PYTHON" "$APP_DIR/testing_app.py" "$@" >> "$APP_DIR/testing/startup.log" 2>&1 < /dev/null &
app_pid=$!
sleep 2
if ! kill -0 "$app_pid" 2>/dev/null; then
    echo "Testing app could not start. See $APP_DIR/testing/startup.log" >&2
    tail -n 20 "$APP_DIR/testing/startup.log" >&2
    exit 1
fi
echo "Testing app is running in the background (PID $app_pid). Open http://<server-ip>:8002"
echo "Logs: $APP_DIR/testing/app.log and $APP_DIR/testing/startup.log"
