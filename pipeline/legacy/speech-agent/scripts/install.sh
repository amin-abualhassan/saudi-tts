#!/usr/bin/env bash
set -euo pipefail
cd /opt/dlami/nvme/speech-agent
mkdir -p bin logs assets/voice knowledge
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/playwright install chromium
if [ ! -x bin/livekit-server ]; then
 curl -fsSL https://github.com/livekit/livekit/releases/download/v1.13.10/livekit_1.13.10_linux_amd64.tar.gz -o /tmp/aljawab-livekit.tar.gz
 tar -xzf /tmp/aljawab-livekit.tar.gz -C bin livekit-server
fi
if [ ! -x bin/cloudflared ]; then
 curl -fsSL https://github.com/cloudflare/cloudflared/releases/download/2026.10.1/cloudflared-linux-amd64 -o bin/cloudflared
 chmod +x bin/cloudflared
fi
if [ ! -x bin/node-v22.21.1-linux-x64/bin/node ]; then
 curl -fsSL https://nodejs.org/dist/v22.21.1/node-v22.21.1-linux-x64.tar.xz -o /tmp/aljawab-node.tar.xz
 tar -xJf /tmp/aljawab-node.tar.xz -C bin
fi
export PATH="$PWD/bin/node-v22.21.1-linux-x64/bin:$PATH"
sha256sum -c deployment/binaries.sha256
npm --prefix web ci
npm --prefix web run check
npm --prefix web run build
.venv/bin/python -m scripts.configure
.venv/bin/python -m scripts.configure_noura
.venv/bin/python -m scripts.refresh_knowledge
.venv/bin/python -m scripts.generate_clips
sudo cp deployment/aljawab-livekit.service deployment/aljawab-app.service deployment/aljawab-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aljawab-livekit aljawab-app aljawab-tunnel
echo 'Aljawab installed. Run scripts/ctl.sh status for the current HTTPS URL.'
