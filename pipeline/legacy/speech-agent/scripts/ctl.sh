#!/usr/bin/env bash
set -euo pipefail
cd /opt/dlami/nvme/speech-agent
turn_units=()
if [ -f /etc/systemd/system/aljawab-turn-proxy.service ]; then
 turn_units=(aljawab-turn-proxy aljawab-turn-renew.timer)
fi
case "${1:-status}" in
 start) sudo systemctl start aljawab-livekit aljawab-app aljawab-tunnel "${turn_units[@]}" ;;
 stop) if [ ${#turn_units[@]} -gt 0 ]; then sudo systemctl stop aljawab-turn-proxy; fi; sudo systemctl stop aljawab-tunnel aljawab-app aljawab-livekit ;;
 restart) sudo systemctl restart aljawab-app; sudo systemctl start aljawab-tunnel "${turn_units[@]}" ;;
 status) systemctl --no-pager --lines=0 status aljawab-livekit aljawab-app aljawab-tunnel "${turn_units[@]}"; rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' logs/tunnel.log | tail -1 ;;
 *) echo 'Usage: scripts/ctl.sh start|stop|restart|status'; exit 2 ;;
esac
