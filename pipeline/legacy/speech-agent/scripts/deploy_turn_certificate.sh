#!/usr/bin/env bash
set -euo pipefail
CERT_DIR=/etc/letsencrypt-aljawab/live/aljawab-turn-ip
TARGET_DIR=/etc/aljawab-turn
install -d -m 700 "$TARGET_DIR"
openssl x509 -in "$CERT_DIR/fullchain.pem" -noout -checkip 98.83.42.166
openssl x509 -in "$CERT_DIR/fullchain.pem" -noout -checkend 86400
umask 077
cat "$CERT_DIR/fullchain.pem" "$CERT_DIR/privkey.pem" > "$TARGET_DIR/turn.pem.next"
mv "$TARGET_DIR/turn.pem.next" "$TARGET_DIR/turn.pem"
if systemctl is-active --quiet aljawab-turn-proxy; then
    systemctl reload aljawab-turn-proxy
fi
