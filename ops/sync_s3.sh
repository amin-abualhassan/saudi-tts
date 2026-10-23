#!/usr/bin/env bash
# Sync everything that must survive instance termination to S3.
# Usage: ops/sync_s3.sh [repo|keep|data|all]   (default: repo)
#   repo: ~/saudi-tts (code, state, reports, configs, manifests, dashboard data) minus .git and venvs
#   keep: /opt/dlami/nvme/keep  (kept checkpoints, voice library, eval renders) -> saudi-tts/keep/
#   data: /opt/dlami/nvme/data/processed (final training audio + manifests) -> saudi-tts/data/
set -euo pipefail
DEST=s3://aljawab-asr-data/saudi-tts
what=${1:-repo}
log() { echo "$(date -u +%FT%TZ) sync_s3 $*" >> "$HOME/saudi-tts/logs/sync_s3.log"; }
if [[ $what == repo || $what == all ]]; then
  aws s3 sync "$HOME/saudi-tts/" "$DEST/repo/" --only-show-errors \
    --exclude ".git/*" --exclude "*/.venv/*" --exclude "*__pycache__*" --exclude "logs/*.tmp" \
    --exclude "dashboard/state/*"
  log "repo ok"
fi
if [[ $what == keep || $what == all ]]; then
  mkdir -p /opt/dlami/nvme/keep
  aws s3 sync /opt/dlami/nvme/keep/ "$DEST/keep/" --only-show-errors
  log "keep ok"
fi
if [[ $what == data || $what == all ]]; then
  mkdir -p /opt/dlami/nvme/data/processed
  aws s3 sync /opt/dlami/nvme/data/processed/ "$DEST/data/" --only-show-errors
  log "data ok"
fi
