#!/usr/bin/env bash
set -euo pipefail
review_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "$review_dir/../.." && pwd)"
review_python="$review_dir/.venv/bin/python"
if [[ ! -x "$review_python" ]]; then
  review_python="$workspace_dir/miniconda3/envs/fish-speech/bin/python"
  echo 'Playback is ready. For speaker analysis, run training/wesam_review/setup.sh first.'
fi
export HF_HUB_CACHE="$review_dir/cache/huggingface/hub"
export PYANNOTE_METRICS_ENABLED=0
exec "$review_python" "$review_dir/review.py" --port 3000 "$@"
