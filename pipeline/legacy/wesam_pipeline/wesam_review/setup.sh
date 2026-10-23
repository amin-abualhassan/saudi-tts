#!/usr/bin/env bash
set -euo pipefail
review_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "$review_dir/../.." && pwd)"
export PIP_CACHE_DIR="$workspace_dir/pip_cache"
"$workspace_dir/miniconda3/envs/fish-speech/bin/python" -m venv "$review_dir/.venv"
"$review_dir/.venv/bin/python" -m pip install 'torch==2.8.0' 'torchaudio==2.8.0' 'torchcodec==0.7.0' \
  'pyannote.audio>=4,<5' soundfile scipy 'silero-vad>=6,<7' \
  'speechbrain>=1.1,<2' 'transformers==4.57.3'
"$review_dir/.venv/bin/python" "$review_dir/review.py" --check

"$review_dir/.venv/bin/python" -m pip freeze > "$review_dir/.venv/installed-requirements.txt"
