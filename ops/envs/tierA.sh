#!/usr/bin/env bash
# Create the per-model environments for the tier-A bake-off (one uv venv each; never shared).
set -u
UV=$HOME/.local/bin/uv
SRC=/opt/dlami/nvme/src; ENVS=/opt/dlami/nvme/envs
export UV_CACHE_DIR=/opt/dlami/nvme/uv-cache UV_LINK_MODE=copy
mk() { local name=$1 dir=$2; shift 2
  echo "== $(date -u +%T) $name"
  ( cd "$SRC/$dir" && UV_PROJECT_ENVIRONMENT=$ENVS/$name $UV sync --python 3.12 "$@" ) && echo "OK $name" || echo "FAIL $name"
}
mk fish fish-speech --extra cu129
mk omnivoice OmniVoice
mk voxcpm VoxCPM
for e in fish omnivoice voxcpm; do
  $ENVS/$e/bin/python -c "import torch;print('$e', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
done
