#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${CODEC_PATH:?Set CODEC_PATH to the verified existing Linux codec.pth path}"
: "${REFERENCE_ROOT:=$bundle_dir/reference}"
python3 - "$CODEC_PATH" <<'PY'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert p.stat().st_size==1871099728 and h.hexdigest()=='74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f','Wrong codec'
PY
exec docker run --rm --name aljawab-nf4-laptop --gpus all \
  --read-only --tmpfs /tmp:rw,nosuid,size=512m --shm-size=256m \
  --security-opt no-new-privileges --cap-drop ALL \
  -p 127.0.0.1:3012:3012 \
  --mount "type=bind,source=$bundle_dir,target=/data,readonly" \
  --mount "type=bind,source=$CODEC_PATH,target=/original/codec.pth,readonly" \
  --mount "type=bind,source=$REFERENCE_ROOT,target=/references,readonly" \
  aljawab-fish-nf4:20261017
