#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# != 2 ]]; then
  echo 'Usage: bash convert.sh /verified/BF16-parent /NEW/output-directory' >&2
  exit 2
fi
export PYTHONPATH="$bundle_dir/fish-source:$bundle_dir/runtime"
exec python "$bundle_dir/runtime/weights.py" --parent "$1" --output "$2"
