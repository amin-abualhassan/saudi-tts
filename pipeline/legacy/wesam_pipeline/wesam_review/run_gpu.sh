#!/usr/bin/env bash
set -euo pipefail
review_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_CACHE="$review_dir/cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export PYANNOTE_METRICS_ENABLED=0
export MPLCONFIGDIR=/tmp/wesam-matplotlib
"$review_dir/.venv/bin/python" "$review_dir/stop_cpu.py"
"$review_dir/.venv/bin/python" "$review_dir/benchmark_gpu.py"
gpu_output="$review_dir/../wesam_dataset_gpu"
mkdir -p "$gpu_output"
"$review_dir/.venv/bin/python" "$review_dir/prepare_dataset.py" \
    --device cuda --threads 4 --segmentation-batch-size 64 --embedding-batch-size 64 \
    --out "$gpu_output" 2>&1 | tee -a "$gpu_output/prepare.log"
