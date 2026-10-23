#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omnivoice
export PYTHONPATH=/opt/dlami/nvme/OmniVoice:${PYTHONPATH:-}

OUTPUT_DIR=/opt/dlami/nvme/training/omnivoice/speaker1/exp/full_finetune

accelerate launch \
  --gpu_ids 0 \
  --num_processes 1 \
  -m omnivoice.cli.train \
  --train_config /opt/dlami/nvme/training/omnivoice/speaker1/config/train_config.json \
  --data_config /opt/dlami/nvme/training/omnivoice/speaker1/config/data_config.json \
  --output_dir "${OUTPUT_DIR}"
