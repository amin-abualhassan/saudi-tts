#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omnivoice
export PYTHONPATH=/opt/dlami/nvme/OmniVoice:${PYTHONPATH:-}

python /opt/dlami/nvme/training/omnivoice/speaker1/prepare_dataset.py

CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
  --input_jsonl /opt/dlami/nvme/training/omnivoice/speaker1/manifests/train.jsonl \
  --tar_output_pattern /opt/dlami/nvme/training/omnivoice/speaker1/tokens/train/audios/shard-%06d.tar \
  --jsonl_output_pattern /opt/dlami/nvme/training/omnivoice/speaker1/tokens/train/txts/shard-%06d.jsonl \
  --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
  --nj_per_gpu 2 --loader_workers 4 \
  --shuffle True --shuffle-seed 42 \
  --samples_per_shard 48 \
  --min_length 1.0 --max_length 30.0 \
  --min_num_shards 1

CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
  --input_jsonl /opt/dlami/nvme/training/omnivoice/speaker1/manifests/dev.jsonl \
  --tar_output_pattern /opt/dlami/nvme/training/omnivoice/speaker1/tokens/dev/audios/shard-%06d.tar \
  --jsonl_output_pattern /opt/dlami/nvme/training/omnivoice/speaker1/tokens/dev/txts/shard-%06d.jsonl \
  --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
  --nj_per_gpu 2 --loader_workers 4 \
  --shuffle False \
  --samples_per_shard 64 \
  --min_length 1.0 --max_length 30.0 \
  --min_num_shards 1
