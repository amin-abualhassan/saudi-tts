# Fish Audio Experiment Notes

Inspected 2026-10-09. Miniconda 26.1.1 is installed at
`/opt/dlami/nvme/miniconda3`; the `fish-speech` environment uses Python 3.10.20.

## Activate the environment

```bash
source /opt/dlami/nvme/training/activate_fish.sh
cd /opt/dlami/nvme/training
python --version
```

The activation script uses workspace-local caches and disables environment
registration in the read-only home directory. It does not change shell profiles.
NumPy 1.26.4, PyYAML 6.0.3, and safetensors 0.7.0 are installed for inspection.
**The environment is not yet ready for training.** Package-server DNS fails,
and the local cached Torch/Torchaudio wheels are 2.1.2+cu121, whereas this
checkout and the old log require 2.8.0. GPU devices are unavailable here and
`nvidia-smi` cannot communicate with the driver.

Once package access is available, install the checkout's locked dependencies
using its documented CUDA extra; do not substitute the old cached Torch wheels.
A possible setup command, not yet executed or verified here, is:

```bash
python -m pip install uv
cd /opt/dlami/nvme/training/fish-speech
uv sync --active --python "$CONDA_PREFIX/bin/python" --extra cu128
```

Confirm GPU/driver compatibility before selecting a CUDA extra, then verify
`import torch, torchaudio, lightning` and `torch.cuda.is_available()`.

## Workspace map and data flow

- `training/fish-speech/`: the experiment checkout. Existing modified configs
  and web UI code were preserved. The top-level `fish-speech/` is a separate
  checkout, mainly containing additional server and studio work.
- `training/audio-preprocess/`: preprocessing package.
- `training/splitted_audios_original/`: original audio clips;
  `splitted_audios_original_transcripts/`: transcript labels.
- `training/splitted_ready_norm/`: normalized audio.
- `training/setup_data_dir.py`: copies normalized WAVs and matching `.lab`
  files into `data/speaker1/`. It can replace existing files if rerun.
- `training/data/speaker1/`: 367 WAVs, 367 labels, and 367 `.npy` token files;
  approximately 49.92 minutes, mono, 44,100 Hz. No missing paired files.
- `training/data/protos/00000000.protos`: packed training data already exists.
  The source pipeline is audio/label pairing → codec token extraction →
  protobuf packing → semantic-model LoRA training → adapter merge → inference.
- `training/checkpoints/s2-pro/`: S2 Pro base weights, tokenizer, and codec.
  Other S1 Mini and merged experiment directories also exist; do not mix them.
- `training/omnivoice/` and `chatter/`: separate experiments.

## Most recent identified Fish Speech run

`results/s2pro_speaker_lora_v2/` contains its resolved configuration in
`.hydra/config.yaml`, TensorBoard events, and checkpoints at steps
2,600, 2,700, 2,800, 2,900, and 3,000. The April 16 log records successful
completion at the configured limit of 3,000 steps, about 1 hour 50 minutes.
Final displayed validation loss was 11.30 and top-5 accuracy 0.754.

Saved settings:

- Base: `checkpoints/s2-pro`; maximum sequence length: 4,096.
- LoRA config: `r_32_alpha_16_fast`, but its actual saved rank **and alpha are
  both 32**. Targets: fast attention, MLP, embeddings, and output.
- Batch size: 4; accumulation: 1; workers: 4; precision: `bf16-mixed`.
- AdamW LR: `5e-5`; weight decay: `0.01`; epsilon: `1e-8`;
  betas: `[0.9, 0.95]`; warmup: 200 steps.
- Validation and checkpointing every 100 steps; five checkpoints retained.

The latest checkpoint archive contains 44 LoRA tensor names plus optimizer,
scheduler, and loop metadata, and records Lightning 2.6.1/global step 3,000.
This was a metadata inspection, not a numerical validation of the weights.
`checkpoints/s2-pro-speaker-merged/` also exists, but which adapter produced
it has not been established.

## Continuing the experiment

Use the saved resolved config as the source of truth. The current
`fish_speech/configs/text2semantic_finetune.yaml` instead defaults to 2,000
steps, `bf16-true`, LR `1e-4`, and `checkpoints/openaudio-s2-pro`, which does
not match the saved run or the available `s2-pro` directory.

The training entry point is `fish-speech/fish_speech/train.py`; launch from
`training/` with the checkout available on `PYTHONPATH` so relative data,
checkpoint, and output paths resolve correctly. Explicitly pass
`--config-name text2semantic_finetune`.

Before a restart:

1. Restore compatible training dependencies and GPU access.
2. Evaluate the saved adapters with the S2 Pro base to choose a starting point.
3. Decide between a full-state resume and an adapter-only continuation with
   a fresh optimizer. Full resume beyond step 3,000 requires a larger total
   `trainer.max_steps`; this value is the total target, not additional steps.
4. Check adapter-only state loading: `on_save_checkpoint` removes base tensors,
   so full resume may require non-strict model loading. The weights-only path
   uses `torch.load` without explicit `weights_only=False`; the saved checkpoint
   contains OmegaConf objects that can conflict with Torch 2.8's default loader.
5. Select an explicit checkpoint and separate output directory. The trainer
   auto-selects checkpoints by filesystem modification time and overrides an
   explicit `ckpt_path` when that output directory already has checkpoints.
   Several copied checkpoints have identical modification times.
6. Create a held-out evaluation set for the next experiment: this run's train
   and validation datasets both point at `data/protos`, so its validation
   metrics do not measure performance on unseen recordings.

`training/test.sh` is an old diagnostic script, not a test suite. Despite its
name, `fish_diag_report.json` contains Python source rather than a JSON report.
Its marker-based bug detector does not recognize the current code's direct
weight-copy pattern. Current `lora.py` copies base embedding/linear weights
when attaching adapters; no conclusion about older runs follows from that alone.

Reference pipeline: [Fish Speech fine-tuning guide](https://github.com/fishaudio/fish-speech/blob/main/docs/en/finetune.md).
No training, merging, dataset regeneration, or existing source edits were performed.
