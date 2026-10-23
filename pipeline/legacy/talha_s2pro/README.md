# Talha Ahmad / S2 Pro preparation and review

Prepared 2026-10-09. **No bulk audio preparation, codec extraction, protobuf
packing, model merging, or training has run during setup verification.** Only metadata/transcript planning
and small CPU previews/quality-replacement checks have run. Existing experiment code and data
were preserved.

## Listening lab on port 8002

Start the testing app now, independently of the training screen:

```bash
bash /opt/dlami/nvme/training/talha_s2pro/start_testing.sh
```

Open `http://<server-ip>:8002`. No additional web packages or conda activation are
needed: the background launcher uses this workspace's existing Fish environment.
The page refreshes every three seconds without clearing text or restarting audio.
It displays training progress/loss, GPU memory, disk space, model readiness and logs.

The app follows the current run recorded in `monitor/status.json`, waits for its
completion checkpoint and atomic `merged.json` receipt, verifies merged weights,
tokenizer, provenance and the original codec, then automatically loads inference
in a separate process. It defers loading while training workers run or GPU free
memory is below 16 GB. A new completed model reloads automatically. Loading and
generation failures stay visible and the worker retries after 60 seconds.

Choose one of three paired training references from distinct videos, enter plain
text, and generate/download a WAV. Reference audio, labels and cached codec tokens
are checked before use. Seeds, temperature, Top P and audio token limits are saved
with each result under `testing/results/`; GPU requests run one at a time. The app
uses native upstream generation/codec code with BF16 and a 4096-token context.
No intermediate training checkpoint is merged or loaded by the testing app.

Use `python testing_app.py --check` from this directory for discovery/system checks
without loading weights. Pass `--run-dir /absolute/path/to/runs/<run>` to the launch
script to pin an experiment. Logs are `testing/app.log` and `testing/startup.log`.
Stop it with `kill "$(cat /opt/dlami/nvme/training/talha_s2pro/testing/app.pid)"`.
The GPU worker is stopped with the app. The existing training dashboard remains
on port 7860.

CPU regression tests cover model discovery, integrity, training deferral, automatic
loading, request validation, serial generation, HTTP handlers and WAV writing with
mocked GPU models. Real fine-tuned synthesis requires training/merge completion and
the server's GPU. This Codex sandbox denies listening sockets, so launch from the
normal server shell if socket creation reports `Operation not permitted`.

## One-command launcher

`run.py` is the entry point for preparation, codec extraction, training, checkpoint
resume, merging, and a read-only dashboard on **port 7860**. It uses the adjacent
pipeline/configuration helpers and the pinned upstream checkout; keep these files
together. Running without a check/preview flag starts real work.

On your GPU server, start a screen session and run:

```bash
screen -S talha-s2pro
source /opt/dlami/nvme/training/activate_fish.sh
cd /opt/dlami/nvme
python training/talha_s2pro/run.py --install
```

`--install` bootstraps uv and installs the training/codec dependencies into this
workspace's `fish-speech` conda environment. It pins Torch/Torchaudio 2.8,
Lightning 2.6.1, TensorBoard 2.20.0, Transformers 4.57.3 and datasets 2.18, and applies upstream's
protobuf override. Network access and a working NVIDIA driver are required.
Once dependencies are installed, subsequent runs can omit `--install`.
TensorBoard is pinned because 2.21 requires Protobuf 6.31.1+, while upstream's
codec dependency override keeps Protobuf below 6. Do not upgrade TensorBoard
independently. The preflight imports its generated protos and the training schema.
Detach screen with **Ctrl-A, then D**; reconnect with `screen -r talha-s2pro`.
Open `http://<server-ip>:7860`. The server must permit access to that port.

The launcher runs these steps automatically:

1. Bind the dashboard, acquire an exclusive lock, and check imports, model shards,
   the clean pinned checkout, CUDA/BF16 support, and available storage.
2. Select Talha-only reference IDs, deduplicate repeated source titles, enforce
   5–30 seconds, and partition by original video. Default: about **10 hours total**,
   approximately 9 training hours plus 0.5 hours each for validation/test.
3. Before bulk preparation, use six longer clips to test normalization, original
   SRT/PCM correspondence, codec encoding/decoding, packing, two training steps,
   checkpoint saving, and a third step resumed with optimizer/scheduler state.
   The pilot pads sequences to the configured context length to check memory use.
4. Prepare all selected paired clips without moving their speech boundaries.
   Verify mono 44.1 kHz audio, finite loudness, clipping limits, duplicate waveforms,
   transcript identity, codec ranges/frame counts, and exact protobuf contents.
   Verify the original slice using nearest or truncated integer sample boundaries;
   this accommodates the splitter's one-sample floating-point rounding differences.
   Both conventions must match the SRT start/end and original PCM within one
   16-bit quantization step; no wider alignment search or boundary edits are used.
   Reject unverified source slices, clips with more than 0.1% full-scale input samples or unmeasurable loudness,
   record the reason, and replace them with clean clips from videos already assigned
   to that same split. The pilot preserves six accepted clips; full preparation
   replenishes each split's planned duration. Never loosen clipping limits to fill
   the budget. Source transcript changes, cross-split duplicates and insufficient
   clean reserves still stop the run with a specific error.
5. Train fast-only LoRA (rank/alpha 32, LR `5e-5`, batch 1, accumulation 4,
   BF16 mixed, **2048 context tokens**, 2,000 optimizer steps). Training samples clips
   randomly within the single speaker; validation uses each held-out clip once,
   in the same order every evaluation. Validate/save every 100 optimizer steps
   (400 training batches). Test clips remain excluded from model selection.
6. Preserve the latest five checkpoints and `last.ckpt`, select the best periodic
   validation checkpoint or better final checkpoint, and merge using the actual
   saved LoRA settings. Keep the original codec linked into the result.

These are conservative starting settings, not an empirically established optimum.
Use `--hours 11.1112` for about 10 **training** hours plus evaluation audio;
`--steps` and `--max-length` adjust the experiment. Configuration changes create a
separate experiment; data preparation is reused when its recipe and sources match.

### Reuse, monitoring, and failure handling

`cache/<fingerprint>/` stores selection, waveforms, per-clip receipts, codec tokens,
and packed splits. Source additions/edits and preprocessing changes invalidate
selection/preparation; codec changes invalidate tokens. Content hashes verify
source/prepared audio, tokens, model weights, and completed merged artifacts.
Reruns perform integrity checks but skip unchanged normalization, codec extraction,
and packing. Interrupted preparation retains completed clip/token receipts.
`qc.json` and `rejected/<clip-id>.json` record rejected candidates; unchanged
rejections are reused on restart. `reserves.jsonl` records deterministic replacement
candidates from the assigned source-video partitions. The source audio is untouched.
Dashboard/launcher edits no longer invalidate data automatically; incompatible
preparation changes require a new recipe revision. This rounding fix migrates the
inspected preceding cache in place, retaining its verified clips and passed pilot.
New receipts record the actual source sample boundaries and rounding convention.

`runs/<fingerprint>/` contains the resolved `config.yaml`, TensorBoard events,
`metrics.jsonl`, checkpoints, and `merged/`. Only a matching experiment resumes;
the launcher rebuilds frozen base weights and restores adapter, optimizer,
scheduler and Lightning loop state. It skips unreadable checkpoints when a valid
one remains. Iterable sampling is reseeded on resume; its exact data cursor is not
restored. Repeating a completed run reuses its verified result.

The frozen slow transformer can emit PyTorch's checkpoint warning about inputs
without gradients. The completed three-step GPU pilot was checked directly:
all 44 fast-model LoRA tensors updated between steps 1 and 3 and all 44 had
nonzero optimizer moments. The warning did not prevent adapter training.

The dashboard shows stage/clip progress, train/validation loss curves, token
accuracy, step count, estimated remaining training time, GPU memory, checkpoint
paths, and log tails. Prepared-reference and codec-reconstruction players appear
after those stages. **These audio checks are not trained-model samples.** Loss
does not establish voice similarity or Arabic pronunciation quality; listening
comparisons against the base model remain necessary.

Errors stop subsequent stages and remain visible with recent logs; the dashboard
stays open until Ctrl-C. Successful runs also keep the dashboard available.
Use `--exit-on-finish` for an exiting batch job.

```bash
# Prerequisites only; --no-dashboard works in socket-restricted environments.
python training/talha_s2pro/run.py --check --no-dashboard
# Six real CPU previews, with no codec extraction or training.
python training/talha_s2pro/run.py --preview --no-dashboard
# Dashboard only; no data/model work.
python training/talha_s2pro/run.py --dashboard-only
```

**Verification limits here:** CPU preparation and cache reuse have passed on six
real clips. Automated checks cover cache repair, transcript/alignment mismatch,
packed-content validation (synthetic tokens), subprocess supervision, and dashboard
HTTP responses/JavaScript rendering. The current sandbox blocks listening sockets,
cannot access an NVIDIA driver, and cannot resolve package servers. Dependency
installation was attempted and failed on DNS. Consequently, codec execution,
GPU training/resume, merging, and an actual browser connection on 7860 remain
unverified here. The launcher tests those GPU stages before scaling up; it cannot
promise an error-free full run in this environment.

## What the previous process did

The original experiment used normalized WAVs and matching `.lab` files, codec
`.npy` extraction, protobuf packing, then S2 Pro LoRA training. The saved
3,000-step run trained only fast-transformer adapters (rank/alpha 32).
`training/setup_data_dir.py` copied already normalized audio and labels.
The workspace also contains NeMo speaker splitting and silence slicing tools;
their presence alone does not establish that either tool produced every old clip.

For the new dataset, `README.md` documents CTC alignment, 5–31 second filtering,
48 kHz FLAC, and mixed Talha/Wesam sources. `refrence.txt` provides source identity.
The actual clips match the aligned `.ar-new.srt` texts and durations. **Clip
numbers differ from original subtitle cue numbers after short cues were dropped.**
The pipeline recovers provenance using exact whitespace-normalized text,
duration within 50 ms, and monotonically increasing cue order, and records
`srt_cue_number`, `start`, and `end`. It does not assume filename index = cue index.

## Is this the best approach?

The paired audio → loudness normalization → codec → protobuf → LoRA sequence
matches [upstream fine-tuning instructions](https://github.com/fishaudio/fish-speech/blob/214da3cd841bda85da2496b96cd3c4d7edb1337e/docs/en/finetune.md).
However, that guide explicitly discourages fine-tuning RL-trained models, and
[S2 Pro uses RL alignment](https://github.com/fishaudio/fish-speech/blob/214da3cd841bda85da2496b96cd3c4d7edb1337e/README.md).
This is a controlled experiment, not an established best recipe. Compare the
base model using a clean Talha reference against a short fine-tuning pilot before
extending training. Fast-only LoRA follows your previous experiment; adapting
the slow transformer too is a separate ablation, not a demonstrated improvement.

Recommended changes to your previous process:

- Reuse aligned 48 kHz segments. Do not run silence slicing, diarization, or
  short-clip merging on paired clips without regenerating their transcripts.
- Deduplicate before splitting. The two Talha playlists repeat 155 complete
  video titles under different IDs. Select the 2025 ID for each repeated title.
  This is source-title deduplication, not exhaustive acoustic deduplication;
  preparation additionally detects identical decoded waveforms.
- Split by original video, not by random segment or existing ASR Parquet split.
  The former experiment used the same protobuf directory for train/validation.
- Check voice identity, other speakers, music, overlaps, clipped speech, and
  ASR accuracy. Channel membership and aligned timing do not prove Talha is the
  only speaker or that transcripts are correct. Review representative beginnings,
  middles, and endings of each selected video and reject unsuitable recordings.
- Preserve mono wideband audio. Use 48 kHz source clips, not the telephony copy.
  The S2 codec config consumes 44.1 kHz; the preview resamples once using polyphase
  filtering, normalizes toward -23 LUFS, and caps estimated true peak at -1 dB.
  Peak protection can result in quieter output; it does not repair input clipping.
- Avoid automatic denoising by default; it can alter the voice. Prefer cleaner
  recordings. Listen to representative outputs when assessing dataset quality.

## Selection already proposed

`plan/manifest.jsonl` is a **metadata proposal**, not a materialized training dataset.
It excludes 6,365 other-channel clips, 13,547 duplicate-title clips, and eight
clips exceeding the configurable-in-code heuristic of six transcript words/sec.
There are 24,376 remaining candidates (87.95 hours).

| Split | Clips | Hours | Independent source videos |
| --- | ---: | ---: | ---: |
| Training | 2,477 | 9.003 | 246 |
| Validation | 131 | 0.505 | 14 |
| Test | 144 | 0.503 | 14 |

Assumption: **10 hours total**, with 90/5/5 duration budgets. To obtain about
10 hours of training plus evaluation audio, generate a new plan with
`--hours 11.1112`. Quality rejection will reduce final duration; replenish only
within the assigned source split. Test recordings remain excluded from tuning
and from any training/validation voice reference.

Selection is deterministic (seed 42), spread across source videos, and retains
complete aligned segments rather than cutting mid-word to hit an exact duration.
Selected videos still require identity/transcript listening checks. Exact text
agreement with SRT validates pairing, not linguistic accuracy.

## Latest upstream checkout

`../fish-speech-upstream/` contains upstream commit
`214da3cd841bda85da2496b96cd3c4d7edb1337e` (2026-09-16). Normal Git fetch failed
because GitHub DNS is unavailable in the shell. The GitHub connector retrieved
the latest revision and missing blobs; every blob, complete tree, and signed
commit hash was verified. This is a shallow checkout of the exact upstream
revision, with older local history retained. `upstream.json` records provenance.
Your modified `../fish-speech/` was not changed or merged.

Do not copy upstream's S1 Mini defaults verbatim for S2 Pro. The overlay fixes
the base path, Hugging Face tokenizer directory, 10 codebooks, separate protobuf
paths, and named experiment output. It sets `train: false`. Draft pilot settings
are batch 1 / accumulation 4, BF16 mixed, LR `5e-5`, and fast-only rank/alpha 32.
The 2,000-step cap is a starting experiment setting, not a validated optimum.
Hardware fit and gradient behavior must be checked before choosing a final budget.

## Commands and later stages

```bash
source /opt/dlami/nvme/training/activate_fish.sh
cd /opt/dlami/nvme
python training/talha_s2pro/pipeline.py check
python -m unittest discover -s training/talha_s2pro/tests -v
```

Planning reads headers/transcripts and writes manifests only:

```bash
python training/talha_s2pro/pipeline.py plan --hours 10 \
  --output training/talha_s2pro/another_plan
```

Small preview (default six, hard limit twelve):

```bash
python training/talha_s2pro/pipeline.py prepare \
  --manifest training/talha_s2pro/plan/manifest.jsonl \
  --output training/talha_s2pro/another_preview --limit 6
```

`smoke_verified/` already contains six paired WAV/`.lab` previews and `qc.json`.
All six retained duration, produced finite audio, stayed below the sample-peak
ceiling, and reached approximately -23 LUFS. Only ~56 seconds of distinct audio
were previewed; these results cannot establish quality of the full dataset.
The same six source slices were checked against their full recordings: all
matched the recovered start time with zero sample offset and maximum amplitude
difference below one 16-bit quantization step. `alignment_smoke.json` records
this check; it verifies slicing offsets, not transcript accuracy by listening.

Later, after quality review, `prepare --all` can materialize the selected clips
into a fresh `dataset/` directory; **that option has not been run**. The tool
refuses existing output directories, checks that labels have not changed, rejects
possible severe full-scale clipping (>0.1% samples, a heuristic), and writes
`prepared.jsonl`. Its statuses remain pending listening review.

`pipeline.py commands --dataset training/talha_s2pro/dataset` prints subsequent
commands without executing anything: extract tokens with **S2 Pro's own codec**,
then pack each split separately. Run `validate-tokens --prepared-manifest ...`
after codec extraction and before packing; it checks `(10, T)` integer tensors
and semantic/residual codebook ID ranges. Its successful synthetic tests do not
replace an actual codec encode/decode test. Test the real codec on preview clips
and verify round-trip audio and packed sample counts before scaling up.

The printed model command uses `train=false`; it still needs full dependencies
and loads model/data if executed. Only enable training in a later, explicit run.
Keep a new output project and do not resume adapters belonging to the old speaker.

## Validation and remaining blockers

The original sixteen helper tests cover source grouping, filtered-cue provenance, reproducibility,
split leakage, insufficient data, FLAC metadata, resampling/peak ceiling, silence
rejection, preview limits, and malformed codec tensors. Hydra composition of
`config/talha_s2pro.yaml` passed without loading a model; `resolved_config.yaml`
records it. `readiness.json` reports the environment truthfully.

Training remains blocked by unavailable NVIDIA driver/device access and absent
Torch 2.8/Torchaudio 2.8, Lightning, datasets, and Transformers. Package-server
DNS is unavailable. Installed cached packages support audio preparation and Hydra
configuration checking; cached Torch 2.1.2 was deliberately not substituted.
No GPU memory estimate has been validated. Once GPU/package access is restored,
install the pinned upstream dependencies using an appropriate CUDA extra, then
test tokenizer loading, codec round-trip, protobuf reading, and one training
batch on the preview dataset before a longer pilot.

Evaluate voice similarity, Arabic pronunciation/word accuracy, fluency, prosody,
and clipping on held-out prompts using the same clean reference for base and
adapter comparisons. Include non-football text to expose domain overfitting.
Use listening comparisons and actual held-out metrics; training loss alone is
insufficient to choose the best checkpoint.

## Final-stage audit

The original dataset remains untouched. Future plans now enforce **5–30 seconds**;
all 2,752 entries in the existing plan already meet that range, so it is retained.
`pipeline.py audit` reports implemented stages separately from tested stages and
continues to report `ready_for_training: false` while the blockers remain.

Additional checks now verify prepared WAV/label identity and duration, actual
codec frame counts, and exact packed transcript/token fingerprints and sentence
counts per split. `validate-packed --dataset PATH` detects silently skipped or
repeated samples, stale labels, invalid codebooks, and truncated protobuf streams.
Its reader was also checked against the existing real shard (367 sentences).
Synthetic packing tests do not constitute codec or training validation.

`merge_model.py` reads the chosen run's resolved `.hydra/config.yaml`, including
the actual rank, alpha, dropout and targets. Do not use upstream's named
`r_32_alpha_16_fast` for this experiment: it has alpha **16**, whereas the prepared
experiment has alpha **32**. The helper checks adapter keys, shapes and finite
values before merging. It links the original S2 Pro codec into the merged directory
because upstream inference expects `codec.pth` there. The symlink depends on the
base directory remaining available; include the codec when exporting elsewhere.
The merge itself has not run.

`pipeline.py evaluation-commands --merged MODEL_DIR --reference REFERENCE.wav
--prompt-text 'reference transcription' --text 'held-out Arabic text' --output DIR`
prints matched base/adapter inference commands with the same reference and seed.
It never executes inference. Pair listening review with independent ASR/WER or
speaker-similarity evaluation when those models are available; automatic evaluation
models and Arabic text normalization have not yet been configured.

The aligner referenced by the dataset README could not be retrieved at its pinned
URL (404 via GitHub); its original word-level export is also absent. Reusing the
existing segment alignment is supported by the timing checks. Newly cut or edited
transcripts would require a new alignment stage, which is not implemented here.
Do not describe this setup as a freshly executed forced-alignment pipeline or a
fully validated end-to-end training run. Current stage details are in `audit.json`.
