# Jordanian S2 Pro experiment suite

Updated 2026-10-11. The corrected H100 run completed all four 1,500-step
experiments and their base/early/best/final matched synthesis. The user selected
`fast_r32_final` and `attention_mlp_r16_early` by listening; the voice server below
serves those exact adapters with the original S2 Pro base and codec.

The initial loader and attention-backward failures are fixed. Evidence and
regression details remain in the [loader report](reports/dataloader_fix.json)
and [checkpointing report](reports/checkpointing_fix.json).

## SGLang voice studio on port 3000

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server_sglang.sh
```

This uses the exact selected checkpoints, existing speaker samples and voice
uploads. See [SGLANG.md](SGLANG.md) for setup, comparison playback, validation and
fallback. Native exports and CPU checks are complete; Docker/GPU startup awaits
running this command on the host.

## Serve the two selected models on port 7860

```bash
screen -S jordanian-voice
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server.sh
```

Open `http://<server-ip>:7860`. The script uses the existing Fish environment and
automatically stops this suite's **finished** dashboard if it occupies the port.
It refuses to stop an active training campaign or an unrelated app. Detach with
**Ctrl-A, D**; reconnect with `screen -r jordanian-voice`. Wait for the models and
codec to load before generating; requests submitted during loading are queued.

- Select `fast_r32_final` or `attention_mlp_r16_early`.
- Choose Talha Ahmad, Wesam Ali, or the single speaker. These use the identical
  reference WAVs and transcripts from Listen and compare.
- To clone your own voice, upload a clear 3–30 second single-speaker recording
  (25 MB maximum), then enter its exact transcript. WAV, MP3, FLAC, M4A, OGG,
  AAC and WebM are supported. Microphone recording is available on HTTPS or
  localhost; uploading also works over HTTP. New uploads clear the previous
  reference transcript.
- Enter the new speech text and generate. Playback and WAV downloads appear
  below; the original comparison dashboard remains at `/compare`.

Defaults match the existing comparison: seed 42, temperature/top-p 0.7,
1,024 maximum new audio tokens, uncompiled upstream generation, and the
original codec. Each request records checkpoint/reference hashes and generation
settings under `voice/results/`. Custom references remain in `voice/uploads/`.
Generation is serialized through one persistent worker, with up to eight queued
requests. Both models stay resident when memory allows; with less than 32 GiB
free at startup, the worker uses one resident model. To force that mode:

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server.sh --model-cache 1
```

`voice/server.log` and `voice/status.json` track the server. Use the UI's
**Retry server** after a worker failure. Inspect selected checkpoint/reference
integrity without CUDA or a listening port:

```bash
/opt/dlami/nvme/miniconda3/envs/fish-speech/bin/python \
  /opt/dlami/nvme/training/jordanian_s2pro/voice_server.py --check
```

The implementation, real upload/HTTP handler checks, actual tiny-model LoRA
loading, and UI flow checks were validated in this session. Subsequent host logs
verified both models resident and successful GPU synthesis with an uploaded voice.
The Codex sandbox cannot create network sockets or access CUDA. See the
[voice server verification report](reports/voice_server_validation.json).

## Acceleration rollback

The compiled-decoding experiment was withdrawn after the user reported that the
fast server was not working. Serving is restored to the original decoder, without
compilation or startup benchmarks. Start with `start_voice_server.sh` as above.
The former `start_voice_server_fast.sh` is now a compatibility entrypoint to the
original server; old optimization flags are accepted but cannot enable compilation.

The same checkpoints, codec, reference samples, uploaded voices and generated
clips are preserved. A verified idle server can be replaced using:

```bash
bash /opt/dlami/nvme/training/jordanian_s2pro/start_voice_server.sh --replace-voice-server
```

Latest persisted host evidence shows both models ready and a successful original
(eager) decoder request. All 33 baseline/restart regression checks passed after
rollback. See the [rollback report](reports/voice_server_rollback_20261010.json).
The withdrawn implementation is archived under `reports/withdrawn_fast_server_20261010/`.

## Run inside screen

```bash
screen -S jordanian-s2pro
source /opt/dlami/nvme/training/activate_fish.sh
cd /opt/dlami/nvme
python training/jordanian_s2pro/run.py
```

Open `http://<server-ip>:7860`. Detach with **Ctrl-A, D**; reconnect with
`screen -r jordanian-s2pro`. The dashboard stays available after completion or
failure. Ctrl-C exits it; restarting the same command resumes compatible runs.
Do not keep the old Talha launcher running: this suite acquires its training
lock too. Stop that launcher with Ctrl-C in its screen. If another unrelated
application owns the port, use `--port 7861`.

If dependency checks fail, use `python training/jordanian_s2pro/run.py --install`
inside the activated Fish environment. This reuses Talha's pinned dependency
installer. The current CPU inspection found the required packages, but CUDA and
the NVIDIA driver cannot be accessed from this Codex sandbox. `--install` does
not fix a missing driver.

Useful commands:

```bash
# Metadata-only plan: no codec extraction or training.
python training/jordanian_s2pro/run.py --plan --no-dashboard
# Real CPU audio previews including each source and split.
python training/jordanian_s2pro/run.py --preview --no-dashboard
# GPU/dependency checks only.
python training/jordanian_s2pro/run.py --check --no-dashboard
# Run one experiment; repeat with other names to fill the same comparison.
python training/jordanian_s2pro/run.py --only fast_r16
# Optional short complete smoke run (separate campaign from the default).
python training/jordanian_s2pro/run.py --only fast_r16 --steps 100
# Regenerate/repair comparison samples for completed runs; no training.
python training/jordanian_s2pro/run.py --compare-only
# Serve the latest persisted status and samples without starting training.
python training/jordanian_s2pro/run.py --dashboard-only
```

Use `--exit-on-finish` to close the dashboard automatically. `--no-dashboard`
also works for full headless runs. The default behavior performs real work;
the explicit `--plan`, `--preview`, and `--check` flags are the safe inspection modes.

## Experiments and research basis

[experiments.json](experiments.json) is the editable plan. Each default trial
starts independently from the original S2 Pro weights, using the same selected
data, seed, rank/alpha ratio, learning rate, batch size, context and step limit.

| Experiment | LoRA rank / alpha | Targets | LR |
| --- | --- | --- | --- |
| `fast_r16` | 16 / 16 | Fast attention, MLP, embeddings, output | `2e-5` |
| `fast_r32` | 32 / 32 | Fast attention, MLP, embeddings, output | `2e-5` |
| `attention_r16` | 16 / 16 | Slow and fast attention | `2e-5` |
| `attention_mlp_r16` | 16 / 16 | Slow and fast attention and MLP | `2e-5` |

The first pair isolates adapter rank. The third/fourth pair isolates adding MLP
adapters. Fast-only versus both-transformer trials compare target strategies;
they are not a pure test of slow attention alone. The pinned upstream treats
unprefixed `attention`/`mlp` as targeting both transformers. Its `fast_*` names
target only the fast transformer.

Defaults: 1,500 **optimizer** steps, batch 1, accumulation 4, BF16 mixed,
2,048 context tokens, up to 100 warmup steps, gradient clipping 1.0, and fixed
validation every 100 optimizer steps (400 training batches). Training is step
bounded, so adding source hours does not turn it into an unbounded epoch run.
The codec, base model weights, and attention-head count stay fixed.

These are controlled pilot settings, not a verified best Jordanian recipe.
The [official S2 Pro model card](https://huggingface.co/fishaudio/s2-pro)
describes the slow semantic transformer and fast acoustic transformer.
The [official fine-tuning guide](https://github.com/fishaudio/fish-speech/blob/main/docs/en/finetune.md)
warns about degraded performance when fine-tuning RL-trained weights and
recommends the earliest satisfactory checkpoint. A
[first-hand fast-only experiment](https://github.com/fishaudio/fish-speech/discussions/1234)
reported voice-cloning improvements and degradation when adapting the slow
transformer. A
[recent first-hand failure report](https://github.com/fishaudio/fish-speech/discussions/1345)
reported better teacher-forced loss but collapsed generated audio. These are
individual reports, not proof of a universal recipe. They motivate bounded
trials and matched listening, rather than choosing by loss alone.

The suite reuses the Talha worker's fixed validation, full-state resume and
preprocessing utilities against upstream commit
`214da3cd841bda85da2496b96cd3c4d7edb1337e`. It does not edit either Fish Speech
checkout or the Talha launcher. Its compatibility bridge replaces the packed
validator only in this suite process so speaker identities are checked too.

## Data limits and identity

Default **total** budgets, including validation/test, are 6 hours Talha,
4 hours podcast, and up to 1 hour from the existing 50-minute speaker. The
current verified plan selects approximately:

| Source | Training | Validation | Test |
| --- | --- | --- | --- |
| Talha | 4.800 h | 0.600 h | 0.599 h |
| Podcast, including Wesam and guests | 2.758 h | 0.335 h | 0.349 h |
| Existing 50-minute speaker | 0.614 h | 0.099 h | 0.093 h |
| Total | **8.172 h** | **1.034 h** | **1.041 h** |

Actual accepted hours may decrease after waveform QC. Budgets are hard upper
bounds, not minimums filled by duplicating audio. The podcast selection reserves
half its budget for Wesam and half for guests; it does not fill a Wesam shortage
with more guests. The current plan has 479 Wesam training clips. Wesam is not
the speaker in every podcast clip.

- Talha uses source-title deduplication and original-video splits; its candidate
  inventory is about 88 hours, but the default selects only six.
- Podcast clips retain the existing episode-disjoint train/validation/test
  assignments. Guest IDs remain episode local. This means validation includes
  unseen guests, rather than claiming every guest has held-out recordings.
- The 367 existing single-speaker clips total 49.92 minutes. Removing the
  splitter's final `_number` suffix gives 71 recording groups; whole groups
  are held out. This relies on the existing filename grouping, not independently
  recovered source timestamps. The previous single-speaker adapter is not used
  as initialization because its evaluation data leaked into training.
- Paired clips must be 5–30 seconds with finite mono audio, matching labels,
  plausible text rates, and no control tokens. Original Talha PCM crops are
  rechecked. Existing podcast source hashes are checked. Normalize to -23 LUFS
  with a -1 dB peak limit and resample to 44.1 kHz without changing speech cuts.
- Input clipping above 0.1% and silence are rejected. Exact decoded duplicates
  are removed within splits and stop the run across splits. This is not an
  exhaustive acoustic near-duplicate search. If a family/split loses over 20%
  of its selected duration, stop for QC review rather than weakening thresholds.
- Datasets have a hard 20-hour total cap, contexts are limited to 1,024–4,096,
  and full batches contain whole paired sentences rather than truncated clips.
  Host RAM and disk reserves are checked. An out-of-memory pilot fails that
  experiment; the launcher does not silently change its settings.

Training probabilities are Talha 40%, podcast 45%, and single speaker 15%.
Within podcast, Wesam gets half the probability; the remaining half is shared
by guests in proportion to the square root of their durations. This tempers
dominance by long guest recordings. Speaker IDs group samples, while voice
selection at inference still uses reference audio; they are not trained named
voice commands. The loader uses zero extra workers, avoiding upstream shard
partitioning that could omit some families from a worker.

The default uses **automatic candidate transcripts and speaker labels**, as
authorized for this experiment. They have not all passed listening review.
Downloaded Wesam decisions can be applied:

```bash
python training/jordanian_s2pro/run.py --review /absolute/path/wesam-review-decisions.json
# Restrict the podcast pool to explicit accepts:
python training/jordanian_s2pro/run.py --review /absolute/path/wesam-review-decisions.json --accepted-only
```

Reject/uncertain decisions are always excluded. Accepted-only mode may leave too
little data for the required splits/references and will fail clearly. Talha and
the existing speaker still need listening review. Preserve spoken Jordanian
wording; do not rewrite it into formal Arabic.

## Execution, recovery, and comparison

1. Bind the dashboard and acquire launcher locks. Validate settings and resources.
2. Prepare a small sample covering each source and split, using clean reserves
   where available. Extract and verify S2 Pro tokens/protobuf, then decode a
   real codec round-trip and check duration, sample rate and finite/non-silent audio.
3. For each configuration, pad pilot batches to the full context, train two
   steps, then restore adapter/optimizer/scheduler/loop state and reach step
   three. Check finite gradients, LoRA-only trainable parameters, and actual
   adapter updates. Failed pilots do not start full trials.
4. Prepare and encode the bounded full subset once. Verify exact packed speaker,
   text, tokens and sentence counts. Reuse this data across all successful trials.
5. Generate the base-model samples and run each experiment sequentially in a
   fresh subprocess. Save two latest periodic checkpoints, `last.ckpt`,
   validation-best, an early checkpoint at step 200, and final checkpoint.
6. Compare early, validation-best and final checkpoints after each experiment.
   No training/inference models run concurrently in this launcher.

Data caches contain per-clip content receipts. Changed source audio/labels,
preprocessing, base weights, code, settings, or accepted/QC-filtered data create
new identities or invalidate affected stages. Training resumes only in its
matching directory; it restores optimizer and scheduler as well as adapters.
The iterable sampler's exact cursor is not restored. Interrupted work may repeat
samples, but does not discard valid checkpoints. Restart with the same config;
changing the step limit starts a separate campaign rather than extending a
checkpoint under a different identity.

Comparison uses a fixed approximately 12-second training reference for Talha,
Wesam and the existing speaker. Each receives the same four Jordanian prompts
and one held-out transcript from that speaker, where available: **15 cases per
model** under the current data. This includes numbers, code switching and a
longer paragraph. Settings/reference hashes are saved. Generated audio can be
downloaded on port 7860; ratings for accent, clarity, voice match and stability
are saved to server JSON. Fixed references let you compare adapters fairly;
three references and one seed are a first comparison, not a comprehensive
generalization benchmark. Edit evaluation prompts/settings and run `--compare-only`
for additional tests without retraining. This refreshes the comparison samples;
copy the comparison directory first if you want to retain earlier evaluations.

Waveform metrics show duration, RMS, quiet fraction and quiet ending, and expose
silent generation. They do **not** measure accent correctness, speaker similarity
or intelligibility. No automatic quality winner is declared. Listen to the
reference and outputs, then choose the earliest satisfactory checkpoint. Do not
use the test transcript cases to repeatedly redesign the recipe; retain another
untouched evaluation set for a final production assessment.

Adapters are loaded directly for comparison, avoiding four full merged model
copies. To export a winner, use the suite path shown by the dashboard:

```bash
python training/jordanian_s2pro/merge.py \
  --suite /opt/dlami/nvme/training/jordanian_s2pro/suites/<suite-id> \
  --model fast_r32_early \
  --output /opt/dlami/nvme/training/checkpoints/jordanian-s2pro-selected
```

Use the actual model name from the comparison. Export validates the checkpoint
hash, uses its exact saved LoRA configuration, links the original codec, and
refuses to overwrite an existing output directory.

## Artifacts and verification

- `monitor/status.json`, `monitor/launcher.log`, `monitor/events.jsonl`: live work.
- `cache/<id>/selection.json`, `manifest.jsonl`, `prepared.jsonl`, `qc.json`:
  selection and audio audit. `receipts/`, `token-receipts/`, `packed.json`: integrity.
- `suites/<id>/status.json`, `settings.json`, `runs/`: experiment state, configs,
  metrics, TensorBoard and checkpoints.
- `suites/<id>/comparison/`: fixed cases, model manifest, generated WAVs, results,
  waveform metrics and listening ratings.
- [CPU validation report](reports/validation.json): current verification evidence.

```bash
python -m unittest discover -s training/jordanian_s2pro/tests -v
```

Seventeen CPU tests cover bounds, recording splits, balanced sampling, complete-sentence
packing, exact packed speaker/content checks, waveform repair, transcript edits,
cross-split duplicates, finite-gradient guards and adapter-update evidence.
Both train and validation loaders now support zero workers, enabling persistent
workers only when the count is positive. They resolve the collator dynamically
so the GPU pilot's full-context padding remains active. Real pilot data produced
a `(1, 11, 1604)` training batch and `(1, 11, 816)` validation batch on CPU, with
all four pilot speaker groups loaded and the intended sampling weights intact.

JavaScript syntax and sample-card rendering passed a DOM-stub smoke check. Real
browser/socket behavior, successful GPU pilot/resume, full training, synthesis and export
remain unverified in this sandbox. The launcher tests the GPU path before full
preparation; it cannot promise that every experiment will produce good audio.
