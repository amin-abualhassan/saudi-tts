# Talha Ahmad · Qwen3-TTS fine-tuning

Run from `/opt/dlami/nvme`:

```bash
python3 training/talha_qwen3/run.py
```

Open `http://<server-ip>:7860`. This command creates its own `.venv`, clones the official Qwen repository at a fixed revision, installs dependencies, audits and prepares the existing Talha dataset, extracts Qwen codec tokens, trains, validates, exports a CustomVoice checkpoint, evaluates test loss, reloads the export and synthesizes listening samples. The dashboard remains available after completion or failure; Ctrl-C closes it. An existing service on port 7860 must be stopped first, or select another `--port`. The runner does not stop existing services.

The default input is the same prepared **10-hour S2 Pro pilot** at `training/talha_s2pro/cache/54f80bc308a0fefd781c55c1/prepared.jsonl`: 2,482 training, 131 validation and 141 test clips (9.006 / 0.505 / 0.507 hours). The source manifests, 44.1 kHz waveforms and previous training outputs stay in place. Qwen gets a separate mono 24 kHz copy and its own 16-codebook tokens. It never consumes Fish Speech tokens. This workspace directory is outside a Git checkout; no commit was made.

Arabic is **not one of the ten languages documented for this Base model**. This experiment adapts both language and voice and uses `language="Auto"`; it does not add a supported Arabic language token. Clean input and low validation loss do not establish intelligible Arabic speech. Source metadata still requests human speaker/transcript review. Listen to the selected reference and the generated held-out samples, then measure Arabic CER/WER with a suitable ASR system and check speaker similarity before selecting a production checkpoint.

## Research and choice

Sources were inspected on 2026-10-10; there is no established universally best Qwen fine-tuning recipe.

| Approach | Appropriate use | Evidence and choice here |
| --- | --- | --- |
| Official full SFT, one speaker | A baseline for adapting one voice, especially with substantial clean data | [Official guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) documents Base single-speaker training, recommends one fixed reference, and uses 2e-5 LR / 3 epochs. We preserve its data format and use a fixed training reference, with explicit alignment corrections and validation. |
| Talker LoRA | First candidate for a small voice-only dataset or limited memory | [Instavar companion repo](https://github.com/instavar/qwen3-tts-lora-finetuning) reports useful LoRA voice adaptation, alignment/projection problems and a more conservative 2e-6 LR. Its results are speaker-specific, not proof for Jordanian Arabic. This runner uses full talker SFT because the current task also requires language adaptation; LoRA is researched but not implemented here. |
| Full SFT with one identity per speaker | One checkpoint with multiple permanent named voices | [vspeech trainer](https://github.com/vspeech/Qwen3-TTS-Train) provides community multi-speaker SFT and speaker registration. We implement a small extension over the pinned official dataset: fixed references per speaker, cached speaker vectors, distinct export IDs and optional balanced sampling. This is experimental custom multi-speaker support, not an official supported training feature. |
| EasyFinetuning UI | An alternative community UI for data and speaker management | [Project repository](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) advertises multi-speaker tooling. It was considered, but its UI is not needed for this one-command workspace runner. |

[Model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) and [tokenizer configuration](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/blob/main/config.json) define the supported language set and codec format. The pinned upstream revision is `0c6a7cbb6c8421a46332f8c2434c7825c4c855ef`, the revision identified by the companion project and inspected directly through its public source. No community trainer is executed or patched into the checkout.

The implementation applies `text_projection`, shifts codec-0 targets exactly once, and explicitly aligns sub-codebook targets without Transformers' causal loss shift. It pairs each target frame with the preceding talker hidden state. Canonical speaker embeddings are extracted once from each fixed training reference; variable reference lengths therefore never get concatenated into a padded multi-speaker batch. The codec and speaker encoder are frozen. Every talker parameter, including the text projection and code predictor, is fine-tuned.

For multiple speakers, use clean identity labels, one reliable reference per identity, validation/test examples for every speaker, and source-recording splits to prevent leakage. Turn on `--balanced` when hours differ materially, and compare held-out speech per speaker. The implementation rejects source-video overlap, duplicate clip IDs and duplicated prepared audio. It rejects unsupported input duration/context instead of silently truncating speech or text.

## Defaults and hardware

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; tokenizer: `Qwen/Qwen3-TTS-Tokenizer-12Hz`.
- Batch 1, gradient accumulation 4, LR 2e-6, 3 epochs, upper bound 2,000 optimizer steps, bfloat16, gradient clipping 1.0, gradient checkpointing.
- Validation/checkpoint every 100 optimizer steps and at each epoch end. The same first 20 validation examples are used for selection; use `--validation-batches 0` for full validation. Final test loss uses every test clip once the best validation checkpoint is selected. Loss is a codec-0-token-weighted average of per-clip composite losses, not ASR WER.
- SDPA attention avoids requiring a FlashAttention compilation. Optional `--attention flash_attention_2` requires installing `flash-attn` in the new environment.
- One CUDA GPU process with bfloat16 support. Preflight requires at least 20 GiB free VRAM; 24–48 GiB is a practical starting range, **not a measured fit guarantee**. Sequence length and the full optimizer can increase the requirement. At least 80 GiB free disk is required; reserve 100 GiB or more for snapshots, audio and atomic checkpoints.
- Torch/torchaudio 2.8.0 CUDA 12.8, Transformers 4.57.3, Accelerate 1.12.0 from the pinned upstream package. Setup uses a compatible Python 3.10–3.13 interpreter even when the launcher is Python 3.14. It creates a separate environment without modifying the Fish Speech environment. The resolved installed dependency versions are saved in `.venv/installed-requirements.txt`.

Three epochs can finish before the 2,000-step cap. The number of optimizer steps depends on batching and, for balanced multi-speaker runs, oversampling. The conservative LR is an initial experiment choice, not a validated optimum for this corpus.

## Checks, CPU preparation and bounded GPU qualification

Dependency/GPU preflight without installing anything:

```bash
python3 training/talha_qwen3/run.py --check --no-dashboard
```

Before `.venv` exists, use an existing compatible environment to inspect hardware:

```bash
miniconda3/envs/fish-speech/bin/python training/talha_qwen3/run.py --check --no-setup --no-dashboard
```

CPU preview (up to two clips per split plus the fixed reference):

```bash
miniconda3/envs/fish-speech/bin/python training/talha_qwen3/run.py --preview --no-dashboard
```

Prepare the entire corpus without loading or training a model:

```bash
miniconda3/envs/fish-speech/bin/python training/talha_qwen3/run.py --prepare-only --exit-on-finish
```

A small GPU run checks codec extraction, two optimizer steps, checkpoint export/reload and synthesis before a long run:

```bash
python3 training/talha_qwen3/run.py --smoke --exit-on-finish
```

Use `--no-setup` only with a compatible Qwen environment and the clean pinned upstream checkout. CPU preparation needs numpy, scipy and soundfile. If the default manifest is absent, provide another explicit `--manifest`; the runner fails rather than silently choosing a different dataset.

## Multi-speaker input

Provide a JSONL with these fields. Audio paths must be absolute or relative to the current working directory. `prepared_audio` takes precedence over `audio` if present.

```json
{"clip_id":"speaker_a_001","source_group":"recording_a_train","split":"train","speaker":"speaker_a","audio":"/absolute/path/a.wav","text":"The matching transcript","duration":7.0}
```

Create train, validation and test records for each speaker. Distinct clips from the same recording must use the same `source_group` and split. Speaker names use lowercase letters, numbers and underscores; clip IDs must be safe filenames. Audio must contain 2–30 seconds of finite, non-silent speech. A manifest can use one language or multiple transcript languages; training uses the Auto prefix consistently. Explicit language/dialect/instruction token training is outside this implementation.

```bash
python3 training/talha_qwen3/run.py --manifest /absolute/path/speakers.jsonl --balanced
```

To choose references manually, create a JSON mapping speaker names to **training clip IDs**, then pass `--references references.json`:

```json
{"talha_ahmad":"nz7e0fvf1n-41"}
```

Without overrides, the runner selects a training clip closest to seven seconds, deterministically. Inspect its audio; automatic duration selection does not establish the best acoustic reference.

## Artifacts and resume

The dashboard reports preparation progress, logs, optimizer steps, train/validation/test loss, gradient norm, learning rate, GPU memory, ETA, checkpoint paths, reference audio and generated/test audio. Raw loss events are persisted in `monitor/events.jsonl`, training metrics in the run's `metrics.jsonl`, and status in `monitor/status.json`.

`cache/<key>/` stores audited 24 kHz WAVs, raw manifests, per-clip token receipts and split codec manifests. Waveform content hashes and tokenizer weight identities control cache reuse. A changed transcript/reference/speaker/split changes the manifest identity. Model snapshots resolve to local files and are fingerprinted. `runs/<key>/recipe.json` includes model/codec/data identities, source code hash, upstream commit and training controls.

Checkpoints are atomically committed after validation. Rerunning the **same command** resumes the last committed checkpoint, including raw talker weights, optimizer state, torch RNG state and the deterministic shuffled data offset. An interrupted accumulation window is replayed. Changing data, model, code or controls creates a separate experiment. An explicit `--warm-start /absolute/path/training_talker.safetensors` initializes a new experiment from raw talker weights, with a fresh optimizer and step counter. This is weights-only recovery, not exact resume. The latest two checkpoints and the best validation checkpoint are retained; older ones are pruned inside this runner's own experiment directory.

Each `checkpoint-step-N/` contains a reloadable CustomVoice export, a separate unmodified `training_talker.safetensors`, local optimizer/RNG state and integrity receipts. Speaker rows are patched in the **export copy**, never the live training weights or raw resume weights. Local optimizer state uses PyTorch serialization and must not be replaced with files from untrusted sources. `best.json`, `evaluation.json`, `samples.json` and `samples/` identify the selected model and its held-out outputs. The exported checkpoint includes its speech tokenizer.

You can load the best export in the Qwen environment with `Qwen3TTSModel.from_pretrained(checkpoint_path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")` and call `generate_custom_voice(text=..., speaker="talha_ahmad", language="Auto", max_new_tokens=512)`. There is no reference-audio requirement for this named-speaker export.

## Inference testing GUI · port 3000

From `/opt/dlami/nvme`, launch:

```bash
bash training/talha_qwen3/start_testing.sh
```

Open `http://<server-ip>:3000`. The app uses the existing Qwen environment, automatically selects the completed run's best checkpoint (currently step 1863), loads it on `cuda:0` with SDPA, and keeps it loaded between requests. Wait for **Model ready**, enter text, then generate. Choose **Auto** for Arabic.

The GUI includes speaker/language and checkpoint selection, Arabic text presets, temperature/Top P/repetition/seed/token controls, queued requests, live status and logs, audio playback with waveform, WAV downloads, generation timing, persistent history, and the existing generated-versus-original held-out recording. Load a newly selected checkpoint before generating; checkpoint changes wait until queued generation finishes. Audio and metadata are saved under `testing/results/`.

Optional flags: `--host 127.0.0.1`, `--port 3000`, `--device cpu` (slower), or `--no-load` to browse the GUI and saved samples before loading the model. `python3 training/talha_qwen3/testing_app.py` is an equivalent launcher that switches into the Qwen environment automatically. `--check` lists completed checkpoints without loading a model or opening a socket. Stop with Ctrl-C.

Seven inference regression tests cover checkpoint selection, request validation, generation state, WAV/metadata persistence, silent-output failures, queued checkpoint protection, HTTP routes, audio range requests and file isolation. GPU synthesis and live browser rendering remain unverified in this sandbox because CUDA and listening sockets are unavailable.

## Validation in this session

The default manifest was inspected, all 2,754 Talha clips were converted successfully to mono 24 kHz under `cache/46d905ba71bec46da6ac4224/`, and 13 regression tests cover split leakage, references, multi-speaker balancing, token corruption, waveform cache invalidation, silence rejection, loss shifting/EOS/padding, projected embeddings, speaker assignment, frame-to-hidden-state alignment worker error propagation and dashboard routes/audio path isolation. Dashboard request handlers passed in memory; opening a live listening socket was denied by the sandbox, so port 7860 could not be exercised here.

```bash
miniconda3/envs/fish-speech/bin/python -m unittest discover -s training/talha_qwen3/tests -v
```

No GPU training, checkpoint reload or generated-speech quality result is claimed for this session: `nvidia-smi` cannot communicate with the NVIDIA driver, `torch.cuda.is_available()` is false, and shell DNS cannot resolve GitHub. The web research tools could inspect the public source, but the official repository and model weights could not be installed through this shell. Run the bounded GPU qualification command after driver and network access are restored.

## Checkpoint export fix (2026-10-10)

The user-launched run reached step 100 but failed in Transformers 4.57.3 `recursive_diff_dict` with `KeyError: dtype`. Qwen's speaker-encoder config does not initialize the standard config fields or accept arbitrary constructor keywords. A runtime dtype on that nested config triggers the diff error.

The exporter now writes the full config without diffing, filters the speaker-config fields to its actual constructor arguments, saves HF-compatible safetensors with `format=pt`, and saves the processor separately. It uses a copied config and copied speaker row, preserving live/raw training weights. Optimizer/RNG state and pending training position are now written before export. Incomplete directories are still not automatically accepted as committed checkpoints.

The original step-100 `training_talker.safetensors` survived at `runs/972c2b02a4f1b7a685e3ecd0/checkpoint-step-100.tmp/`. Its optimizer/RNG state was never saved, so exact step-100 resume is unavailable. To reuse these trained weights, close the failed dashboard runner with Ctrl-C, then launch a new experiment:

```bash
python3 training/talha_qwen3/run.py --warm-start training/talha_qwen3/runs/972c2b02a4f1b7a685e3ecd0/checkpoint-step-100.tmp/training_talker.safetensors
```

This reuses the prepared audio and codec caches and preserves the recovered source file. The optimizer and step counter restart; the default schedule runs as a new experiment from the recovered weights. Do not relabel the temporary directory as a resumable checkpoint.

The added regression test reproduces the actual Qwen dtype serialization failure and verifies export config reload, registered speaker rows, safetensors metadata and unmodified source config/weights. A separate CPU verification exports the real saved step-100 tensors into a temporary test directory, reloads the real config/processor and verifies every weight name and shape; its dummy speaker vector is used only for serialization testing, and the temporary export is removed. Results are in `checkpoint_fix_validation.json`. GPU continuation and synthesis remain unverified in this sandbox.
