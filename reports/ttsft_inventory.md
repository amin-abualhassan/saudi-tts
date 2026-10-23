# TTS-FT server inventory (`Majed-AWS`, i-0bf8e6a113fff4d51)

Inspected 2026-10-23 with read-only commands (`ls/find/du/cat/head/grep`) and `rsync` pulls into `pipeline/legacy/`. Paths below are on the remote box under `/opt/dlami/nvme/` unless they start with `pipeline/legacy/`. The local copies are listed in section 5.

**Bottom line.** The box ran a controlled Jordanian **Fish S2 Pro LoRA suite** with four 1,500-step trials on 8.2 h of data and 24 speakers. It also hosts a live SGLang-Omni S2 Pro studio and a public **Saudi-voice ("Noura") voice-agent demo**. Two findings matter most for our run:
1. The pinned training code feeds the Slow-AR **differently from generation**. This is an upstream bug.
2. Every Slow-AR LoRA collapsed to silent output by step 1,500, even while its validation loss kept improving.

All raw Jordan data is already in S3. `/opt/dlami/nvme` is an **instance-store volume (`lv_ephemeral`, 563 GB used), so stopping the instance wipes it**.

---

## 1. What was run and how

### 1.1 Fish S2 Pro pipeline (`training/talha_s2pro/`, `training/jordanian_s2pro/`)
- **Code base:** upstream fish-speech commit `214da3cd841bda85da2496b96cd3c4d7edb1337e` (2026-09-16), checked out at `training/fish-speech-upstream/`. Provenance is in `training/talha_s2pro/upstream.json`. The top-level `fish-speech/` (HEAD `3dd1f85c…`) is a separate, modified checkout with a web studio and API server. It was not used for the suite.
- **Data steps:** paired `*.wav` + `*.lab` → `pipeline.normalize()` → codec tokens → protobuf → LoRA. `pipeline.normalize()` (`talha_s2pro/pipeline.py:207`) uses pyloudnorm to reach **-23 LUFS**, then a polyphase resample to **44.1 kHz** and a **-1 dB true-peak** cap; speech cuts are not moved.
  - Codec extraction: `tools/vqgan/extract_vq.py --config-name modded_dac_vq --checkpoint-path checkpoints/s2-pro/codec.pth --batch-size 1 --num-workers 1` (`talha_s2pro/run.py:543-545`). **This confirms the plan's day-0 check: `extract_vq` uses `s2-pro/codec.pth`.** Its sha256 is `74fc41c5…bb2f` (`suites/36aa…/settings.json`). Tokens are `(10, T)` int64, i.e. 10 codebooks.
  - Packing: `tools/llama/build_dataset.py --text-extension .lab`, with a separate proto directory for each split.
  - A codec round-trip (encode, then decode at 44.1 kHz) runs before every campaign.
- **Training config** (resolved: `suites/36aa33e6739594a4858d620c/runs/*/config.yaml`):

| Setting | Value |
| --- | --- |
| Base | `checkpoints/s2-pro` (`fishaudio/s2-pro`) |
| Precision, context | `bf16-mixed`, `max_length` 2048 |
| Batch | 1 × `accumulate_grad_batches` 4; `num_workers` 0 |
| Optimiser | AdamW, β (0.9, 0.95), wd 0.01, eps 1e-8 |
| Schedule | constant LR after 100 warm-up steps; grad clip 1.0 (norm) |
| Steps | 1,500 optimizer steps; validation every 100 steps (400 batches); seed 42 |
| Dataset | `interactive_prob` 0.7, `use_speaker` false, `num_codebooks` 10 |
| LoRA | dropout 0 |

- **Experiments** (`experiments.json`; LR 2e-5 for all four):

| Trial | r / α | Targets | Adapter params | Final val loss (total / slow-base / semantic / top-5) | GPU reserved | Speed (opt steps/s) |
| --- | --- | --- | --- | --- | --- | --- |
| fast_r16 | 16/16 | fast_attention, fast_mlp, fast_embeddings, fast_output | — | 18.68 / 12.457 / 6.22 / 0.077 | 13.0 GB | 0.88 |
| fast_r32 | 32/32 | same | 7.1 M (44 tensors) | 18.64 / 12.457 / 6.18 / 0.080 | 13.1 GB | 0.88 |
| attention_r16 | 16/16 | attention (Slow + Fast) | — | 9.08 / 4.04 / 5.04 / 0.208 | 14.4 GB | 0.50 |
| attention_mlp_r16 | 16/16 | attention + mlp (Slow + Fast) | 33.4 M (400 tensors) | 8.87 / 3.91 / 4.96 / 0.219 | 20.2 GB | 0.41 |

  - Final validation losses come from `runs/*/completed.json`, and the curves from `runs/*/metrics.jsonl`.
  - The fast-only "slow-base" loss stays at **12.457** from step 100 to 1,500: the frozen Slow-AR's token loss on validation.
  - Wall time on one H100 is about 28 min (fast-only) to 61 min (attn+mlp) per trial.
- **Generation check** (`suites/36aa…/comparison/*/results.jsonl`): 15 fixed cases per variant (3 speakers × 4 Jordanian prompts + 1 held-out line), seed 42, temperature/top-p 0.7.
  - Base, all fast-only variants and both `*_early` checkpoints (step 200) give normal audio: mean RMS 0.048–0.063, quiet fraction 0.12–0.17.
  - **`attention_r16` and `attention_mlp_r16` at val-best and final (step 1,500) give near-silent output in 15/15 cases** (RMS 0.000–0.001, quiet fraction ≥ 0.998). This happened while their validation loss was the best of the suite.
- **Listening selection** (`reports/voice_server_validation.json`): the user chose **`fast_r32_final`** and **`attention_mlp_r16_early`** (step 200). The second one is the model the Saudi "Noura" voice agent uses in production (`speech-agent/.env.example`: `TTS_MODEL=attention_mlp_r16_early`).
- **Evaluation gaps:** there was no CER, WER or speaker-similarity evaluation of the suite. The only ASR check is Whisper-large-v3-turbo on 4 BF16-vs-NF4 sentences: BF16 0 % WER on all 4 (`training/attention_mlp_r16_nf4_20261017/bundle/VALIDATION.md`).
- **Earlier runs:**
  - `talha_s2pro` run `51b79183…`: Talha only, about 9 h, fast-only r32/α32, LR 5e-5, 2,000 steps, final val loss 18.47, merged to `runs/…/merged/`.
  - April run `results/s2pro_speaker_lora_v2`: fast-only r32/α32, LR 5e-5, batch 4, 3,000 steps, val loss 11.30, top-5 0.754. **Its train and validation sets were the same proto directory** (`training/EXPERIMENT_NOTES.md`).

### 1.2 What `PROJECT_STATUS.md` (2026-10-18) says worked and failed
- **Worked:**
  - The bounded suite with recording-disjoint splits, full-state resume, and LoRA-only gradients checked by pilots.
  - Native-HF export of the adapters for SGLang, with all 358 tensors and slow/fast logits checked on CPU.
  - SGLang-Omni serving with FA3 and CUDA graphs: about **98–100 tokens/s, median RTF 0.25**, warm **TTFT 90–125 ms**, but **first audio about 0.9–1.0 s**. A new reference costs **13–19 s of CPU encoding** (`training/jordanian_s2pro/SGLANG.md`, `reports/sglang_*_20261011.json`). The original eager decoder ran at about 12 tokens/s (RTF 1.84).
- **Failed or fixed along the way:**
  1. Upstream `SemanticDataModule` sets `persistent_workers=True` with 0 workers, which crashes (`reports/dataloader_fix.json`).
  2. Attention-only LoRA failed with `element 0 of tensors does not require grad`. The cause is reentrant activation checkpointing with frozen embeddings; the fix is `use_reentrant=False` (`reports/checkpointing_fix.json`, `training_compat.py`).
  3. A compiled "fast server" was withdrawn after the user reported it broken.
  4. The SGLang Docker build needed a protobuf override (`>=6.31.1,<7`) against audiotools 0.7.2. Image extraction also ran out of space on the 96 GB root disk, so Docker and containerd were moved to NVMe.
  5. Users reported fading or quiet endings and clipped last syllables. The fixes were output-side: a playback leveler and a `[short breath]` tag appended to every sentence. Mispronunciation (النص) was fixed by prompting for selective diacritics.
- **Open issue flagged but never fixed:** "Pinned `BaseTransformer.embed` omits the `scale_codebook_embeddings` normalisation present in `forward_generate`" (`PROJECT_STATUS.md`, Open issues; `checkpointing_fix.json`). I verified this in the upstream file `fish_speech/models/text2semantic/llama.py`:
  - `forward_generate` (lines 390–432) divides VQ-position embeddings by `sqrt(num_codebooks+1)` (√11) and audio embeddings by √2.
  - `embed()` (lines 326–345), which `forward()` uses for training, does neither.
  - The S2 Pro config sets `scale_codebook_embeddings=True` (line 126).
  - Supporting evidence: the base model's step-0 training loss is near chance (Talha `base_loss` 10.70, top-5 **0.0075**; suite 10.45 / 0.0175), and the frozen Slow-AR's validation loss is 12.46. This suggests the slow-AR LoRAs spent their capacity learning to compensate for the mis-scaled inputs and then broke under the correctly scaled generation path. That would explain the silent output. This is likely but not proven.

### 1.3 Other pipelines
- **Splitter/ (NeMo diarisation)**: NeMo 1.23 MSDD `diar_msdd_telephonic`, VAD `vad_multilingual_marblenet` (window 0.15 s, shift 0.01 s, `min_duration_off` 0.2 s), embeddings `titanet_large` (multi-scale windows 1.5–0.5 s), oracle 2 speakers, 16 kHz, `MIN_CHUNK_SEC` 1.0. It was tested on one Emirati podcast (`UAE_1.wav`) and is superseded.
- **Wesam podcast pipeline** (`training/wesam_review/`, `wesam_dataset_gpu/`, `wesam_dataset_words/`): this is the reusable one.
  - Diarisation: **pyannote `speaker-diarization-community-1`** (pyannote.audio 4.0.7, torch 2.8.0+cu128) on whole episodes at 16 kHz, batch 64.
  - Host identification: **WeSpeaker `pyannote/wespeaker-voxceleb-resnet34-LM`** embeddings. ECAPA and WavLM-SV backends are also coded. Rule: 3-s windows from the longest clean turns (up to 60 s), ≥ 9 s of evidence, **cosine ≥ 0.5** and ≥ 80 % window agreement; ambiguous clusters are excluded.
  - Clip cutting: from YouTube VTT word timestamps, **3–30 s**, ≥ 3 s of detected speech, no speaker change or overlap inside a clip, native 48 kHz PCM-16, no loudness normalisation or denoising. Splits are by episode (19/2/2).
  - Speed: an H100 diarises 300 s in 4.15 s against 253 s on CPU (61×); about 0.44 h for the full corpus (`reports/gpu_benchmark.json`).
  - Output: **7,102 clips, 20.0 h, 23 episodes, 1,061 Wesam clips (1 h 59 m) plus 25 episode-local guest IDs**; recording-disjoint train 5,790 / 16.57 h, val 718 / 1.68 h, test 594 / 1.75 h (`PROJECT_STATUS.md`).
- **speaker_embedings.ipynb**: a Colab-style notebook. SpeechBrain **SepFormer `sepformer-whamr16k`** separates two-speaker mixtures, then **ECAPA `spkrec-ecapa-voxceleb`** keeps the stream nearest a reference (3 s chunks, 1.5 s hop, confidence margin 0.10). Each stream is then transcribed by **Gemini (`gemini-3.7-flash`)** and scored with jiwer. The API key is a placeholder (`"xxx"`); no real credential is present.
- **OmniVoice** (`OmniVoice/` HEAD `4a4b2295`; `training/omnivoice/speaker1/`):
  - Data: the 50-min Jordanian female voice-over set, 330/37 rows, `language_id: "apc"`, 24 kHz.
  - Method: **full fine-tune** from `k2-fsa/OmniVoice` on the Qwen3-0.6B LLM backbone, LR 5e-5 cosine, warm-up 1 %, `batch_tokens` 8192, bf16, 2,000 steps (about 10 min).
  - **Eval loss 4.553 at step 500, then 4.631, 5.031 and 5.316 at step 2,000**, i.e. overfitting after about 500 steps (`exp/full_finetune/train.log`).
- **chatter/**: Chatterbox-multilingual LoRA (r 32, α 64, dropout 0.05, LR 2e-5 cosine, 20 epochs, batch 1 × accum 8, 3–30 s) on the same 50-min speaker. It left 29 GB of checkpoints, `fix_tokenizer_corrupted.py` and `fix_merged_model.py`, and no recorded result.
- **talha_qwen3/**: Qwen3-TTS full-talker SFT on Talha (LR 2e-6, 3 epochs / 2,000 steps, `language="Auto"`). Test loss 3.39 (`runs/9074…/evaluation.json`). Its README warns that Arabic is not a supported language. This agrees with the plan excluding Qwen3.
- **speech-agent/** (live, port 8002 behind a Cloudflare tunnel, plus LiveKit and TURN on 443): a voice agent built from OpenAI `gpt-live-transcribe`, `gpt-6-luna`, and S2 Pro over SGLang (`POST :3000/api/stream`, SSE base64 PCM16 at 44.1 kHz). The voice is **Noura**, cloned from `speech-agent/saudi_female_1minute-trimmed2.wav`: a 9.46 s reference with transcript in `assets/voice/reference.json`.
  - Audio post-processing: compressor/limiter at -6 dBFS, quiet boost ≤ 3 dB, 250 ms leveler, 40 ms onset ramp.
  - `.env` contains credentials (not copied).
- **NF4 derivative** (`training/attention_mlp_r16_nf4_20261017/`): a bitsandbytes NF4 build of `attention_mlp_r16_early` for a laptop. It peaks at 5.7 GiB of GPU memory but is **slower than real time on H100** (RTF about 2.7, first PCM about 3 s).

## 2. Data on the box

| Dir | Content | Count / size | Format | In S3 `Jordan/`? |
| --- | --- | --- | --- | --- |
| `full-audios-transcript/audios-48khz` | full YouTube recordings (Talha + Wesam channels) | 452 files, 61 GiB | FLAC 48 kHz mono **24-bit** | **Yes**: 452 flac + 452 srt + 452 vtt, 65.45 GB (byte totals match) |
| `full-audios-transcript/transcript` | `*.ar-new.srt` (refined cues) + `*.ar.vtt` (YouTube captions with per-word `<c>` timestamps) | 904 files, 131 MB | SRT/VTT | Yes (same) |
| `splitted-audio-transcript/Jordan-youtube-48khz` | CTC-aligned segments | 44,296 files, 23 GiB | FLAC 48 kHz mono 16-bit | **Yes**: 132,888 objects = 3 × 44,296, 32.52 GB |
| `splitted-audio-transcript/Jordan-youtube-telephony` | same segments | 44,296 files, 8.2 GiB | FLAC **16 kHz** | Yes |
| `splitted-audio-transcript/transcript` | one `.txt` per segment | 44,296 files | plain text | Yes |
| `youtube-jordan-48khz` | `Jordan-youtube-{train,validation}.parquet` (`audio_name`, `audio_bytes`, `transcript`) | 23.53 GB + 0.25 GB | Parquet | **Yes**, identical byte sizes |
| `youtube-jordan-telephony` | telephony parquet | 8.56 GB + 0.09 GB | Parquet | **Yes**, identical byte sizes |
| `training/wesam_dataset_words` | diarised podcast clips (section 1.3) | 7,102 WAV + `.lab`, 20.0 h, 6.7 GB | WAV 48 kHz PCM-16 + JSONL | **No** (derived; manifests pulled) |
| `training/jordanian_s2pro/cache/eef569…` | suite dataset: 44.1 kHz normalised WAV, `.npy` VQ, protos | train 2,429 / val 321 / test 308 sentences, 24 train speakers, 7.5 GB | WAV / npy / protobuf | No (re-derivable) |
| `training/data/speaker1`, `TTS_data.zip` | Jordanian female voice-over ads, 367 clips, 49.9 min, 44.1 kHz | 514 MB + 405 MB | WAV + `.lab` | Not under `Jordan/` |

- **Corpus totals** (`README.md`): 44,296 segments, **158 h 37 m**, 5–31 s, from the wesamali.official channel and two TalhaAhmad81 playlists.
- **Duplicated videos.** The two Talha playlists repeat **155 videos under different IDs**: 13,547 segments are duplicates, so only about 88 h of Talha audio is unique (`talha_s2pro/plan/summary.json`). The train/validation split of the ASR parquet is therefore probably leaky.
- **Transcripts are YouTube-caption orthography.** They have no punctuation and are normalised (ة→ه, أ/إ→ا), for example "ورحمه الله … التاخير يا جماعه". The S2 Pro suite trained on this text. Inference text from the LLM has punctuation and hamza.
- **QC rejects** in the suite: 95 of the selected clips (86 with more than 0.1 % clipped samples, 9 that did not match the SRT interval) (`cache/eef569…/qc.json`).
- **Dialect.** None of this data is Saudi; it is all Jordanian/Levantine. The only Saudi audio found is the 1-min Noura reference.

## 3. Install-time savers

- **wheelhouse/** (193 wheels, 2.5 GB): **stale.** It holds torch 2.1.2+cu121, torchaudio 2.1.2, triton 2.1.0 (cp310), nemo_toolkit 1.23.0, pyannote.audio 3.1.1, transformers 4.36.2 and 5.6.1, and lightning 2.6.1. There is no flash-attn, sglang or vllm. `EXPERIMENT_NOTES.md` explicitly says not to use these torch wheels. **Do not pull.**
- **uv_cache/** (7.1 GB): torch **2.8.0** cp310 manylinux_2_28 (requires `nvidia-cuda-runtime-cu12==12.8.90`), torchaudio 2.8.0, triton 3.4.0, cuDNN 9.10.2.21, lightning 2.6.1, transformers 4.57.3, protobuf 5.29.6 and descript-audio-codec 1.0.0. All of it is on PyPI, so pulling 7 GB saves nothing. **Do not pull.**
- **pip_cache/** (7.8 GB) and `conda/pkgs` (571 MB): generic caches. Skip them.
- **sglang_docker_storage/**: root-owned and unreadable. It is the Docker/containerd data-root that was moved to NVMe and holds the built image `sha256:64f376e7…`. It cannot be exported without writing on the box. **Rebuild it from the pulled `sglang.Dockerfile` instead.**
- **Environment versions proven to work on H100:**
  - **Fish training:** Python 3.10.20 (Miniconda) with `protobuf>=3.20,<6`, torch/torchaudio 2.8.0 (cu128), lightning 2.6.1, transformers 4.57.3, datasets 2.18.0, `numpy<2`, `pyarrow<21`, hydra-core 1.3, tensorboard 2.20.0 (2.21 needs protobuf ≥ 6.31), einx 0.2.2, descript-audio-codec, pyloudnorm (`talha_s2pro/monitor/training-requirements.in`, `dependency-overrides.txt`).
  - **Serving:** Docker `lmsysorg/sglang:v0.5.16` + `sglang-omni==0.1.3`, `descript-audio-codec==1.0.0`, `descript-audiotools==0.7.2`, protobuf override `>=6.31.1,<7`, torch 2.11.0+cu130, transformers 5.12.1. It needs a host driver that supports **CUDA 13**.
  - Engine settings: bf16, FA3, CUDA graphs (`cuda_graph_bs_decode`, `cuda_graph_max_bs_decode`), 4,096 context, 8,192 KV tokens, 1 running request, `mem_fraction_static` 0.32 per engine.
  - **Diarisation:** pyannote.audio 4.0.7, speechbrain 1.1.1, torch 2.8.0, numpy 2.2.6 (`wesam_review/venv-installed-requirements.txt`).

## 4. Recommendations for the Saudi run

**Adopt:**
1. The data recipe: 44.1 kHz mono, -23 LUFS, -1 dBTP, polyphase resampling, 5–30 s clips. Reject inputs with more than 0.1 % clipped samples. Deduplicate by decoded-PCM hash and by source title before splitting. **Split by source recording.** Keep held-out test protos that are never used for selection. Run a codec round-trip before each campaign.
2. The safety rails: pilot runs that pad to full context, run 2 steps, resume and reach step 3; checks that gradients are finite and adapters are actually updated; content receipts and hashes for clips, tokens and base weights.
3. The trainer fixes: `use_reentrant=False` checkpointing, `persistent_workers=(num_workers>0)`, and merging with the *saved* LoRA α. Upstream's `r_32_alpha_16_fast` config name hides that α was 32. Also: torch 2.8 `torch.load` needs `weights_only=False` for checkpoints containing OmegaConf objects, and the upstream trainer auto-resumes the newest-mtime checkpoint even when `ckpt_path` is given. Use fresh output directories.
4. The evaluation pattern: fixed references, fixed prompts, and base/early/best/final checkpoints all synthesised. Waveform metrics (RMS, quiet fraction, trailing silence) are what caught the silent-output collapse. Add our WER-O, SIM and UTMOS on top.
5. Serving: the SGLang-Omni Dockerfile, the native-HF exporter (`sglang_export.py`, `sglang_validate_exports.py`), and the streaming/telemetry code. **Cache reference encodings**: first use costs 13–19 s of CPU.
6. Diarisation and identity for multi-speaker Saudi sources: Community-1 + WeSpeaker with a cosine threshold of 0.5, ≥ 9 s of evidence and ≥ 80 % agreement; on GPU at batch 64.

**What the plan missed:**
- **P1, train/generate embedding mismatch** (§1.2). Before any Slow-AR LoRA:
  - Patch `BaseTransformer.embed()` to apply the same √11 and √2 scaling as `forward_generate`, or confirm that current upstream HEAD has fixed it.
  - Add a regression test that teacher-forced `forward` logits equal `forward_generate` logits.
  - Gate on the base model's step-0 loss and top-5 accuracy, which are near chance today.
- **P2, Slow-AR LoRA at LR 2e-5 collapsed to silence within 1,500 steps** (75 % of a 2,000-step plan), and only step 200 was usable. The plan's 10–20k steps at 5e-5 to 1e-4 is therefore untested and risky until P1 is fixed. Re-pilot with generation checks (silence fraction and ASR CER) every 100–200 steps, and keep the plan's "earliest acceptable checkpoint" rule.
- **P3, fast-only LoRA barely changes output.** Waveform stats stay close to the base model and Slow-AR token loss is unchanged. It cannot teach dialect, because pronunciation and lexis live in the Slow-AR.
- **P4, first audio is not TTFT.** Their SGLang TTFT was 90–125 ms but first audio arrived after about 0.9–1.0 s. The plan's 200 ms gate must measure first PCM, and needs codec-chunk tuning. Concurrency was never tested (1 running request per engine).
- **P5, S2 Pro output artefacts seen in production:** fading or quiet endings, clipped final syllables, overlong pauses, and Arabic homograph errors. The workarounds were a `[short breath]` suffix tag, a leveler, and selective diacritics. Add these to the hard set.
- **P6, data hygiene:** YouTube-caption orthography (no punctuation, ة→ه, hamza dropped); 155 duplicated videos across playlists; VTT word ends that are inferred, not aligned. The `aljawab-ai/ctc-forced-aligner` pinned commit returned 404.
- **P7, the April adapter's validation set was its training set.** None of its loss numbers mean anything.
- **P8, OmniVoice full FT overfits after about 500 steps** on 50 min of one speaker (LR 5e-5, `batch_tokens` 8192). For our quality track, use eval-loss-based early stopping. They used `language_id "apc"` for Jordanian; use a Saudi ISO code consistently.

**Still worth pulling** (not done, because they are checkpoints or audio; total about 7.2 GB):

| What | Remote path | Size | Why |
| --- | --- | --- | --- |
| Noura Saudi reference | `speech-agent/saudi_female_1minute-trimmed2.wav` | 1.7 MB | Saudi reference for the bake-off and continuity with the demo voice; transcript is in `assets/voice/reference.json` |
| Selected adapters | `training/jordanian_s2pro/suites/36aa33e6739594a4858d620c/runs/fast_r32_d1eaf54e40cd7adf/checkpoints/final.ckpt`; `…/attention_mlp_r16_8ccaa9923c3a033a/checkpoints/early.ckpt` | 85 MB + 402 MB | Lets us reproduce the current production voice as a bake-off baseline (merge onto HF s2-pro with `merge.py`, after checking the base hashes in `settings.json`) |
| Wesam diarised clips (optional) | `training/wesam_dataset_words/audio/` | 6.7 GB | 20 h of speaker-labelled Levantine audio, usable as pan-Arabic or replay data. It can also be re-cut from S3 FLACs using the pulled manifests |

- **Not needed:** the S2 Pro base (11 GB; on HF), the exported or merged models (`sglang/models` 13 GB, `speech-agent/transfers` 62 GB, the NF4 bundle 31 GB, `talha_s2pro/runs` 10 GB), the old S1-mini and April checkpoints (57 GB), `talha_qwen3` (48 GB), the chatter checkpoints (29 GB), the OmniVoice speaker1 checkpoints (28 GB) and all caches.

**Can Amin stop the box?**
- **For this project: yes, once the three items above are pulled** (or knowingly skipped). Everything else we need is in S3 or `pipeline/legacy/`.
- **Two caveats before stopping:**
  1. **Stopping wipes `/opt/dlami/nvme`**, an instance-store LVM holding 563 GB. That includes every Jordanian model, the NF4 delivery ZIP, the Docker images and the speech-agent code it runs from.
  2. **The public Noura voice-agent demo and the port-3000 SGLang studio are live on this box** and will die. Their systemd units point at NVMe paths, so they will not come back after a restart.
- Amin should confirm with the owner of the Jordanian/speech-agent work that they have what they need (for example, an S3 copy of `training/jordanian_s2pro/sglang/models/` and `speech-agent/`) before stopping.

## 5. Pulled into `pipeline/legacy/` (162 MB, code, configs, docs and JSON metadata only; no audio, no checkpoints, no `.env`)

| Local dir | Source on Majed-AWS (`/opt/dlami/nvme/…`) | Content |
| --- | --- | --- |
| `ttsft-docs/` | `PROJECT_STATUS.md`, `AGENTS.md`, `README.md`, `training/EXPERIMENT_NOTES.md`, `training/experiment_inventory.json`, `training/fish_diag_report.json`, `training/activate_fish.sh`, `speaker_embedings.ipynb` | Status docs; the notebook was copied with outputs stripped (its key is a placeholder) |
| `jordanian_s2pro/` | `training/jordanian_s2pro/` (excluding `suites/*/checkpoints`, `voice/`, `cache/`, `sglang/`, WAVs, PNGs) | Suite runner, `experiments.json`, data QC and balancing, merge, SGLang Dockerfile/launcher/exporter/stream, `reports/*.json`, tests; `suites/*/` holds configs, metrics, comparison results |
| `jordanian_s2pro/cache_meta/` | `training/jordanian_s2pro/cache/eef569b20c6e7d3f45ac876d/{selection,qc,packed,token_recipe}.json` | Dataset selection and QC receipts |
| `talha_s2pro/` (+ `plan/`, `run_51b79183605dbfee8d0b/`) | `training/talha_s2pro/` (excluding runs, cache, smoke, testing) | `pipeline.py` (normalise, extract_vq, pack, validate), `run.py`, `merge_model.py`, pinned requirements and overrides, Talha dedup plan, run config and metrics |
| `wesam_pipeline/` | `training/wesam_review/` (excluding `.venv`, models), `training/wesam_dataset_words/*.json(l)`, `training/wesam_dataset_gpu/` (diarisation JSONs, summaries), `training/wesam_exploration/` | Diarisation and identity code, Wesam profiles, per-episode diarisation, word-level clip manifests |
| `splitter/` | `Splitter/*.py`, `diar_config.yaml`, `input_manifest.json` | NeMo MSDD scripts |
| `omnivoice_speaker1/` | `training/omnivoice/speaker1/{config,manifests,*.sh,prepare_dataset.py,exp/full_finetune/{train.log,initial_config.json}}` | OmniVoice full-FT recipe and log |
| `chatter/` | `chatter/test.py`, `chatter/chatterbox-finetuning-multilingual/{lora.py,test.py,fix_*.py,diagnose_and_fix.py}` | Chatterbox LoRA script |
| `talha_qwen3/` | `training/talha_qwen3/{README.md,run.py,validation_report.json}`, `runs/9074c7c19faa68bc3976472d/{evaluation,recipe,best}.json` | Qwen3-TTS attempt (reference only) |
| `speech-agent/` | `speech-agent/` (excluding `.env`, venvs, `bin`, `transfers`, `exports`, `logs`, media) | Voice-agent app, specs, `.env.example` (keys empty), voice manifests, verification |
| `nf4_bundle/` | `training/attention_mlp_r16_nf4_20261017/bundle/` (excluding model, samples, reference) | NF4 conversion, launch scripts, `VALIDATION.md`, `requirements.lock` |

Credit: this work was done on TTS-FT by the Aljawab team (Jordanian S2 Pro suite, Wesam diarisation pipeline, speech-agent) on top of upstream fish-speech `214da3cd`, sglang-omni 0.1.3, pyannote and OmniVoice. Retain these attributions when we adopt code from `pipeline/legacy/`. Note that `wesam_pipeline/wesam_dataset_words/*.jsonl` (about 80 MB) is metadata; consider git-ignoring it rather than committing it.
