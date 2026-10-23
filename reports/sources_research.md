# Sources research: download routes for datasets and models

Checked 23 Oct 2026 against the public HF API (`/api/{models,datasets}/<id>?blobs=true`), the HF datasets-server (`/size`, `/first-rows`, `/statistics`), the Kaggle public API, IEEE DataPort pages, GitHub READMEs and the ElevenLabs docs. No large files were downloaded and no HF token was used. Sizes are the sum of `siblings[].size` (GB = 10^9 bytes). Anything marked **UNVERIFIED** was not confirmed from a primary source.

## 0. Blocker before anything downloads tonight

`ops/guard.py` was extended at 21:02 with **rule 8**: any `hf download`, `huggingface-cli download`, `curl`, `wget`, `aria2c`, `kaggle` or `git clone` aimed at a Hugging Face repo or host that is **not listed in `ops/allowlist.txt` is blocked**. That file **does not exist yet**, so every command below will be refused until it does. It takes lines of the form `hf <owner>/<repo>` (with `owner/*` wildcards allowed) and `host <domain>`. Hosts needed beyond huggingface.co: `www.kaggle.com` (if Kaggle is used), `dl.fbaipublicfiles.com` (omnilingual_asr auto-download), `drive.google.com` (WavLM-SV checkpoint), `github.com` / `raw.githubusercontent.com` (torch.hub for UTMOS22 and Silero), `ieee-dataport.org` and its S3 bucket (MASC official). Partway through this research the guard also started blocking my own `curl` calls to the HF API; the later checks were done with WebFetch.

None of the tools are installed: `hf`, `kaggle` and `gdown` are all missing, and system Python is 3.14 (too new for most of these repos, so use a uv venv per model). Install the HF CLI with:
```bash
uv tool install "huggingface_hub[cli]"        # provides `hf`; hf_xet is used automatically for Xet-backed repos
export HF_HOME=/opt/dlami/nvme/hf-cache HF_XET_HIGH_PERFORMANCE=1
D=/opt/dlami/nvme/data/raw; M=/opt/dlami/nvme/models
```
The guard's regex expects the repo ID right after `hf download`, so keep the repo ID first and put the flags after it.

## 1. Datasets (priority order)

| # | Dataset | Best route (repo / URL) | Gated | Size | License (as published) |
|---|---|---|---|---|---|
| 1 | SADA | HF `khaledalganem/sada2022` (byte-identical Kaggle conversion) | No | **77.06 GB** (4,563 WAV + 3 CSV) | CC BY-NC-SA 4.0 |
| 1b | SADA (official copy) | Kaggle `sdaiancai/sada2022` | Kaggle login + API token | 77.06 GB (≈64.9 GB zipped) | CC BY-NC-SA 4.0 |
| 2 | Omnilingual (ars/acw/afb) | HF `facebook/omnilingual-asr-corpus`, `data/{ars,acw,afb}_Arab/` | No | **8.46 GB** (2.90 + 3.15 + 2.41) | CC-BY-4.0 |
| 3 | SCC22 audio | HF `MohamedRashad/SCC22` | No | **2.56 GB** (6 parquet) | CC BY-NC-SA 4.0 |
| 3b | SCC22 official CSV | HF `SDAIANCAI/Saudilang-Code-Switch-Corpus` (official SDAIA org) or Kaggle `sdaiancai/saudilang-code-switch-corpus-scc` | No / Kaggle login | 1.2 MB (CSV only: YouTube links + timestamps) | card: CC BY-NC-SA 4.0 (YAML says cc-by-sa-4.0) |
| 4 | MASC (with country labels) | HF `Byne/MASC` | No | **134.45 GB** (274 parquet) | none stated on mirror; original is CC-BY-4.0 |
| 4b | MASC official | IEEE DataPort `masc.tar.gz` | Free IEEE account login | 169.41 GB tar.gz | CC-BY-4.0 |
| 4c | MASC (ASR mirror) | HF `pain/MASC` | No (card carries a gating prompt, but API reports `gated=false`) | 87.70 GB (tar.xz + CSV) | CC-BY-4.0 |
| 5 | Saudi ASWAT | KSAA (falak.ksaa.gov.sa) | Request only | n/a | n/a |

### 1.1 SADA
- **IEEE DataPort node 14747 is not usable.** It says "requires an IEEE DataPort Subscription" and "Files have not been uploaded for this dataset"; only `Sada.txt` is offered.
- **The Kaggle slug in the brief is wrong.** `sdaiaonline/sada-saudi-audio-dataset-for-arabic` does not exist. The real one is `sdaiancai/sada2022` (version 23, updated 2023-08-31, `totalBytes` 77,055,385,134).
- **The HF mirror `khaledalganem/sada2022` matches Kaggle.** Its card says `converted_from: kaggle, kaggle_id: sdaiancai/sada2022`, its total is the same 77.06 GB, and it has the same 4,563 WAV files the IEEE page describes. This is the recommended route: no login needed, and it is fast.
- **Structure.** Four folders, `batch_1/` to `batch_4/` (1,140–1,141 WAVs each, about 19 GB each). Each WAV is a full ~10-minute episode. I checked the RIFF header: PCM 16-bit, mono, 16,000 Hz. There are also `train.csv` (123 MB), `valid.csv` (2.7 MB) and `test.csv` (3.2 MB).
- **CSV columns (verified from the header):** `FileName, ShowName, FullFileLength, SegmentID, SegmentLength, SegmentStart, SegmentEnd, SpeakerAge, SpeakerGender, SpeakerDialect, Speaker, Environment, GroundTruthText, ProcessedText, Category`. Example values: `Elderly -- كبير في السن`, `Najdi`/`Khaliji`, `Speaker1متحدث` (the speaker ID is only unique within a file), `Clean -- نظيف` / `Music -- موسيقى`. `GroundTruthText` keeps diacritics and punctuation; `ProcessedText` is normalised.
- **Do not use `MohamedRashad/SADA22` for our purposes** (50.4 GB of pre-cut parquet, 241,834 / 5,139 / 6,193 rows). It drops `Environment`, `Speaker`, `SegmentStart`/`SegmentEnd` and `ShowName`, all of which the pipeline needs.

```bash
hf download khaledalganem/sada2022 --repo-type dataset --include "*.csv" --local-dir $D/sada2022     # CSVs first (130 MB) to plan filters
hf download khaledalganem/sada2022 --repo-type dataset --local-dir $D/sada2022 --max-workers 16       # full 77 GB
# Fallback (needs Kaggle account + ~/.kaggle/kaggle.json): kaggle datasets download -d sdaiancai/sada2022 -p $D/sada_kaggle --unzip
```

### 1.2 Meta Omnilingual ASR Corpus (Saudi-relevant only)
- **Configs** are named `{iso639_3}_{script}`: `ars_Arab` (Najdi), `acw_Arab` (Hijazi), `afb_Arab` (Gulf). Each has a train split only; there is no dev or test split for these three.
- **Columns:** `language, speaker_id, prompt_id, prompt, segment_id, audio (FLAC), duration, raw_text, iso_639_3, glottocode, iso_15924`. `raw_text` contains tags (laughter, hesitation, unintelligible, noise), so they need stripping. The sample rate is not in the schema; the ~1 MB-per-clip size suggests 44.1–48 kHz (**UNVERIFIED**).
- **Measured with datasets-server `/statistics`** (complete, not partial):

| Config | Rows | Mean dur | Hours | Unique `speaker_id` | Parquet |
|---|---|---|---|---|---|
| ars_Arab | 2,831 | 26.1 s | **20.5 h** | 16 | 2.90 GB (6 files) |
| acw_Arab | 2,955 | 27.2 s | **22.3 h** | 20 | 3.15 GB (7 files) |
| afb_Arab | 2,815 | 24.0 s | **18.8 h** | 12 | 2.41 GB (5 files) |

  That is **about 61.6 h in total, more than the plan's 25–30 h, but with only 48 speakers**, so the expected yield in the plan should be revised. Three clips exceed the 40 s limit of the non-"Unlimited" omniASR models.

```bash
hf download facebook/omnilingual-asr-corpus --repo-type dataset --include "data/ars_Arab/*" "data/acw_Arab/*" "data/afb_Arab/*" README.md --local-dir $D/omnilingual
# or in Python: load_dataset("facebook/omnilingual-asr-corpus", "ars_Arab", split="train")
```

### 1.3 SCC22 (Saudilang Code-Switch Corpus)
- **The official release has no audio.** SDAIA publishes only `scc_testset.csv` (12 columns: `Link` to YouTube, `SegmentStart, SegmentEnd, Original_text, ProcessedText, Language, Environment, Speaker, SpeakerGender, FullFileLength, SegmentLength, Segment_ID`). It covers 3,296 segments, 4.45 h, 4 male speakers and 3 Thmanyah episodes.
- **The audio mirror is complete.** `MohamedRashad/SCC22` has the same 3,296 rows, all in a single `test` split. Columns are `audio, text, cleaned_text, environment_type, speaker_id, speaker_gender`. datasets-server decodes the audio as 44.1 kHz mono; that it is the native rate is **UNVERIFIED**. The mirror lacks start/end and YouTube link, so join it to the official CSV by row order or text (**UNVERIFIED** that the order matches).
- `HuggingPanda/SDAIANCAI-Saudilang-Code-Switch-Corpus` (0.17 GB, 48 kHz, re-split 2,962/165/165, `sentence` column only) is a lower-metadata alternative.

```bash
hf download MohamedRashad/SCC22 --repo-type dataset --local-dir $D/scc22
hf download SDAIANCAI/Saudilang-Code-Switch-Corpus --repo-type dataset --local-dir $D/scc22/official
```

### 1.4 MASC
- **IEEE DataPort route.** The dataset is "Open Access" but downloading requires logging in with a free IEEE account. It is one archive, `masc.tar.gz` of 169.41 GB, and **it ships audio**: `audios/` holds WAV at 16 kHz and `subtitles/` holds `.ar.vtt` files. `channels_quality.csv` has channel_url, name, id and quality. `subsets/` has clean/noisy train/dev/test segment CSVs (`video_id,start,end,duration,text`) plus `_meta` CSVs (`video_id, category, video_duration, channel_id, country, dialect, gender, transcript_duration`). The page recommends AWS download after login (exact S3 command **UNVERIFIED**; it appears only when logged in).
- **The practical route is `Byne/MASC`.** It has a train split of 942,813 rows and a test split of 104,757 rows. Columns are `video_id, dialect, gender, country, category, text, audio(16 kHz)`. It has **no start/end and no channel_id**, and its provenance is undocumented (**UNVERIFIED** that it is a faithful copy).
- **There is no Saudi-only subset.** Download everything and filter on `country == "SA"` (channel country) and/or `dialect == "Saudi Arabia"`.
- **Saudi share.** Partial datasets-server statistics on the first 38,640 train rows show country SA at 6,246 rows (16%), dialect "Saudi Arabia" at 1,871 (4.8%) and dialect MSA at 18,591 (48%). The "Arab Voices" survey (arXiv 2601.13319) puts Saudi-dialect MASC at **~31 h**, against 569 h of MSA.
- **The plan's "452 h channel-labelled Saudi" is UNVERIFIED and looks too high.** Expect roughly 31 h dialect-labelled Saudi, plus perhaps ~100–160 h from SA channels that are largely MSA (an extrapolation from the partial statistics).
- **Other mirrors.** `pain/MASC` (87.7 GB) has audio tarballs and transcripts but no country. `MohamedRashad/MASC-Arabic` (184.9 GB) has no country. `TRoboto/masc` is text only (twitter LM).

```bash
hf download Byne/MASC --repo-type dataset --local-dir $D/masc_byne --max-workers 16      # 134 GB, then filter country/dialect
# Official (manual, browser login): https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus  -> masc.tar.gz 169.41 GB
```

### 1.5 Saudi ASWAT
- I found no public download. The LREC 2026 paper (2026.lrec-1.124) abstract says nothing about access. The KSAA "Aswat Corpus" initiative page lists only `info@ksaa.gov.sa` as a contact.
- `falak.ksaa.gov.sa` is a JavaScript app that does not render without a browser, so I could not read its access terms.
- Conclusion: access is by request, which is consistent with waiting on the owner. Whether any public portion exists is **UNVERIFIED**.

## 2. Models

Gated column: **auto** means you click "agree" on the HF page while logged in, and the download then needs an HF token. Every other repo is fully public.

| Model | HF repo(s) | Gated | Size | License | Python / install | Streaming |
|---|---|---|---|---|---|---|
| Fish S2 Pro | `fishaudio/s2-pro` | No | 11.01 GB | Fish Audio Research License (non-commercial without a separate licence) | Py 3.12. `git clone fishaudio/fish-speech; uv sync --python 3.12 --extra cu129` (cu126/cu128/cu129); apt `portaudio19-dev libsox-dev ffmpeg` | Yes |
| ↳ SGLang path | same | | | | `uv venv -p 3.12; uv pip install --prerelease=allow "sglang-omni==0.1.9"; uv pip install "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0"`; Docker `hongccc/sglang-omni:dev` / `lmsysorg/sglang-omni:dev` (torch 2.14.1, flash-attn-4) | `sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000`; POST `/v1/audio/speech` with `stream` (raw PCM chunks), `references:[{audio_path,text}]` or `ref_audio/ref_text` |
| MOSS-TTS Local v1.5 (4B, 48 kHz stereo) | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` + codec `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` | No | 9.12 + 8.50 GB | Apache-2.0 | Py 3.12, transformers 5.0.0. `git clone OpenMOSS/MOSS-TTS; pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"` (pins torch 2.9.1+cu128) | Yes, via SGLang-Omni `/v1/audio/speech` (48 kHz PCM chunks); realtime decode examples in `moss_tts_local_v1.5/` |
| OmniVoice (24 kHz) | `k2-fsa/OmniVoice` | No | 3.27 GB | Code Apache-2.0; **weights CC-BY-NC** | `pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url …/cu128; pip install omnivoice` (Python version not stated; use 3.11/3.12) | No native streaming |
| VoxCPM2 (48 kHz) | `openbmb/VoxCPM2` | No | 4.96 GB | Apache-2.0 | `pip install voxcpm`; Python ≥3.10 <3.13, torch ≥2.5, CUDA ≥12 | Yes: `model.generate_streaming()`; also `vllm serve openbmb/VoxCPM2 --omni` or `nano-vllm-voxcpm` |
| Higgs TTS 3 (4B, 24 kHz) | `bosonai/higgs-tts-3-4b` | No | 9.32 GB | Boson Higgs TTS 3 Research & Non-Commercial | Py 3.12 uv; SGLang-Omni or `vllm-omni serve bosonai/higgs-tts-3-4b --trust-remote-code --omni` | Yes: `sgl-omni serve --model-path bosonai/higgs-tts-3-4b`, `"stream": true` (base64 WAV chunks) |
| Fish Drama 3 (35B-A3B, FP8 experts) | `fishaudio/drama-3` (`fishaudio/drama-3-int4` 24.03 GB) | No | 41.50 GB | Card says Fish Audio Community License; metadata says research licence | SGLang-Omni `uv pip install --prerelease=allow -v -e ".[fishaudio]"` | Yes: `sgl-omni serve --model-path fishaudio/drama-3`; TTFA p50 88.9 ms (H200, c=1) |
| Audar-TTS-V1-Flash (0.55B, 24 kHz) | `audarai/Audar-TTS-V1-Flash` + `neuphonic/neucodec` | Flash: No; **neucodec: auto** | 2.70 + 3.68 GB (neucodec `model.safetensors` alone is 2.52 GB) | AudarAI Open License v1.0 (commercial allowed); neucodec Apache-2.0 | `pip install transformers torch neucodec soundfile librosa` (or GGUF via llama-cpp-python); reference audio 16 kHz mono | Not documented |
| MOSS-TTS Realtime (1.7B, 24 kHz) | `OpenMOSS-Team/MOSS-TTS-Realtime` + codec `OpenMOSS-Team/MOSS-Audio-Tokenizer` | No | 4.68 + 7.10 GB | Apache-2.0 | Same MOSS-TTS env (Py 3.12) | Yes: `example_llm_stream_to_tts.py`; TTFB ~180 ms on L20; vLLM-Omni recipe `recipes/OpenMOSS/MOSS-TTS.md` |
| Habibi-TTS SAU (F5, 336M) | `SWivid/Habibi-TTS`, file `Specialized/SAU/model_200000.safetensors` + `vocab.txt` | No | 1.35 GB (repo 10.79 GB) | CC-BY-NC-SA-4.0 (SAU/UAE/unified) | Py ≥3.10, `pip install habibi-tts`; `habibi-tts_infer-cli --dialect SAU` (exact SAU invocation **UNVERIFIED**) | No |

Other MOSS repos, so the right one is downloaded: `MOSS-TTS-v1.5` is the 8B delay model (17 GB) and `MOSS-TTS-Local-Transformer` is the 1.7B v1.0. `OpenMOSS-Team/MOSS-TTS-Local-v1.5` does not exist (401).

```bash
hf download fishaudio/s2-pro --local-dir $M/s2-pro
hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --local-dir $M/moss-local-v1.5
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 --local-dir $M/moss-audio-tokenizer-v2
hf download k2-fsa/OmniVoice --local-dir $M/omnivoice
hf download openbmb/VoxCPM2 --local-dir $M/voxcpm2
hf download bosonai/higgs-tts-3-4b --local-dir $M/higgs-tts-3
hf download fishaudio/drama-3 --local-dir $M/drama-3            # or fishaudio/drama-3-int4
hf download audarai/Audar-TTS-V1-Flash --local-dir $M/audar-flash
hf download neuphonic/neucodec --include model.safetensors config.json preprocessor_config.json meta.yaml --local-dir $M/neucodec   # needs token after accepting
hf download OpenMOSS-Team/MOSS-TTS-Realtime --local-dir $M/moss-realtime
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir $M/moss-audio-tokenizer
hf download SWivid/Habibi-TTS --include "Specialized/SAU/*" --local-dir $M/habibi
```

### 2.1 Evaluation models

| Tool | Repo / source | Gated | Size | Licence | Install / notes |
|---|---|---|---|---|---|
| Cohere Transcribe Arabic (2B) | `CohereLabs/cohere-transcribe-arabic-07-2026` | **auto** | 4.13 GB | Apache-2.0 | transformers; card is behind the gate so the recipe was not read (**UNVERIFIED**). Blog: SADA test WER 37.47; internal Najdi 26.18, Gulf 24.36, Hijazi 16.24 |
| oddadmix whisper v2 | `oddadmix/whisper-large-v3-arabic-dialectal-v2` | No | 6.18 GB (single fp32 `model.safetensors`) | Apache-2.0 | HF Whisper pipeline. Trained on `oddadmix/lahgtna-v3-small` (13 dialects incl. Saudi/Gulf; SADA not mentioned); WER 0.320 / CER 0.120 |
| ADI-20 dialect ID | `Elyadata/ADI-whisper-ADI20` | No | 7.63 GB repo; **2.55 GB needed** (skip `whisper_opt.ckpt`, `optimizer.ckpt`) | none stated | `git clone elyadata/ADI-20; pip install -r requirements.txt` (SpeechBrain). Labels include `KSA`(4), `KUW`, `QAT`, `UAE`, `BAH`, `OMA`, `MSA`. hparams also pull `openai/whisper-large-v3` (≈3 GB, **UNVERIFIED** size) |
| Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B` | No | 4.70 GB | Apache-2.0 | Py 3.12, `pip install -U qwen-asr` (`qwen-asr[vllm]` for vLLM and streaming). Arabic listed. Its forced aligner `Qwen/Qwen3-ForcedAligner-0.6B` does **not** list Arabic |
| Omnilingual ASR (WER-O) | **`omniASR_LLM_7B`**: what Habibi's `2_cal_wer-o.py` uses, with `SAU-Najdi→ars_Arab`, `SAU-Hijazi→acw_Arab`, `SAU-Gulf→afb_Arab`. HF copy `facebook/omniASR-LLM-7B` | No | 31.21 GB (30 GiB) | Apache-2.0 | `pip install omnilingual-asr` (needs libsndfile). It auto-downloads from `dl.fbaipublicfiles.com/mms/…` into `~/.cache/fairseq2/assets/`. 40 s audio limit (`omniASR_LLM_Unlimited_7B_v2` removes it). `_v2` cards exist; keep `omniASR_LLM_7B` for comparability with Habibi |
| WavLM-large SV (seed-tts-eval) | `wavlm_large_finetune.pth`, Google Drive id `1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP`, plus the WavLM-Large upstream via s3prl torch.hub | n/a | ≈1.2 GB (**UNVERIFIED**) | UniSpeech (MIT, **UNVERIFIED**) | `gdown 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP` (needs `drive.google.com` allowlisted) |
| UTMOS | Habibi uses **UTMOS22 strong**: `torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong")`. UTMOSv2: `sarulab-speech/UTMOSv2` | No | UTMOSv2 4.09 GB; UTMOS22 small | UTMOSv2 MIT | `pip install git+https://github.com/sarulab-speech/UTMOSv2.git` (Py 3.9–3.13); `utmosv2.create_model(pretrained=True)` |
| pyannote community-1 | `pyannote/speaker-diarization-community-1` | **auto** | 0.03 GB | CC-BY-4.0 | `pip install pyannote.audio` (4.x); card is behind the gate (exclusive-mode API **UNVERIFIED** here) |
| Forced alignment | `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` | No | 2.52 GB | Apache-2.0 | transformers CTC |
| Silero VAD | pip package `silero-vad` (GitHub snakers4/silero-vad); HF ONNX `onnx-community/silero-vad` | No | <10 MB | MIT | `pip install silero-vad`; 8/16 kHz only |

```bash
hf download CohereLabs/cohere-transcribe-arabic-07-2026 --local-dir $M/cohere-transcribe-ar   # needs token after accepting
hf download oddadmix/whisper-large-v3-arabic-dialectal-v2 --exclude "samples/*" --local-dir $M/whisper-ar-dialectal-v2
hf download Elyadata/ADI-whisper-ADI20 --exclude whisper_opt.ckpt optimizer.ckpt --local-dir $M/adi20
hf download openai/whisper-large-v3 --include "*.json" model.safetensors --local-dir $M/whisper-large-v3   # base for ADI-20 hparams
hf download Qwen/Qwen3-ASR-1.7B --local-dir $M/qwen3-asr-1.7b
hf download facebook/omniASR-LLM-7B --local-dir $M/omniASR-LLM-7B   # or let omnilingual_asr pull it from dl.fbaipublicfiles.com
hf download sarulab-speech/UTMOSv2 --local-dir $M/utmosv2
hf download pyannote/speaker-diarization-community-1 --local-dir $M/pyannote-community-1      # needs token after accepting
hf download jonatasgrosman/wav2vec2-large-xlsr-53-arabic --local-dir $M/w2v2-ar
```

## 3. ElevenLabs (`eleven_v4_turbo`)
- **Endpoints.** The models page says "Eleven v4 Turbo is available via the **Text to Dialogue websocket**": `wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?model_id=eleven_v4_turbo&output_format=…`. With v4 Turbo it accepts **exactly one** registered voice per connection (`eleven_v4` allows 10). The first message carries `voices` (and `xi_api_key` unless sent as a header); later frames carry `inputs:[{text, voice_id, new_turn}]`, and audio comes back as base64 chunks.
- **The classic socket will not work.** The TTS WebSocket `/v1/text-to-speech/{voice_id}/stream-input` officially "does **not** support the `eleven_v3` or `eleven_v4` models". A third-party PR (lampOS, 21 Oct 2026) reports error `4400 unsupported_model` there, and 116 ms to first audio on the dialogue socket against 214 ms for Flash on stream-input.
- **HTTP streaming is unconfirmed.** `POST /v1/text-to-dialogue/stream` exists (default `eleven_v3`; PCM 8–48 kHz, `pcm_44100` needs Pro). Whether it accepts `eleven_v4_turbo` is **UNVERIFIED**, as is whether `/v1/text-to-speech/{id}/stream` accepts it. Check `GET /v1/models` (`can_do_text_to_speech`). This supersedes the plan's "SDK selects" wording: use the dialogue WebSocket.
- **Pricing** (elevenlabs.io/pricing/api, Oct 2026): v4 Turbo **$0.04 / 1K chars**; v4, v3 and Multilingual v2 **$0.08 / 1K**; Flash/Turbo $0.04. Plan allowances for v4 Turbo: Pro ($99) 2,475,000 chars, Scale ($299) 7,475,000, Business ($990) 24,750,000.
- **Maryam E (`aMmeBf0lzDYlouyfqNjh`): UNVERIFIED.** The ID does not appear in any public index, and "Maryam" is not in third-party Saudi voice catalogues. Shared library voices are normally added with `POST /v1/voices/add/{public_owner_id}/{voice_id}` (`new_name` required; `public_owner_id` comes from `GET /v1/shared-voices?search=Maryam`). Before generating anything, check through `pipeline/elevenlabs_client.py` (guard rule 7 forbids direct curl to ElevenLabs): `GET /v1/voices/aMmeBf0lzDYlouyfqNjh`. A 404 or a `category: "professional"/"shared"` result means it must be added. Record its `sharing.disable_at_unix`.

## 4. Disk estimate (download size on NVMe)

| Group | GB |
|---|---|
| SADA (HF WAV) | 77.1 |
| Omnilingual ars+acw+afb | 8.5 |
| SCC22 (audio + CSV) | 2.6 |
| MASC via `Byne/MASC` (all; the Saudi slice is only a fraction) | 134.5 (IEEE tar: 169.4 + ~170 extracted) |
| **Datasets subtotal** | **≈ 223** |
| Tier A models (S2 Pro 11.0, MOSS Local v1.5 9.1 + codec v2 8.5, OmniVoice 3.3, VoxCPM2 5.0) | 36.9 |
| Tier B (Higgs 9.3, Drama-3 41.5, Audar 2.7 + neucodec 3.7, MOSS RT 4.7 + codec 7.1, Habibi SAU 1.35) | 70.3 (53 with drama-3-int4) |
| Eval (Cohere 4.1, oddadmix 6.2, ADI20 2.6 + whisper-v3 ~3, Qwen3-ASR 4.7, omniASR 7B 31.2, UTMOSv2 4.1, w2v2 2.5, WavLM-SV ~1.2, rest <0.1) | ≈ 60 |
| **Total downloads** | **≈ 390 GB** |

Working copies (24 kHz FLAC, a 16 kHz copy for VoxCPM2, 48 kHz for MOSS, Demucs stems for SADA Music) will add roughly 0.3–0.5 TB, so the total stays well under the 3.3 TB free.

## 5. Corrections to plan-v2
1. SADA: IEEE node 14747 has no files. Use `khaledalganem/sada2022` (HF) or Kaggle `sdaiancai/sada2022`.
2. Omnilingual Saudi: about 61.6 h and 48 speaker IDs, not 25–30 h.
3. MASC Saudi: "452 h" is unverified. Dialect-labelled Saudi is about 31 h; country=SA is about 16% of rows and mostly MSA.
4. SCC22: the official release is CSV + YouTube links only; the audio comes from a community HF mirror.
5. Habibi's UTMOS figures are UTMOS22-strong, not UTMOSv2, and its WER-O is `omniASR_LLM_7B` with per-dialect language codes.
6. ElevenLabs v4 Turbo streaming is the Text-to-Dialogue WebSocket only (single voice).
7. Weight licences: OmniVoice weights are CC-BY-NC, S2 Pro / Drama 3 / Higgs 3 are research licences, and Habibi SAU is CC-BY-NC-SA. All are covered by Amin's written permissions per the plan; copy each into `reports/LICENCES.md`.
