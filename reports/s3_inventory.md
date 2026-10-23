# S3 inventory — `s3://aljawab-asr-data/` (23 Oct 2026, read-only)

Source: `aws s3 ls --recursive --summarize` per prefix (raw listings in `/opt/dlami/nvme/inventory/s3/*.txt`, synced to `s3://aljawab-asr-data/saudi-tts/inventory/`), plus each prefix's README. `S3 directory sizes.txt` (Mar 2026) is stale: it lacks `Jordan/` and `multidialectal-parquet/`.

| Prefix | Objects | Size | What it is | Saudi content? | Use in this project |
|---|---:|---:|---|---|---|
| `Casablanca/` | 88 | 28.1 GiB | HF parquet export of Casablanca: 8 dialect configs (Algeria, Egypt, Jordan, Mauritania, Morocco, Palestine, UAE, Yemen) + `*-CS` code-switch configs; columns `audio, seg_id, transcription, gender, duration`; train/validation/test splits | No (confirms plan §6) | Stage-A auxiliary (UAE, Yemen = Gulf-adjacent); `Jordan` config (2.26 GiB) feeds the Jordanian dry run |
| `Egypt/` | 68,415 | 27.6 GiB | `combined-egypt-dataset` (ArzEn + Kaggle YouTube + MasriSpeech; HF `save_to_disk`, wav folder) and FLEURS `ar_eg` | No | Not used (not needed for Saudi or Jordanian) |
| `Jordan/` | 134,256 | 121.4 GiB | Mirror of the TTS-FT server data: `full-audios-transcript/` (452 long 48 kHz FLAC + SRT/VTT, 60.8 GiB), `splitted-audio-transcript/` (44,296 segments, 48 kHz + telephony copies, `.txt` transcripts), `youtube-jordan-48khz/` and `-telephony/` parquet (`audio_name, audio_bytes, transcript`). README: 158 h 37 m, 5–31 s segments, CTC-aligned with aljawab-ai/ctc-forced-aligner; channels wesamali.official and TalhaAhmad81 | No | **Primary Jordanian dry-run data** (already segmented and aligned, 48 kHz). Means the TTS-FT box's data is already backed up — nothing data-wise has to be pulled from it |
| `Levant/` | 43 | 1.8 GiB | IWSLT 2024 dev/test wav (Levantine) | No | Not used |
| `finance/` | 4,725 | 17.6 GiB | 2,362 YouTube mp3 + SRT (video-ID names, no channel metadata) | Unknown | Optional dialect-ID screen if real Saudi hours fall short (plan §6) |
| `jordanian/` | 103,238 | 773.8 GiB | 51,478 YouTube mp3 + 51,759 SRT | No (Jordanian) | Dry-run top-up only if needed |
| `multidialectal/` | 394,968 | 2,444 GiB | ~130k YouTube videos in 14 shards (`<id>.mp3`, `.ar.vtt`, `.ar-new.srt`), no channel metadata | Unknown — likely mixed | Optional dialect-ID screen (sample-based) only if the day-2 gap analysis asks; screening 17k h is not affordable this week |
| `multidialectal-parquet/` | 16,525 | 300.1 GiB | Parquet shards; `language_distribution_0.tsv` says 17,378.8 h labelled `apc_Arab` (Levantine) for corpus `aljawab_multidialectal` | Labelled Levantine | Not used for Saudi |
| `tiktok/jordan/` | 133 | 56.7 GiB | HF parquet: `id, channel_name, video_id, segment_index, language, audio (16 kHz), text, …`; 14,997 train + 307 validation rows | No (Jordanian) | 16 kHz — too low for TTS; not used |

## Takeaways
1. **No Saudi corpus exists in the bucket.** Saudi data must come from SADA, Omnilingual, SCC22, MASC (and ASWAT when it lands). `multidialectal/` and `finance/` could hide Saudi speakers, but with no channel metadata a screen means dialect-ID over sampled audio; it is parked behind the day-2 gap analysis.
2. **Jordanian dry-run data is ready**: `Jordan/splitted-audio-transcript` (158 h, 48 kHz, aligned) + `Casablanca/Jordan`. A ~20 h speaker-diverse subset is enough for the one-day dry run.
3. **Stage-A auxiliary**: `Casablanca/UAE` and `Casablanca/Yemen` (Gulf-adjacent), ~6.7 GiB.
4. New outputs of this project go only under `s3://aljawab-asr-data/saudi-tts/` (guard-enforced: no deletes elsewhere).
