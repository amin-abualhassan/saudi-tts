# Jordan-YouTube ASR Dataset

An Automatic Speech Recognition (ASR) dataset built from Jordanian Arabic YouTube content.

## Overview

| Property | Value |
|---|---|
| **Samples** | 44,296 |
| **Total Duration** | 158h 37m 18s |
| **Sample Rate** | 48 kHz |
| **Audio Format** | FLAC |
| **File Format** | Parquet |

## Sources

The audio was collected from the following YouTube channels and playlists:

- **wesamali.official** — all videos
- **TalhaAhmad81** — two playlists:
  - talha-screenshot-2025
  - talha-screenshot-2024

## Alignment

Transcript alignment was performed using CTC forced alignment via [aljawab-ai/ctc-forced-aligner](https://github.com/aljawab-ai/ctc-forced-aligner/tree/915f1672a43c536eebb0e82835bb917583d2ad06).

## Dataset Schema

The Parquet file contains three columns:

| Column | Type | Description |
|---|---|---|
| `audio_name` | string | Name of the original source file (e.g., `first.flac`) |
| `audio_bytes` | binary | Raw bytes of the sliced FLAC sub-audio segment |
| `transcript` | string | Text transcription for the corresponding audio segment |

## Filtering

Only segments between **5 seconds** and **31 seconds** in length were kept. Segments outside this range were discarded.

## Storage

The dataset is stored at:

```
s3://aljawab-asr-data/Jordan/youtube-jordan-48khz/Jordan-youtube.parquet
```

Region: `eu-north-1`
# Project updates

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the latest verified results,
current focus, open issues, and next steps across this workspace.

