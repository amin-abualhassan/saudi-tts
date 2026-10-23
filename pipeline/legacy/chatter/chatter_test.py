#!/usr/bin/env python3
"""
Prepare Chatterbox Multilingual LoRA dataset from:
  /opt/dlami/nvme/training/data/speaker1
where each utterance has:
  - audio: *.wav
  - transcription: matching *.lab

Output:
  /opt/dlami/nvme/training/data/speaker1_chatterbox/
    ├── metadata.csv
    └── audio/
         ├── xxx.wav
         └── ...

Uses ffprobe for duration checks.
Creates symlinks by default for speed and disk efficiency.

Run:
  python prepare_chatterbox_dataset_ffmpeg.py
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

# ========= CONFIG =========
SRC_DIR = Path("/opt/dlami/nvme/training/data/speaker1")
DST_DIR = Path("/opt/dlami/nvme/training/data/speaker1_chatterbox")
AUDIO_DIR = DST_DIR / "audio"

USE_SYMLINKS = True        # False = copy wav files instead
MIN_DURATION = 1.0         # seconds
MAX_DURATION = 30.0        # seconds
OVERWRITE_OUTPUT = True
# =========================


def normalize_arabic_text(text: str) -> str:
    text = text.strip()
    text = text.replace("ـ", "")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_duration_ffprobe(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    if not value:
        raise ValueError("ffprobe returned empty duration")
    return float(value)


def safe_unlink(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        pass


def prepare_dataset() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    if OVERWRITE_OUTPUT and DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []

    wav_files = sorted(SRC_DIR.glob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in {SRC_DIR}")

    for wav_path in wav_files:
        lab_path = wav_path.with_suffix(".lab")

        if not lab_path.exists():
            skipped.append((wav_path.name, "missing .lab"))
            continue

        try:
            text = lab_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            try:
                text = lab_path.read_text(encoding="utf-8-sig").strip()
            except Exception as e:
                skipped.append((wav_path.name, f"lab read error: {e}"))
                continue
        except Exception as e:
            skipped.append((wav_path.name, f"lab read error: {e}"))
            continue

        text = normalize_arabic_text(text)
        if not text:
            skipped.append((wav_path.name, "empty transcription"))
            continue

        try:
            duration = get_duration_ffprobe(wav_path)
        except Exception as e:
            skipped.append((wav_path.name, f"ffprobe error: {e}"))
            continue

        if duration < MIN_DURATION:
            skipped.append((wav_path.name, f"too short ({duration:.2f}s)"))
            continue

        if duration > MAX_DURATION:
            skipped.append((wav_path.name, f"too long ({duration:.2f}s)"))
            continue

        out_wav = AUDIO_DIR / wav_path.name
        safe_unlink(out_wav)

        if USE_SYMLINKS:
            os.symlink(wav_path, out_wav)
        else:
            shutil.copy2(wav_path, out_wav)

        rows.append(
            {
                "file_name": f"audio/{wav_path.name}",
                "transcription": text,
                "duration_seconds": f"{duration:.2f}",
            }
        )

    metadata_path = DST_DIR / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "transcription", "duration_seconds"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {metadata_path}")
    print(f"Usable samples: {len(rows)}")
    print(f"Skipped: {len(skipped)}")

    if rows:
        print("\nFirst 5 rows:")
        for row in rows[:5]:
            print(row)

    if skipped:
        print("\nFirst 20 skipped:")
        for name, reason in skipped[:20]:
            print(f"- {name}: {reason}")

    print("\nSet this in lora.py:")
    print(f'AUDIO_DATA_DIR = "{DST_DIR}"')


if __name__ == "__main__":
    prepare_dataset()