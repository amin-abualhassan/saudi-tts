#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SOURCE_DIR = Path("/opt/dlami/nvme/training/data/speaker1")
ROOT_DIR = Path("/opt/dlami/nvme/training/omnivoice/speaker1")
WAV24K_DIR = ROOT_DIR / "wav24k"
MANIFEST_DIR = ROOT_DIR / "manifests"
TRAIN_JSONL = MANIFEST_DIR / "train.jsonl"
DEV_JSONL = MANIFEST_DIR / "dev.jsonl"
OUTLIERS_TXT = MANIFEST_DIR / "outliers.txt"

TARGET_SR = 24_000
MIN_DURATION = 1.0
MAX_DURATION = 30.0
LANGUAGE_ID = "apc"
ID_PREFIX = "speaker1"

DEV_FAMILIES = {
    "2AXIS",
    "ME2B",
    "apollo gym",
    "fainal mnasef2_01",
    "madooo",
    "newtin  ensurance",
    "ابلكيشن غازك",
    "ابو شيخو (2)_01",
    "اصل العسل",
    "التنقل من غير ميوزك ومن غير الجملة",
    "تعاون4",
}


def family_from_stem(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def load_resample_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1, keepdims=True).T
    tensor = torch.from_numpy(mono)
    if sample_rate != TARGET_SR:
        tensor = torchaudio.functional.resample(
            tensor, orig_freq=sample_rate, new_freq=TARGET_SR
        )
    audio_24k = tensor.squeeze(0).numpy()
    return np.clip(audio_24k, -1.0, 1.0), sample_rate


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Missing source directory: {SOURCE_DIR}")

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    WAV24K_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    for old_wav in WAV24K_DIR.glob("*.wav"):
        old_wav.unlink()

    wav_files = sorted(SOURCE_DIR.glob("*.wav"), key=lambda p: p.name)
    lab_files = sorted(SOURCE_DIR.glob("*.lab"), key=lambda p: p.name)

    if not wav_files:
        raise RuntimeError(f"No .wav files found under {SOURCE_DIR}")

    wav_stems = {p.stem for p in wav_files}
    lab_stems = {p.stem for p in lab_files}
    missing_labs = sorted(wav_stems - lab_stems)
    missing_wavs = sorted(lab_stems - wav_stems)

    if missing_labs or missing_wavs:
        problems = []
        if missing_labs:
            problems.append(f"missing labs for: {missing_labs[:20]}")
        if missing_wavs:
            problems.append(f"missing wavs for: {missing_wavs[:20]}")
        raise RuntimeError("; ".join(problems))

    train_records: list[dict] = []
    dev_records: list[dict] = []
    outlier_rows: list[list[str]] = []

    for index, wav_path in enumerate(wav_files):
        sample_id = f"{ID_PREFIX}_{index:04d}"
        lab_path = SOURCE_DIR / f"{wav_path.stem}.lab"
        text = lab_path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Empty transcript for {lab_path}")

        family = family_from_stem(wav_path.stem)
        split = "dev" if family in DEV_FAMILIES else "train"

        audio_24k, original_sr = load_resample_mono(wav_path)
        duration_sec = len(audio_24k) / TARGET_SR

        if not (MIN_DURATION <= duration_sec <= MAX_DURATION):
            outlier_rows.append(
                [
                    sample_id,
                    split,
                    str(wav_path.resolve()),
                    str(original_sr),
                    f"{duration_sec:.6f}",
                    "duration_out_of_range",
                ]
            )
            out_path = WAV24K_DIR / f"{wav_path.stem}.wav"
            if out_path.exists():
                out_path.unlink()
            continue

        out_path = WAV24K_DIR / f"{wav_path.stem}.wav"
        sf.write(str(out_path), audio_24k, TARGET_SR, subtype="PCM_16")

        record = {
            "id": sample_id,
            "audio_path": str(out_path.resolve()),
            "text": text,
            "language_id": LANGUAGE_ID,
        }

        if split == "train":
            train_records.append(record)
        else:
            dev_records.append(record)

    write_jsonl(TRAIN_JSONL, train_records)
    write_jsonl(DEV_JSONL, dev_records)

    with OUTLIERS_TXT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["id", "split", "orig_path", "orig_sr", "resampled_sec", "reason"]
        )
        writer.writerows(outlier_rows)

    print(f"Prepared train records: {len(train_records)}")
    print(f"Prepared dev records: {len(dev_records)}")
    print(f"Logged outliers: {len(outlier_rows)}")
    print(f"Train manifest: {TRAIN_JSONL}")
    print(f"Dev manifest: {DEV_JSONL}")
    print(f"Outlier log: {OUTLIERS_TXT}")


if __name__ == "__main__":
    main()
