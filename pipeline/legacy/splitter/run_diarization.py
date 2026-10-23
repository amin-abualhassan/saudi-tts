import os
import json
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

# ─── CONFIG ───────────────────────────────────────────────────────────────────
AUDIO_FILE   = "podcast_16k.wav"      # your converted audio
CONFIG_FILE  = "diar_config.yaml"
OUTPUT_DIR   = "./diar_output"
MIN_CHUNK_SEC = 1.0   # drop speaker segments shorter than this (cleanup)
# ──────────────────────────────────────────────────────────────────────────────


def build_manifest(audio_path: str, out_dir: str) -> str:
    """NeMo needs a JSON manifest pointing to the audio file."""
    audio_path = str(Path(audio_path).resolve())
    info = sf.info(audio_path)
    duration = info.duration

    manifest_path = os.path.join(out_dir, "input_manifest.json")
    os.makedirs(out_dir, exist_ok=True)

    entry = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "num_speakers": 2,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w") as f:
        json.dump(entry, f)
        f.write("\n")

    print(f"[manifest] duration={duration:.1f}s → {manifest_path}")
    return manifest_path


def run_diarization(audio_file: str, config_file: str, output_dir: str):
    print(f"\n{'='*60}")
    print(f"  NeMo MSDD Meeting Diarization")
    print(f"  Audio : {audio_file}")
    print(f"  Device: {'CUDA – ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    cfg = OmegaConf.load(config_file)
    cfg.diarizer.out_dir = output_dir
    cfg.diarizer.manifest_filepath = build_manifest(audio_file, output_dir)

    diarizer = NeuralDiarizer(cfg=cfg)
    diarizer.diarize()

    # NeMo writes RTTM to: <out_dir>/pred_rttms/<audio_stem>.rttm
    audio_stem = Path(audio_file).stem
    rttm_path = os.path.join(output_dir, "pred_rttms", f"{audio_stem}.rttm")
    print(f"\n[done] RTTM written to: {rttm_path}")
    return rttm_path


def parse_rttm(rttm_path: str):
    """Parse RTTM into a list of (start, end, speaker) tuples."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            segments.append((start, start + duration, speaker))
    segments.sort(key=lambda x: x[0])
    return segments


def drop_overlapping(segments):
    """
    Remove any segment that temporally overlaps with another segment.
    Since we set ignore_overlap=True in NeMo, overlaps should be rare,
    but this is an extra safety pass.
    """
    clean = []
    for i, (s1, e1, spk1) in enumerate(segments):
        overlap = False
        for j, (s2, e2, spk2) in enumerate(segments):
            if i == j:
                continue
            # overlap: intervals share any time
            if s1 < e2 and s2 < e1:
                overlap = True
                break
        if not overlap:
            clean.append((s1, e1, spk1))
    dropped = len(segments) - len(clean)
    print(f"[overlap] dropped {dropped} overlapping segments, kept {len(clean)}")
    return clean


def split_audio_by_speaker(audio_file: str, segments, output_dir: str,
                            min_chunk_sec: float = 1.0):
    """
    Write one WAV per clean segment, organised by speaker:
        diar_output/
            speaker_0/  chunk_0000_0.00-5.32.wav
            speaker_1/  chunk_0000_0.00-4.10.wav
    Also saves a manifest JSON for downstream use.
    """
    audio, sr = sf.read(audio_file)

    speakers = sorted(set(spk for _, _, spk in segments))
    speaker_dirs = {}
    for spk in speakers:
        d = os.path.join(output_dir, spk)
        os.makedirs(d, exist_ok=True)
        speaker_dirs[spk] = d

    counters   = {spk: 0 for spk in speakers}
    manifest   = []
    total_kept = 0

    for start, end, spk in segments:
        dur = end - start
        if dur < min_chunk_sec:
            continue

        s_idx = int(start * sr)
        e_idx = int(end   * sr)
        chunk = audio[s_idx:e_idx]

        fname = f"chunk_{counters[spk]:04d}_{start:.2f}-{end:.2f}.wav"
        fpath = os.path.join(speaker_dirs[spk], fname)
        sf.write(fpath, chunk, sr)

        manifest.append({
            "audio_filepath": fpath,
            "speaker": spk,
            "start": start,
            "end": end,
            "duration": dur,
        })
        counters[spk] += 1
        total_kept += 1

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[split] saved {total_kept} chunks")
    for spk in speakers:
        print(f"  {spk}: {counters[spk]} chunks  →  {speaker_dirs[spk]}/")
    print(f"  manifest: {manifest_path}")
    return manifest


def print_summary(segments):
    from collections import defaultdict
    totals = defaultdict(float)
    counts = defaultdict(int)
    for s, e, spk in segments:
        totals[spk] += e - s
        counts[spk] += 1

    print("\n── Speaker summary ──────────────────────────────")
    for spk in sorted(totals):
        m, sec = divmod(totals[spk], 60)
        print(f"  {spk}: {counts[spk]:4d} segments  |  {int(m):02d}:{sec:05.2f} total speech")
    print("─────────────────────────────────────────────────\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Run MSDD diarization
    rttm_path = run_diarization(AUDIO_FILE, CONFIG_FILE, OUTPUT_DIR)

    # 2. Parse result
    segments = parse_rttm(rttm_path)
    print(f"\n[rttm] {len(segments)} raw segments found")

    # 3. Drop overlapping segments
    clean_segments = drop_overlapping(segments)

    # 4. Summary
    print_summary(clean_segments)

    # 5. Split audio
    split_audio_by_speaker(AUDIO_FILE, clean_segments, OUTPUT_DIR,
                           min_chunk_sec=MIN_CHUNK_SEC)

    print("\n✓ All done!")