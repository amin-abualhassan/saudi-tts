#!/usr/bin/env python3
"""Plan and preview Talha preparation. Never starts codec extraction or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / "training/fish-speech-upstream"
BASE = ROOT / "training/checkpoints/s2-pro"
PIN = "214da3cd841bda85da2496b96cd3c4d7edb1337e"


def read_reference(path):
    section = ""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "old_name - new_key" or line.startswith("talha-screenshot-"):
            section = line
        elif line and set(line) != {"="}:
            title, sep, key = line.rpartition(" - ")
            if not sep or not re.fullmatch(r"[a-z0-9]{10}", key):
                raise ValueError(f"Invalid reference row: {line}")
            if key in rows:
                raise ValueError(f"Repeated source ID: {key}")
            rows[key] = {"section": section, "title": title}
    return rows


def flac_info(path):
    with path.open("rb") as stream:
        head = stream.read(42)
    if len(head) != 42 or head[:4] != b"fLaC" or head[4] & 127 != 0:
        raise ValueError(f"FLAC STREAMINFO missing: {path}")
    bits = int.from_bytes(head[18:26], "big")
    rate = bits >> 44
    samples = bits & ((1 << 36) - 1)
    if rate == 0 or samples == 0:
        raise ValueError(f"Invalid audio length: {path}")
    return {"duration": samples / rate, "sample_rate": rate,
            "channels": ((bits >> 41) & 7) + 1}


def srt_cues(path):
    def seconds(timestamp):
        h, m, s = timestamp.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    cues = {}
    if not path.exists():
        return cues
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and lines[0].strip().isdigit() and " --> " in lines[1]:
            start, end = lines[1].split(" --> ")
            cues[int(lines[0])] = (seconds(start), seconds(end.split()[0]),
                                      " ".join(" ".join(lines[2:]).split()))
    return cues


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inventory(root=ROOT):
    refs = read_reference(root / "refrence.txt")
    by_title = defaultdict(list)
    for key, row in refs.items():
        if row["section"].startswith("talha-screenshot-"):
            by_title[row["title"]].append(key)
    # Prefer 2025 IDs where the same original video appears in both playlists.
    canonical = {}
    for title, keys in by_title.items():
        representative = min(keys, key=lambda key: (refs[key]["section"] !=
                                                   "talha-screenshot-2025", key))
        canonical[representative] = digest(title)
    cues = {key: srt_cues(root / "full-audios-transcript/transcript" /
                         f"{key}.ar-new.srt") for key in canonical}
    cue_lookup = {}
    for key, source_cues in cues.items():
        lookup = defaultdict(list)
        for number, cue in source_cues.items():
            lookup[cue[2]].append((number, cue))
        cue_lookup[key] = lookup
    previous_cue = defaultdict(int)
    rows = []
    rejected = Counter()
    paths = (root / "splitted-audio-transcript/Jordan-youtube-48khz").glob("*.flac")
    def order(path):
        key, _, index = path.stem.partition("-")
        return key, int(index) if index.isdigit() else -1
    for path in sorted(paths, key=order):
        key, sep, index = path.stem.partition("-")
        if key not in refs:
            rejected["unmapped_source"] += 1
            continue
        if not refs[key]["section"].startswith("talha-screenshot-"):
            rejected["other_channel"] += 1
            continue
        if key not in canonical:
            rejected["duplicate_source_title"] += 1
            continue
        label = root / "splitted-audio-transcript/transcript" / f"{path.stem}.txt"
        if not label.exists():
            rejected["missing_label"] += 1
            continue
        text = " ".join(label.read_text(encoding="utf-8").split())
        if not text or re.search(r"<\|.*?\|>|[<>]", text):
            rejected["empty_or_control_text"] += 1
            continue
        info = flac_info(path)
        if info["sample_rate"] != 48000 or info["channels"] != 1:
            rejected["unexpected_audio_format"] += 1
            continue
        if not 5 <= info["duration"] <= 30:
            rejected["duration_outside_5_30_seconds"] += 1
            continue
        if len(text.split()) / info["duration"] > 6:
            rejected["implausible_text_rate"] += 1
            continue
        matches = [(number, cue) for number, cue in cue_lookup[key].get(text, [])
                   if number > previous_cue[key] and
                   abs(cue[1] - cue[0] - info["duration"]) <= 0.05]
        if not matches:
            rejected["missing_alignment_provenance"] += 1
            continue
        cue_number, cue = min(matches)
        previous_cue[key] = cue_number
        start, end, cue_text = cue
        rows.append({"clip_id": path.stem, "source_id": key,
                     "source_title": refs[key]["title"],
                     "source_group": canonical[key], "audio": str(path.resolve()),
                     "label": str(label.resolve()), "text": text,
                     "start": start, "end": end, **info,
                     "srt_cue_number": cue_number,
                     "review_status": "pending_speaker_and_transcript_review"})
    return rows, dict(rejected)


def select(rows, hours=10, seed=42):
    if hours <= 0 or not math.isfinite(hours):
        raise ValueError("Hours must be positive and finite")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Need at least three independent source videos for splits")
    random.Random(seed).shuffle(keys)
    n_eval = max(1, round(len(keys) * 0.05))
    partitions = {"test": keys[:n_eval], "validation": keys[n_eval:2*n_eval],
                  "train": keys[2*n_eval:]}
    selected = []
    summary = {}
    for split, source_keys in partitions.items():
        target = hours * 3600 * (0.9 if split == "train" else 0.05)
        queues = []
        for key in source_keys:
            queue = sorted(groups[key], key=lambda row: digest(f"{seed}:{row['clip_id']}"))
            queues.append(queue)
        duration = 0
        chosen = []
        while duration < target and any(queues):
            for queue in queues:
                if queue and duration < target:
                    row = dict(queue.pop())
                    row["split"] = split
                    chosen.append(row)
                    duration += row["duration"]
        if duration < target:
            raise ValueError(f"Not enough eligible audio for {split}: {duration/3600:.2f}h")
        selected.extend(chosen)
        summary[split] = {"clips": len(chosen), "hours": duration / 3600,
                          "source_videos": len({r['source_group'] for r in chosen})}
    return selected, summary


def assert_disjoint(rows):
    groups = defaultdict(set)
    clips = set()
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        groups[row["source_group"]].add(row["split"])
        if row["clip_id"] in clips:
            raise ValueError("Duplicate clip ID")
        clips.add(row["clip_id"])
        if "duration" in row and not 5 <= row["duration"] <= 30:
            raise ValueError("Clip duration must be between 5 and 30 seconds")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Original video leaks across splits")


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert_disjoint(rows)
    return rows


def normalize(audio, rate, target_lufs=-23, peak_db=-1):
    import numpy as np
    import pyloudnorm as pyln
    from scipy.signal import resample_poly
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be finite, nonempty and mono")
    meter = pyln.Meter(rate)
    input_lufs = meter.integrated_loudness(audio)
    if not np.isfinite(input_lufs):
        raise ValueError("Audio is silent or has no measurable loudness")
    gain = 10 ** ((target_lufs - input_lufs) / 20)
    divisor = math.gcd(rate, 44100)
    result = resample_poly(audio * gain, 44100 // divisor, rate // divisor)
    # Oversample to approximate true peak; resampling can otherwise overshoot.
    peak = float(np.max(np.abs(resample_poly(result, 4, 1))))
    limit = 10 ** (peak_db / 20)
    if peak > limit:
        result *= limit / peak
    output_lufs = pyln.Meter(44100).integrated_loudness(result)
    return result, {"input_lufs": float(input_lufs), "output_lufs": float(output_lufs),
                    "output_sample_rate": 44100, "target_lufs": target_lufs,
                    "true_peak_limit_db": peak_db,
                    "peak_limited": peak > limit}


def smoke(rows, output, limit=6, all_clips=False):
    import numpy as np
    import soundfile as sf
    if not all_clips and not 1 <= limit <= 12:
        raise ValueError("Smoke preview is limited to 1–12 clips")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    picks = list(rows) if all_clips else []
    if not all_clips:
        for split in ("train", "validation", "test"):
            choices = [r for r in rows if r["split"] == split]
            picks.extend(choices[:max(1, limit // 3)])
        picks = picks[:limit]
    if not picks:
        raise ValueError("No clips selected for preparation")
    output.mkdir(parents=True)
    reports = []
    prepared = []
    audio_hashes = {}
    for row in picks:
        if " ".join(Path(row["label"]).read_text(encoding="utf-8").split()) != row["text"]:
            raise ValueError(f"Transcript changed after planning: {row['clip_id']}")
        audio, rate = sf.read(row["audio"], dtype="float64")
        if rate != row["sample_rate"] or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid decoded audio: {row['clip_id']}")
        pcm_hash = hashlib.sha256(audio.tobytes()).hexdigest()
        if pcm_hash in audio_hashes:
            if audio_hashes[pcm_hash] != row["split"]:
                raise ValueError("Identical decoded audio leaks across splits")
            reports.append({"clip_id": row["clip_id"], "qc_status": "duplicate_decoded_audio"})
            continue
        audio_hashes[pcm_hash] = row["split"]
        fraction = float(np.mean(np.abs(audio) >= 1 - 1 / 32768))
        report = {"clip_id": row["clip_id"], "split": row["split"],
                  "duration": len(audio) / rate,
                  "input_fullscale_fraction": fraction,
                  "review_status": row["review_status"],
                  "source_sha256": hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest()}
        if fraction > 0.001:
            report["qc_status"] = "rejected_possible_clipping"
            reports.append(report)
            continue
        normalized, metrics = normalize(audio, rate)
        folder = output / "audio" / row["split"] / "talha_ahmad"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{row['clip_id']}.wav"
        sf.write(dest, normalized, 44100, subtype="PCM_16")
        dest.with_suffix(".lab").write_text(row["text"] + "\n", encoding="utf-8")
        check, check_rate = sf.read(dest)
        if check_rate != 44100 or abs(len(check)/44100-row["duration"]) > 1/44100:
            raise ValueError("Resampling changed clip timing")
        if np.max(np.abs(check)) > 10 ** (-1/20) + 1/32768:
            raise ValueError("Output exceeds peak ceiling")
        report.update(metrics)
        report.update({"qc_status": "preview_only_needs_listening", "output": str(dest)})
        reports.append(report)
        prepared.append({**row, "prepared_audio": str(dest.resolve()),
                         "prepared_label": str(dest.with_suffix('.lab').resolve()),
                         "decoded_audio_sha256": pcm_hash})
    (output / "qc.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    (output / "prepared.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in prepared))
    return reports


def validate_tokens(path, duration=None):
    import numpy as np
    codes = np.load(path, allow_pickle=False)
    if codes.ndim != 2 or codes.shape[0] != 10 or codes.shape[1] == 0:
        raise ValueError(f"Expected S2 Pro tokens shaped (10, T): {path}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"Tokens must be integer IDs: {path}")
    if codes.min() < 0 or codes[0].max() >= 4096 or codes[1:].max() >= 1024:
        raise ValueError(f"Token IDs outside codec codebook range: {path}")
    if duration is not None and abs(codes.shape[1] - duration * 44100 / 2048) > 2:
        raise ValueError(f"Token length does not match audio duration: {path}")
    return codes.shape[1]


def readiness():
    import importlib.util
    checks = {}
    revision = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    checks["pinned_upstream"] = revision.returncode == 0 and revision.stdout.strip() == PIN
    for name in ("torch", "torchaudio", "lightning", "hydra", "datasets", "transformers"):
        checks[f"module_{name}"] = importlib.util.find_spec(name) is not None
    checks["base_files"] = all((BASE/name).exists() for name in
                               ("config.json", "codec.pth", "tokenizer.json", "model.safetensors.index.json"))
    index_path = BASE / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    checks["base_shards"] = bool(index) and all((BASE/name).exists() for name in set(index.get("weight_map", {}).values()))
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True)
        checks["nvidia_driver"] = smi.returncode == 0
    except FileNotFoundError:
        checks["nvidia_driver"] = False
    if checks["module_torch"]:
        import torch
        checks["torch_2_8"] = torch.__version__.split("+")[0] == "2.8.0"
        checks["cuda_available"] = torch.cuda.is_available()
        checks["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return checks


def commands(dataset):
    # Only display commands. No subprocess can start an expensive stage here.
    python = str(ROOT / "miniconda3/envs/fish-speech/bin/python")
    prefix = f"PYTHONPATH={shlex.quote(str(UPSTREAM))} {shlex.quote(python)}"
    result = []
    for split in ("train", "validation", "test"):
        folder = dataset / "audio" / split
        result.append(f"{prefix} {shlex.quote(str(UPSTREAM/'tools/vqgan/extract_vq.py'))} "
                      f"{shlex.quote(str(folder))} --num-workers 1 --batch-size 2 "
                      f"--config-name modded_dac_vq --checkpoint-path {shlex.quote(str(BASE/'codec.pth'))}")
    result.append(f"{shlex.quote(python)} {shlex.quote(str(HERE/'pipeline.py'))} validate-tokens "
                  f"--prepared-manifest {shlex.quote(str(dataset/'prepared.jsonl'))}")
    for split in ("train", "validation", "test"):
        folder = dataset / "audio" / split
        result.append(f"{prefix} {shlex.quote(str(UPSTREAM/'tools/llama/build_dataset.py'))} "
                      f"--input {shlex.quote(str(folder))} --output {shlex.quote(str(dataset/'protos'/split))} "
                      "--text-extension .lab --num-workers 1")
    result.append(f"{shlex.quote(python)} {shlex.quote(str(HERE/'pipeline.py'))} validate-packed "
                  f"--dataset {shlex.quote(str(dataset))}")
    result.append(f"{prefix} {shlex.quote(str(UPSTREAM/'fish_speech/train.py'))} "
                  f"--config-path {shlex.quote(str(HERE/'config'))} --config-name talha_s2pro "
                  f"train=false train_dataset.proto_files=[{shlex.quote(str(dataset/'protos/train'))}] "
                  f"val_dataset.proto_files=[{shlex.quote(str(dataset/'protos/validation'))}]")
    result.extend([
        "# After a later training run: merge with experiment settings, NOT upstream alpha-16 defaults.",
        f"{shlex.quote(python)} {shlex.quote(str(HERE/'merge_model.py'))} "
        "--experiment-config <SAVED_RUN_CONFIG.yaml> --checkpoint <CHOSEN_ADAPTER.ckpt> --output <NEW_MERGED_DIRECTORY>",
        "# Compare base and merged inference on held-out prompts with the same clean reference.",
    ])
    return result


def evaluation_commands(merged, reference, prompt_text, text, output):
    python = ROOT / "miniconda3/envs/fish-speech/bin/python"
    prefix = f"PYTHONPATH={shlex.quote(str(UPSTREAM))} {shlex.quote(str(python))}"
    result = [f"mkdir -p {shlex.quote(str(output))}"]
    for label, model in (("base", BASE), ("adapter", merged)):
        result.append(f"{prefix} {shlex.quote(str(UPSTREAM/'fish_speech/models/text2semantic/inference.py'))} "
                      f"--checkpoint-path {shlex.quote(str(model))} "
                      f"--prompt-audio {shlex.quote(str(reference))} "
                      f"--prompt-text {shlex.quote(prompt_text)} --text {shlex.quote(text)} "
                      f"--seed 42 --num-samples 1 --no-compile "
                      f"--output-dir {shlex.quote(str(output/label))} "
                      f"--output {shlex.quote(str(output/(label+'.wav')))}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Read-only metadata audit and proposed selection")
    plan.add_argument("--hours", type=float, default=10)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--output", type=Path, required=True)
    preview = sub.add_parser("smoke", help="Decode/normalize at most 12 clips; no tokens/training")
    preview.add_argument("--manifest", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--limit", type=int, default=6)
    sub.add_parser("check", help="Check environment and model availability")
    cmd = sub.add_parser("commands", help="Print later commands; never execute them")
    cmd.add_argument("--dataset", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="Normalize selected clips; defaults to a small preview")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=6)
    prepare.add_argument("--all", action="store_true", help="Explicitly process full selection")
    tokens = sub.add_parser("validate-tokens", help="Validate actual codec outputs before packing")
    tokens.add_argument("--prepared-manifest", type=Path, required=True)
    packed = sub.add_parser("validate-packed", help="Check protobuf records exactly match prepared tokens/texts")
    packed.add_argument("--dataset", type=Path, required=True)
    audit = sub.add_parser("audit", help="Report implemented and validated stages; no processing")
    audit.add_argument("--dataset", type=Path)
    evaluation = sub.add_parser("evaluation-commands", help="Print matched base/adapter inference commands; never execute")
    evaluation.add_argument("--merged", type=Path, required=True)
    evaluation.add_argument("--reference", type=Path, required=True)
    evaluation.add_argument("--prompt-text", required=True)
    evaluation.add_argument("--text", required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        if args.output.exists():
            raise ValueError("Plan output already exists; refusing to overwrite")
        rows, rejected = inventory()
        selected, summary = select(rows, args.hours, args.seed)
        assert_disjoint(selected)
        args.output.mkdir(parents=True)
        (args.output / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in selected))
        report = {"seed": args.seed, "target_total_hours": args.hours, "splits": summary,
                  "eligible_clips": len(rows), "eligible_hours": sum(r['duration'] for r in rows)/3600,
                  "excluded": rejected, "source_title_dedup_only": True,
                  "review_status": "proposal_only_not_approved_training_data"}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke(read_manifest(args.manifest), args.output, args.limit), indent=2))
    elif args.command == "check":
        checks = readiness()
        print(json.dumps({"checks": checks, "ready_for_training": all(checks.values())}, indent=2))
    elif args.command == "prepare":
        print(json.dumps(smoke(read_manifest(args.manifest), args.output, args.limit, args.all), indent=2))
    elif args.command == "validate-tokens":
        rows = read_manifest(args.prepared_manifest)
        from validation import validate_prepared
        validate_prepared(rows)
        frames = 0
        for row in rows:
            frames += validate_tokens(Path(row["prepared_audio"]).with_suffix(".npy"), row["duration"])
        print(json.dumps({"validated_files": len(rows), "frames": frames}))
    elif args.command == "validate-packed":
        from validation import validate_packed
        print(json.dumps(validate_packed(args.dataset), indent=2))
    elif args.command == "audit":
        from validation import audit
        print(json.dumps(audit(args.dataset), indent=2))
    elif args.command == "evaluation-commands":
        print("# Preview only. Use the same reviewed reference, target text and seed for both models.")
        print("\n".join(evaluation_commands(args.merged.resolve(), args.reference.resolve(),
                                           args.prompt_text, args.text, args.output.resolve())))
    else:
        print("# Preview only. Review/QC and token validation are required before executing.")
        print("\n".join(commands(args.dataset.resolve())))


if __name__ == "__main__":
    main()
