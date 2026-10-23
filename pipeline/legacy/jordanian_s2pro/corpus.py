"""Bounded selection, recording-disjoint splits and content-verified preparation."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from common import ROOT, HERE, atomic_json, file_hash, identity, read_json, read_rows, shared, talha_pipeline, talha_validation, write_rows

SPLITS = ("train", "validation", "test")


def valid_row(row):
    duration = row["duration"]
    text = " ".join(row["text"].split())
    return (math.isfinite(duration) and 5 <= duration <= 30 and bool(text)
            and not re.search(r"[<>{}]", text) and len(text.split()) / duration <= 6)


def inventory(review=None, accepted_only=False):
    import soundfile as sf
    decisions = {}
    if review:
        payload = json.loads(Path(review).read_text(encoding="utf-8"))
        decisions = {r["clip_id"]: r["decision"] for r in payload["decisions"]}
        if any(v not in {"accept", "reject", "uncertain", "pending"} for v in decisions.values()):
            raise ValueError("Invalid review decision")
    candidates, rejected = talha_pipeline.inventory()
    rows = [{**r, "clip_id": "talha_" + r["clip_id"], "speaker": "talha_ahmad",
             "family": "talha", "origin": "talha", "original_clip_id": r["clip_id"],
             "source_group": "talha_" + r["source_group"]} for r in candidates]
    for r in read_rows(HERE.parent / "wesam_dataset_words/manifest.jsonl"):
        decision = decisions.get(r["clip_id"], "pending")
        if decision in {"reject", "uncertain"} or (accepted_only and decision != "accept"):
            continue
        rows.append({**r, "clip_id": "podcast_" + r["clip_id"], "original_clip_id": r["clip_id"],
                     "family": "podcast", "origin": "podcast", "label": str(Path(r["audio"]).with_suffix(".lab")),
                     "source_group": "podcast_" + r["source_group"], "review_decision": decision})
    for path in sorted((HERE.parent / "data/speaker1").glob("*.wav")):
        label = path.with_suffix(".lab")
        if not label.is_file():
            raise ValueError(f"Missing single-speaker transcript: {label}")
        info = sf.info(path)
        # The splitter's numeric suffix identifies consecutive clips of a recording.
        recording = re.sub(r"_\d+$", "", path.stem)
        rows.append({"clip_id": "single_" + identity(path.stem)[:24], "original_clip_id": path.stem,
                     "audio": str(path.resolve()), "label": str(label.resolve()),
                     "text": " ".join(label.read_text(encoding="utf-8").split()),
                     "duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels,
                     "speaker": "single_speaker", "family": "single", "origin": "single",
                     "source_id": recording, "source_group": "single_" + identity(recording),
                     "review_status": "existing_paired_audio_identity_not_independently_verified"})
    eligible = [r for r in rows if valid_row(r)]
    rejected["mixed_duration_text_filter"] = len(rows) - len(eligible)
    if any(not re.fullmatch(r"[a-zA-Z0-9_]+", r["speaker"]) for r in eligible):
        raise ValueError("Unsafe speaker folder name")
    for r in eligible:
        r["text"] = " ".join(r["text"].split())
    return eligible, rejected


def select(rows, budgets, seed):
    """Assign each family by recording, then round-robin speakers and recordings.

    Podcast's existing recording splits are retained. Budgets are hard upper
    bounds across all three splits, not requested minimums requiring oversampling.
    """
    chosen = []
    report = {}
    for family in ("talha", "podcast", "single"):
        pool = [dict(r) for r in rows if r["family"] == family]
        groups = sorted({r["source_group"] for r in pool})
        if len(groups) < 3:
            raise ValueError(f"{family}: need at least three recording groups")
        if family != "podcast":
            random.Random(seed).shuffle(groups)
            n_eval = max(1, round(len(groups) * .10))
            assignments = {g: ("test" if i < n_eval else "validation" if i < 2*n_eval else "train")
                           for i, g in enumerate(groups)}
            for r in pool:
                r["split"] = assignments[r["source_group"]]
        totals = {}
        for split, fraction in (("train", .8), ("validation", .1), ("test", .1)):
            available = [r for r in pool if r["split"] == split]
            speakers = defaultdict(lambda: defaultdict(list))
            for r in sorted(available, key=lambda r: identity([seed, r["clip_id"]])):
                speakers[r["speaker"]][r["source_group"]].append(r)
            target = budgets[family] * 3600 * fraction
            bucket_totals = Counter()
            has_wesam = any(r["speaker"] == "wesam_ali" for r in available)
            total = 0
            selection = []
            queues = [list(recordings.values()) for _, recordings in sorted(speakers.items())]
            # A round grants one clip per speaker and rotates that speaker's recordings.
            while any(any(q for q in records) for records in queues):
                for records in queues:
                    records[:] = [q for q in records if q]
                    if not records:
                        continue
                    q = records.pop(0)
                    r = q.pop()
                    records.append(q)
                    bucket = "wesam" if r["speaker"] == "wesam_ali" else "other"
                    bucket_limit = target*.5 if family == "podcast" and has_wesam else target
                    if total + r["duration"] <= target and bucket_totals[bucket] + r["duration"] <= bucket_limit:
                        selection.append(r)
                        total += r["duration"]
                        bucket_totals[bucket] += r["duration"]
            if not selection:
                raise ValueError(f"{family}/{split}: empty budget or eligible pool")
            chosen.extend(selection)
            totals[split] = {"clips": len(selection), "hours": total / 3600,
                             "available_hours": sum(r["duration"] for r in available)/3600,
                             "speakers": dict(Counter(r["speaker"] for r in selection))}
        report[family] = totals
    talha_pipeline.assert_disjoint(chosen)
    return sorted(chosen, key=lambda r: r["clip_id"]), report


def plan(config, review=None, accepted_only=False):
    rows, rejected = inventory(review, accepted_only)
    selected, summary = select(rows, config["hours"], config["seed"])
    # Include actual labels and audio metadata; per-clip content receipts are stronger.
    signature = []
    for r in selected:
        stat = Path(r["audio"]).stat()
        if " ".join(Path(r["label"]).read_text(encoding="utf-8").split()) != r["text"]:
            raise ValueError(f"Transcript mismatch: {r['clip_id']}")
        signature.append({**r, "source_size": stat.st_size, "source_mtime": stat.st_mtime_ns,
                          "label_sha256": file_hash(r["label"])})
    recipe = {"revision": 1, "selected": signature, "seed": config["seed"],
              "budgets": config["hours"], "implementation": file_hash(Path(__file__)),
              "normalization": file_hash(talha_pipeline.HERE / "pipeline.py"),
              "review": file_hash(review) if review else None, "accepted_only": accepted_only}
    dataset = HERE / "cache" / identity(recipe)[:24]
    dataset.mkdir(parents=True, exist_ok=True)
    write_rows(dataset / "manifest.jsonl", selected)
    atomic_json(dataset / "selection.json", {"recipe_sha256": identity(recipe), "summary": summary,
                                           "rejected": rejected, "review_applied": bool(review),
                                           "quality": "Automatic candidates unless explicitly reviewed; listening remains necessary"})
    return dataset, selected, summary


def prepare(rows, dataset, monitor, pilot=False):
    import numpy as np
    import soundfile as sf
    talha_pipeline.assert_disjoint(rows)
    monitor.update(stage="audio preparation", completed=0, total=len(rows), rejected=0)
    prepared, qc, seen = [], [], {}
    for i, row in enumerate(rows):
        path = dataset / "audio" / row["split"] / row["speaker"] / (row["clip_id"] + ".wav")
        receipt_path = dataset / "receipts" / (row["clip_id"] + ".json")
        receipt = read_json(receipt_path)
        source_hash = file_hash(row["audio"])
        if " ".join(Path(row["label"]).read_text(encoding="utf-8").split()) != row["text"]:
            raise ValueError(f"Source label changed: {row['clip_id']}; replan")
        if row.get("audio_sha256") and row["audio_sha256"] != source_hash:
            raise ValueError(f"Source audio differs from podcast export: {row['clip_id']}")
        try:
            if not shared.receipt_valid(receipt, row, path):
                audio, rate = sf.read(row["audio"], dtype="float64")
                if audio.ndim != 1 or not np.isfinite(audio).all():
                    raise shared.QualityRejection("Nonfinite or non-mono source")
                if rate != row["sample_rate"] or abs(len(audio)/rate-row["duration"]) > 1.1/rate:
                    raise ValueError(f"Source format/duration changed: {row['clip_id']}")
                if np.mean(np.abs(audio) >= 1-1/32768) > .001:
                    raise shared.QualityRejection("Input clipping exceeds 0.1%")
                alignment = {}
                if row["origin"] == "talha":
                    alignment = shared.verify_source_interval(audio, rate, row,
                        ROOT / "full-audios-transcript/audios-48khz" / (row["source_id"] + ".flac"))
                try:
                    normalized, metrics = talha_pipeline.normalize(audio, rate)
                except ValueError as error:
                    raise shared.QualityRejection(str(error)) from error
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp.wav")
                sf.write(temp, normalized, 44100, subtype="PCM_16")
                temp.replace(path)
                path.with_suffix(".lab").write_text(row["text"] + "\n", encoding="utf-8")
                path.with_suffix(".npy").unlink(missing_ok=True)
                receipt = {"source_sha256": source_hash, "text": row["text"],
                           "wav_sha256": file_hash(path), "lab_sha256": file_hash(path.with_suffix(".lab")),
                           "decoded_audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                           **alignment, **metrics}
                atomic_json(receipt_path, receipt)
            pcm_hash = receipt["decoded_audio_sha256"]
            if pcm_hash in seen:
                if seen[pcm_hash]["split"] != row["split"]:
                    raise ValueError(f"Duplicate audio across splits: {row['clip_id']}")
                raise shared.QualityRejection("Duplicate audio within split")
            output, out_rate = sf.read(path)
            if out_rate != 44100 or not np.isfinite(output).all() or np.max(np.abs(output)) > 10**(-1/20)+1/32768:
                raise ValueError(f"Invalid prepared waveform: {path}")
            seen[pcm_hash] = row
            prepared.append({**row, "prepared_audio": str(path), "prepared_label": str(path.with_suffix(".lab"))})
        except shared.QualityRejection as error:
            qc.append({"clip_id": row["clip_id"], "reason": str(error)})
            for suffix in (".wav", ".lab", ".npy"):
                path.with_suffix(suffix).unlink(missing_ok=True)
            receipt_path.unlink(missing_ok=True)
            monitor.log(f"Rejected {row['clip_id']}: {error}")
        monitor.update(completed=i+1, rejected=len(qc), accepted=len(prepared))
    atomic_json(dataset / "qc.json", qc)
    for family in (() if pilot else ("talha", "podcast", "single")):
        for split in SPLITS:
            original = [r for r in rows if r["family"] == family and r["split"] == split]
            accepted = [r for r in prepared if r["family"] == family and r["split"] == split]
            if not accepted or sum(r["duration"] for r in accepted) < .8*sum(r["duration"] for r in original):
                raise ValueError(f"Too much rejected audio in {family}/{split}; inspect {dataset/'qc.json'}")
    talha_validation.validate_prepared(prepared)
    # Remove stale derived pairs so the recursive codec/packer sees exactly this manifest.
    keep = {Path(r["prepared_audio"]) for r in prepared}
    for wav in (dataset / "audio").rglob("*.wav"):
        if wav not in keep:
            for ext in (".wav", ".lab", ".npy"):
                wav.with_suffix(ext).unlink(missing_ok=True)
    write_rows(dataset / "prepared.jsonl", prepared)
    return prepared


def validate_packed(dataset):
    import numpy as np
    rows = read_rows(dataset / "prepared.jsonl")
    talha_validation.validate_prepared(rows)
    summaries = {}
    for split in SPLITS:
        expected, actual = Counter(), Counter()
        for r in (r for r in rows if r["split"] == split):
            path = Path(r["prepared_audio"])
            talha_pipeline.validate_tokens(path.with_suffix(".npy"), r["duration"])
            text = re.sub(r"\s+", " ", path.with_suffix(".lab").read_text(encoding="utf-8"))
            expected[(r["speaker"], talha_validation.fingerprint([text], np.load(path.with_suffix(".npy"), allow_pickle=False)))] += 1
        for path in sorted((dataset / "protos" / split).glob("*.protos")):
            for speaker, records in talha_validation.packed_records(path):
                actual.update((speaker, record) for record in records)
        if not expected or actual != expected:
            raise ValueError(f"Packed speaker/text/token/count mismatch: {split}")
        summaries[split] = {"verified_sentences": sum(actual.values()), "speakers": len({k[0] for k in actual})}
    return summaries
