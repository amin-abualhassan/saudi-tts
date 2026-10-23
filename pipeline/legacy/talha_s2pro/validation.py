"""Dependency-light corpus and protobuf validation; never loads a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

import pipeline


def varint(data, offset=0):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
    raise ValueError("Protobuf varint too long")


def fields(data):
    offset = 0
    result = []
    while offset < len(data):
        tag, offset = varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("Invalid protobuf field zero")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            size, offset = varint(data, offset)
            if offset + size > len(data):
                raise ValueError("Truncated protobuf field")
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f"Unexpected wire type {wire} in text-data schema")
        result.append((number, wire, value))
    return result


def code_values(data):
    values = []
    for number, wire, value in fields(data):
        if number != 1:
            raise ValueError("Unexpected semantic field")
        if wire == 0:
            values.append(value)
        else:
            offset = 0
            while offset < len(value):
                code, offset = varint(value, offset)
                values.append(code)
    return values


def fingerprint(texts, codes):
    return tuple(texts), hashlib.sha256(np.asarray(codes, dtype="<u4").tobytes()).hexdigest()


def validate_prepared(rows):
    import soundfile as sf
    if not rows:
        raise ValueError("Empty prepared manifest")
    pipeline.assert_disjoint(rows)
    totals = Counter()
    seen_audio = set()
    for row in rows:
        path = Path(row["prepared_audio"])
        if str(path.resolve()) in seen_audio:
            raise ValueError("Repeated prepared audio path")
        seen_audio.add(str(path.resolve()))
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels != 1:
            raise ValueError("Prepared audio must be mono 44.1 kHz")
        if abs(info.duration - row["duration"]) > 1.1/44100:
            raise ValueError("Prepared audio duration changed")
        label = Path(row["prepared_label"])
        if label != path.with_suffix(".lab"):
            raise ValueError("Prepared label basename does not match audio")
        if " ".join(label.read_text(encoding="utf-8").split()) != row["text"]:
            raise ValueError("Prepared label changed")
        totals[row["split"]] += 1
    return dict(totals)


def sentence(data):
    texts, codes = [], []
    for number, wire, value in fields(data):
        if number == 1 and wire == 2:
            texts.append(value.decode("utf-8"))
        elif number == 3 and wire == 2:
            codes.append(code_values(value))
        else:
            raise ValueError("Unexpected sentence field")
    if len(texts) != 1 or not texts[0].strip():
        raise ValueError("Expected one nonempty transcript per sentence")
    if len(codes) != 10 or not codes[0] or len({len(row) for row in codes}) != 1:
        raise ValueError("Packed codes must have shape (10, T)")
    if min(min(row) for row in codes) < 0 or max(codes[0]) >= 4096:
        raise ValueError("Invalid packed semantic IDs")
    if max(max(row) for row in codes[1:]) >= 1024:
        raise ValueError("Invalid packed residual IDs")
    return fingerprint(texts, codes)


def packed_records(path):
    with path.open("rb") as stream:
        while True:
            header = stream.read(4)
            if not header:
                return
            if len(header) != 4:
                raise ValueError("Truncated protobuf stream header")
            size = struct.unpack("<I", header)[0]
            if not 0 < size <= 256 * 1024 * 1024:
                raise ValueError("Invalid protobuf record length")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("Truncated protobuf stream record")
            speaker = None
            sentences = []
            for number, wire, value in fields(data):
                if number in (1, 2) and wire == 2:
                    if number == 2:
                        speaker = value.decode("utf-8")
                elif number == 4 and wire == 2:
                    sentences.append(sentence(value))
                else:
                    raise ValueError("Unexpected TextData field")
            if not speaker or not sentences:
                raise ValueError("Empty speaker group")
            yield speaker, sentences


def validate_packed(dataset):
    rows = pipeline.read_manifest(dataset / "prepared.jsonl")
    if not rows:
        raise ValueError("Empty prepared dataset")
    validate_prepared(rows)
    summaries = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise ValueError(f"Empty {split} split")
        expected = Counter()
        for row in selected:
            path = Path(row["prepared_audio"])
            label = path.with_suffix(".lab")
            raw_text = label.read_text(encoding="utf-8")
            text = " ".join(raw_text.split())
            if text != row["text"]:
                raise ValueError("Prepared transcript differs from planned transcript")
            token_path = path.with_suffix(".npy")
            pipeline.validate_tokens(token_path, row["duration"])
            # Match the upstream packer's documented cleaning, without changing source labels.
            clean = re.sub(r"\{.*?\}", " ", raw_text)
            clean = re.sub(r"<.*?>", " ", clean)
            clean = re.sub(r"\s+", " ", clean)
            expected[fingerprint([clean], np.load(token_path, allow_pickle=False))] += 1
        actual = Counter()
        files = sorted((dataset / "protos" / split).glob("*.protos"))
        if not files:
            raise ValueError(f"No packed shards for {split}")
        for file in files:
            for speaker, records in packed_records(file):
                if speaker != "talha_ahmad":
                    raise ValueError(f"Unexpected packed speaker: {speaker}")
                actual.update(records)
        if expected != actual:
            raise ValueError(f"{split}: missing/changed={sum((expected-actual).values())}, "
                             f"extra/repeated={sum((actual-expected).values())}")
        summaries[split] = {"verified_sentences": sum(actual.values()), "shards": len(files)}
    return summaries


def audit(dataset=None):
    checks = pipeline.readiness()
    plan_path = pipeline.HERE / "plan/manifest.jsonl"
    plan = pipeline.read_manifest(plan_path) if plan_path.exists() else []
    report = {
        "upstream_pin": pipeline.PIN,
        "selected_clips": len(plan),
        "selected_duration_range": [min(r['duration'] for r in plan), max(r['duration'] for r in plan)] if plan else None,
        "stages": {
            "existing_ctc_alignment": "documented in dataset README; original aligner export not found",
            "source_filter_dedup_and_video_splits": "validated metadata proposal",
            "audio_transcript_pairing": "all proposed clips checked against SRT text/duration/order",
            "word_timing": "VTT caption timing exists; CTC word export unverified; not required by training format",
            "preparation": "six real clips validated; full subset not prepared",
            "codec_extraction_roundtrip": "implemented by pinned upstream; NOT RUN",
            "token_validation": "implemented; malformed tensor tests pass; actual new tokens NOT GENERATED",
            "protobuf_packing": "implemented by pinned upstream; new dataset NOT PACKED",
            "protobuf_validation": "exact transcript/token/count validator implemented",
            "training": "configuration composed; disabled; GPU/model batch NOT TESTED",
            "adapter_merge": "experiment-config helper implemented; NOT RUN",
            "held_out_inference": "matched base/adapter command generator implemented; NOT RUN",
        },
        "environment": checks,
        "ready_for_training": False,
        "blockers": [name for name, passed in checks.items() if not passed],
        "quality_review": "speaker identity, first/last word boundaries and transcript correctness pending listening review",
    }
    if dataset:
        try:
            report["packed_validation"] = validate_packed(dataset)
        except (ValueError, OSError) as error:
            report["packed_validation_error"] = str(error)
    return report
