#!/usr/bin/env python3
"""One-command Talha S2 Pro preparation, training and read-only monitoring.

Default invocation performs real work. --check only checks prerequisites;
--preview prepares six real clips without importing Torch or training.
"""
import argparse
from collections import Counter, deque
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline

ROOT, UPSTREAM, BASE = pipeline.ROOT, pipeline.UPSTREAM, pipeline.BASE
VERSION = 1
# Bump when prepared waveforms or their acceptance rules become incompatible.
PREPARATION_REVISION = 2
LEGACY_RUNNER_SHA256 = "9e06febcfe6c2aea4d3a12f077c39d185cebc3674f1104f5f0e3210bb268e2e7"
SPLITS = ("train", "validation", "test")
EVENT = "TALHA_EVENT "
TRAINING_REQUIREMENTS = """torch==2.8.0
torchaudio==2.8.0
lightning==2.6.1
transformers==4.57.3
datasets==2.18.0
pyarrow<21
numpy<2
hydra-core>=1.3.2,<1.4
tensorboard==2.20.0
natsort>=8.4.0
einops>=0.7.0
einx[torch]==0.2.2
librosa>=0.10.1
rich>=13.5.3
loguru>=0.6.0
loralib>=0.1.2
pyrootutils>=1.0.4
resampy>=0.4.3
zstandard>=0.22.0
ormsgpack
tiktoken>=0.8.0
safetensors
soundfile>=0.13
scipy
pyloudnorm
click
descript-audio-codec
protobuf>=3.20,<6
"""


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


class Monitor:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state = {"status": "starting", "stage": "startup", "started": time.time(),
                      "updated": time.time(), "step": 0, "history": [], "logs": [],
                      "message": "Initializing", "dashboard_port": 7860}
        self.logfile = (directory / "launcher.log").open("a", encoding="utf-8", buffering=1)

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["updated"] = time.time()
            atomic_json(self.directory / "status.json", self.state)

    def log(self, message):
        message = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(message)).strip()
        if not message:
            return
        line = time.strftime("%H:%M:%S ", time.gmtime()) + message
        with self.lock:
            self.state["logs"] = (self.state["logs"] + [line])[-100:]
            self.logfile.write(line + "\n")
        print(line, flush=True)

    def event(self, value):
        event_record = dict(value)
        with self.lock:
            history = self.state["history"]
            if "metrics" in value:
                point = {"step": value.get("step", self.state["step"]), **value["metrics"]}
                if history and point["step"] < history[-1]["step"]:
                    history = [p for p in history if p["step"] <= point["step"]]
                if history and history[-1]["step"] == point["step"]:
                    history[-1].update(point)
                else:
                    history.append(point)
                value["history"] = history[-3000:]
            self.update(**value)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event_record, allow_nan=False) + "\n")

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state, allow_nan=False))


def dashboard(monitor, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?")[0]
            audio_keys = {"/audio/reference.wav": "reference_audio", "/audio/codec.wav": "codec_audio"}
            if route in audio_keys:
                value = monitor.snapshot().get(audio_keys[route])
                path = Path(value).resolve() if value else None
                if not path or not path.is_relative_to((HERE/"cache").resolve()) or not path.is_file():
                    self.send_error(404)
                    return
                payload = path.read_bytes()
                content_type = "audio/wav"
            elif route == "/api/status":
                payload = json.dumps(monitor.snapshot(), allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            elif self.path == "/" or self.path.startswith("/?"):
                payload = DASHBOARD.encode()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def process(command, monitor, progress=None):
    """Stream both stdout/stderr, consume structured worker events, stop children on interrupt."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UPSTREAM) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["USE_LIBUV"] = "0"
    for name in ("SLURM_NTASKS", "SLURM_PROCID", "SLURM_JOB_NAME", "SLURM_NTASKS_PER_NODE"):
        env.pop(name, None)
    monitor.log("Running: " + " ".join(map(str, command)))
    child = subprocess.Popen(list(map(str, command)), cwd=UPSTREAM, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", start_new_session=True)
    lines = queue.Queue()

    def reader():
        try:
            for line in child.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    last_tick = 0
    ended = False
    try:
        while not ended:
            try:
                line = lines.get(timeout=1)
                if line is None:
                    ended = True
                elif line.startswith(EVENT):
                    monitor.event(json.loads(line[len(EVENT):]))
                else:
                    monitor.log(line)
            except queue.Empty:
                pass
            if progress and time.monotonic() - last_tick >= 5:
                progress()
                last_tick = time.monotonic()
        code = child.wait()
        if code:
            raise RuntimeError(f"Stage failed with exit code {code}; see {monitor.directory/'launcher.log'}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        child.stdout.close()


def source_signature(root=ROOT):
    """Detect additions, replacements and edits before selecting/reusing data."""
    digest = hashlib.sha256()
    directories = [root / "splitted-audio-transcript/Jordan-youtube-48khz",
                   root / "splitted-audio-transcript/transcript",
                   root / "full-audios-transcript/transcript",
                   root / "full-audios-transcript/audios-48khz"]
    digest.update((root / "refrence.txt").read_bytes())
    for folder in directories:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if path.suffix not in (".flac", ".txt", ".srt"):
                continue
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}\n".encode())
            if path.suffix in (".txt", ".srt"):
                digest.update(path.read_bytes())
    return digest.hexdigest()


def recipe_signature():
    return {"version": VERSION, "pipeline": file_hash(HERE / "pipeline.py"),
            "validation": file_hash(HERE / "validation.py"), "preparation_revision": PREPARATION_REVISION,
            "sample_rate": 44100, "lufs": -23, "peak_db": -1,
            "alignment": "existing_srt_plus_original_pcm_check_round_or_truncate", "upstream": pipeline.PIN}


def compatible_selection(saved, recipe):
    if saved == recipe:
        return True
    # Only migrate the inspected preceding launcher: all its accepted PCM checks
    # were stricter, and normalization, selection and training settings are unchanged.
    if recipe.get("preparation_revision") != 2:
        return False
    legacy = {k:v for k,v in recipe.items() if k != "preparation_revision"}
    legacy.update(runner=LEGACY_RUNNER_SHA256, alignment="existing_srt_plus_original_pcm_check")
    return saved == legacy


def make_plan(args, monitor):
    monitor.update(stage="inventory", status="running", message="Checking source files and transcript provenance")
    recipe = {**recipe_signature(), "source": source_signature(), "hours": args.hours, "seed": args.seed}
    key = identity(recipe)[:24]
    dataset = HERE / "cache" / key
    saved = read_json(dataset / "selection.json")
    if not saved:
        # Preserve paths referenced by existing pilot checkpoints and token receipts.
        for candidate in sorted((HERE / "cache").glob("*/selection.json")):
            previous = read_json(candidate)
            if previous and compatible_selection(previous.get("recipe"), recipe):
                dataset, saved = candidate.parent, previous
                key = dataset.name
                break
    if saved and saved.get("recipe") != recipe and compatible_selection(saved.get("recipe"), recipe):
        saved = {**saved, "legacy_recipe": saved["recipe"], "recipe": recipe}
        atomic_json(dataset / "selection.json", saved)
        monitor.log("Preserving verified preparation and GPU pilot from the compatible preceding launcher")
    if saved and saved.get("recipe") == recipe:
        monitor.log(f"Reusing selected manifest: {dataset}")
        rows = pipeline.read_manifest(dataset / "manifest.jsonl")
        summary = saved["summary"]
    else:
        candidates, rejected = pipeline.inventory()
        rows, summary = pipeline.select(candidates, args.hours, args.seed)
        pipeline.assert_disjoint(rows)
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        atomic_json(dataset / "selection.json", {"recipe": recipe, "summary": summary, "rejected": rejected})
    reserve_path = dataset / "reserves.jsonl"
    if not reserve_path.exists():
        candidates, _ = pipeline.inventory()
        reserves = make_reserves(rows, candidates, args.seed)
        temporary = reserve_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in reserves),encoding="utf-8")
        temporary.replace(reserve_path)
    reserves = pipeline.read_manifest(reserve_path)
    monitor.update(dataset=str(dataset), dataset_key=key, selected=len(rows), splits=summary)
    return dataset, rows, reserves


def make_reserves(planned, candidates, seed):
    """Reserve only clips from videos already assigned to a single split."""
    pipeline.assert_disjoint(planned)
    assigned = {r["source_group"]:r["split"] for r in planned}
    used = {r["clip_id"] for r in planned}
    reserves = [{**row,"split":assigned[row["source_group"]]} for row in candidates
                if row["source_group"] in assigned and row["clip_id"] not in used]
    reserves.sort(key=lambda r:pipeline.digest(f"{seed}:reserve:{r['clip_id']}"))
    pipeline.assert_disjoint([*planned,*reserves])
    return reserves


class QualityRejection(ValueError):
    """An unsuitable or unverified clip can be replaced within its assigned split."""


def verify_source_interval(audio, rate, row, original):
    """Accept only exact SRT slices using nearest or truncated sample boundaries."""
    import numpy as np
    import soundfile as sf
    rounded_start = round(row["start"] * rate)
    with sf.SoundFile(original) as full:
        if full.samplerate != rate or full.channels != 1:
            raise ValueError(f"Unexpected original audio format: {original}")
        for policy, convert in (("nearest", round), ("truncate", int)):
            start, end = convert(row["start"] * rate), convert(row["end"] * rate)
            if start < 0 or end > len(full) or end-start != len(audio):
                continue
            full.seek(start)
            aligned = full.read(len(audio), dtype="float64")
            difference = float(np.max(np.abs(audio-aligned))) if len(aligned) == len(audio) else float("inf")
            if difference <= 1/32768 + 1e-8:
                return {"source_rounding":policy, "source_start_frame":start, "source_end_frame":end,
                        "source_offset_samples":start-rounded_start, "source_max_pcm_difference":difference}
    raise QualityRejection(f"Clip does not match original SRT interval: {row['clip_id']}")


def receipt_valid(receipt, row, path):
    if not receipt or receipt.get("source_sha256") != file_hash(row["audio"]):
        return False
    if receipt.get("text") != row["text"]:
        return False
    try:
        return (receipt["wav_sha256"] == file_hash(path) and
                receipt["lab_sha256"] == file_hash(path.with_suffix(".lab")))
    except (FileNotFoundError, KeyError):
        return False


def prepare(rows, dataset, monitor, root=ROOT, replacements=(), target="duration"):
    import numpy as np
    import soundfile as sf
    import validation
    if target not in ("duration","count"):
        raise ValueError("Preparation target must be duration or count")
    pipeline.assert_disjoint([*rows,*replacements])
    targets = Counter()
    for row in rows:
        targets[row["split"]] += 1 if target == "count" else row["duration"]
    queues = {split:deque(r for r in replacements if r["split"] == split) for split in SPLITS}
    pending = deque(rows)
    accepted = Counter()
    attempted = set()
    monitor.update(stage="audio preparation", completed=0, total=len(rows), reused=0, rejected=0, accepted=0,
                   message="Preserving aligned cuts; normalizing and resampling to 44.1 kHz")
    prepared, reports, seen = [], [], {}
    reused, rejected, processed = 0, 0, 0
    while pending or any(accepted[s]+1e-8 < amount for s,amount in targets.items()):
        if not pending:
            split = next(s for s,amount in targets.items() if accepted[s]+1e-8 < amount)
            if not queues[split]:
                atomic_json(dataset / "qc.json", reports)
                raise ValueError(f"Insufficient clean audio in {split} after rejecting {rejected} clips; "
                                 f"accepted {accepted[split]:.3f} of required {targets[split]:.3f} {target}. "
                                 f"See {dataset/'qc.json'}")
            pending.append(queues[split].popleft())
        row = pending.popleft()
        if row["clip_id"] in attempted:
            raise ValueError(f"Repeated preparation candidate: {row['clip_id']}")
        attempted.add(row["clip_id"])
        processed += 1
        if " ".join(Path(row["label"]).read_text(encoding="utf-8").split()) != row["text"]:
            raise ValueError(f"Source transcript changed: {row['clip_id']}; restart to replan")
        folder = dataset / "audio" / row["split"] / "talha_ahmad"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{row['clip_id']}.wav"
        receipt_path = dataset / "receipts" / f"{row['clip_id']}.json"
        rejected_path = dataset / "rejected" / f"{row['clip_id']}.json"
        source_hash = file_hash(row["audio"])
        receipt = read_json(receipt_path)
        previous_rejection = read_json(rejected_path)
        try:
            if (previous_rejection and previous_rejection.get("source_sha256") == source_hash and
                    previous_rejection.get("text") == row["text"]):
                raise QualityRejection(previous_rejection["reason"])
            if receipt_valid(receipt, row, path):
                reused += 1
            else:
                audio, rate = sf.read(row["audio"], dtype="float64")
                if rate != 48000 or audio.ndim != 1:
                    raise ValueError(f"Invalid decoded source format: {row['clip_id']}")
                if not np.isfinite(audio).all():
                    raise QualityRejection("Nonfinite input audio")
                # Confirm that the supplied clip really came from its recovered SRT interval.
                original = root / "full-audios-transcript/audios-48khz" / f"{row['source_id']}.flac"
                alignment = verify_source_interval(audio, rate, row, original)
                fraction = float(np.mean(np.abs(audio) >= 1-1/32768))
                if fraction > 0.001:
                    raise QualityRejection(f"Possible input clipping: {fraction:.6%} full-scale samples (limit 0.1%)")
                try:
                    normalized, metrics = pipeline.normalize(audio, rate)
                except ValueError as error:
                    if str(error) == "Audio is silent or has no measurable loudness":
                        raise QualityRejection(str(error)) from error
                    raise
                temp = path.with_suffix(".tmp.wav")
                sf.write(temp, normalized, 44100, subtype="PCM_16")
                temp.replace(path)
                path.with_suffix(".lab").write_text(row["text"]+"\n", encoding="utf-8")
                # Any regenerated waveform invalidates its codec tokens, even if shape was valid.
                path.with_suffix(".npy").unlink(missing_ok=True)
                receipt = {"source_sha256": source_hash, "text": row["text"],
                           "wav_sha256": file_hash(path), "lab_sha256": file_hash(path.with_suffix(".lab")),
                           "decoded_audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                           "input_fullscale_fraction":fraction, **alignment, **metrics}
                atomic_json(receipt_path, receipt)
            pcm_hash = receipt["decoded_audio_sha256"]
            if pcm_hash in seen:
                if seen[pcm_hash]["split"] != row["split"]:
                    raise ValueError(f"Duplicate decoded audio leaks across splits: {row['clip_id']}")
                raise QualityRejection(f"Duplicate decoded audio within split: {seen[pcm_hash]['clip_id']}")
        except QualityRejection as error:
            rejected += 1
            report = {"clip_id":row["clip_id"],"split":row["split"],"source_sha256":source_hash,
                      "text":row["text"],"qc_status":"rejected","reason":str(error)}
            atomic_json(rejected_path, report)
            for suffix in (".wav",".lab",".npy"):
                path.with_suffix(suffix).unlink(missing_ok=True)
            receipt_path.unlink(missing_ok=True)
            reports.append(report)
            monitor.log(f"Excluded {row['clip_id']} ({row['split']}): {error}; selecting a clean replacement")
            monitor.update(completed=processed,total=max(len(rows),processed),reused=reused,rejected=rejected,accepted=len(prepared))
            continue
        seen[pcm_hash] = row
        output, out_rate = sf.read(path)
        if (out_rate != 44100 or not np.isfinite(output).all() or
                np.max(np.abs(output)) > 10**(-1/20)+1/32768):
            raise ValueError(f"Invalid prepared audio: {path}")
        prepared.append({**row, "prepared_audio": str(path.resolve()),
                         "prepared_label": str(path.with_suffix('.lab').resolve()),
                         "decoded_audio_sha256": pcm_hash})
        reports.append({"clip_id": row["clip_id"], "split":row["split"],"qc_status":"prepared", **receipt})
        accepted[row["split"]] += 1 if target == "count" else row["duration"]
        monitor.update(completed=processed,total=max(len(rows),processed),reused=reused,rejected=rejected,accepted=len(prepared))
    validation.validate_prepared(prepared)
    manifest = dataset / "prepared.jsonl"
    temp = manifest.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in prepared), encoding="utf-8")
    temp.replace(manifest)
    atomic_json(dataset / "qc.json", reports)
    return prepared


def artifact_hashes(dataset, rows):
    paths = [Path(r["prepared_audio"]).with_suffix(".npy") for r in rows]
    paths += sorted((dataset / "protos").glob("*/*.protos"))
    return {str(p.relative_to(dataset)): file_hash(p) for p in paths}


def encode_and_pack(dataset, rows, monitor, codec_key):
    import validation
    marker = dataset / "packed.json"
    previous = read_json(marker)
    monitor.update(stage="codec tokens", completed=0, total=len(rows), message="Checking reusable codec tokens")
    if previous and previous.get("codec") == codec_key:
        try:
            if previous["artifacts"] == artifact_hashes(dataset, rows):
                summary = validation.validate_packed(dataset)
                monitor.log("Reusing verified codec tokens and packed data; no extraction required")
                monitor.update(completed=len(rows), packed=summary)
                return
        except (FileNotFoundError, ValueError, KeyError):
            monitor.log("Cached artifacts incomplete or changed; repairing them")
    token_recipe = read_json(dataset / "token_recipe.json")
    def token_receipt(row):
        return dataset / "token-receipts" / f"{row['clip_id']}.json"
    def token_identity(row, path):
        return {"codec": codec_key, "tokens": file_hash(path),
                "wav": file_hash(Path(row["prepared_audio"]))}
    for row in rows:
        path = Path(row["prepared_audio"]).with_suffix(".npy")
        if path.exists():
            try:
                if token_recipe != {"codec": codec_key}:
                    raise ValueError("Codec recipe changed")
                if previous and previous.get("artifacts", {}).get(str(path.relative_to(dataset))) != file_hash(path):
                    raise ValueError("Token content changed after verification")
                if read_json(token_receipt(row)) != token_identity(row, path):
                    raise ValueError("Missing/changed token provenance")
                pipeline.validate_tokens(path, row["duration"])
            except (ValueError, OSError, EOFError):
                path.unlink()
    atomic_json(dataset / "token_recipe.json", {"codec": codec_key})
    recorded = set()
    def progress():
        count = 0
        for row in rows:
            path = Path(row["prepared_audio"]).with_suffix(".npy")
            if not path.exists():
                continue
            if row["clip_id"] not in recorded:
                try:
                    pipeline.validate_tokens(path, row["duration"])
                    atomic_json(token_receipt(row), token_identity(row, path))
                    recorded.add(row["clip_id"])
                except (ValueError, OSError, EOFError):
                    continue  # Upstream may still be writing this file; verify it on the next tick.
            count += 1
        monitor.update(completed=count)
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        if all(Path(r["prepared_audio"]).with_suffix(".npy").exists() for r in selected):
            continue
        monitor.update(message=f"Extracting S2 Pro 10-codebook tokens: {split}")
        process([sys.executable, UPSTREAM / "tools/vqgan/extract_vq.py", dataset / "audio" / split,
                 "--num-workers", "1", "--batch-size", "1", "--config-name", "modded_dac_vq",
                 "--checkpoint-path", BASE / "codec.pth"], monitor, progress)
        for row in selected:
            pipeline.validate_tokens(Path(row["prepared_audio"]).with_suffix(".npy"), row["duration"])
        progress()
    monitor.update(stage="packing", message="Packing and checking all transcripts and token records")
    for split in SPLITS:
        output = dataset / "protos" / split
        if output.exists():
            shutil.rmtree(output)  # Derived cache only; upstream packer otherwise leaves stale shards.
        process([sys.executable, UPSTREAM / "tools/llama/build_dataset.py", "--input", dataset / "audio" / split,
                 "--output", output, "--text-extension", ".lab", "--num-workers", "1"], monitor)
    summary = validation.validate_packed(dataset)
    atomic_json(marker, {"codec": codec_key, "artifacts": artifact_hashes(dataset, rows), "summary": summary})
    monitor.update(packed=summary)


def preflight(monitor):
    checks = pipeline.readiness()
    clean = subprocess.run(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    checks["upstream_clean"] = clean.returncode == 0 and not clean.stdout.strip()
    required = ("numpy", "scipy", "pyloudnorm", "soundfile", "tensorboard", "loralib",
                "pyrootutils", "einops", "einx", "tiktoken", "safetensors", "loguru", "click", "google.protobuf",
                "dac", "audiotools", "natsort", "rich", "pyarrow")
    for name in required:
        try:
            checks["import_"+name] = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            checks["import_"+name] = False
    failures = [name for name, okay in checks.items() if not okay]
    monitor.update(checks=checks)
    if failures:
        raise RuntimeError("Preflight failed: " + ", ".join(failures) +
                           ". Activate fish-speech, install training dependencies, and verify nvidia-smi before a full run.")
    # Import actual code, rather than treating an installed package as a working installation.
    sys.path.insert(0, str(UPSTREAM))
    import torch
    import torchaudio
    import lightning
    from torch.utils.tensorboard import SummaryWriter
    from fish_speech.datasets.protos.text_data_pb2 import TextData
    from lightning.pytorch import Trainer
    from fish_speech.models.text2semantic.lit_module import TextToSemantic
    from fish_speech.models.dac.modded_dac import DAC
    from fish_speech.tokenizer import FishTokenizer
    if torchaudio.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError("S2 Pro checkout requires torchaudio==2.8.0 alongside torch==2.8.0")
    if "weights_only" not in inspect.signature(Trainer.fit).parameters:
        raise RuntimeError("Use lightning==2.6.1: this launcher requires Trainer.fit(weights_only=...).")
    tokenizer = FishTokenizer.from_pretrained(str(BASE))
    if (tokenizer.semantic_end_id-tokenizer.semantic_begin_id+1 != 4096 or
            not tokenizer.encode("هذا اختبار للصوت.",add_special_tokens=False)):
        raise RuntimeError("The base S2 Pro tokenizer does not expose the expected 4096 semantic IDs")
    probe = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    (probe * probe).sum().item()
    device = torch.cuda.get_device_properties(0)
    monitor.update(gpu_name=device.name, gpu_total_gb=round(device.total_memory/2**30, 2),
                   versions={"torch": torch.__version__, "torchaudio": torchaudio.__version__, "lightning": lightning.__version__})
    if shutil.disk_usage(HERE).free < 30 * 2**30:
        raise RuntimeError("At least 30 GiB free space is required for prepared audio and checkpoints")


def install_dependencies(monitor):
    """Install the training/codec subset, including upstream's protobuf override."""
    if Path(sys.prefix).resolve() != (ROOT / "miniconda3/envs/fish-speech").resolve():
        raise RuntimeError("--install requires the workspace fish-speech conda environment; source training/activate_fish.sh first")
    monitor.update(stage="dependencies", status="running", message="Installing the S2 Pro training dependencies in fish-speech")
    if importlib.util.find_spec("uv") is None:
        process([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--retries", "1", "--timeout", "20", "uv"], monitor)
    requirements = monitor.directory / "training-requirements.in"
    overrides = monitor.directory / "dependency-overrides.txt"
    requirements.write_text(TRAINING_REQUIREMENTS, encoding="utf-8")
    # This is the same override as upstream pyproject.toml; audiotools pins protobuf<3.20.
    # TensorBoard is pinned above: 2.21+ generates protos requiring protobuf 6.31.1.
    overrides.write_text("protobuf>=3.20,<6\n", encoding="utf-8")
    process([sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable,
             "--cache-dir", ROOT / "uv_cache", "--override", overrides, "-r", requirements], monitor)
    process([sys.executable, "-m", "uv", "pip", "freeze", "--python", sys.executable], monitor)


def composed_config(dataset, run_dir, args, pilot=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    overrides = ["train=true", f"seed={args.seed}", f"max_length={args.max_length}",
                 f"paths.run_dir={run_dir}", f"train_dataset.proto_files=[{dataset/'protos/train'}]",
                 f"val_dataset.proto_files=[{dataset/'protos/validation'}]", f"trainer.max_steps={args.steps}",
                 "trainer.log_every_n_steps=1", "trainer.enable_progress_bar=false",
                 "+trainer.num_sanity_val_steps=2",
                 "callbacks.model_checkpoint.save_last=true",
                 f"trainer.val_check_interval={args.validation_steps * 4}",
                 "train_dataset.causal=false", "trainer.limit_val_batches=1.0",
                 f"callbacks.model_checkpoint.every_n_train_steps={args.validation_steps}"]
    # Keys absent in the original config need Hydra's + prefix.
    overrides = [("+"+o if o.startswith(("trainer.log_every", "trainer.enable_progress")) else o) for o in overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(HERE / "config")):
        config = compose(config_name="talha_s2pro", overrides=overrides)
    del config.callbacks.progress_bar
    if pilot:
        config.trainer.max_steps = 2
        config.trainer.val_check_interval = 4
        config.trainer.limit_val_batches = 2
        config.callbacks.model_checkpoint.every_n_train_steps = 1
    return OmegaConf.to_container(config, resolve=True)


def emit(**value):
    print(EVENT + json.dumps(value, allow_nan=False), flush=True)


def worker(config_path, pilot=False):
    """Load upstream classes without changing either Fish Speech checkout."""
    sys.path.insert(0, str(UPSTREAM))
    import torch
    import lightning as L
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from lightning.pytorch.callbacks import ModelCheckpoint
    import numpy as np
    from torch.utils.data import Dataset
    from types import SimpleNamespace
    from fish_speech.text.clean import clean_text
    config = OmegaConf.load(config_path)
    run_dir = Path(config.paths.run_dir)
    checkpoint_dir = Path(config.paths.ckpt_dir)
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    emit(message="Loading base S2 Pro and single-speaker LoRA")
    datamodule = instantiate(config.data)
    # The upstream iterable validation sampler changes its clips between evaluations.
    # Use its current ContentSequence encoder with each held-out clip exactly once.
    dataset_root = Path(config.val_dataset.proto_files[0]).parent.parent
    val_rows = [r for r in pipeline.read_manifest(dataset_root / "prepared.jsonl") if r["split"] == "validation"]
    encoder = datamodule.val_dataset

    class FixedValidation(Dataset):
        def __len__(self):
            return len(val_rows)

        def __getitem__(self, index):
            row = val_rows[index]
            codes = np.load(Path(row["prepared_audio"]).with_suffix(".npy"), allow_pickle=False)
            semantics = [SimpleNamespace(values=values.tolist()) for values in codes]
            tokens, labels = encoder.pack_sentences([clean_text(row["text"])], [semantics], skip_text=False)
            if tokens.shape[1] > config.max_length:
                raise ValueError(f"Validation clip exceeds context length: {row['clip_id']}")
            return {"tokens": tokens, "labels": labels}

    datamodule.val_dataset = FixedValidation()
    if pilot:
        # The tiny corpus can produce shorter training sequences than the full corpus.
        # Pad to the real context length, testing full-length memory use as well.
        import fish_speech.datasets.semantic as semantic
        original_collator = semantic.TextDataCollator

        class FullLengthPilotCollator(original_collator):
            def __call__(self, examples):
                batch = super().__call__(examples)
                padding = self.max_length - batch["inputs"].shape[-1]
                if padding > 0:
                    batch["inputs"] = torch.nn.functional.pad(batch["inputs"], (0,padding), value=0)
                    batch["labels"] = torch.nn.functional.pad(batch["labels"], (0,padding), value=-100)
                    batch["attention_masks"] = torch.nn.functional.pad(batch["attention_masks"], (0,padding), value=True)
                return batch

        semantic.TextDataCollator = FullLengthPilotCollator
    model = instantiate(config.model)
    # Upstream saves adapter-only state_dicts. Rebuild frozen base, allow missing base keys,
    # and check adapter keys explicitly before restoring optimizer, scheduler and loop state.
    model.strict_loading = False
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    resume, resume_step = None, -1
    expected = {name: tensor.shape for name, tensor in model.state_dict().items() if "lora" in name}
    for candidate in sorted(checkpoints, key=lambda p: p.stat().st_mtime_ns):
        try:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
            actual = checkpoint.get("state_dict", {})
            if (set(actual) != set(expected) or not checkpoint.get("optimizer_states") or
                    not checkpoint.get("lr_schedulers") or not checkpoint.get("loops")):
                raise ValueError("Incomplete adapter or training state")
            if any(t.shape != expected[name] or not torch.isfinite(t).all() for name,t in actual.items()):
                raise ValueError("Invalid adapter tensor shape or values")
            if checkpoint["global_step"] >= resume_step:
                resume, resume_step = candidate, checkpoint["global_step"]
            del checkpoint
        except Exception as error:
            print(f"Skipping unreadable/incompatible checkpoint {candidate.name}: {error}", flush=True)
    if checkpoints and resume is None:
        raise RuntimeError("No valid resume checkpoint remains; inspect the saved checkpoints before starting a new run")
    callbacks = [instantiate(c) for c in config.callbacks.values() if c and "_target_" in c]
    best = ModelCheckpoint(dirpath=str(checkpoint_dir), filename="best_{step:09d}",
                           monitor="val/loss", mode="min", save_top_k=1,
                           save_on_train_epoch_end=False, auto_insert_metric_name=False)
    callbacks.append(best)

    class LiveStats(L.Callback):
        def __init__(self):
            self.last_step = -1
            self.started = time.monotonic()
            self.start_step = None

        def on_train_start(self, trainer, module):
            self.started = time.monotonic()
            self.start_step = trainer.global_step

        def update(self, trainer):
            step = trainer.global_step
            if self.start_step is None:
                self.start_step = step
            metrics = {}
            for key, value in trainer.callback_metrics.items():
                if hasattr(value, "numel") and value.numel() == 1:
                    number = float(value.detach().cpu())
                    if not math.isfinite(number):
                        raise RuntimeError(f"Nonfinite metric: {key}; stopping training")
                    metrics[key] = number
            seconds = time.monotonic()-self.started
            rate = (step-self.start_step)/seconds if seconds else 0
            event = dict(step=step, max_steps=config.trainer.max_steps, metrics=metrics,
                 steps_per_second=rate, eta_seconds=(config.trainer.max_steps-step)/rate if rate > 0 else None,
                 gpu_allocated_gb=round(torch.cuda.memory_allocated()/2**30, 2),
                 gpu_reserved_gb=round(torch.cuda.memory_reserved()/2**30, 2),
                 best_checkpoint=best.best_model_path, resume_checkpoint=str(resume) if resume else None,
                 learning_rate=trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None,
                 message="GPU pilot" if pilot else "Training Talha adapter")
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event,allow_nan=False)+"\n")
            emit(**event)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.global_step != self.last_step:
                self.update(trainer)
                self.last_step = trainer.global_step

        def on_validation_end(self, trainer, module):
            if not trainer.sanity_checking:
                self.update(trainer)

    callbacks.append(LiveStats())
    loggers = [instantiate(c) for c in config.logger.values() if c and "_target_" in c]
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(resume) if resume else None, weights_only=False)
    final = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(final)
    final_loss = None
    if not pilot:
        # Evaluate the final adapter on validation data, allowing it to compete for best.
        # Test videos remain untouched for an honest later voice-quality comparison.
        result = trainer.validate(model=model, datamodule=datamodule)
        final_loss = result[0].get("val/loss") if result else None
    chosen = best.best_model_path or str(final)
    if final_loss is not None and (best.best_model_score is None or final_loss < float(best.best_model_score)):
        chosen = str(final)
    atomic_json(run_dir / "completed.json", {"step": trainer.global_step,
                "final_checkpoint": str(final), "best_checkpoint": chosen,
                "periodic_best_checkpoint": best.best_model_path,
                "chosen_checkpoint": chosen, "final_validation_loss": final_loss})
    emit(step=trainer.global_step, checkpoint=str(final), best_checkpoint=chosen,
         message="Pilot passed" if pilot else "Training finished")


def base_signature():
    base_identity = {p.name: file_hash(p) for p in sorted(BASE.glob("*.json"))}
    base_identity["codec.pth"] = file_hash(BASE / "codec.pth")
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        base_identity[name] = file_hash(BASE / name)
    return base_identity


def gpu_pilot(dataset, rows, args, monitor, base_identity, codec_key, reserves=()):
    from omegaconf import OmegaConf
    key = identity({"base": base_identity, "config": composed_config(dataset, Path("PILOT"), args, True),
                    "environment": monitor.snapshot().get("versions")})[:20]
    pilot_dir = dataset / "gpu-pilots" / key
    passed = pilot_dir / "passed.json"
    saved = read_json(passed)
    if saved:
        audio = pilot_dir / "codec-reconstruction.wav"
        if audio.is_file() and saved.get("roundtrip_sha256") == file_hash(audio):
            monitor.log("Reusing passed GPU pipeline pilot for these data, settings and model weights")
            monitor.update(codec_audio=str(audio), reference_audio=saved["reference_audio"])
            return
    pilot_dataset = dataset / "pilot-data"
    # Long clips exercise the codec; padding in worker() exercises full context memory.
    picks = [r for split in SPLITS for r in sorted([r for r in rows if r["split"] == split],
                                                  key=lambda r:r["duration"], reverse=True)[:{"train":3,"validation":2,"test":1}[split]]]
    picked = {r["clip_id"] for r in picks}
    pilot_reserves = sorted([r for r in [*rows,*reserves] if r["clip_id"] not in picked],
                            key=lambda r:(-r["duration"],r["clip_id"]))
    pilot_rows = prepare(picks, pilot_dataset, monitor, replacements=pilot_reserves, target="count")
    encode_and_pack(pilot_dataset, pilot_rows, monitor, codec_key)
    pilot_dir.mkdir(parents=True, exist_ok=True)
    reference = Path(pilot_rows[0]["prepared_audio"])
    reconstruction = pilot_dir / "codec-reconstruction.wav"
    monitor.update(stage="codec round-trip", message="Decoding real S2 Pro tokens; the comparison audio appears below",
                   reference_audio=str(reference), codec_audio=None)
    process([sys.executable, UPSTREAM / "fish_speech/models/dac/inference.py", "--input-path", reference.with_suffix(".npy"),
             "--output-path", reconstruction, "--config-name", "modded_dac_vq", "--checkpoint-path", BASE / "codec.pth"], monitor)
    import numpy as np
    import soundfile as sf
    decoded, rate = sf.read(reconstruction)
    if (rate != 44100 or decoded.ndim != 1 or not np.isfinite(decoded).all() or
            abs(len(decoded)/rate-pilot_rows[0]["duration"]) > .1 or np.sqrt(np.mean(decoded**2)) < 1e-5):
        raise ValueError("Codec round-trip produced invalid, silent or incorrectly timed audio")
    monitor.update(codec_audio=str(reconstruction))
    config = composed_config(pilot_dataset, pilot_dir, args, pilot=True)
    OmegaConf.save(OmegaConf.create(config), pilot_dir / "config.yaml")
    monitor.update(stage="GPU pilot", message="Testing forward/backward, validation and checkpoint saving on six clips")
    process([sys.executable, Path(__file__), "--worker", pilot_dir / "config.yaml", "--pilot-worker"], monitor)
    monitor.update(message="Testing optimizer, scheduler and adapter resume in the GPU pilot")
    config["trainer"]["max_steps"] = 3
    OmegaConf.save(OmegaConf.create(config), pilot_dir / "config.yaml")
    process([sys.executable, Path(__file__), "--worker", pilot_dir / "config.yaml", "--pilot-worker"], monitor)
    completed = read_json(pilot_dir / "completed.json")
    if not completed or completed["step"] < 3:
        raise RuntimeError("GPU pilot failed to complete its checkpoint resume test")
    atomic_json(passed, {"completed": time.time(), "checks": "codec encode/decode, packing, fit, validation, save and resume",
                        "roundtrip_sha256": file_hash(reconstruction), "reference_audio": str(reference)})


def train(dataset, rows, args, monitor, base_identity):
    config_identity = composed_config(dataset, Path("RUN_DIRECTORY"), args)
    run_key = identity({"dataset": dataset.name, "base": base_identity, "config": config_identity,
                        "environment": monitor.snapshot().get("versions")})[:20]
    run_dir = HERE / "runs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    config = composed_config(dataset, run_dir, args)
    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create(config), run_dir / "config.yaml")
    atomic_json(run_dir / "identity.json", {"dataset": str(dataset), "base": base_identity, "run_key": run_key})
    monitor.update(run_dir=str(run_dir), max_steps=args.steps)
    monitor.update(stage="training", step=0, max_steps=args.steps, history=[], metrics={},
                   message="Training starts after the GPU pilot passes")
    if (run_dir / "metrics.jsonl").exists():
        points, latest = {}, None
        for line in (run_dir / "metrics.jsonl").read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            points.setdefault(item["step"], {"step":item["step"]}).update(item["metrics"])
            latest = item
        if latest:
            monitor.update(history=list(points.values())[-3000:], metrics=latest["metrics"], step=latest["step"])
    completed = read_json(run_dir / "completed.json")
    if completed and completed["step"] >= args.steps:
        monitor.log("This exact training run already completed; reusing its checkpoint")
    else:
        process([sys.executable, Path(__file__), "--worker", run_dir / "config.yaml"], monitor)
        completed = read_json(run_dir / "completed.json")
    if not completed or not Path(completed["chosen_checkpoint"]).is_file():
        raise RuntimeError("Training did not produce a verified completion checkpoint")
    monitor.update(step=completed["step"], checkpoint=completed["final_checkpoint"],
                   best_checkpoint=completed["best_checkpoint"])
    monitor.update(stage="merge", message="Merging the chosen adapter with the exact training configuration")
    merged = run_dir / "merged"
    merge_receipt = read_json(run_dir / "merged.json")
    expected = ("model.pth", "config.json", "tokenizer.json", "merge_provenance.json")
    merge_valid = bool(merge_receipt)
    if merge_valid:
        try:
            merge_valid = (merge_receipt["checkpoint"] == completed["chosen_checkpoint"] and
                           merge_receipt["files"] == {name:file_hash(merged/name) for name in expected} and
                           (merged/"codec.pth").resolve() == (BASE/"codec.pth").resolve())
        except (FileNotFoundError, KeyError):
            merge_valid = False
    if not merge_valid:
        if merged.exists():
            merged.rename(run_dir / f"merged-incomplete-{time.time_ns()}")
        process([sys.executable, HERE / "merge_model.py", "--experiment-config", run_dir / "config.yaml",
                 "--checkpoint", completed["chosen_checkpoint"], "--output", merged], monitor)
        atomic_json(run_dir / "merged.json", {"checkpoint": completed["chosen_checkpoint"],
                                              "files": {name:file_hash(merged/name) for name in expected}})
    monitor.update(merged_model=str(merged))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--hours", type=float, default=10, help="Total selected hours: 90%% train, 5%% validation, 5%% test")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=2048, help="Context tokens; conservative single-GPU default")
    parser.add_argument("--validation-steps", type=int, default=100, help="Optimizer steps between validation/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--install", action="store_true", help="Install required training dependencies into the active fish-speech env before preflight")
    parser.add_argument("--check", action="store_true", help="Preflight only; no data preparation or training")
    parser.add_argument("--preview", action="store_true", help="Prepare six real clips only; no GPU required")
    parser.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true", help="Headless checks/previews in environments that prohibit listening sockets")
    parser.add_argument("--exit-on-finish", action="store_true", help="Close dashboard after success/failure (default: keep serving)")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pilot-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0 or args.steps < 1 or args.validation_steps < 1:
        parser.error("hours, steps and validation-steps must be positive")
    if args.max_length < 1024 or args.max_length > 4096:
        parser.error("max-length must be between 1024 and 4096")
    if sum((args.check, args.preview, args.dashboard_only)) > 1:
        parser.error("Choose only one of --check, --preview or --dashboard-only")
    if args.no_dashboard and not (args.check or args.preview):
        parser.error("--no-dashboard is allowed only with --check or --preview")
    return args


def main():
    args = arguments()
    if args.worker:
        worker(args.worker, args.pilot_worker)
        return 0
    runtime = HERE / "monitor"
    runtime.mkdir(exist_ok=True)
    with (runtime / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Talha launcher is running; open its dashboard on port 7860", file=sys.stderr)
            return 1
        monitor = Monitor(runtime)
        server = None
        if not args.no_dashboard:
            try:
                server = dashboard(monitor, args.host, args.port)
            except OSError as error:
                monitor.log(f"Cannot bind dashboard port {args.port}: {error}")
                monitor.update(status="failed", error=str(error), message=f"Dashboard could not bind port {args.port}")
                monitor.logfile.close()
                return 1
            monitor.update(dashboard_port=server.server_port)
            monitor.log(f"Dashboard: http://<server-ip>:{server.server_port}; logs: {runtime/'launcher.log'}")
        code = 0
        try:
            if args.dashboard_only:
                monitor.update(status="monitor only", message="Dashboard test; no preparation or training running")
            else:
                if args.install:
                    install_dependencies(monitor)
                if not args.preview:
                    monitor.update(stage="preflight", message="Checking dependencies, checkpoint files and GPU")
                    preflight(monitor)
                if args.check:
                    monitor.update(status="checked", message="Preflight passed; no data preparation or training started")
                else:
                    dataset, rows, reserves = make_plan(args, monitor)
                    if not args.preview:
                        monitor.update(stage="model fingerprints", message="Hashing base and codec weights for safe cache/resume reuse")
                        base_identity = base_signature()
                        codec_key = identity({"codec_sha256": base_identity["codec.pth"], "upstream": pipeline.PIN})
                        gpu_pilot(dataset, rows, args, monitor, base_identity, codec_key, reserves)
                    if args.preview:
                        planned = rows
                        rows = [r for split in SPLITS for r in [r for r in rows if r["split"] == split][:2]]
                        picked = {r["clip_id"] for r in rows}
                        reserves = [r for r in [*planned,*reserves] if r["clip_id"] not in picked]
                        dataset = dataset / "preview"
                        monitor.update(dataset=str(dataset), mode="six-clip CPU preview")
                    rows = prepare(rows, dataset, monitor,replacements=reserves,
                                   target="count" if args.preview else "duration")
                    if args.preview:
                        monitor.update(reference_audio=rows[0]["prepared_audio"])
                    monitor.update(splits={split: {"clips": sum(r["split"] == split for r in rows),
                                                  "hours": sum(r["duration"] for r in rows if r["split"] == split)/3600,
                                                  "source_videos": len({r["source_group"] for r in rows if r["split"] == split})}
                                          for split in SPLITS})
                    if not args.preview:
                        encode_and_pack(dataset, rows, monitor, codec_key)
                        train(dataset, rows, args, monitor, base_identity)
                    monitor.update(status="complete", message="Six-clip CPU preview passed; training was not started" if args.preview
                                   else "Training and merge complete; checkpoint and model paths below")
            if not args.exit_on_finish and not args.check and not args.preview:
                monitor.log("Dashboard stays available. Ctrl-C closes it; rerunning resumes this experiment.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            monitor.update(status="stopped", message="Stopped; verified preparation and saved checkpoints will be reused")
            code = 130
        except Exception as error:
            code = 1
            monitor.log(traceback.format_exc())
            monitor.update(status="failed", error=str(error), message="Stopped on error; no later stages will run")
            if not args.exit_on_finish and not args.check and not args.preview:
                monitor.log("Failure details remain visible on the dashboard. Ctrl-C closes it.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        finally:
            if server:
                server.shutdown()
                server.server_close()
            monitor.logfile.close()
        return code


DASHBOARD = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Talha · S2 Pro training</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#0c1422;color:#e3eaf5}*{box-sizing:border-box}
body{margin:0}main{max-width:1200px;margin:auto;padding:32px 24px}header{display:flex;justify-content:space-between;gap:16px;align-items:center}
h1{font-size:28px;margin:0}h2{font-size:18px;margin:0 0 16px}p{color:#a8b8cd;line-height:1.6}.pill{border:1px solid #476080;border-radius:24px;padding:9px 15px;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.card,section{background:#152237;border:1px solid #293b54;border-radius:14px;padding:20px}
.label{color:#a8b8cd;font-size:14px}.value{font-size:26px;font-weight:650;margin-top:10px}.small{font-size:13px;color:#a8b8cd;overflow-wrap:anywhere}
.bar{height:10px;background:#293b54;border-radius:8px;overflow:hidden;margin:18px 0}.bar>div{height:100%;background:#45cfb4;transition:width .5s}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}svg{width:100%;height:200px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace;max-height:340px;overflow:auto;margin:0}
.error{color:#ffb1b1}.ok{color:#68e3c4}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #293b54}.paths{display:grid;gap:10px;margin-top:14px}
@media(max-width:740px){.grid{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}header{align-items:start;flex-direction:column}main{padding:20px 14px}}
</style><main><header><div><h1>Talha Ahmad · S2 Pro</h1><p>Single-speaker fine-tuning · live updates every 2 seconds</p></div><span id="status" class="pill">Connecting…</span></header>
<section><h2 id="stage">Starting</h2><div id="message"></div><div class="bar"><div id="progress" style="width:0"></div></div><div class="small" id="detail"></div><p id="error" class="error"></p></section>
<div class="grid"><div class="card"><div class="label">Optimizer steps</div><div class="value" id="steps">—</div><div class="small" id="eta">Waiting for training</div></div>
<div class="card"><div class="label">Training loss</div><div class="value" id="trainloss">—</div><div class="small">Lower is better; compare validation too</div></div>
<div class="card"><div class="label">Validation loss</div><div class="value" id="valloss">—</div><div class="small" id="accuracy">Held-out source videos</div></div>
<div class="card"><div class="label">GPU memory allocated</div><div class="value" id="gpu">—</div><div class="small" id="gpuname">Checking GPU</div></div></div>
<div class="charts"><section><h2>Training loss</h2><svg id="trainchart" viewBox="0 0 500 200" role="img" aria-label="Training loss by optimizer step"></svg><div class="small">Horizontal: optimizer step · vertical: loss</div></section>
<section><h2>Validation loss</h2><svg id="valchart" viewBox="0 0 500 200" role="img" aria-label="Validation loss by optimizer step"></svg><div class="small">Loss measures fit; listening determines voice quality</div></section></div>
<section><h2>Dataset and saved results</h2><table><thead><tr><th>Split</th><th>Clips</th><th>Hours</th><th>Source videos</th></tr></thead><tbody id="splits"></tbody></table>
<div class="paths" id="paths"></div><p class="small">Aligned 5–30 second clips · mono 44.1 kHz · S2 Pro 10-codebook codec · cached preparation · video-level splits</p></section>
<section id="audio-section" hidden style="margin-top:16px"><h2>Preparation audio checks</h2><p class="small">These compare prepared audio with codec reconstruction before training. They are not samples from the trained adapter.</p>
<div class="charts"><div><p>Prepared reference</p><audio id="referenceaudio" controls preload="none" style="width:100%"></audio></div>
<div id="codec-player" hidden><p>S2 Pro codec reconstruction</p><audio id="codecaudio" controls preload="none" style="width:100%"></audio></div></div></section>
<section style="margin-top:16px"><h2>Recent activity</h2><pre id="logs">Waiting for launcher…</pre></section><p class="small" id="updated"></p></main>
<script>
const el=id=>document.getElementById(id), set=(id,v)=>el(id).textContent=v??'—';
const fmt=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
function chart(id,history,key,color){let points=history.filter(p=>Number.isFinite(p[key]));let svg=el(id);svg.replaceChildren();if(!points.length){let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',12);t.setAttribute('y',100);t.setAttribute('fill','#a8b8cd');t.textContent='Waiting for real measurements';svg.append(t);return;}
let xs=points.map(p=>p.step),ys=points.map(p=>p[key]),lo=Math.min(...ys),hi=Math.max(...ys),xlo=Math.min(...xs),xhi=Math.max(...xs);let line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',points.map(p=>`${45+440*(p.step-xlo)/Math.max(1,xhi-xlo)},${175-145*(p[key]-lo)/Math.max(.001,hi-lo)}`).join(' '));line.setAttribute('stroke',color);line.setAttribute('fill','none');line.setAttribute('stroke-width','2.5');svg.append(line);
for(let [x,y,text] of [[2,25,hi.toFixed(2)],[2,180,lo.toFixed(2)],[45,198,String(xlo)],[445,198,String(xhi)]]){let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',x);t.setAttribute('y',y);t.setAttribute('fill','#a8b8cd');t.setAttribute('font-size',12);t.textContent=text;svg.append(t);}}
async function refresh(){try{let response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);let s=await response.json(),m=s.metrics||{};
set('status',s.status);el('status').className='pill '+(s.status==='failed'?'error':s.status==='complete'?'ok':'');set('stage',s.stage);set('message',s.message);set('error',s.error||'');
let training=['training','merge','GPU pilot'].includes(s.stage),a=training?s.step:s.completed,b=training?s.max_steps:s.total;el('progress').style.width=(b?Math.min(100,100*(a||0)/b):0)+'%';set('detail',b?`${a||0} / ${b} ${training?'optimizer steps':'clips checked'}${s.reused?' · '+s.reused+' prepared clips reused':''}${s.rejected?' · '+s.rejected+' clips excluded':''}`:'Checking prerequisites');
set('steps',s.max_steps?`${s.step||0} / ${s.max_steps}`:'—');set('eta',s.eta_seconds!=null?`ETA ${Math.ceil(s.eta_seconds/60)} min · ${fmt(s.steps_per_second,2)} steps/s`:'Waiting for training');
set('trainloss',fmt(m['train/loss']));set('valloss',fmt(m['val/loss']));set('accuracy',Number.isFinite(m['val/top_5_accuracy'])?`Token top-5 accuracy ${(100*m['val/top_5_accuracy']).toFixed(1)}%`:'Held-out source videos');
set('gpu',Number.isFinite(s.gpu_allocated_gb)?`${s.gpu_allocated_gb} GiB`:'—');set('gpuname',s.gpu_name||'GPU not yet available');
el('audio-section').hidden=!s.reference_audio;el('codec-player').hidden=!s.codec_audio;
for(let [id,key,url] of [['referenceaudio','reference_audio','/audio/reference.wav'],['codecaudio','codec_audio','/audio/codec.wav']])if(s[key]&&el(id).dataset.path!==s[key]){el(id).src=url+'?v='+encodeURIComponent(s[key]);el(id).dataset.path=s[key];}
el('splits').replaceChildren();for(let [name,r] of Object.entries(s.splits||{})){let tr=document.createElement('tr');for(let v of [name,r.clips,fmt(r.hours,2),r.source_videos]){let td=document.createElement('td');td.textContent=v;tr.append(td);}el('splits').append(tr);}
el('paths').replaceChildren();for(let [label,key] of [['Prepared data','dataset'],['Experiment','run_dir'],['Latest checkpoint','checkpoint'],['Best validation checkpoint','best_checkpoint'],['Merged model','merged_model']])if(s[key]){let div=document.createElement('div');div.className='small';div.textContent=label+': '+s[key];el('paths').append(div);}
chart('trainchart',s.history||[],'train/loss','#45cfb4');chart('valchart',s.history||[],'val/loss','#b7a6ff');let logs=el('logs'),bottom=logs.scrollTop+logs.clientHeight>=logs.scrollHeight-30;logs.textContent=(s.logs||[]).join('\n');if(bottom)logs.scrollTop=logs.scrollHeight;set('updated','Last update: '+new Date(s.updated*1000).toLocaleString());
}catch(e){set('status','Connection lost');set('updated','Dashboard cannot reach launcher: '+e.message);}finally{setTimeout(refresh,2000);}}refresh();
</script></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
