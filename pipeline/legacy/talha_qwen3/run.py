#!/usr/bin/env python3
"""One command: isolated setup, Talha data, Qwen codec, SFT, resume, evaluation, dashboard."""
import argparse
import copy
from collections import Counter, defaultdict
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
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = ROOT / "training/qwen3-tts-upstream"
PIN = "0c6a7cbb6c8421a46332f8c2434c7825c4c855ef"
DEFAULT_MANIFEST = ROOT / "training/talha_s2pro/cache/54f80bc308a0fefd781c55c1/prepared.jsonl"
ENV = HERE / ".venv"
EVENT = "QWEN_EVENT "
SPLITS = ("train", "validation", "test")
REVISION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def emit(**values):
    print(EVENT + json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


class Monitor:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state = {"status": "starting", "stage": "startup", "started": time.time(),
                      "step": 0, "history": [], "logs": [], "message": "Initializing",
                      "language_note": "Arabic adaptation is experimental; generation uses Auto."}
        self.logfile = (directory / "launcher.log").open("a", encoding="utf-8", buffering=1)

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["updated"] = time.time()
            atomic_json(self.directory / "status.json", self.state)

    def event(self, values):
        with self.lock:
            if "metrics" in values:
                point = {"step": values.get("step", self.state["step"]), **values["metrics"]}
                self.state["history"].append(point)
                self.state["history"] = self.state["history"][-3000:]
            self.update(**values)
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(values, ensure_ascii=False, allow_nan=False) + "\n")

    def log(self, line):
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(line)).strip()
        if line:
            line = time.strftime("%H:%M:%S ", time.gmtime()) + line
            with self.lock:
                self.state["logs"] = (self.state["logs"] + [line])[-100:]
                self.logfile.write(line + "\n")
            print(line, flush=True)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))


def dashboard(monitor, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/":
                payload, content = DASHBOARD.encode(), "text/html; charset=utf-8"
            elif route == "/api/status":
                payload, content = json.dumps(monitor.snapshot()).encode(), "application/json"
            elif route.startswith("/audio/"):
                # Only serve explicitly registered audio, never arbitrary local paths.
                key = route[len("/audio/"):]
                record = monitor.snapshot().get("audio", {}).get(key)
                path = Path(record["path"]).resolve() if record else None
                if not path or not path.is_relative_to(HERE.resolve()) or not path.is_file():
                    self.send_error(404)
                    return
                payload, content = path.read_bytes(), "audio/wav"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def process(command, monitor, cwd=ROOT):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", HF_HOME=str(HERE / "hf_cache"), TOKENIZERS_PARALLELISM="false")
    monitor.log("Running: " + " ".join(map(str, command)))
    child = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    lines = queue.Queue()

    def reader():
        for line in child.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                monitor.update()
                continue
            if line is None:
                break
            if line.startswith(EVENT):
                monitor.event(json.loads(line[len(EVENT):]))
            else:
                monitor.log(line)
        if child.wait():
            raise RuntimeError(f"Command failed ({child.returncode}); see {monitor.directory / 'launcher.log'}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        child.stdout.close()


def setup(monitor):
    monitor.update(stage="setup", message="Preparing a separate Qwen environment and pinned official repository")
    if not (UPSTREAM / ".git").exists():
        if UPSTREAM.exists():
            raise RuntimeError(f"{UPSTREAM} exists but is not a Git checkout")
        process(["git", "clone", "https://github.com/QwenLM/Qwen3-TTS.git", UPSTREAM], monitor)
        process(["git", "checkout", "--detach", PIN], monitor, UPSTREAM)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=UPSTREAM, text=True)
    if head != PIN or dirty.strip():
        raise RuntimeError(f"Expected clean Qwen checkout at {PIN}; found {head}. Use a separate checkout.")
    python = ENV / "bin/python"
    requirements = {"revision": REVISION, "upstream": PIN, "torch": "2.8.0", "transformers": "4.57.3"}
    if not python.exists():
        # The workspace launcher may be 3.14, which has no Torch 2.8 wheels.
        candidates = [ROOT / "miniconda3/envs/fish-speech/bin/python", Path(sys.executable)]
        candidates.extend(Path(path) for name in ("python3.12", "python3.11", "python3.10", "python3.13")
                          if (path := shutil.which(name)))
        interpreter = None
        for candidate in candidates:
            if candidate.is_file():
                version = json.loads(subprocess.check_output(
                    [candidate, "-c", "import json,sys;print(json.dumps(list(sys.version_info[:2])))"], text=True))
                if version[0] == 3 and 10 <= version[1] <= 13:
                    interpreter = candidate
                    break
        if interpreter is None:
            raise RuntimeError("Install Python 3.10–3.13 to create the isolated Torch 2.8 environment")
        process([interpreter, "-m", "venv", ENV], monitor)
    if read_json(ENV / "setup.json") != requirements:
        process([python, "-m", "pip", "install", "--upgrade", "pip"], monitor)
        process([python, "-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0",
                 "--index-url", "https://download.pytorch.org/whl/cu128"], monitor)
        process([python, "-m", "pip", "install", "-e", UPSTREAM, "scipy", "numpy", "safetensors"], monitor)
        process([python, "-m", "pip", "check"], monitor)
        freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
        (ENV / "installed-requirements.txt").write_text(freeze, encoding="utf-8")
        atomic_json(ENV / "setup.json", requirements)
    return python


def verify_checkout():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=UPSTREAM, text=True)
    if head != PIN or dirty.strip():
        raise RuntimeError(f"Qwen checkout must be clean and pinned at {PIN}")


def load_manifest(path, speaker="talha_ahmad"):
    rows = jsonl(path)
    if not rows:
        raise ValueError("Empty manifest")
    groups, ids, decoded = defaultdict(set), set(), {}
    for row in rows:
        for field in ("clip_id", "source_group", "split", "text"):
            if not row.get(field):
                raise ValueError(f"Missing {field} in manifest")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Transcript must be a nonempty string")
        if row.get("prepared_label"):
            label = Path(row["prepared_label"]).read_text(encoding="utf-8")
            if " ".join(label.split()) != " ".join(row["text"].split()):
                raise ValueError(f"Prepared transcript changed: {row['clip_id']}")
        if row["split"] not in SPLITS or row["clip_id"] in ids:
            raise ValueError("Unknown split or duplicate clip_id")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["clip_id"]):
            raise ValueError("clip_id must be a safe filename")
        ids.add(row["clip_id"])
        groups[row["source_group"]].add(row["split"])
        row["speaker"] = row.get("speaker", speaker).lower()
        if not re.fullmatch(r"[a-z0-9_]+", row["speaker"]):
            raise ValueError("Speaker names must contain lowercase letters, digits or underscores")
        source = Path(row.get("prepared_audio", row.get("audio", "")))
        if not source.is_file():
            raise ValueError(f"Missing audio: {source}")
        row["source_audio"] = str(source.resolve())
        pcm_hash = row.get("decoded_audio_sha256")
        if pcm_hash and pcm_hash in decoded:
            raise ValueError("Duplicate decoded audio in manifest")
        if pcm_hash:
            decoded[pcm_hash] = row["split"]
    if any(len(value) != 1 for value in groups.values()):
        raise ValueError("Source video leaks across splits")
    if set(row["split"] for row in rows) != set(SPLITS):
        raise ValueError("Manifest must include train, validation and test")
    speakers = {row["speaker"] for row in rows}
    for name in speakers:
        if {row["split"] for row in rows if row["speaker"] == name} != set(SPLITS):
            raise ValueError(f"Speaker {name} needs train, validation and test examples")
    return rows


def select_references(rows, overrides=None):
    overrides = overrides or {}
    if not isinstance(overrides, dict):
        raise ValueError("Reference overrides must be a JSON object mapping speakers to clip IDs")
    speakers = sorted({row["speaker"] for row in rows})
    if set(overrides) - set(speakers):
        raise ValueError("Unknown speaker in reference overrides")
    result = {}
    for speaker in speakers:
        train = [row for row in rows if row["speaker"] == speaker and row["split"] == "train"]
        if speaker in overrides:
            matches = [row for row in train if row["clip_id"] == overrides[speaker]]
            if len(matches) != 1:
                raise ValueError(f"Reference for {speaker} must be a training clip_id")
            result[speaker] = matches[0]
        else:
            result[speaker] = min(train, key=lambda row: (abs(row.get("duration", 7) - 7), row["clip_id"]))
    return result


def resample(source, dest):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    duration = len(audio) / rate
    if not 2 <= duration <= 30 or not np.isfinite(audio).all():
        raise ValueError(f"Audio must be finite and 2–30 seconds: {source}")
    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError(f"Silent audio: {source}")
    divisor = math.gcd(rate, 24000)
    audio = resample_poly(audio, 24000 // divisor, rate // divisor).astype("float32")
    peak = float(np.max(np.abs(audio)))
    if peak > 10 ** (-1 / 20):
        audio *= 10 ** (-1 / 20) / peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp.wav")
    sf.write(temp, audio, 24000, subtype="PCM_16")
    temp.replace(dest)
    return duration


def prepare(args, monitor):
    rows = load_manifest(args.manifest, args.speaker)
    overrides = json.loads(args.references.read_text(encoding="utf-8")) if args.references else None
    references = select_references(rows, overrides)
    if args.preview or args.smoke:
        selected = []
        for speaker in references:
            for split in SPLITS:
                selected.extend([row for row in rows if row["speaker"] == speaker and row["split"] == split][:2])
        for reference in references.values():
            if reference not in selected:
                selected.append(reference)
        rows = selected
    # Content hashes prevent stale reuse after edits to audio, transcript or reference choice.
    monitor.update(stage="data audit", completed=0, total=len(rows), message="Checking selected waveforms and split isolation")
    signatures = {}
    for index, row in enumerate(rows):
        signatures[row["clip_id"]] = file_hash(row["source_audio"])
        if "prepared_audio" in row:
            receipt = Path(row["prepared_audio"]).parents[3] / "receipts" / (row["clip_id"] + ".json")
            saved = read_json(receipt)
            if saved and saved.get("wav_sha256") != signatures[row["clip_id"]]:
                raise ValueError(f"Prepared audio changed since S2 QC: {row['clip_id']}")
        monitor.update(completed=index + 1)
    key = identity({"revision": REVISION, "manifest": file_hash(args.manifest), "audio": signatures,
                    "speakers": sorted(references), "references": {s: r["clip_id"] for s, r in references.items()},
                    "preview": args.preview, "smoke": args.smoke})[:24]
    folder = HERE / "cache" / key
    monitor.update(stage="audio preparation", dataset=str(folder), completed=0, total=len(rows),
                   message="Resampling aligned clips to mono 24 kHz; preserving source splits")
    prepared, audio_registry = [], {}
    ref_paths = {speaker: str(folder / "audio" / "train" / speaker / (row["clip_id"] + ".wav"))
                 for speaker, row in references.items()}
    seen_pcm = set()
    for index, row in enumerate(rows):
        dest = folder / "audio" / row["split"] / row["speaker"] / (row["clip_id"] + ".wav")
        receipt = folder / "receipts" / (row["clip_id"] + ".json")
        saved = read_json(receipt)
        source_hash = signatures[row["clip_id"]]
        if not (saved and saved.get("source_sha256") == source_hash and dest.exists()
                and saved.get("wav_sha256") == file_hash(dest)):
            duration = resample(row["source_audio"], dest)
            saved = {"source_sha256": source_hash, "wav_sha256": file_hash(dest), "duration": duration}
            atomic_json(receipt, saved)
        if saved["wav_sha256"] in seen_pcm:
            raise ValueError(f"Duplicate resampled waveform: {row['clip_id']}")
        seen_pcm.add(saved["wav_sha256"])
        prepared.append({"clip_id": row["clip_id"], "source_group": row["source_group"], "split": row["split"],
                         "speaker": row["speaker"], "text": row["text"], "audio": str(dest),
                         "ref_audio": ref_paths[row["speaker"]], "duration": saved["duration"], "language": "Auto",
                         "wav_sha256": saved["wav_sha256"]})
        monitor.update(completed=index + 1)
    for speaker, path in ref_paths.items():
        audio_registry["reference_" + speaker] = {"path": path, "text": references[speaker]["text"], "speaker": speaker}
    for split in SPLITS:
        write_jsonl(folder / (split + "_raw.jsonl"), [row for row in prepared if row["split"] == split])
    write_jsonl(folder / "prepared.jsonl", prepared)
    summary = {split: {"clips": sum(row["split"] == split for row in prepared),
                       "hours": sum(row["duration"] for row in prepared if row["split"] == split) / 3600,
                       "source_videos": len({row["source_group"] for row in prepared if row["split"] == split})}
               for split in SPLITS}
    atomic_json(folder / "audit.json", {"manifest": str(args.manifest), "key": key, "splits": summary,
                                       "references": {s: r["clip_id"] for s, r in references.items()}})
    monitor.update(splits=summary, speakers=sorted(references), audio=audio_registry)
    return folder


def preflight(args):
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This runner supports one GPU process; do not use torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver; CPU training is not supported.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A GPU with bfloat16 support is required")
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    emit(gpu={"name": torch.cuda.get_device_name(device), "free_gib": free / 2**30, "total_gib": total / 2**30})
    if free < 20 * 2**30:
        raise RuntimeError("Full 1.7B SFT requires substantial VRAM; at least 20 GiB free is required for this runner")
    if shutil.disk_usage(HERE).free < 80 * 2**30:
        raise RuntimeError("At least 80 GiB free disk space is required")
    if args.attention == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("Install flash-attn in the Qwen environment or use --attention sdpa")
    import qwen_tts  # noqa: F401
    import transformers
    if transformers.__version__ != "4.57.3":
        raise RuntimeError("The pinned Qwen code requires transformers==4.57.3")


def snapshot(model, revision):
    from huggingface_hub import snapshot_download
    if Path(model).is_dir():
        return Path(model).resolve()
    return Path(snapshot_download(model, revision=revision, cache_dir=str(HERE / "hf_cache/hub")))


def model_identity(path):
    return identity({str(file.relative_to(path)): file_hash(file) for file in sorted(path.rglob("*"))
                     if file.is_file() and ".cache" not in file.parts and file.suffix in (".json", ".safetensors", ".txt")})


def validate_codes(codes):
    if not isinstance(codes, list) or not codes:
        raise ValueError("Empty codec tokens")
    if any(not isinstance(frame, list) or len(frame) != 16 or
           any(type(code) is not int or not 0 <= code < 2048 for code in frame) for frame in codes):
        raise ValueError("Expected T x 16 integer Qwen 12Hz codes in [0, 2048)")


def encode(folder, tokenizer_path, device):
    from qwen_tts import Qwen3TTSTokenizer
    codec_key = model_identity(tokenizer_path)
    tokenizer = None
    for split in SPLITS:
        rows = jsonl(folder / (split + "_raw.jsonl"))
        emit(stage="codec extraction", message=f"Extracting Qwen 16-codebook tokens: {split}", completed=0, total=len(rows))
        output = []
        for index, row in enumerate(rows):
            key = identity({"codec": codec_key, "wav": row["wav_sha256"], "revision": REVISION})
            receipt_path = folder / "codes" / (key + ".json")
            receipt = read_json(receipt_path)
            try:
                if not receipt or receipt.get("key") != key:
                    raise ValueError("Missing token receipt")
                validate_codes(receipt["audio_codes"])
                if receipt.get("codes_sha256") != identity(receipt["audio_codes"]):
                    raise ValueError("Changed tokens")
            except (ValueError, KeyError):
                if tokenizer is None:
                    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=device)
                codes = tokenizer.encode([row["audio"]]).audio_codes[0].cpu().tolist()
                validate_codes(codes)
                receipt = {"key": key, "audio_codes": codes, "codes_sha256": identity(codes)}
                atomic_json(receipt_path, receipt)
            output.append({**row, "audio_codes": receipt["audio_codes"]})
            emit(completed=index + 1)
        write_jsonl(folder / (split + "_codes.jsonl"), output)
    del tokenizer
    import torch
    torch.cuda.empty_cache()


def codec_loss(logits, labels):
    """Inputs are already shifted once by the caller; never use HF's implicit shift."""
    import torch.nn.functional as F
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def save_custom_voice_export(directory, config, raw_talker, speaker_ids, embeddings, processor):
    """Write HF-compatible weights and full config without nested config diffing.

    Transformers 4.57.3 recursive_diff_dict raises KeyError('dtype') for
    Qwen's loaded nested configs. Full serialization preserves their values.
    """
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    export_config = copy.deepcopy(config)
    export_config.tts_model_type = "custom_voice"
    export_config.talker_config.spk_id = dict(speaker_ids)
    export_config.talker_config.spk_is_dialect = {speaker: False for speaker in speaker_ids}
    export_state = {"talker." + key: value for key, value in raw_talker.items()}
    weight = export_state["talker.model.codec_embedding.weight"].clone()
    for speaker, speaker_id in speaker_ids.items():
        weight[speaker_id] = embeddings[speaker].detach().cpu().to(weight.dtype)
    export_state["talker.model.codec_embedding.weight"] = weight
    export_config.dtype = weight.dtype
    serialized = json.loads(export_config.to_json_string(use_diff=False))
    # The pinned speaker config does not accept **kwargs or call super().__init__.
    # Runtime dtype/attention attributes cause both the diff failure and a reload
    # TypeError if copied into its constructor arguments.
    speaker_signature = inspect.signature(type(export_config.speaker_encoder_config).__init__)
    if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD
               for parameter in speaker_signature.parameters.values()):
        serialized["speaker_encoder_config"] = {
            key: value for key, value in serialized["speaker_encoder_config"].items()
            if key in speaker_signature.parameters and key != "self"
        }
    atomic_json(directory / "config.json", serialized)
    save_file(export_state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    processor.save_pretrained(directory)


def losses(model, batch, embeddings):
    import torch
    talker = model.talker
    ids = batch["input_ids"]
    text = talker.text_projection(talker.model.text_embedding(ids[:, :, 0]))
    text = text * batch["text_embedding_mask"]
    codec = talker.model.codec_embedding(ids[:, :, 1]) * batch["codec_embedding_mask"]
    codec[:, 6] = torch.stack([embeddings[speaker] for speaker in batch["speakers"]])
    combined = text + codec
    codes, mask = batch["codec_ids"], batch["codec_mask"]
    for group in range(1, 16):
        combined = combined + talker.code_predictor.get_input_embeddings()[group - 1](codes[:, :, group]) * mask.unsqueeze(-1)
    attention = batch["attention_mask"][:, :-1]
    positions, _ = talker.get_rope_index(attention)
    hidden = talker.model(inputs_embeds=combined[:, :-1], attention_mask=attention,
                          position_ids=positions, use_cache=False).last_hidden_state
    main = codec_loss(talker.codec_head(hidden), batch["codec_0_labels"][:, 1:])
    # A hidden state before frame t predicts code group 0 of t, then groups 1..15 of t.
    frame_hidden = hidden[mask[:, 1:]]
    frame_codes = codes[mask]
    predictor_inputs = [frame_hidden.unsqueeze(1), talker.get_input_embeddings()(frame_codes[:, :1])]
    for group in range(1, 15):
        predictor_inputs.append(talker.code_predictor.get_input_embeddings()[group - 1](frame_codes[:, group:group + 1]))
    sub_logits = talker.code_predictor.forward_finetune(inputs_embeds=torch.cat(predictor_inputs, dim=1),
                                                      labels=None, use_cache=False).logits
    sub = codec_loss(sub_logits, frame_codes[:, 1:])
    return main + 0.3 * sub, main, sub


def epoch_order(count, seed, epoch, speakers=None):
    rng = random.Random(seed + epoch)
    if speakers is None:
        order = list(range(count))
        rng.shuffle(order)
        return order
    groups = defaultdict(list)
    for index, speaker in enumerate(speakers):
        groups[speaker].append(index)
    # Oversample small speakers so every speaker has equal examples per epoch.
    maximum = max(map(len, groups.values()))
    order = []
    for group in groups.values():
        local = []
        while len(local) < maximum:
            shuffled = group.copy()
            rng.shuffle(shuffled)
            local.extend(shuffled)
        order.extend(local[:maximum])
    rng.shuffle(order)
    return order


def train(args, folder, base_path, tokenizer_path):
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(UPSTREAM / "finetuning"))
    from dataset import TTSDataset
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_rows = jsonl(folder / "train_codes.jsonl")
    val_rows = jsonl(folder / "validation_codes.jsonl")
    test_rows = jsonl(folder / "test_codes.jsonl")
    speakers = sorted({row["speaker"] for row in train_rows})
    counts = Counter(row["speaker"] for row in train_rows)
    epoch_examples = max(counts.values()) * len(speakers) if args.balanced else len(train_rows)
    epoch_steps = math.ceil(math.ceil(epoch_examples / args.batch_size) / args.accumulation)
    target_steps = min(args.steps, args.epochs * epoch_steps)
    config = {"revision": REVISION, "script": file_hash(__file__), "upstream": PIN,
              "base": model_identity(base_path), "codec": model_identity(tokenizer_path),
              "data": {split: file_hash(folder / (split + "_codes.jsonl")) for split in SPLITS},
              "controls": {name: getattr(args, name) for name in ("batch_size", "accumulation", "lr", "epochs", "steps",
                            "seed", "validation_steps", "validation_batches", "attention", "balanced", "max_length")}}
    config["warm_start"] = {"path": str(args.warm_start), "sha256": file_hash(args.warm_start)} if args.warm_start else None
    run_dir = HERE / "runs" / identity(config)[:24]
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "recipe.json", config)
    latest = read_json(run_dir / "latest.json")
    emit(stage="model loading", status="running", run_dir=str(run_dir), max_steps=target_steps,
         message="Loading 1.7B Base in bfloat16")
    tts = Qwen3TTSModel.from_pretrained(str(base_path), device_map=args.device, dtype=torch.bfloat16,
                                       attn_implementation=args.attention)
    model = tts.model
    if args.warm_start:
        model.talker.load_state_dict(load_file(str(args.warm_start), device=args.device))
        emit(message="Loaded recovered talker weights; starting a new optimizer and step counter")
    # The speech codec and speaker encoder are fixed; only the full talker is optimized.
    model.requires_grad_(False)
    model.talker.requires_grad_(True)
    model.talker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.talker.config.use_cache = False
    embeddings = {}
    for speaker in speakers:
        row = next(row for row in train_rows if row["speaker"] == speaker)
        audio, sr = sf.read(row["ref_audio"], dtype="float32")
        with torch.inference_mode():
            embedding = model.extract_speaker_embedding(audio, sr)
        embeddings[speaker] = embedding.detach().clone().reshape(-1)
    speaker_ids = {speaker: 3000 + index for index, speaker in enumerate(speakers)}
    if max(speaker_ids.values()) >= model.talker.model.codec_embedding.num_embeddings:
        raise ValueError("Too many speaker IDs for codec vocabulary")
    if any(int(value) in speaker_ids.values() for value in (model.config.talker_config.spk_id or {}).values()):
        raise ValueError("Reserved speaker IDs overlap existing speakers")

    class CachedDataset(TTSDataset):
        def __getitem__(self, index):
            row = self.data_list[index]
            text_ids = self._tokenize_texts(self._build_assistant_text(row["text"]))[:, :-5]
            codes = torch.tensor(row["audio_codes"], dtype=torch.long)
            if text_ids.shape[1] + codes.shape[0] + 8 > args.max_length:
                raise ValueError(f"Clip {row['clip_id']} exceeds --max-length; no truncation is allowed")
            return {"text_ids": text_ids, "audio_codes": codes,
                    "ref_mel": torch.zeros(1, 1, 128), "speaker": row["speaker"]}

        def collate_fn(self, examples):
            batch = super().collate_fn(examples)
            batch.pop("ref_mels")
            batch["speakers"] = [item["speaker"] for item in examples]
            return batch

    datasets = {name: CachedDataset(rows, tts.processor, model.config)
                for name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows))}
    optimizer = torch.optim.AdamW(model.talker.parameters(), lr=args.lr, weight_decay=0.01)
    start_epoch, offset, step, best = 0, 0, 0, float("inf")
    best_path = None
    if latest:
        checkpoint = Path(latest["path"])
        receipt = read_json(checkpoint / "trainer.json")
        if not receipt or receipt["recipe"] != identity(config):
            raise RuntimeError("Resume recipe mismatch")
        for name, digest in receipt["hashes"].items():
            if file_hash(checkpoint / name) != digest:
                raise RuntimeError(f"Resume checkpoint changed: {name}")
        model.talker.load_state_dict(load_file(str(checkpoint / "training_talker.safetensors"), device=args.device))
        # Local optimizer/RNG state is created by this runner; never accept external state files.
        state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        start_epoch, offset, step, best = receipt["epoch"], receipt["offset"], receipt["step"], receipt["best"]
        best_path = receipt.get("best_path")
        emit(step=step, message=f"Resuming optimizer step {step}")

    def to_device(batch):
        return {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    @torch.no_grad()
    def evaluate(split, limit=0):
        model.talker.eval()
        values, weights = [], []
        loader = DataLoader(datasets[split], batch_size=1, shuffle=False, collate_fn=datasets[split].collate_fn,
                            generator=torch.Generator().manual_seed(args.seed))
        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            batch = to_device(batch)
            value = losses(model, batch, embeddings)[0]
            if not torch.isfinite(value):
                raise RuntimeError(f"Nonfinite {split} loss")
            values.append(float(value))
            weights.append(int((batch["codec_0_labels"] != -100).sum()))
        model.talker.train()
        return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)

    def save(epoch, next_offset, validation):
        nonlocal best, best_path
        emit(stage="checkpoint", message=f"Saving resumable checkpoint at step {step}")
        destination = run_dir / f"checkpoint-step-{step}"
        temporary = run_dir / f"checkpoint-step-{step}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        raw_talker = {key: value.detach().cpu().contiguous().clone() for key, value in model.talker.state_dict().items()}
        save_file(raw_talker, str(temporary / "training_talker.safetensors"))
        # Preserve optimizer/RNG and position before export so a future export
        # error leaves recoverable training state in the unpublished directory.
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, temporary / "optimizer.pt")
        atomic_json(temporary / "pending.json", {"recipe": identity(config), "epoch": epoch,
                    "offset": next_offset, "step": step, "best": best if math.isfinite(best) else None,
                    "best_path": best_path, "validation_loss": validation})
        save_custom_voice_export(temporary, model.config, raw_talker, speaker_ids, embeddings, tts.processor)
        shutil.copytree(base_path / "speech_tokenizer", temporary / "speech_tokenizer", dirs_exist_ok=True)
        for name in ("generation_config.json", "generate_config.json"):
            if (base_path / name).exists():
                shutil.copy2(base_path / name, temporary / name)
        (temporary / "pending.json").unlink()
        if validation < best:
            best, best_path = validation, str(destination)
        hashes = {file.name: file_hash(file) for file in temporary.iterdir() if file.is_file()}
        atomic_json(temporary / "trainer.json", {"recipe": identity(config), "epoch": epoch, "offset": next_offset,
                    "step": step, "best": best, "best_path": best_path, "validation_loss": validation, "hashes": hashes})
        temporary.replace(destination)
        atomic_json(run_dir / "latest.json", {"path": str(destination)})
        atomic_json(run_dir / "best.json", {"path": best_path, "validation_loss": best})
        emit(checkpoint=str(destination), best_checkpoint=best_path, best_validation_loss=best)
        checkpoints = sorted(run_dir.glob("checkpoint-step-*"), key=lambda path: int(path.name.split("-")[-1]) if path.suffix != ".tmp" else -1)
        keep = {str(path) for path in checkpoints[-2:]} | {best_path}
        for path in checkpoints:
            if str(path) not in keep and path.is_dir():
                shutil.rmtree(path)

    started = time.monotonic()
    initial_step = step
    for epoch in range(start_epoch, args.epochs):
        if step >= target_steps:
            break
        order = epoch_order(len(train_rows), args.seed, epoch,
                            [row["speaker"] for row in train_rows] if args.balanced else None)
        if epoch != start_epoch:
            offset = 0
        if offset >= len(order):
            offset = 0
            continue
        loader = DataLoader(datasets["train"], sampler=order[offset:], batch_size=args.batch_size,
                            collate_fn=datasets["train"].collate_fn, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        total_batches = len(loader)
        model.talker.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        window = min(args.accumulation, total_batches)
        for batch_index, batch in enumerate(loader):
            emit(stage="training", epoch=epoch + 1, step=step, message=f"Training epoch {epoch + 1}/{args.epochs}")
            batch = to_device(batch)
            total, main, sub = losses(model, batch, embeddings)
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite training loss; stopping before optimizer update")
            (total / window).backward()
            accumulated_loss += float(total.detach()) / window
            offset += len(batch["speakers"])
            boundary = (batch_index + 1) % args.accumulation == 0 or batch_index + 1 == total_batches
            if not boundary:
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.talker.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.monotonic() - started
            rate = (step - initial_step) / elapsed
            metrics = {"train_loss": accumulated_loss, "codec0_loss": float(main.detach()), "sub_loss": float(sub.detach()),
                       "grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"]}
            emit(step=step, metrics=metrics, steps_per_second=rate, eta_seconds=(target_steps - step) / rate,
                 gpu={"name": torch.cuda.get_device_name(args.device), "allocated_gib": torch.cuda.memory_allocated(args.device) / 2**30,
                      "peak_gib": torch.cuda.max_memory_allocated(args.device) / 2**30})
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "epoch": epoch + 1, **metrics}) + "\n")
            accumulated_loss = 0.0
            window = min(args.accumulation, total_batches - batch_index - 1)
            if step % args.validation_steps == 0 or step >= target_steps or batch_index + 1 == total_batches:
                emit(stage="validation", message="Evaluating fixed held-out validation subset")
                validation = evaluate("validation", args.validation_batches)
                emit(step=step, metrics={"validation_loss": validation})
                save(epoch, offset, validation)
            if step >= target_steps:
                break
    if not best_path:
        raise RuntimeError("No checkpoint was produced")
    emit(stage="test evaluation", message="Evaluating the best validation checkpoint on the untouched test split")
    best_talker = load_file(str(Path(best_path) / "training_talker.safetensors"), device=args.device)
    model.talker.load_state_dict(best_talker)
    del best_talker
    test_loss = evaluate("test")
    atomic_json(run_dir / "evaluation.json", {"checkpoint": best_path, "test_loss": test_loss,
                "validation_loss": best, "test_clips": len(test_rows), "language": "Auto", "arabic_experimental": True})
    emit(step=step, metrics={"test_loss": test_loss}, best_checkpoint=best_path)
    if step > initial_step:
        del total, main, sub, batch, norm
    del optimizer, datasets, model, tts, embeddings
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Fresh reload verifies that the saved CustomVoice export is actually usable.
    emit(stage="sample generation", message="Reloading best export and generating held-out listening samples")
    inference = Qwen3TTSModel.from_pretrained(best_path, device_map=args.device, dtype=torch.bfloat16,
                                             attn_implementation=args.attention)
    generated = {}
    for speaker in speakers:
        row = next(row for row in test_rows if row["speaker"] == speaker)
        torch.manual_seed(args.seed)
        wavs, sr = inference.generate_custom_voice(text=row["text"], speaker=speaker, language="Auto",
                                                    max_new_tokens=512, do_sample=False)
        audio = np.asarray(wavs[0])
        if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
            raise RuntimeError(f"Invalid generated audio for {speaker}")
        output = run_dir / "samples" / (speaker + ".wav")
        output.parent.mkdir(exist_ok=True)
        sf.write(output, audio, sr)
        generated["sample_" + speaker] = {"path": str(output), "text": row["text"], "speaker": speaker}
        truth = output.with_name(speaker + "_truth.wav")
        shutil.copy2(row["audio"], truth)
        generated["truth_" + speaker] = {"path": str(truth), "text": row["text"], "speaker": speaker}
    atomic_json(run_dir / "samples.json", generated)
    emit(stage="complete", status="complete", generated_audio=generated,
         message="Training, test loss, export reload and listening samples completed")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--speaker", default="talha_ahmad")
    parser.add_argument("--references", type=Path, help="JSON mapping speaker name to a training clip_id")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=20, help="0 evaluates all validation clips")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true", help="Balance speakers by oversampling smaller training sets")
    parser.add_argument("--warm-start", type=Path, help="Initialize a new experiment from raw training_talker.safetensors; fresh optimizer and step counter")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only dependency/GPU/data preflight")
    mode.add_argument("--preview", action="store_true", help="CPU-only: prepare a few clips and references without setup or training")
    mode.add_argument("--prepare-only", action="store_true", help="CPU-only: prepare the entire corpus without setup or training")
    mode.add_argument("--smoke", action="store_true", help="Bounded GPU qualification: a few clips, two optimizer steps, export and synthesis")
    mode.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--no-setup", action="store_true", help="Use active Python environment and existing pinned checkout")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--exit-on-finish", action="store_true")
    parser.add_argument("--worker", choices=("check", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.epochs = 1
        args.validation_steps = 1
        args.validation_batches = 2
    for name in ("batch_size", "accumulation", "epochs", "steps", "validation_steps", "max_length"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.validation_batches < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Invalid validation-batches or learning rate")
    if not 0 <= args.port <= 65535 or not args.device.startswith("cuda:"):
        parser.error("Invalid port or device; use cuda:N")
    if args.warm_start:
        args.warm_start = args.warm_start.resolve()
        if not args.warm_start.is_file():
            parser.error("warm-start weights do not exist")
    return args


def worker(args):
    preflight(args)
    if args.worker == "check":
        return
    verify_checkout()
    emit(stage="model download", message="Resolving and downloading model/tokenizer snapshots")
    base = snapshot(args.model, args.model_revision)
    tokenizer = snapshot(args.tokenizer, args.tokenizer_revision)
    atomic_json(args.dataset / "models.json", {"base_path": str(base), "tokenizer_path": str(tokenizer),
                                              "upstream": PIN})
    encode(args.dataset, tokenizer, args.device)
    train(args, args.dataset, base, tokenizer)


def main():
    args = arguments()
    if args.worker:
        worker(args)
        return 0
    runtime = HERE / ("preview_monitor" if args.preview else "monitor")
    runtime.mkdir(parents=True, exist_ok=True)
    with (HERE / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Qwen launcher is running", file=sys.stderr)
            return 1
        monitor, server, code = Monitor(runtime), None, 0
        try:
            if not args.no_dashboard:
                server = dashboard(monitor, args.host, args.port)
                monitor.log(f"Dashboard: http://<server-ip>:{server.server_port}")
                monitor.update(dashboard_port=server.server_port)
            if args.dashboard_only:
                previous = read_json(HERE / "monitor/status.json", {})
                monitor.update(**{key: value for key, value in previous.items() if key not in ("logs", "status", "started")})
                monitor.update(status="monitor only", message="Read-only dashboard; training is not running")
            elif args.preview or args.prepare_only:
                monitor.update(status="running")
                prepare(args, monitor)
                monitor.update(status="complete", message="CPU preparation passed; training was not started")
            else:
                monitor.update(status="running")
                rows = load_manifest(args.manifest, args.speaker)
                monitor.update(selected=len(rows), speakers=sorted({row["speaker"] for row in rows}))
                python = sys.executable if args.no_setup else (ENV / "bin/python" if args.check and (ENV / "bin/python").exists() else None)
                if args.check and python is None:
                    python = sys.executable
                if python is None:
                    python = setup(monitor)
                if not args.check:
                    verify_checkout()
                worker_args = [value for value in sys.argv[1:] if value not in ("--check", "--no-setup")]
                monitor.update(stage="preflight", message="Checking CUDA, dependencies and free disk space")
                process([python, Path(__file__).resolve(), *worker_args, "--worker", "check"], monitor)
                if args.check:
                    monitor.update(status="checked", message="Preflight passed; no training started")
                else:
                    folder = prepare(args, monitor)
                    process([python, Path(__file__).resolve(), *worker_args, "--worker", "train", "--dataset", folder], monitor)
                    generated = monitor.snapshot().get("generated_audio", {})
                    monitor.update(audio={**monitor.snapshot().get("audio", {}), **generated})
            if server and not (args.exit_on_finish or args.check or args.preview):
                monitor.log("Dashboard remains available. Ctrl-C closes it. Rerun the same command to resume.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            monitor.update(status="stopped", message="Stopped; rerun to resume the last committed checkpoint")
            code = 130
        except Exception as error:
            code = 1
            monitor.log(traceback.format_exc())
            monitor.update(status="failed", error=str(error), message=str(error))
            if server and not (args.exit_on_finish or args.check or args.preview):
                monitor.log("Failure report stays on the dashboard. Ctrl-C closes it.")
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


DASHBOARD = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Talha · Qwen3-TTS training</title><style>
body{margin:0;background:#101827;color:#e5eaf4;font:15px system-ui}main{max-width:1120px;margin:auto;padding:30px}h1{font-size:26px}h2{font-size:17px}.card{background:#1a2538;border:1px solid #34445e;border-radius:12px;padding:20px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.value{font-size:24px;font-weight:600}.muted{color:#a7b6ce}progress{width:100%;height:18px;accent-color:#40caba}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:300px;overflow:auto}canvas{width:100%;height:230px}table{width:100%;text-align:left;border-spacing:0 10px}audio{width:100%}.error{color:#ffaaa5}.path{overflow-wrap:anywhere}p[dir=auto]{line-height:1.7}
</style><main><h1>Talha Ahmad · Qwen3-TTS 1.7B</h1><p class="muted">Full speaker and language fine-tuning · live reporting</p>
<section class="card"><div class="grid"><div><p class="muted">Status</p><div id="status" class="value">Connecting</div></div><div><p class="muted">Stage</p><div id="stage" class="value">—</div></div><div><p class="muted">Optimizer steps</p><div id="step" class="value">—</div></div><div><p class="muted">ETA</p><div id="eta" class="value">—</div></div></div><p id="message"></p><progress id="progress" max="100" value="0"></progress><p class="muted" id="detail"></p><p id="error" class="error"></p><p id="note" class="muted"></p></section>
<section class="card"><h2>Loss · train / validation</h2><canvas id="chart"></canvas><div class="grid"><div id="train"></div><div id="validation"></div><div id="gpu"></div></div></section>
<section class="card"><h2>Dataset and checkpoints</h2><table><thead><tr><th>Split</th><th>Clips</th><th>Hours</th><th>Videos</th></tr></thead><tbody id="splits"></tbody></table><p class="path" id="dataset"></p><p class="path" id="checkpoint"></p><p class="path" id="best"></p></section>
<section class="card"><h2>References and held-out listening samples</h2><p class="muted">References are selected from training only. Generated samples appear after the best checkpoint is reloaded.</p><div id="audio"></div></section><section class="card"><h2>Live log</h2><pre id="logs"></pre></section></main><script>
const el=id=>document.getElementById(id),set=(id,v)=>el(id).textContent=v??'—';let audioKey='';
function chart(points){const c=el('chart'),d=devicePixelRatio||1;c.width=c.clientWidth*d;c.height=230*d;const x=c.getContext('2d');x.scale(d,d);const w=c.clientWidth,h=230;let vals=points.flatMap(p=>[p.train_loss,p.validation_loss]).filter(Number.isFinite);if(!vals.length)return;let low=Math.min(...vals),high=Math.max(...vals),steps=Math.max(1,...points.map(p=>p.step));for(let [key,color]of [['train_loss','#40caba'],['validation_loss','#f8bd65']]){x.beginPath();x.strokeStyle=color;x.lineWidth=2;let first=true;for(let p of points){if(!Number.isFinite(p[key]))continue;let a=20+(w-40)*p.step/steps,b=205-(p[key]-low)/(high-low||1)*180;if(first){x.moveTo(a,b);first=false}else x.lineTo(a,b);}x.stroke()}x.fillStyle='#a7b6ce';x.fillText(high.toFixed(3),0,16);x.fillText(low.toFixed(3),0,225)}
async function refresh(){try{let r=await fetch('/api/status');if(!r.ok)throw Error(r.status);let s=await r.json();set('status',s.status);set('stage',s.stage);set('step',`${s.step||0} / ${s.max_steps||'—'}`);set('eta',s.eta_seconds?`${Math.round(s.eta_seconds/60)} min`:'—');set('message',s.message);set('error',s.error||'');set('note',s.language_note);let prep=['data audit','audio preparation','codec extraction'].includes(s.stage),a=prep?s.completed:s.step,b=prep?s.total:s.max_steps;el('progress').value=b?100*a/b:0;set('detail',b?`${a||0} / ${b} ${prep?'clips':'steps'}`:'');set('dataset',s.dataset);set('checkpoint',s.checkpoint?`Latest: ${s.checkpoint}`:'');set('best',s.best_checkpoint?`Best: ${s.best_checkpoint}`:'');let hist=s.history||[],last=key=>[...hist].reverse().find(p=>Number.isFinite(p[key]))?.[key];set('train',`Train loss: ${last('train_loss')?.toFixed(4)||'—'}`);set('validation',`Validation loss: ${last('validation_loss')?.toFixed(4)||'—'}`);set('gpu',s.gpu?`${s.gpu.name} · ${Number(s.gpu.allocated_gib??s.gpu.free_gib??0).toFixed(1)} GiB ${s.gpu.allocated_gib?'allocated':'free'}`:'');chart(hist);el('splits').replaceChildren();for(let [name,v]of Object.entries(s.splits||{})){let tr=document.createElement('tr');for(let value of [name,v.clips,v.hours.toFixed(3),v.source_videos]){let td=document.createElement('td');td.textContent=value;tr.append(td)}el('splits').append(tr)}let aud={...(s.audio||{}),...(s.generated_audio||{})},key=JSON.stringify(aud);if(key!==audioKey){audioKey=key;el('audio').replaceChildren();for(let [id,v]of Object.entries(aud)){let div=document.createElement('div'),p=document.createElement('p'),t=document.createElement('p'),player=document.createElement('audio');p.textContent=id.replaceAll('_',' ');t.textContent=v.text;t.dir='auto';player.controls=true;player.preload='none';player.src='/audio/'+encodeURIComponent(id);div.append(p,t,player);el('audio').append(div)}}set('logs',(s.logs||[]).join('\n'));}catch(e){set('error','Dashboard disconnected: '+e.message)}}refresh();setInterval(refresh,2000);
</script></html>'''

if __name__ == "__main__":
    raise SystemExit(main())
