#!/usr/bin/env python3
"""Serve the two listening-selected adapters and reference-conditioned synthesis."""
import argparse
from collections import OrderedDict
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from urllib.parse import parse_qs, urlsplit
import uuid

from common import BASE, HERE, UPSTREAM, atomic_json, file_hash, read_json, read_rows

SELECTED = ("fast_r32_final", "attention_mlp_r16_early")
SPEAKERS = {"talha_ahmad": "Talha Ahmad", "wesam_ali": "Wesam Ali",
            "single_speaker": "Single speaker"}
PREFIX = "JORDANIAN_VOICE_EVENT "
MAX_UPLOAD = 25 * 1024 * 1024
ID_PATTERN = re.compile(r"[a-f0-9]{32}")


def selected_bundle(campaign=None):
    campaign = Path(campaign or read_json(HERE / "monitor/status.json", {}).get("campaign", "")).resolve()
    if not campaign.is_relative_to((HERE / "suites").resolve()):
        raise ValueError("Select a campaign inside this suite's suites directory")
    comparison = campaign / "comparison"
    available = {m["name"]: m for m in read_rows_json(comparison / "models.json")}
    models = {}
    for name in SELECTED:
        model = available.get(name)
        if not model or model.get("status") != "complete":
            raise ValueError(f"The listening-selected checkpoint is unavailable: {name}")
        checkpoint = Path(model["checkpoint"]).resolve()
        config = Path(model["config"]).resolve()
        if not all(p.is_relative_to(campaign / "runs") and p.is_file() for p in (checkpoint, config)):
            raise ValueError(f"Missing checkpoint/configuration for {name}")
        if file_hash(checkpoint) != model.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint integrity changed: {name}")
        models[name] = {"name": name, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": model["checkpoint_sha256"], "config": str(config),
                        "config_sha256": file_hash(config)}
    references = {}
    for case in read_rows(comparison / "cases.jsonl"):
        speaker = case["speaker"]
        if speaker in references:
            continue
        ref = case["reference"]
        wav = Path(ref["prepared_audio"]).resolve()
        if not wav.is_relative_to((HERE / "cache").resolve()) or not wav.is_file():
            raise ValueError(f"Missing comparison reference for {speaker}")
        dataset = wav.parents[3]
        receipt = read_json(dataset / "receipts" / (ref["clip_id"] + ".json"), {})
        if receipt.get("wav_sha256") != file_hash(wav) or receipt.get("text") != ref["text"]:
            raise ValueError(f"Comparison reference integrity changed: {speaker}")
        references[speaker] = {"id": speaker, "name": SPEAKERS.get(speaker, speaker),
                               "clip_id": ref["clip_id"], "text": ref["text"],
                               "duration": ref["duration"], "prepared_audio": str(wav),
                               "wav_sha256": receipt["wav_sha256"], "dataset": str(dataset)}
    if set(references) != set(SPEAKERS):
        raise ValueError("The three Listen and compare speaker references are required")
    return {"campaign": str(campaign), "models": models, "references": references,
            "base": read_json(comparison / "base_identity.json", {})}


def read_rows_json(path):
    value = read_json(path, [])
    if not isinstance(value, list):
        raise ValueError(f"Expected a model list: {path}")
    return value


def plain_text(value, label, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must contain 1–{maximum} characters")
    if "<|" in value or "|>" in value:
        raise ValueError(f"Use plain text for {label.lower()}")
    return value.strip()


def validate_request(payload, models, references):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object")
    model = payload.get("model")
    reference = payload.get("reference_id")
    if not isinstance(model, str) or model not in models:
        raise ValueError("Choose one of the two selected models")
    if not isinstance(reference, str) or reference not in references:
        raise ValueError("Choose a speaker sample or upload your voice")
    result = {"model": model, "reference_id": reference,
              "text": plain_text(payload.get("text"), "Speech text", 1000)}
    if references[reference].get("uploaded"):
        result["reference_text"] = plain_text(payload.get("reference_text"), "Reference transcript", 1000)
    else:
        result["reference_text"] = references[reference]["text"]
    for name, default, low, high, integer in (
            ("seed", 42, 0, 2**32 - 1, True), ("max_new_tokens", 1024, 128, 2048, True),
            ("temperature", .7, .1, 1.0, False), ("top_p", .7, .1, 1.0, False)):
        value = payload.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {name}")
        if not low <= value <= high or integer and value != int(value):
            raise ValueError(f"{name} must be between {low} and {high}")
        result[name] = int(value) if integer else float(value)
    return result


def prepare_upload(raw, output, upload_id):
    """Decode bounded uploads to codec-rate mono WAV; never guess their transcript."""
    import numpy as np
    import soundfile as sf

    if not raw or len(raw) > MAX_UPLOAD or not ID_PATTERN.fullmatch(upload_id):
        raise ValueError("Upload an audio file smaller than 25 MB")
    directory = Path(output) / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / (upload_id + ".input")
    temporary = directory / (upload_id + ".tmp.wav")
    destination = directory / (upload_id + ".wav")
    source.write_bytes(raw)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-protocol_whitelist", "file,pipe",
                        "-format_whitelist", "wav,mp3,flac,ogg,mov,matroska,webm,aac", "-i", str(source),
                        "-t", "31", "-vn", "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le",
                        str(temporary)], capture_output=True, timeout=45, check=True)
        audio, rate = sf.read(temporary, dtype="float32")
        duration = len(audio) / rate
        if not 3 <= duration <= 30:
            raise ValueError("Use a reference clip between 3 and 30 seconds")
        if audio.ndim != 1 or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < .0001:
            raise ValueError("The reference clip is silent or invalid")
        temporary.replace(destination)
        reference = {"id": "upload_" + upload_id, "name": "My voice", "uploaded": True,
                     "prepared_audio": str(destination), "duration": duration,
                     "wav_sha256": file_hash(destination)}
        atomic_json(directory / (upload_id + ".json"), reference)
        return reference
    except (subprocess.SubprocessError, RuntimeError) as error:
        raise ValueError("Could not decode the audio. Try WAV, MP3, FLAC, M4A, OGG or WebM") from error
    finally:
        source.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


class VoiceLab:
    def __init__(self, bundle, output, gpu=0, model_cache=2, timeout=900):
        self.bundle, self.output = bundle, Path(output)
        self.gpu, self.model_cache, self.timeout = gpu, model_cache, timeout
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.pending = queue.Queue(maxsize=8)
        self.events = queue.Queue()
        self.references = dict(bundle["references"])
        self.jobs = OrderedDict()
        self.worker = None
        self.active = None
        self.restart_requested = False
        self.state = {"status": "starting", "message": "Starting speech generation", "loaded_models": []}
        self.logfile = (self.output / "server.log").open("a", encoding="utf-8", buffering=1)
        for path in sorted((self.output / "results").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)[-30:]:
            job = read_json(path, {})
            if ID_PATTERN.fullmatch(job.get("id", "")):
                if job.get("status") in ("queued", "generating"):
                    job.update(status="failed", error="Server restarted; submit this request again")
                    atomic_json(path, job)
                self.jobs[job["id"]] = job
        for path in (self.output / "uploads").glob("*.json"):
            ref = read_json(path, {})
            wav = Path(ref.get("prepared_audio", ""))
            if ref.get("id") == "upload_" + path.stem and wav.resolve().is_relative_to(self.output.resolve() / "uploads") and wav.is_file():
                self.references[ref["id"]] = ref
        atomic_json(self.output / "selection.json", bundle)

    def log(self, message):
        with self.lock:
            if not self.logfile.closed:
                self.logfile.write(time.strftime("%H:%M:%S ") + str(message).rstrip() + "\n")

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return {**self.state, "models": [{"id": name, "name": name} for name in self.bundle["models"]],
                    "references": [self.public_reference(r) for r in self.bundle["references"].values()],
                    "jobs": [dict(job) for job in list(self.jobs.values())[-30:]]}

    @staticmethod
    def public_reference(ref):
        return {k: ref[k] for k in ("id", "name", "duration")} | {
            "text": ref.get("text", ""), "audio_url": "/api/reference/" + ref["id"] + ".wav"}

    def upload(self, data):
        ref = prepare_upload(data, self.output, uuid.uuid4().hex)
        with self.lock:
            self.references[ref["id"]] = ref
        return self.public_reference(ref)

    def submit(self, payload):
        with self.lock:
            request = validate_request(payload, self.bundle["models"], self.references)
            if self.stop.is_set() or self.state["status"] == "error":
                raise RuntimeError(self.state.get("message", "Speech server is unavailable"))
            if self.pending.full():
                raise RuntimeError("The queue is full; wait for a request to finish")
            ref = dict(self.references[request["reference_id"]])
            ref["text"] = request["reference_text"]
            job = {"id": uuid.uuid4().hex, "status": "queued", "request": request,
                   "reference_name": ref["name"], "created_at": time.time(),
                   "checkpoint_sha256": self.bundle["models"][request["model"]]["checkpoint_sha256"],
                   "reference_sha256": ref["wav_sha256"], "campaign": self.bundle["campaign"]}
            self.jobs[job["id"]] = job
            atomic_json(self.output / "results" / (job["id"] + ".json"), job)
            self.pending.put_nowait((job["id"], ref))
            # Keep the live response bounded; all results remain on disk.
            while len(self.jobs) > 100:
                first = next(iter(self.jobs))
                if self.jobs[first]["status"] not in ("complete", "failed"):
                    break
                self.jobs.pop(first)
            return dict(job)

    def start_worker(self):
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONPATH=str(UPSTREAM),
                   MPLCONFIGDIR=str(self.output / "matplotlib"))
        worker = subprocess.Popen([sys.executable, str(HERE / "voice_worker.py"),
            "--selection", str(self.output / "selection.json"), "--gpu", str(self.gpu),
            "--model-cache", str(self.model_cache)], cwd=UPSTREAM, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        self.worker = worker
        self.worker_started = time.monotonic()
        self.update(status="loading", message="Loading the voice models and audio codec", loaded_models=[])

        def reader():
            for line in worker.stdout:
                self.log(line)
                if line.startswith(PREFIX):
                    try:
                        self.events.put((worker.pid, json.loads(line[len(PREFIX):])))
                    except ValueError:
                        self.log("Invalid worker event")
        threading.Thread(target=reader, daemon=True).start()

    def fail_pending(self, reason):
        with self.lock:
            for job in self.jobs.values():
                if job["status"] in ("queued", "generating"):
                    job.update(status="failed", error=reason, finished_at=time.time())
                    atomic_json(self.output / "results" / (job["id"] + ".json"), job)
            while not self.pending.empty():
                self.pending.get_nowait()
            self.active = None

    def stop_worker(self):
        worker = self.worker
        self.worker = None
        if worker:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            if worker.stdin:
                worker.stdin.close()

    def drain_events(self):
        while not self.events.empty():
            pid, event = self.events.get_nowait()
            if not self.worker or pid != self.worker.pid:
                continue
            kind = event.get("type")
            if kind == "ready":
                self.update(status="ready", message="Ready to generate speech", loaded_models=event["models"])
            elif kind == "loading":
                self.update(message=event["message"])
            elif kind == "model_loaded":
                self.update(loaded_models=event["models"])
            elif kind in ("result", "job_error"):
                with self.lock:
                    job = self.jobs.get(event.get("id"))
                    if not job or job["id"] != self.active:
                        continue
                    if kind == "result":
                        job.update(status="complete", audio_url="/api/audio/" + job["id"] + ".wav",
                                   duration=event["duration"], elapsed=event["elapsed"],
                                   sample_rate=event["sample_rate"], warning=event.get("warning"))
                        for field in ("backend", "rtf", "seed_supported", "first_response_bytes_seconds",
                                      "time_to_first_token_seconds", "engine_time_to_first_token_seconds",
                                      "time_to_first_audio_seconds", "streamed_pcm"):
                            if field in event:
                                job[field] = event[field]
                    else:
                        job.update(status="failed", error=event["error"])
                    job["finished_at"] = time.time()
                    atomic_json(self.output / "results" / (job["id"] + ".json"), job)
                    self.active = None
                    self.update(status="ready", message="Ready to generate speech")

    def tick(self):
        self.drain_events()
        if self.restart_requested:
            self.restart_requested = False
            self.stop_worker()
            self.fail_pending("Generation worker restarted; submit again")
            self.start_worker()
        if self.worker and self.worker.poll() is not None:
            code = self.worker.returncode
            self.stop_worker()
            reason = f"Speech worker exited ({code}). See voice/server.log and use Retry server."
            self.fail_pending(reason)
            self.update(status="error", message=reason, loaded_models=[])
        if self.worker and self.state["status"] == "loading" and time.monotonic() - self.worker_started > self.timeout:
            raise RuntimeError("Model loading timed out")
        if self.worker and self.state["status"] == "ready" and not self.active and not self.pending.empty():
            job_id, ref = self.pending.get_nowait()
            with self.lock:
                job = self.jobs[job_id]
                job.update(status="generating", started_at=time.time())
                self.active = job_id
                self.update(status="generating", message="Generating speech with " + job["request"]["model"])
                message = {"id": job_id, "request": job["request"], "reference": ref,
                           "output": str(self.output / "results" / (job_id + ".wav"))}
                self.worker.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.worker.stdin.flush()
        if self.active and time.time() - self.jobs[self.active]["started_at"] > self.timeout:
            raise RuntimeError("Speech generation timed out")

    def supervise(self):
        try:
            self.start_worker()
            while not self.stop.wait(.5):
                try:
                    self.tick()
                except Exception as error:
                    self.log(traceback.format_exc())
                    self.stop_worker()
                    self.fail_pending(str(error))
                    self.update(status="error", message=str(error), loaded_models=[])
                atomic_json(self.output / "status.json", self.snapshot())
        except Exception as error:
            self.log(traceback.format_exc())
            self.fail_pending(str(error))
            self.update(status="error", message=str(error))
            atomic_json(self.output / "status.json", self.snapshot())
        finally:
            self.stop_worker()
            self.fail_pending("Speech server stopped; submit again after restarting")
            if self.stop.is_set():
                self.update(status="stopped", message="Speech server stopped", loaded_models=[])
            atomic_json(self.output / "status.json", self.snapshot())


def make_handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def file(self, path, content_type="audio/wav"):
            if not path.is_file():
                return self.send(404, {"error": "Audio is unavailable"})
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/":
                return self.file(HERE / "voice_ui.html", "text/html; charset=utf-8")
            if route.path == "/api/status":
                return self.send(200, lab.snapshot())
            match = re.fullmatch(r"/api/audio/([a-f0-9]{32})\.wav", route.path)
            if match:
                return self.file(lab.output / "results" / (match[1] + ".wav"))
            match = re.fullmatch(r"/api/reference/([a-zA-Z0-9_-]+)\.wav", route.path)
            if match:
                with lab.lock:
                    ref = lab.references.get(match[1])
                if ref:
                    return self.file(Path(ref["prepared_audio"]))
            if route.path == "/compare":
                html = (HERE / "dashboard.html").read_text().replace("fetch('/api/status')", "fetch('/api/comparison')")
                return self.send(200, html.encode(), "text/html; charset=utf-8")
            if route.path == "/api/comparison":
                state = read_json(HERE / "monitor/status.json", {})
                comparison = Path(lab.bundle["campaign"]) / "comparison"
                state.update(models=read_json(comparison / "models.json", []),
                             cases=read_rows(comparison / "cases.jsonl"),
                             ratings=read_json(comparison / "ratings.json", {}))
                return self.send(200, state)
            if route.path == "/audio":
                from dashboard import safe_asset
                try:
                    return self.file(safe_asset(parse_qs(route.query).get("path", [""])[0]))
                except (ValueError, OSError):
                    pass
            return self.send(404, {"error": "Route not found"})

        def do_POST(self):
            route = urlsplit(self.path)
            try:
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    return self.send(403, {"error": "Use this server's page"})
                length = int(self.headers.get("Content-Length", "0"))
                if route.path == "/api/upload":
                    if not 0 < length <= MAX_UPLOAD:
                        raise ValueError("Upload an audio file smaller than 25 MB")
                    data = self.rfile.read(length)
                    if len(data) != length:
                        raise ValueError("Incomplete audio upload")
                    return self.send(201, lab.upload(data))
                if route.path == "/api/retry":
                    with lab.lock:
                        if lab.state["status"] != "error":
                            raise RuntimeError("Server is already running")
                        lab.restart_requested = True
                    return self.send(202, {"restarting": True})
                if route.path not in ("/api/generate", "/api/rating"):
                    return self.send(404, {"error": "Route not found"})
                if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json" or not 0 < length <= 16384:
                    raise ValueError("Send JSON smaller than 16 KB")
                payload = json.loads(self.rfile.read(length))
                if route.path == "/api/generate":
                    return self.send(202, lab.submit(payload))
                comparison = Path(lab.bundle["campaign"]) / "comparison"
                models = {m["name"] for m in read_rows_json(comparison / "models.json")}
                cases = {c["id"] for c in read_rows(comparison / "cases.jsonl")}
                if payload.get("model") not in models or payload.get("case") not in cases:
                    raise ValueError("Unknown model or sample")
                rating = {k: payload[k] for k in ("accent", "clarity", "voice", "stability")}
                if any(type(v) is not int or not 1 <= v <= 5 for v in rating.values()):
                    raise ValueError("Ratings must be integers from 1 to 5")
                with lab.lock:
                    path = comparison / "ratings.json"
                    saved = read_json(path, {})
                    saved[payload["model"] + ":" + payload["case"]] = rating
                    atomic_json(path, saved)
                return self.send(200, {"saved": True})
            except (ValueError, TypeError, KeyError) as error:
                return self.send(400, {"error": str(error)})
            except RuntimeError as error:
                return self.send(409, {"error": str(error)})
            except OSError:
                lab.log(traceback.format_exc())
                return self.send(500, {"error": "Could not save the request; inspect server.log"})
    return Handler


def stop_finished_dashboard(port=7860):
    """Release the requested port only from this suite's completed launcher."""
    state = read_json(HERE / "monitor/status.json", {})
    if state.get("status") not in ("complete", "complete with failures"):
        raise RuntimeError("The experiment launcher has not finished; leave training running")
    stopped = []
    for command in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = command.read_bytes().decode(errors="replace").split("\0")
            if any(arg == "--worker" or arg.startswith("--worker=") for arg in args) or "--no-dashboard" in args:
                continue
            cwd = (command.parent / "cwd").resolve()
            if not any((cwd / arg).resolve() == HERE / "run.py" for arg in args if arg.endswith("run.py")):
                continue
            configured_port = 7860
            for index, argument in enumerate(args):
                if argument == "--port":
                    configured_port = int(args[index + 1])
                elif argument.startswith("--port="):
                    configured_port = int(argument.split("=", 1)[1])
            if configured_port != port:
                continue
            pid = int(command.parent.name)
            if pid != os.getpid():
                os.kill(pid, signal.SIGINT)
                stopped.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return stopped


def stop_idle_voice_server(output, proc_root=Path("/proc")):
    """Signal only the PID of this exact server, and refuse unfinished requests."""
    output = Path(output)
    state = read_json(output / "status.json", {})
    if state.get("status") == "generating" or any(
            job.get("status") in ("queued", "generating") for job in state.get("jobs", [])):
        raise RuntimeError("The voice server has unfinished requests; let them finish before switching launchers")
    try:
        pid = int((output / "server.pid").read_text().strip())
        process = proc_root / str(pid)
        arguments = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
        cwd = (process / "cwd").resolve()
        servers = {HERE / "voice_server.py", HERE / "sglang_server.py"}
        matches = any((cwd / arg).resolve() in servers
                      for arg in arguments if arg.endswith(".py"))
        if pid <= 1 or pid == os.getpid() or not matches:
            raise ValueError("PID does not belong to this voice server")
    except (OSError, ValueError) as error:
        raise RuntimeError("Could not verify the existing voice server PID; stop its screen process before switching") from error
    os.kill(pid, signal.SIGTERM)
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--model-cache", type=int, choices=(1, 2), default=2)
    # Accept old acceleration arguments so existing screen commands roll back too.
    parser.add_argument("--optimization", choices=("eager", "compiled"), default="eager", help=argparse.SUPPRESS)
    parser.add_argument("--compile-mode", help=argparse.SUPPRESS)
    parser.add_argument("--startup-timeout", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--replace-voice-server", action="store_true", help="Gracefully stop this voice server when it has no unfinished requests")
    parser.add_argument("--replace-dashboard", action="store_true", help="Stop this suite's completed dashboard if it occupies the port")
    parser.add_argument("--check", action="store_true", help="Verify the selected checkpoints/references without loading CUDA or serving")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.gpu < 0 or (args.startup_timeout is not None and args.startup_timeout <= 0):
        parser.error("Use a valid port, nonnegative GPU index and positive startup timeout")
    bundle = selected_bundle(args.campaign)
    if args.check:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0
    if args.optimization != "eager":
        print("Acceleration has been rolled back; using the original decoder", flush=True)
    output = HERE / "voice"
    output.mkdir(exist_ok=True)
    with (output / "server.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not args.replace_voice_server:
                raise RuntimeError("The voice server is already running")
            pid = stop_idle_voice_server(output)
            print(f"Stopping idle voice server {pid} before restarting", flush=True)
            for _ in range(240):
                time.sleep(.25)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    continue
            else:
                raise RuntimeError("The previous voice server did not exit within 60 seconds")
        lab = VoiceLab(bundle, output, args.gpu, args.model_cache)
        try:
            try:
                server = ThreadingHTTPServer((args.host, args.port), make_handler(lab))
            except OSError as error:
                if error.errno != 98 or not args.replace_dashboard:
                    raise
                stopped = stop_finished_dashboard(args.port)
                if not stopped:
                    raise RuntimeError(f"Port {args.port} is occupied by another app; stop that app before starting") from error
                for _ in range(20):
                    time.sleep(.25)
                    try:
                        server = ThreadingHTTPServer((args.host, args.port), make_handler(lab))
                        break
                    except OSError as retry_error:
                        if retry_error.errno != 98:
                            raise
                else:
                    raise RuntimeError("The finished dashboard did not release the port") from error
            server.daemon_threads = True
            (output / "server.pid").write_text(str(os.getpid()) + "\n")
            thread = threading.Thread(target=lab.supervise, daemon=True)
            thread.start()
            def stopped(*_):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stopped)
            print(f"Jordanian voice server: http://<server-ip>:{server.server_port}", flush=True)
            try:
                server.serve_forever(poll_interval=.5)
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                lab.stop.set()
                thread.join(timeout=55)
                (output / "server.pid").unlink(missing_ok=True)
        finally:
            lab.logfile.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
