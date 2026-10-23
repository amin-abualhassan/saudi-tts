"""Serial voice RPC bridge to two isolated, bounded SGLang-Omni engines."""
import argparse
import base64
import errno
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

from common import HERE, ROOT, atomic_json, file_hash, identity, read_json, read_rows
from voice_server import PREFIX, validate_request

MAX_AUDIO = 32 * 1024 * 1024
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
EMIT_LOCK = threading.Lock()


class MemorySampler:
    """Sample total device memory without creating a second CUDA context."""
    def __init__(self, gpu, output):
        self.gpu, self.output, self.peak = gpu, output, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        while not self.stop.is_set():
            try:
                result = subprocess.check_output(["nvidia-smi", f"--id={self.gpu}",
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, text=True, timeout=5)
                current = int(result.strip())
                self.peak = max(current, self.peak or 0)
                atomic_json(self.output / "sglang-gpu-memory.json",
                    {"current_mib": current, "sampled_peak_mib": self.peak,
                     "gpu": self.gpu, "sample_interval_seconds": 1,
                     "note": "Whole-device memory; includes both engines", "updated_at": time.time()})
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self.stop.wait(1)


def uploaded_check(engine, bundle, output):
    for path in sorted((output / "results").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        job = read_json(path, {})
        request = job.get("request", {})
        ref_id = request.get("reference_id", "")
        if not ref_id.startswith("upload_") or not request.get("reference_text"):
            continue
        ref = read_json(output / "uploads" / (ref_id.removeprefix("upload_") + ".json"), {})
        if not ref or ref.get("wav_sha256") != job.get("reference_sha256"):
            continue
        ref["text"] = request["reference_text"]
        settings = validate_request({**request, "model": engine.name}, bundle["models"], {ref_id: ref})
        audio_id = uuid.uuid4().hex
        emit("loading", message=f"Checking a previously uploaded voice with {engine.name}")
        try:
            metrics = engine.generate(settings, ref, output / "sglang-benchmark" / (audio_id + ".wav"))
            return {"audio_id": audio_id, "reference_sha256": ref["wav_sha256"], **metrics}
        except Exception as error:
            return {"error": str(error), "reference_sha256": ref["wav_sha256"]}
    return {"status": "pending", "note": "Upload your voice with its transcript to test custom cloning"}


def emit(kind, **values):
    with EMIT_LOCK:
        print(PREFIX + json.dumps({"type": kind, **values}, allow_nan=False), flush=True)


def docker_mounts(runtime):
    return ["--mount", f"type=bind,src={ROOT},dst={ROOT},readonly",
            "--mount", f"type=bind,src={runtime},dst=/runtime"]


def api_json(port, route, timeout=10):
    with HTTP.open(f"http://127.0.0.1:{port}{route}", timeout=timeout) as response:
        return json.loads(response.read(4 * 1024 * 1024))


def wait_api_port(port, timeout=90):
    """Match Omni's non-reuse bind check, including a retiring socket's wait."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.bind(("127.0.0.1", port))
            return
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Private SGLang port {port} did not become available") from error
            time.sleep(.25)


def speech_fields(schema, declared_fields=()):
    """Follow OpenAPI refs so seed support is reported rather than assumed."""
    route = schema["paths"]["/v1/audio/speech"]["post"]
    if "requestBody" not in route:
        # This release uses a raw FastAPI Request, so OpenAPI omits its body.
        # Preflight obtains the fields from the installed CreateSpeechRequest.
        return set(declared_fields)
    node = route["requestBody"]["content"]["application/json"]["schema"]
    def fields(node):
        if "$ref" in node:
            node = schema["components"]["schemas"][node["$ref"].split("/")[-1]]
        result = set(node.get("properties", {}))
        for child in node.get("allOf", []) + node.get("anyOf", []):
            result.update(fields(child))
        return result
    return fields(node)


def request_audio(port, model, request, reference, output, seed_supported=False, stream=False,
                  on_chunk=None, on_token=None, timing_dir=None):
    import numpy as np
    import soundfile as sf
    from compare import audio_metrics

    if file_hash(Path(reference["prepared_audio"])) != reference["wav_sha256"]:
        raise ValueError("Reference audio changed")
    payload = {"model": model, "input": request["text"], "voice": "default",
               "references": [{"audio_path": reference["prepared_audio"], "text": reference["text"]}],
               "response_format": "pcm" if stream else "wav", "stream": stream,
               "temperature": request["temperature"], "top_p": request["top_p"],
               "top_k": 30, "max_new_tokens": request["max_new_tokens"]}
    if seed_supported:
        payload["seed"] = request["seed"]
    marker = None
    if stream and timing_dir is not None:
        tag = uuid.uuid4().hex
        payload["stage_params"] = {"tts_engine": {"studio_request_id": tag}}
        marker = Path(timing_dir) / (tag + ".json")
    started = time.monotonic()
    token_metrics = {"time_to_first_token_seconds": None,
                     "engine_time_to_first_token_seconds": None}
    stopped = threading.Event()
    def capture_token():
        receipt = json.loads(marker.read_text())
        elapsed = receipt["first_token_monotonic"] - started
        if elapsed < 0:
            raise ValueError("Invalid token timing clock")
        token_metrics.update(time_to_first_token_seconds=elapsed,
            engine_time_to_first_token_seconds=receipt["engine_time_to_first_token_seconds"])
        if on_token:
            on_token(dict(token_metrics))
    def watch_token():
        while not stopped.is_set():
            try:
                capture_token()
                return
            except (OSError, ValueError, KeyError):
                stopped.wait(.02)
    monitor = threading.Thread(target=watch_token, daemon=True) if marker else None
    if monitor:
        monitor.start()
    message = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with HTTP.open(message, timeout=900) as response:
            if stream and on_chunk:
                chunks, pending, total, first_bytes = [], b"", 0, None
                read = getattr(response, "read1", response.read)
                while True:
                    chunk = read(16384)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_AUDIO:
                        raise ValueError("Engine returned too much audio")
                    pending += chunk
                    aligned = len(pending) - len(pending) % 2
                    if aligned:
                        pcm, pending = pending[:aligned], pending[aligned:]
                        elapsed = time.monotonic() - started
                        if first_bytes is None:
                            first_bytes = elapsed
                        chunks.append(pcm)
                        on_chunk(pcm, elapsed)
                if pending:
                    raise ValueError("Incomplete streamed PCM")
                data = b"".join(chunks)
                if first_bytes is None:
                    raise ValueError("Engine returned no streamed audio")
            else:
                first = response.read(4096)
                first_bytes = time.monotonic() - started
                data = first + response.read(MAX_AUDIO + 1 - len(first))
                if len(data) > MAX_AUDIO:
                    raise ValueError("Engine returned too much audio")
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode(errors="replace")
        raise RuntimeError(f"SGLang HTTP {error.code}: {detail}") from error
    finally:
        if monitor:
            stopped.set()
            monitor.join(timeout=1)
            if marker.is_file() and token_metrics["time_to_first_token_seconds"] is None:
                try:
                    capture_token()
                except (OSError, ValueError, KeyError):
                    pass
            marker.unlink(missing_ok=True)
    if stream:
        if len(data) % 2:
            raise ValueError("Incomplete streamed PCM")
        audio, rate = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768, 44100
    else:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    duration = len(audio) / rate
    if rate != 44100 or audio.ndim != 1 or not .1 <= duration <= 120:
        raise ValueError("Engine returned invalid S2 Pro audio")
    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < .0001:
        raise ValueError("Engine returned silent or nonfinite audio")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.wav")
    sf.write(temporary, audio / max(1, float(np.abs(audio).max()) / .98), rate, subtype="PCM_16")
    temporary.replace(output)
    elapsed = time.monotonic() - started
    metrics = audio_metrics(audio, rate)
    return {**metrics, "duration": duration, "sample_rate": rate, "elapsed": elapsed,
            **token_metrics,
            "rtf": elapsed / duration, "first_response_bytes_seconds": first_bytes,
            "streamed_pcm": stream, "time_to_first_audio_seconds": first_bytes if stream else None,
            "seed_supported": seed_supported, "wav_sha256": file_hash(output),
            "backend": "sglang-omni", "warning": metrics.get("warning") or
            (None if seed_supported else "This SGLang API does not expose a reproducible seed")}


class Engine:
    def __init__(self, name, port, manifest, gpu, output):
        self.name, self.port, self.manifest = name, port, manifest
        self.container = "jordanian-s2-" + uuid.uuid4().hex
        self.process = None
        self.log = None
        self.gpu, self.output = gpu, Path(output)
        self.seed_supported = False

    def start(self):
        wait_api_port(self.port)
        runtime = HERE / "sglang/runtime" / self.name
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "timings").mkdir(exist_ok=True)
        self.log = (HERE / "sglang" / (self.name + ".log")).open("a", buffering=1)
        args = ["docker", "run", "--rm", "--init", "--name", self.container,
                "--label", "jordanian.s2pro.owner=" + str(os.getpid()),
                "--label", "jordanian.s2pro.suite=" + str(HERE),
                "--gpus", f"device={self.gpu}", "--network", "host", "--shm-size", "4g",
                *docker_mounts(runtime),
                "-e", "TORCHINDUCTOR_CACHE_DIR=/runtime/inductor",
                "-e", "TRITON_CACHE_DIR=/runtime/triton",
                "-e", "XDG_CACHE_HOME=/runtime/cache", "-e", "HF_HOME=/runtime/hf",
                "-e", f"PYTHONPATH={HERE}",
                "-e", "JORDANIAN_TOKEN_TIMING_DIR=/runtime/timings",
                "-e", "SGLANG_OMNI_STRICT_PORT=1",
                self.manifest["image"], "sgl-omni", "serve",
                "--model-path", self.manifest["exports"][self.name],
                "--config", str(Path(self.manifest["config_dir"]) / (self.name + ".json")),
                "--host", "127.0.0.1", "--port", str(self.port)]
        self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT)
        deadline, started = time.monotonic() + 3600, time.monotonic()
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} engine exited ({self.process.returncode}); see sglang/{self.name}.log")
            try:
                schema = api_json(self.port, "/openapi.json")
                fields = speech_fields(schema, self.manifest["preflight"].get("speech_fields", []))
                required = {"references", "temperature", "top_p", "max_new_tokens"}
                if not required <= fields:
                    raise RuntimeError(f"Pinned speech API lacks required cloning/sampling fields: {sorted(required - fields)}")
                # A published schema can precede engine readiness. Canary generation below
                # is the final readiness gate; some releases expose /health, others /healthz.
                health_ready = False
                for route in ("/health", "/healthz"):
                    try:
                        with HTTP.open(f"http://127.0.0.1:{self.port}{route}", timeout=10):
                            pass
                        health_ready = True
                        break
                    except urllib.error.HTTPError as error:
                        if error.code != 404:
                            raise
                if not health_ready:
                    # Require the model discovery endpoint when no health endpoint exists.
                    if not api_json(self.port, "/v1/models").get("data"):
                        raise ValueError("Models are not ready")
                self.seed_supported = "seed" in fields
                return time.monotonic() - started
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                time.sleep(2)
        raise RuntimeError(f"{self.name} startup timed out; inspect its engine log")

    def generate(self, request, reference, output, stream=False, on_chunk=None, on_token=None):
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("SGLang engine has exited")
        return request_audio(self.port, self.manifest["exports"][self.name], request,
                             reference, output, self.seed_supported, stream, on_chunk, on_token,
                             HERE / "sglang/runtime" / self.name / "timings" if stream else None)

    def close(self):
        if self.process:
            # Bound cleanup below the parent worker's grace period. Only our UUID container.
            try:
                subprocess.run(["docker", "rm", "--force", self.container],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            except subprocess.SubprocessError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()


def benchmark(engine, bundle, output, startup):
    """Resume 15 paired cases per model; distinguish historical from live timing."""
    cases = read_rows(Path(bundle["campaign"]) / "comparison/cases.jsonl")
    baseline_dir = Path(bundle["campaign"]) / "comparison" / engine.name
    baselines = {r["id"]: r for r in read_rows(baseline_dir / "results.jsonl")}
    report_path = output / "sglang-benchmark.json"
    report = read_json(report_path, {"rows": []})
    runtime_identity = {k: engine.manifest[k] for k in ("image", "exports", "checkpoint_hashes")}
    runtime_identity["settings"] = engine.manifest["preflight"]["settings"]
    source = identity({"runtime": runtime_identity, "cases": cases,
                       "references": bundle["references"],
                       "implementation": file_hash(Path(__file__)), "seed_supported": engine.seed_supported})
    for i, case in enumerate(cases):
        ref = bundle["references"][case["speaker"]]
        audio_id = identity({"source": source, "case": case["id"], "model": engine.name})[:32]
        wav = output / "sglang-benchmark" / (audio_id + ".wav")
        old = next((r for r in report["rows"] if r.get("audio_id") == audio_id), None)
        if old and wav.is_file() and file_hash(wav) == old.get("wav_sha256"):
            continue
        emit("loading", message=f"SGLang comparison: {engine.name}, sample {i + 1}/{len(cases)}")
        request = {"model": engine.name, "reference_id": case["speaker"],
                   "text": case["text"], **case["generation"]}
        request = validate_request(request, bundle["models"], bundle["references"])
        try:
            metrics = engine.generate(request, ref, wav)
        except Exception as error:
            if engine.process.poll() is not None:
                raise
            emit("loading", message=f"Comparison sample failed: {case['id']}: {error}; continuing")
            metrics = {"error": str(error), "status": "failed"}
        base = baselines.get(case["id"], {})
        row = {"model": engine.name, "case": case["id"], "text": case["text"],
               "speaker": ref["name"], "audio_id": audio_id, "source": source, **metrics,
               "baseline_wav": base.get("wav"), "baseline_elapsed": base.get("elapsed"),
               "baseline_duration": base.get("duration"), "startup_seconds": startup}
        report["rows"] = [r for r in report["rows"] if not
                          (r["case"] == row["case"] and r["model"] == engine.name)] + [row]
        report.update(note="Original timings are historical. SGLang sampling differs; listen to judge quality.",
                      updated_at=time.time(), manifest=engine.manifest)
        atomic_json(report_path, report)
    # Repeated-reference measurements always run fresh, after warmup.
    case = cases[0]
    request = validate_request({"model": engine.name, "reference_id": case["speaker"],
        "text": case["text"], **case["generation"]}, bundle["models"], bundle["references"])
    repeats = []
    for stream in (False, False, True):
        try:
            repeats.append(engine.generate(request, bundle["references"][case["speaker"]],
                output / "sglang-benchmark" / (uuid.uuid4().hex + ".wav"), stream=stream))
        except Exception as error:
            repeats.append({"error": str(error), "streamed_pcm": stream})
    report = read_json(report_path, {"rows": []})
    report.setdefault("warm_repeats", {})[engine.name] = repeats
    if all("wav_sha256" in r for r in repeats[:2]):
        report.setdefault("repeat_audio_identical", {})[engine.name] = (
            repeats[0]["wav_sha256"] == repeats[1]["wav_sha256"])
    report.setdefault("warm_run", {})[engine.name] = {"manifest": engine.manifest, "time": time.time()}
    report.setdefault("uploaded_voice_checks", {})[engine.name] = uploaded_check(engine, bundle, output)
    atomic_json(report_path, report)


def run(selection, gpu, benchmark_enabled):
    bundle = read_json(selection)
    output = selection.parent
    manifest = read_json(HERE / "sglang/manifest.json")
    if manifest["checkpoint_hashes"] != {n: m["checkpoint_sha256"] for n, m in bundle["models"].items()}:
        raise ValueError("SGLang exports do not match the selected checkpoints")
    engines = {}
    memory = MemorySampler(gpu, output)
    memory.thread.start()
    try:
        for i, name in enumerate(bundle["models"]):
            emit("loading", message=f"Starting SGLang {name}; first launch compiles CUDA kernels")
            engine = Engine(name, manifest["ports"][i], manifest, gpu, output)
            engines[name] = engine
            startup = engine.start()
            # A real request verifies model + tokenizer + reference codec + vocoder.
            case = read_rows(Path(bundle["campaign"]) / "comparison/cases.jsonl")[0]
            request = validate_request({"model": name, "reference_id": case["speaker"],
                "text": case["text"], **case["generation"]}, bundle["models"], bundle["references"])
            emit("loading", message=f"Checking voice cloning with {name}")
            canary = engine.generate(request, bundle["references"][case["speaker"]],
                                    output / "sglang-benchmark" / (uuid.uuid4().hex + ".wav"))
            atomic_json(output / "sglang-benchmark" / (name + "-canary.json"),
                        {"startup_seconds": startup, "metrics": canary, "manifest": manifest})
            emit("model_loaded", models=list(engines))
            if benchmark_enabled:
                benchmark(engine, bundle, output, startup)
        emit("ready", models=list(engines))
        for line in sys.stdin:
            job = json.loads(line)
            try:
                request = job["request"]
                streaming = request.get("stream", False)
                if streaming:
                    emit("stream_event", id=job["id"], event="started")
                metrics = engines[request["model"]].generate(request, job["reference"], job["output"],
                    stream=streaming,
                    on_token=(lambda metrics, job_id=job["id"]: emit("stream_event", id=job_id, event="token", **metrics)) if streaming else None,
                    on_chunk=(lambda pcm, elapsed, job_id=job["id"]: emit("stream_event", id=job_id, event="audio",
                        pcm=base64.b64encode(pcm).decode(), elapsed_seconds=elapsed, sample_rate=44100)) if streaming else None)
                emit("result", id=job["id"], **metrics)
            except Exception as error:
                traceback.print_exc()
                emit("job_error", id=job["id"], error=str(error))
                if any(e.process.poll() is not None for e in engines.values()):
                    raise RuntimeError("A SGLang engine exited; retry the server") from error
    finally:
        for engine in engines.values():
            engine.close()
        memory.stop.set()
        memory.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        run(args.selection, args.gpu, not args.skip_benchmark)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
