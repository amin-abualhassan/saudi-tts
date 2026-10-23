#!/usr/bin/env python3
"""Live Talha listening lab. Starts immediately; loads only a completed merged run."""
import argparse
import csv
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
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
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline
from run import atomic_json, file_hash, identity

RUNS = HERE / "runs"
BASE = pipeline.BASE
UPSTREAM = pipeline.UPSTREAM
PREFIX = "TALHA_TEST_EVENT "
MERGED_FILES = ("model.pth", "config.json", "tokenizer.json", "merge_provenance.json")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def training_workers():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = path.read_bytes().decode(errors="replace").split("\0")
            if "--worker" in arguments and str(HERE / "run.py") in arguments:
                found.append(int(path.parent.name))
        except OSError:
            pass
    return found


def system_status(gpu_index):
    result = {"disk_free_gb": round(shutil.disk_usage(HERE).free / 2**30, 1),
              "python": sys.version.split()[0], "environment": sys.prefix,
              "dependencies": {name: importlib.util.find_spec(name) is not None for name in
                               ("torch", "torchaudio", "numpy", "soundfile", "hydra", "loralib")}}
    try:
        query = subprocess.run(["nvidia-smi", "-i", str(gpu_index),
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        fields = next(csv.reader(query.stdout.splitlines()))
        result["gpu"] = {"name": fields[0].strip(), "total_gb": round(float(fields[1])/1024, 1),
                         "used_gb": round(float(fields[2])/1024, 1),
                         "free_gb": round(float(fields[3])/1024, 1), "utilization": float(fields[4])}
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
        result["gpu_error"] = str(error)
    return result


def find_run(training, explicit=None):
    if explicit:
        return explicit.resolve()
    current = training.get("run_dir")
    if current and within(current, RUNS):
        return Path(current).resolve()
    identities = sorted(RUNS.glob("*/identity.json"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
    return identities[0].parent if identities else None


def candidate_model(run_dir, required_steps=0):
    """The atomic merge receipt is the publication boundary, not model.pth alone."""
    completed = read_json(run_dir / "completed.json")
    receipt = read_json(run_dir / "merged.json")
    if not completed or not receipt or completed.get("step", 0) < required_steps:
        return None
    checkpoint = Path(completed["chosen_checkpoint"]).resolve()
    if not within(checkpoint, run_dir / "checkpoints") or not checkpoint.is_file():
        raise ValueError("Completion checkpoint is missing or outside this run")
    if Path(receipt["checkpoint"]).resolve() != checkpoint:
        raise ValueError("Merged model does not match the chosen completion checkpoint")
    merged = run_dir / "merged"
    if set(receipt.get("files", {})) != set(MERGED_FILES):
        raise ValueError("Incomplete merged-model integrity receipt")
    if (merged / "codec.pth").resolve() != (BASE / "codec.pth").resolve():
        raise ValueError("Merged model is not using the original S2 Pro codec")
    stats = {}
    for name in (*MERGED_FILES, "codec.pth"):
        path = merged / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty merged artifact: {name}")
        stat = path.stat()
        stats[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return {"directory": str(merged), "run_dir":str(run_dir), "step":completed["step"],
            "checkpoint":str(checkpoint), "receipt":receipt,
            "key":identity({"directory":str(merged), "receipt":receipt, "stats":stats})}


def verify_model(candidate):
    merged = Path(candidate["directory"])
    for name, expected in candidate["receipt"]["files"].items():
        if file_hash(merged / name) != expected:
            raise ValueError(f"Merged model integrity check failed: {name}")
    provenance = read_json(merged / "merge_provenance.json", {})
    if Path(provenance.get("checkpoint", "")).resolve() != Path(candidate["checkpoint"]):
        raise ValueError("Merge provenance does not match the completion checkpoint")
    source = read_json(Path(candidate["run_dir"]) / "identity.json", {})
    expected_codec = source.get("base", {}).get("codec.pth")
    if not expected_codec or file_hash(merged / "codec.pth") != expected_codec:
        raise ValueError("Codec integrity differs from the training run")


def references_for(run_dir):
    source = read_json(run_dir / "identity.json", {})
    dataset = Path(source.get("dataset", ""))
    if not within(dataset, HERE / "cache"):
        return []
    manifest = dataset / "prepared.jsonl"
    if not manifest.is_file():
        return []
    rows = pipeline.read_manifest(manifest)
    eligible = [r for r in rows if r["split"] == "train" and 5 <= r["duration"] <= 15
                and within(r["prepared_audio"], dataset / "audio")]
    chosen, sources = [], set()
    for row in sorted(eligible, key=lambda r:(abs(r["duration"]-10),r["clip_id"])):
        if row["source_group"] in sources or not Path(row["prepared_audio"]).is_file():
            continue
        sources.add(row["source_group"])
        chosen.append({**row, "dataset":str(dataset)})
        if len(chosen) == 3:
            break
    return chosen


def validate_request(payload, references):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object")
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip() or len(text) > 1000 or len(text.encode()) > 4000:
        raise ValueError("Enter 1–1000 characters of text (at most 4000 UTF-8 bytes)")
    if "<|" in text or "|>" in text:
        raise ValueError("Enter plain text; the app supplies the single-speaker tag")
    result = {"text":text.strip()}
    for name, default, low, high, integer in (
            ("seed",42,0,2**32-1,True), ("max_new_tokens",1024,128,2048,True),
            ("temperature",.7,.1,1.0,False), ("top_p",.7,.1,1.0,False)):
        value = payload.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {name}")
        if not low <= value <= high or (integer and value != int(value)):
            raise ValueError(f"{name} must be {'an integer ' if integer else ''}between {low} and {high}")
        result[name] = int(value) if integer else float(value)
    reference_id = payload.get("reference_id")
    if reference_id is not None and reference_id not in references:
        raise ValueError("Choose an available reference, or choose no reference")
    result["reference_id"] = reference_id
    return result


class Lab:
    def __init__(self, output, explicit_run=None, gpu_index=0, poll_seconds=5, worker_timeout=600):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.explicit_run, self.gpu_index = explicit_run, gpu_index
        self.poll_seconds, self.worker_timeout = poll_seconds, worker_timeout
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.pending = queue.Queue(maxsize=8)
        self.worker = None
        self.model_key = None
        self.started = 0
        self.active_job = None
        self.references = {}
        self.jobs = {}
        self.retry_at = 0
        self.state = {"status":"waiting", "message":"Checking training and model availability",
                      "training":{}, "system":{}, "model":None, "logs":[]}
        self.logfile = (output / "app.log").open("a", encoding="utf-8", buffering=1)
        for path in sorted((output / "results").glob("*.json"), key=lambda p:p.stat().st_mtime_ns)[-20:]:
            item = read_json(path)
            if item and re.fullmatch(r"[a-f0-9]{32}", item.get("id", "")):
                self.jobs[item["id"]] = item

    def log(self, text):
        with self.lock:
            if self.logfile.closed:
                return
            line = time.strftime("%H:%M:%S") + " " + str(text).rstrip()
            self.logfile.write(line+"\n")
            self.state["logs"] = (self.state["logs"]+[line])[-80:]

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return {**self.state, "updated_at":time.time(), "references":[
                {"id":r["clip_id"],"text":r["text"],"duration":r["duration"],
                 "audio_url":f"/api/reference/{r['clip_id']}.wav"} for r in self.references.values()],
                "jobs":list(self.jobs.values())[-20:]}

    def submit(self, payload):
        with self.lock:
            request = validate_request(payload, self.references)
            if self.state["status"] != "ready" or self.worker is None or self.worker.poll() is not None:
                raise RuntimeError("The completed model is not ready yet")
            if self.pending.full():
                raise RuntimeError("The testing queue is full; wait for the current tests")
            job = {"id":uuid.uuid4().hex, "status":"queued", "created_at":time.time(),
                   "model":self.state["model"], "request":request}
            self.pending.put_nowait(job)
            self.jobs[job["id"]] = job
            # Keep displayed history bounded; result WAV/JSON files remain on disk.
            if len(self.jobs) > 40:
                for key,item in list(self.jobs.items()):
                    if item["status"] in ("complete","failed") and len(self.jobs)>40:
                        del self.jobs[key]
            return dict(job)

    def stop_worker(self, reason):
        worker, self.worker = self.worker, None
        if worker is not None and worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            except ProcessLookupError:
                pass
        with self.lock:
            for job in self.jobs.values():
                if job["status"] in ("queued","generating"):
                    job.update(status="failed", error=reason)
            self.active_job = None
            while not self.pending.empty():
                self.pending.get_nowait()
        while not self.events.empty():
            self.events.get_nowait()
        self.model_key = None

    def start_worker(self, candidate):
        self.update(status="loading",message="Checking completed model and codec integrity before loading")
        verify_model(candidate)
        if self.stop_event.is_set():
            return
        self.update(model={k:candidate[k] for k in ("directory","run_dir","step","checkpoint")},
                    message="Loading the fine-tuned model and S2 Pro codec onto the GPU")
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONPATH=str(UPSTREAM), USE_LIBUV="0",
                   MPLCONFIGDIR=str(self.output / "matplotlib"))
        for key in list(env):
            if key.startswith("SLURM_"):
                del env[key]
        self.worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
             candidate["directory"], "--gpu", str(self.gpu_index)], cwd=UPSTREAM, env=env,
             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
             text=True, bufsize=1, start_new_session=True)
        self.model_key, self.started = candidate["key"], time.monotonic()
        worker = self.worker
        def reader():
            for line in worker.stdout:
                self.log(line)
                if line.startswith(PREFIX):
                    try:
                        self.events.put((worker.pid,json.loads(line[len(PREFIX):])))
                    except ValueError:
                        self.log("Invalid inference-worker event")
        threading.Thread(target=reader,daemon=True).start()

    def drain_events(self):
        while not self.events.empty():
            pid,event = self.events.get_nowait()
            if not self.worker or pid != self.worker.pid:
                continue
            if event["type"] == "ready":
                self.update(status="ready",message="Fine-tuned Talha model is ready for listening tests",
                            inference=event)
            elif event["type"] == "loading":
                self.update(message=event["message"])
            elif event["type"] in ("result","job_error"):
                with self.lock:
                    job = self.jobs.get(event["id"])
                    if not job or event["id"] != self.active_job:
                        continue
                    if event["type"] == "result":
                        job.update(status="complete",audio_url=f"/api/audio/{job['id']}.wav",
                                   duration=event["duration"],elapsed=event["elapsed"],
                                   sample_rate=event["sample_rate"],finished_at=time.time())
                    else:
                        job.update(status="failed",error=event["error"],finished_at=time.time())
                    atomic_json(self.output / "results" / f"{job['id']}.json", job)
                    self.active_job = None

    def tick(self):
        self.drain_events()
        training = read_json(HERE / "monitor/status.json", {})
        brief = {k:training.get(k) for k in ("status","stage","step","max_steps","metrics",
                 "message","eta_seconds","run_dir","error","updated")}
        self.update(training=brief)
        run_dir = find_run(training, self.explicit_run)
        self.update(watching_run=str(run_dir) if run_dir else None)
        refs = references_for(run_dir) if run_dir else []
        with self.lock:
            self.references = {r["clip_id"]:r for r in refs}
        active = training_workers()
        if active:
            if self.worker:
                self.stop_worker("Training is using the GPU; testing will reload after completion")
            self.update(status="waiting",message="Training is running. Waiting for its completed, verified merged model",
                        training_worker_pids=active,model=None)
            return
        self.update(training_worker_pids=[])
        required = (training.get("max_steps",0) or 0) if not self.explicit_run else 0
        candidate = candidate_model(run_dir,required) if run_dir else None
        if candidate is None:
            if self.worker:
                self.stop_worker("Waiting for the selected training run to complete")
            self.update(status="waiting",model=None,message="Waiting for training completion and the atomic merged-model receipt")
            return
        if self.worker and self.model_key != candidate["key"]:
            self.stop_worker("A new completed model is available; reloading")
        if self.worker and self.worker.poll() is not None:
            code = self.worker.returncode
            self.stop_worker(f"Inference worker stopped with exit code {code}; see the app logs")
            self.update(status="error",message=f"Inference worker exited ({code}); retrying in 60 seconds")
            self.retry_at = time.monotonic()+60
        if self.worker is None and time.monotonic() >= self.retry_at:
            gpu = self.state.get("system",{}).get("gpu")
            if gpu and gpu["free_gb"] < 16:
                self.update(status="waiting",message="Waiting for at least 16 GB of free GPU memory")
                return
            self.log(f"Loading completed model: {candidate['directory']}")
            self.start_worker(candidate)
        if self.worker and self.state["status"] == "loading" and time.monotonic()-self.started > self.worker_timeout:
            self.stop_worker("Model loading timed out")
            raise RuntimeError("Model loading exceeded 10 minutes; inspect the app log")
        if self.worker and self.state["status"] == "ready" and self.active_job is None and not self.pending.empty():
            job = self.pending.get_nowait()
            request = job["request"]
            reference = self.references.get(request["reference_id"])
            if request["reference_id"] and not reference:
                with self.lock:
                    job.update(status="failed",error="Reference is no longer available")
                return
            self.worker.stdin.write(json.dumps({"id":job["id"],"request":request,"reference":reference,
                 "output":str(self.output / "results" / f"{job['id']}.wav")}, ensure_ascii=False)+"\n")
            self.worker.stdin.flush()
            with self.lock:
                job.update(status="generating",started_at=time.time())
                self.active_job = job["id"]
        if self.active_job and time.time()-self.jobs[self.active_job]["started_at"] > self.worker_timeout:
            self.stop_worker("Speech generation timed out; reloading the inference worker")
            raise RuntimeError("Generation exceeded 10 minutes; worker stopped to free GPU memory")

    def supervise(self):
        last_system = 0
        try:
            while not self.stop_event.is_set():
                try:
                    if time.monotonic()-last_system >= 15:
                        self.update(system=system_status(self.gpu_index))
                        last_system = time.monotonic()
                    self.tick()
                except Exception as error:
                    self.log(traceback.format_exc())
                    self.stop_worker(str(error))
                    self.update(status="error",message=str(error)+"; checking again in 60 seconds")
                    self.retry_at = time.monotonic()+60
                atomic_json(self.output / "status.json", self.snapshot())
                self.stop_event.wait(self.poll_seconds)
        finally:
            self.stop_worker("Testing app stopped")


def make_handler(lab):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            data = json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path, content_type):
            if not path.is_file():
                return self.send_json(404,{"error":"File not available"})
            self.send_response(200)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(path.stat().st_size))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            with path.open("rb") as source:
                shutil.copyfileobj(source,self.wfile)

        def do_GET(self):
            path = self.path.split("?",1)[0]
            if path == "/":
                return self.send_file(HERE / "testing_ui.html","text/html; charset=utf-8")
            if path == "/api/status":
                return self.send_json(200,lab.snapshot())
            match = re.fullmatch(r"/api/audio/([a-f0-9]{32})\.wav",path)
            if match:
                return self.send_file(lab.output / "results" / f"{match[1]}.wav","audio/wav")
            match = re.fullmatch(r"/api/reference/([a-z0-9]+-\d+)\.wav",path)
            if match:
                with lab.lock:
                    row = lab.references.get(match[1])
                if row and within(row["prepared_audio"], HERE / "cache"):
                    return self.send_file(Path(row["prepared_audio"]),"audio/wav")
            self.send_json(404,{"error":"Route not found"})

        def do_POST(self):
            if self.path != "/api/generate":
                return self.send_json(404,{"error":"Route not found"})
            try:
                if self.headers.get("Content-Type","").split(";",1)[0] != "application/json":
                    raise ValueError("Use application/json")
                size = int(self.headers.get("Content-Length","0"))
                if not 0 < size <= 8192:
                    raise ValueError("Request must be 1–8192 bytes")
                payload = json.loads(self.rfile.read(size))
                job = lab.submit(payload)
                self.send_json(202,job)
            except (ValueError, TypeError) as error:
                self.send_json(400,{"error":str(error)})
            except RuntimeError as error:
                self.send_json(409,{"error":str(error)})
    return Handler


def emit(kind, **values):
    print(PREFIX+json.dumps({"type":kind,**values},allow_nan=False),flush=True)


def inference_worker(model_dir, gpu_index):
    """Native upstream generation and codec, isolated from the always-available UI."""
    sys.path.insert(0,str(UPSTREAM))
    import numpy as np
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.llama import DualARTransformer
    from fish_speech.models.text2semantic.inference import (
        decode_one_token_ar, generate_long, load_codec_model, decode_to_audio, encode_audio)
    device = f"cuda:{gpu_index}"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this S2 Pro testing worker")
    torch.cuda.set_device(gpu_index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required for this S2 Pro testing worker")
    precision = torch.bfloat16
    torch.set_float32_matmul_precision("high")
    emit("loading",message="Loading completed fine-tuned weights")
    model = DualARTransformer.from_pretrained(str(model_dir),load_weights=True,max_length=4096)
    model = model.to(device=device,dtype=precision).eval()
    if model.tokenizer is None or model.config.num_codebooks != 10:
        raise ValueError("Invalid S2 Pro tokenizer or codebook configuration")
    emit("loading",message="Loading the original S2 Pro DAC codec")
    codec = load_codec_model(model_dir / "codec.pth",device,precision)
    emit("ready",device=device,precision="BF16",context_tokens=4096,
         gpu_allocated_gb=round(torch.cuda.memory_allocated(gpu_index)/2**30,2))
    references = {}
    for line in sys.stdin:
        item = json.loads(line)
        request = item["request"]
        started = time.monotonic()
        try:
            prompt_tokens, prompt_text = None,None
            reference = item.get("reference")
            if reference:
                key = reference["clip_id"]
                if key not in references:
                    wav = Path(reference["prepared_audio"])
                    dataset = Path(reference["dataset"])
                    receipt = read_json(dataset / "receipts" / f"{key}.json",{})
                    if receipt.get("wav_sha256") != file_hash(wav) or receipt.get("text") != reference["text"]:
                        raise ValueError("Reference audio or transcript integrity check failed")
                    label = Path(reference["prepared_label"])
                    if file_hash(label) != receipt.get("lab_sha256"):
                        raise ValueError("Reference label integrity check failed")
                    tokens = wav.with_suffix(".npy")
                    if tokens.is_file():
                        token_receipt = read_json(dataset / "token-receipts" / f"{key}.json",{})
                        if token_receipt.get("tokens") != file_hash(tokens) or token_receipt.get("wav") != file_hash(wav):
                            raise ValueError("Reference codec-token integrity check failed")
                        pipeline.validate_tokens(tokens,reference['duration'])
                        codes = torch.from_numpy(np.load(tokens,allow_pickle=False)).long().cpu()
                    else:
                        codes = encode_audio(wav,codec,device).cpu()
                    references[key] = codes
                prompt_tokens, prompt_text = [references[key]], [reference["text"]]
            torch.manual_seed(request["seed"])
            torch.cuda.manual_seed_all(request["seed"])
            codes = []
            with torch.inference_mode():
                for response in generate_long(model=model,device=device,decode_one_token=decode_one_token_ar,
                        text="<|speaker:0|>"+request["text"],num_samples=1,
                        max_new_tokens=request["max_new_tokens"],top_p=request["top_p"],
                        top_k=30,temperature=request["temperature"],
                        compile=False,iterative_prompt=True,chunk_length=200,
                        prompt_tokens=prompt_tokens,prompt_text=prompt_text):
                    if response.action == "sample":
                        codes.append(response.codes)
                if not codes:
                    raise RuntimeError("The model did not generate audio tokens")
                tokens = torch.cat(codes,dim=1)
                audio = decode_to_audio(tokens.to(device),codec).float().cpu().numpy()
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 1e-6:
                raise ValueError("Generated audio is empty, silent or nonfinite")
            # Keep playback within PCM bounds without changing the relative waveform.
            audio = audio / max(1.0,float(np.max(np.abs(audio)))/.98)
            output = Path(item["output"])
            output.parent.mkdir(parents=True,exist_ok=True)
            temporary = output.with_suffix(".tmp.wav")
            sf.write(temporary,audio,codec.sample_rate,subtype="PCM_16")
            temporary.replace(output)
            emit("result",id=item["id"],duration=len(audio)/codec.sample_rate,
                 sample_rate=codec.sample_rate,elapsed=round(time.monotonic()-started,2))
        except Exception as error:
            traceback.print_exc()
            emit("job_error",id=item["id"],error=str(error))
            if isinstance(error,torch.cuda.OutOfMemoryError):
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="0.0.0.0")
    parser.add_argument("--port",type=int,default=8002)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--poll-seconds",type=float,default=5)
    parser.add_argument("--check",action="store_true",help="Check discovery/system without serving or loading weights")
    parser.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        inference_worker(args.worker,args.gpu)
        return 0
    if not 1 <= args.port <= 65535 or args.poll_seconds < 1 or args.gpu < 0:
        parser.error("Use a valid port, nonnegative GPU index and polling interval of at least one second")
    if args.run_dir and not within(args.run_dir,RUNS):
        parser.error("--run-dir must select an experiment inside training/talha_s2pro/runs")
    if args.check:
        training = read_json(HERE / "monitor/status.json",{})
        run_dir = find_run(training,args.run_dir)
        print(json.dumps({"run_dir":str(run_dir) if run_dir else None,"training_workers":training_workers(),
                          "system":system_status(args.gpu),"completed_model":candidate_model(run_dir) if run_dir else None},indent=2))
        return 0
    output = HERE / "testing"
    output.mkdir(parents=True,exist_ok=True)
    with (output / "app.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("The testing app is already running; open port 8002",file=sys.stderr)
            return 1
        lab = Lab(output,args.run_dir,args.gpu,args.poll_seconds)
        try:
            server = ThreadingHTTPServer((args.host,args.port),make_handler(lab))
        except OSError:
            lab.logfile.close()
            raise
        server.daemon_threads = True
        (output / "app.pid").write_text(str(os.getpid())+"\n")
        thread = threading.Thread(target=lab.supervise,daemon=True)
        thread.start()
        def stopped(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stopped)
        print(f"Talha listening lab: http://<server-ip>:{args.port}; waiting/loading is automatic",flush=True)
        try:
            server.serve_forever(poll_interval=.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            lab.stop_event.set()
            thread.join(timeout=10)
            lab.stop_worker("Testing app stopped")
            lab.logfile.close()
            (output / "app.pid").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
