"""Existing voice studio, with the SGLang worker and a paired listening page."""
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from common import HERE, atomic_json, read_json
from sglang_stream import StreamHub, sse_bytes
import voice_server as original

OriginalLab = original.VoiceLab
original_handler = original.make_handler


class SGLangLab(OriginalLab):
    def __init__(self, *args, **kwargs):
        self.streams = StreamHub()
        # Original main acquires the voice lock and stops the old worker first.
        source = os.environ.get("JORDANIAN_SGLANG_MANIFEST")
        if source:
            path = Path(source).resolve()
            if not path.is_relative_to(HERE / "sglang/preflight"):
                raise ValueError("Unexpected SGLang manifest location")
            atomic_json(HERE / "sglang/manifest.json", read_json(path))
        super().__init__(*args, **kwargs)

    def submit_stream(self, payload):
        with self.lock:
            # The existing dispatcher also acquires this lock before reading
            # a queued job, so its stream flag/subscription are ready first.
            if len(self.streams.clients) >= self.streams.limit:
                raise RuntimeError("Too many open streams; wait for one to finish")
            job = super().submit(payload)
            saved = self.jobs[job["id"]]
            saved["request"]["stream"] = True
            saved["streaming"] = True
            channel = self.streams.open(job["id"])
            atomic_json(self.output / "results" / (job["id"] + ".json"), saved)
            self.streams.publish(job["id"], {"event": "accepted", "id": job["id"],
                "sample_rate": 44100, "channels": 1, "encoding": "pcm_s16le"})
            return dict(saved), channel

    def stream_references(self):
        with self.lock:
            refs = []
            for ref in self.references.values():
                public = self.public_reference(ref)
                public["uploaded"] = ref.get("uploaded", False)
                if public["uploaded"]:
                    public["name"] = "My voice · " + ref["id"][-6:]
                    for job in reversed(list(self.jobs.values())):
                        request = job.get("request", {})
                        if (request.get("reference_id") == ref["id"] and
                                job.get("reference_sha256") == ref["wav_sha256"]):
                            public["text"] = request.get("reference_text", "")
                            break
                refs.append(public)
            return refs

    def route_worker_event(self, pid, event):
        kind, job_id = event.get("type"), event.get("id")
        if kind == "stream_event":
            if event["event"] == "started":
                with self.lock:
                    job = self.jobs.get(job_id, {})
                    event["queue_seconds"] = max(0, job.get("started_at", time.time()) -
                                                 job.get("created_at", time.time()))
            self.streams.publish(job_id, event)
            return
        if kind == "result":
            self.streams.publish(job_id, {**event, "event": "complete",
                "audio_url": "/api/audio/" + job_id + ".wav",
                "leveled_audio_url": "/api/leveled-audio/" + job_id + ".wav"})
        elif kind == "job_error":
            self.streams.publish(job_id, {"event": "error", "error": event["error"]})
        self.events.put((pid, event))

    def fail_pending(self, reason):
        self.streams.fail_all(reason)
        return super().fail_pending(reason)

    def start_worker(self):
        command = [sys.executable, str(HERE / "sglang_worker.py"),
                   "--selection", str(self.output / "selection.json"), "--gpu", str(self.gpu)]
        if os.environ.get("JORDANIAN_SGLANG_SKIP_BENCHMARK") == "1":
            command.append("--skip-benchmark")
        self.worker = subprocess.Popen(command, cwd=HERE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, start_new_session=True)
        self.worker_started = time.monotonic()
        self.update(status="loading", message="Starting SGLang and checking voice cloning", loaded_models=[])
        worker = self.worker
        def reader():
            for line in worker.stdout:
                if line.startswith(original.PREFIX):
                    try:
                        event = json.loads(line[len(original.PREFIX):])
                    except ValueError:
                        self.log("Invalid SGLang worker event")
                        continue
                    if not (event.get("type") == "stream_event" and event.get("event") == "audio"):
                        self.log(line)
                    self.route_worker_event(worker.pid, event)
                else:
                    self.log(line)
        threading.Thread(target=reader, daemon=True).start()

    def tick(self):
        # First startup includes two graph/compile warmups and paired samples.
        timeout = self.timeout
        if self.state["status"] == "loading":
            self.timeout = 10800
        try:
            super().tick()
        finally:
            self.timeout = timeout

    def stop_worker(self):
        self.streams.fail_all("Speech worker stopped; retry the stream when ready")
        worker, self.worker = self.worker, None
        if worker:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            if worker.stdin:
                worker.stdin.close()
            # If the worker needed SIGKILL, its finally block may not have run.
            # Remove only containers labelled with this worker PID and this suite.
            try:
                found = subprocess.check_output(["docker", "ps", "-aq", "--filter",
                    f"label=jordanian.s2pro.owner={worker.pid}", "--filter",
                    f"label=jordanian.s2pro.suite={HERE}"], text=True, timeout=5).split()
                if found and len(found) <= 2 and all(re.fullmatch(r"[a-f0-9]{12,64}", x) for x in found):
                    subprocess.run(["docker", "rm", "--force", *found], check=True,
                                   capture_output=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as error:
                self.log(f"Could not finish SGLang container cleanup: {error}")

    def snapshot(self):
        return super().snapshot() | {"backend": "sglang-omni", "benchmark_url": "/sglang-compare"}


def handler_for(lab):
    parent = original_handler(lab)
    class Handler(parent):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                html = (HERE / "voice_ui.html").read_text()
                html = html.replace("Your two selected models. A familiar voice, or your own.",
                                    "SGLang · Your two selected models. A familiar voice, or your own.")
                html = html.replace("</header>", '<a href="/sglang-compare" target="_blank" rel="noopener">SGLang comparison ↗</a></header>')
                html = html.replace("</header>", '<a href="/stream" target="_blank" rel="noopener">Streaming studio ↗</a></header>')
                html = html.replace("The defaults match Listen and compare. Keep the same seed when comparing models.",
                    "SGLang sampling can change the sound. Seed support depends on the engine API; results show a warning when unavailable.")
                html = html.replace("</details>", '</details><label><input id="steady-volume" type="checkbox" checked style="width:auto;margin-right:8px"> Keep ending volume steady</label><p class="muted">Gently lifts quieter final words. Uncheck to hear the original audio.</p>')
                html = html.replace("audio.src=job.audio_url;",
                    "audio.src=$('steady-volume').checked?job.audio_url.replace('/api/audio/','/api/leveled-audio/'):job.audio_url;")
                html = html.replace("link.href=job.audio_url;", "link.href=audio.src;")
                html = html.replace("row.append(link,element('span',",
                    "const raw=element('a','Original WAV');raw.href=job.audio_url;raw.download=job.id+'_original.wav';row.append(raw);row.append(link,element('span',")
                html = html.replace("countText();refresh();", "$('steady-volume').onchange=()=>{resultsSignature='';if(state)renderResults(state.jobs||[])};countText();refresh();")
                return self.send(200, html.encode(), "text/html; charset=utf-8")
            if path == "/stream":
                return self.file(HERE / "sglang_stream.html", "text/html; charset=utf-8")
            if path == "/api/stream/references":
                return self.send(200, {"references": lab.stream_references()})
            match = re.fullmatch(r"/api/leveled-audio/([a-f0-9]{32})\.wav", path)
            if match:
                source = lab.output / "results" / (match[1] + ".wav")
                if not source.is_file():
                    return self.send(404, {"error": "Audio is unavailable"})
                from audio_finish import leveled_wav
                try:
                    return self.file(leveled_wav(source, lab.output / "leveled"))
                except (ValueError, OSError) as error:
                    lab.log(f"Could not level ending volume: {error}")
                    return self.send(500, {"error": "Could not prepare leveled audio; the original WAV is available"})
            if path == "/sglang-compare":
                return self.file(HERE / "sglang_compare.html", "text/html; charset=utf-8")
            if path == "/api/sglang-comparison":
                report = read_json(lab.output / "sglang-benchmark.json", {"rows": []})
                report["gpu_memory"] = read_json(lab.output / "sglang-gpu-memory.json", {})
                return self.send(200, report)
            match = re.fullmatch(r"/api/sglang-audio/([a-f0-9]{32})\.wav", path)
            if match:
                return self.file(lab.output / "sglang-benchmark" / (match[1] + ".wav"))
            return super().do_GET()

        def do_POST(self):
            if urlsplit(self.path).path != "/api/stream":
                return super().do_POST()
            try:
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    return self.send(403, {"error": "Use this server's page"})
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json" or not 0 < length <= 16384:
                    raise ValueError("Send JSON smaller than 16 KB")
                payload = json.loads(self.rfile.read(length))
                job, channel = lab.submit_stream(payload)
            except (ValueError, TypeError, KeyError) as error:
                return self.send(400, {"error": str(error)})
            except RuntimeError as error:
                return self.send(409, {"error": str(error)})
            except OSError as error:
                lab.log(f"Could not submit stream: {error}")
                return self.send(500, {"error": "Could not save the stream request"})
            deadline = time.monotonic() + 1050
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.connection.settimeout(15)
                while time.monotonic() < deadline:
                    try:
                        event = channel.get(timeout=5)
                    except queue.Empty:
                        self.wfile.write(b": waiting\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(sse_bytes(event))
                    self.wfile.flush()
                    if event["event"] in ("complete", "error"):
                        break
                else:
                    self.wfile.write(sse_bytes({"event": "error", "error": "Stream timed out; check the studio for the completed WAV"}))
                    self.wfile.flush()
            except (OSError, TimeoutError):
                pass
            finally:
                lab.streams.close(job["id"])
    return Handler


def main():
    original.VoiceLab = SGLangLab
    original.make_handler = handler_for
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())

