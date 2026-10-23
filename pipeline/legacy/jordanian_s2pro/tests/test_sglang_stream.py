"""Incremental PCM delivery, real token markers and bounded stream lifecycle."""
import io
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.request
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import file_hash
from sglang_server import SGLangLab, handler_for
from sglang_stream import StreamHub, sse_bytes
from sglang_telemetry import install_timing_hook
import sglang_worker as worker


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_restart_waits_for_private_port_release_instead_of_using_another(self):
        held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        released = threading.Event()
        def retire():
            time.sleep(.03)
            held.close()
            released.set()
        retiring = threading.Thread(target=retire)
        retiring.start()
        try:
            worker.wait_api_port(port, timeout=1)
            self.assertTrue(released.is_set())
        finally:
            retiring.join(timeout=1)
            held.close()

    def test_first_real_token_recorded_once_without_altering_outputs(self):
        class Runner:
            def _collect_step_outputs(self, result, requests):
                for request in requests:
                    request.data.output_codes.append(result)
                return result
        data = SimpleNamespace(output_codes=[], engine_start_s=time.perf_counter(),
            stage_payload=SimpleNamespace(request_id="native-request", request=SimpleNamespace(
                params={"stage_params": {"tts_engine": {"studio_request_id": "a" * 32}}})))
        install_timing_hook(Runner, self.root)
        runner, request = Runner(), SimpleNamespace(data=data)
        value = object()
        self.assertIs(runner._collect_step_outputs(value, [request]), value)
        marker = self.root / ("a" * 32 + ".json")
        receipt = marker.read_bytes()
        self.assertGreater(json.loads(receipt)["first_token_monotonic"], 0)
        runner._collect_step_outputs(value, [request])
        self.assertEqual(marker.read_bytes(), receipt)
        self.assertEqual(data.output_codes, [value, value])

    def test_untagged_or_invalid_tags_do_not_write_timing(self):
        for tag in (None, "../../outside"):
            class Runner:
                def _collect_step_outputs(self, result, requests):
                    requests[0].data.output_codes.append(result)
            data = SimpleNamespace(output_codes=[], engine_start_s=0,
                stage_payload=SimpleNamespace(request=SimpleNamespace(
                    params={"studio_request_id": tag})))
            install_timing_hook(Runner, self.root)
            Runner()._collect_step_outputs(1, [SimpleNamespace(data=data)])
        self.assertEqual(list(self.root.iterdir()), [])

    def fixture(self):
        ref = self.root / "reference.wav"
        ref.write_bytes(b"verified reference")
        return ({"prepared_audio": str(ref), "wav_sha256": file_hash(ref), "text": "نص التسجيل"},
                {"text": "كيفك؟", "temperature": .7, "top_p": .7, "seed": 42, "max_new_tokens": 1024})

    def test_incremental_delivery_precedes_completed_wav_and_has_true_token_timing(self):
        reference, request = self.fixture()
        audio = (np.sin(np.arange(44100) * .03) * .1 * 32767).astype("<i2")
        pcm = audio.tobytes()
        chunks, tokens = [], []
        output = self.root / "generated.wav"
        timing = self.root / "timings"
        timing.mkdir()
        def opened(message, **kwargs):
            payload = json.loads(message.data)
            self.assertTrue(payload["stream"])
            tag = payload["stage_params"]["tts_engine"]["studio_request_id"]
            (timing / (tag + ".json")).write_text(json.dumps({
                "first_token_monotonic": time.monotonic(),
                "engine_time_to_first_token_seconds": .001}))
            return io.BytesIO(pcm)
        def chunk(data, elapsed):
            self.assertFalse(output.exists())
            self.assertEqual(len(data) % 2, 0)
            chunks.append(data)
        with patch.object(worker.HTTP, "open", side_effect=opened):
            metrics = worker.request_audio(3001, "/model", request, reference, output,
                seed_supported=True, stream=True, on_chunk=chunk, on_token=tokens.append,
                timing_dir=timing)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(b"".join(chunks), pcm)
        self.assertEqual(len(tokens), 1)
        self.assertGreaterEqual(metrics["time_to_first_token_seconds"], 0)
        self.assertLessEqual(metrics["time_to_first_token_seconds"], metrics["time_to_first_audio_seconds"])
        self.assertTrue(output.is_file())
        self.assertEqual(list(timing.iterdir()), [])

    def test_odd_transport_chunks_are_reassembled_and_partial_sample_rejected(self):
        reference, request = self.fixture()
        pcm = (np.sin(np.arange(4410) * .03) * 3000).astype("<i2").tobytes()
        class OddResponse(io.BytesIO):
            def read1(self, size):
                return self.read(min(size, 511))
        for value in (pcm, pcm[:-1]):
            chunks = []
            with patch.object(worker.HTTP, "open", return_value=OddResponse(value)):
                if len(value) % 2:
                    with self.assertRaisesRegex(ValueError, "Incomplete"):
                        worker.request_audio(3001, "/model", request, reference,
                            self.root / "odd.wav", stream=True, on_chunk=lambda x, t: chunks.append(x))
                else:
                    worker.request_audio(3001, "/model", request, reference,
                        self.root / "even.wav", stream=True, on_chunk=lambda x, t: chunks.append(x))
                    self.assertEqual(b"".join(chunks), pcm)
        self.assertFalse((self.root / "odd.wav").exists())

    def test_missing_timing_is_unavailable_instead_of_audio_latency(self):
        reference, request = self.fixture()
        pcm = (np.sin(np.arange(4410) * .03) * 3000).astype("<i2").tobytes()
        with patch.object(worker.HTTP, "open", return_value=io.BytesIO(pcm)):
            metrics = worker.request_audio(3001, "/model", request, reference,
                self.root / "without-marker.wav", stream=True,
                on_chunk=lambda x, t: None, timing_dir=self.root)
        self.assertIsNone(metrics["time_to_first_token_seconds"])
        self.assertIsNotNone(metrics["time_to_first_audio_seconds"])

    def test_slow_clients_and_disconnects_do_not_block_or_grow_buffers(self):
        hub = StreamHub(limit=1, buffer=2)
        channel = hub.open("job")
        with self.assertRaises(RuntimeError):
            hub.open("another")
        for _ in range(3):
            hub.publish("job", {"event": "audio", "pcm": "data"})
        self.assertEqual(channel.get_nowait()["event"], "error")
        self.assertEqual(hub.clients, {})
        channel = hub.open("another")
        hub.fail_all("worker stopped")
        self.assertEqual(channel.get_nowait()["error"], "worker stopped")
        hub.close("another")
        hub.publish("another", {"event": "audio"})

    def test_stream_uses_existing_queue_and_normal_submission_stays_normal(self):
        ref = {"id": "talha_ahmad", "name": "Talha", "duration": 12,
               "text": "نص التسجيل", "wav_sha256": "reference"}
        bundle = {"campaign": "/campaign", "models": {"fast_r32_final": {"checkpoint_sha256": "checkpoint"}},
                  "references": {"talha_ahmad": ref}}
        lab = SGLangLab(bundle, self.root / "voice")
        try:
            payload = {"model": "fast_r32_final", "reference_id": "talha_ahmad", "text": "كيفك؟"}
            normal = lab.submit(payload)
            streaming, channel = lab.submit_stream(payload)
            self.assertNotIn("stream", normal["request"])
            self.assertTrue(streaming["request"]["stream"])
            self.assertEqual(lab.pending.get_nowait()[0], normal["id"])
            self.assertEqual(lab.pending.get_nowait()[0], streaming["id"])
            self.assertEqual(channel.get_nowait()["event"], "accepted")
            lab.route_worker_event(1, {"type": "stream_event", "id": streaming["id"],
                                      "event": "audio", "pcm": "data"})
            self.assertEqual(channel.get_nowait()["event"], "audio")
            self.assertTrue(lab.events.empty())
            lab.route_worker_event(1, {"type": "result", "id": streaming["id"], "duration": 1})
            self.assertEqual(channel.get_nowait()["event"], "complete")
            self.assertEqual(lab.events.get_nowait()[1]["type"], "result")
        finally:
            lab.logfile.close()

    def test_sse_json_cannot_inject_an_event_from_speech_text(self):
        event = {"event": "error", "error": "text\nevent: complete\n"}
        encoded = sse_bytes(event).decode()
        self.assertEqual(encoded.count("\nevent:"), 0)
        self.assertEqual(json.loads(encoded.split("data: ")[1]), event)

    def test_pcm_is_delivered_but_not_duplicated_into_progress_logs(self):
        lab = SGLangLab({"campaign": "/campaign", "models": {}, "references": {}},
                        self.root / "voice")
        events = [{"type": "stream_event", "id": "job", "event": "audio", "pcm": "VOICE_PAYLOAD"},
                  {"type": "stream_event", "id": "job", "event": "token", "time_to_first_token_seconds": .1},
                  {"type": "ready", "models": []}]
        process = SimpleNamespace(pid=1, stdout=[worker.PREFIX + json.dumps(e) + "\n" for e in events])
        channel = lab.streams.open("job")
        try:
            with patch('sglang_server.subprocess.Popen', return_value=process):
                lab.start_worker()
            self.assertEqual(channel.get(timeout=1)["pcm"], "VOICE_PAYLOAD")
            self.assertEqual(channel.get(timeout=1)["event"], "token")
            self.assertEqual(lab.events.get(timeout=1)[1]["type"], "ready")
            log = (lab.output / "server.log").read_text()
            self.assertNotIn("VOICE_PAYLOAD", log)
            self.assertIn("time_to_first_token_seconds", log)
        finally:
            lab.logfile.close()

    def test_http_delivers_audio_before_generation_finishes_and_releases_subscription(self):
        ref = {"id": "talha_ahmad", "name": "Talha", "duration": 12,
               "text": "نص التسجيل", "wav_sha256": "reference"}
        bundle = {"campaign": "/campaign", "models": {"fast_r32_final": {"checkpoint_sha256": "checkpoint"}},
                  "references": {"talha_ahmad": ref}}
        lab = SGLangLab(bundle, self.root / "voice")
        lab.update(status="ready")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(lab))
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        release = threading.Event()
        def generate():
            job_id, _ = lab.pending.get(timeout=2)
            # Match VoiceLab.tick's locked job read before dispatch to worker.
            with lab.lock:
                self.assertTrue(lab.jobs[job_id]["request"]["stream"])
            lab.route_worker_event(1, {"type": "stream_event", "id": job_id, "event": "token",
                                      "time_to_first_token_seconds": .01})
            lab.route_worker_event(1, {"type": "stream_event", "id": job_id, "event": "audio",
                                      "pcm": "AAAA", "sample_rate": 44100})
            release.wait(2)
            lab.route_worker_event(1, {"type": "result", "id": job_id, "duration": 1})
        generating = threading.Thread(target=generate, daemon=True)
        serving.start()
        generating.start()
        try:
            payload = {"model": "fast_r32_final", "reference_id": "talha_ahmad", "text": "كيفك؟"}
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/stream",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertTrue(response.headers["Content-Type"].startswith("text/event-stream"))
                events = []
                while "audio" not in events:
                    line = response.readline().decode()
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:])["event"])
                self.assertEqual(events, ["accepted", "token", "audio"])
                self.assertFalse(release.is_set())
                release.set()
                self.assertIn(b'"event": "complete"', response.read())
            deadline = time.monotonic() + 1
            while lab.streams.clients and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(lab.streams.clients, {})
        finally:
            release.set()
            generating.join(timeout=2)
            server.shutdown()
            server.server_close()
            serving.join(timeout=2)
            lab.logfile.close()


if __name__ == "__main__":
    unittest.main()
