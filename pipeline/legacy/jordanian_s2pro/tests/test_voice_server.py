"""Selected checkpoint identity, real reference uploads, HTTP and serial job lifecycle."""
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voice_server as app
from common import atomic_json, file_hash, write_rows


class VoiceServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / "suites/campaign"
        self.comparison = self.campaign / "comparison"
        self.comparison.mkdir(parents=True)
        self.here = patch.object(app, "HERE", self.root)
        self.here.start()
        self.models = []
        for name in (*app.SELECTED, "attention_mlp_r16_final"):
            run = self.campaign / "runs" / name
            run.mkdir(parents=True)
            checkpoint = run / ("early.ckpt" if name.endswith("early") else "final.ckpt")
            checkpoint.write_bytes(name.encode())
            config = run / "config.yaml"
            config.write_text("configuration: test\n")
            self.models.append({"name": name, "status": "complete", "checkpoint": str(checkpoint),
                                "config": str(config), "checkpoint_sha256": file_hash(checkpoint)})
        atomic_json(self.comparison / "models.json", self.models)
        atomic_json(self.comparison / "base_identity.json", {"codec.pth": "test"})
        cases = []
        for speaker in app.SPEAKERS:
            dataset = self.root / "cache/data"
            wav = dataset / "audio/train" / speaker / "reference.wav"
            wav.parent.mkdir(parents=True)
            sf.write(wav, np.sin(np.arange(44100 * 4) * .04) * .1, 44100)
            ref = {"clip_id": speaker + "_clip", "text": "هذه الكلمات في التسجيل",
                   "duration": 4, "prepared_audio": str(wav)}
            atomic_json(dataset / "receipts" / (ref["clip_id"] + ".json"),
                        {"wav_sha256": file_hash(wav), "text": ref["text"]})
            cases.append({"speaker": speaker, "reference": ref, "id": speaker + "_0"})
        write_rows(self.comparison / "cases.jsonl", cases)
        self.bundle = app.selected_bundle(self.campaign)
        self.lab = app.VoiceLab(self.bundle, self.root / "voice")

    def tearDown(self):
        self.lab.worker = None
        self.lab.logfile.close()
        self.here.stop()
        self.temp.cleanup()

    def request(self, **values):
        return {"model": "attention_mlp_r16_early", "reference_id": "wesam_ali",
                "text": "كيفك اليوم؟", **values}

    def wav_bytes(self, seconds=4, silence=False):
        buffer = io.BytesIO()
        audio = np.zeros(16000 * seconds) if silence else np.sin(np.arange(16000 * seconds) * .04) * .1
        sf.write(buffer, audio, 16000, format="WAV")
        return buffer.getvalue()

    def handler(self, path, method="GET", data=b"", headers=None):
        handler = app.make_handler(self.lab).__new__(app.make_handler(self.lab))
        handler.path, handler.command = path, method
        handler.request_version, handler.requestline = "HTTP/1.1", method + " " + path
        handler.headers = {"Content-Length": str(len(data)), "Host": "localhost:7860", **(headers or {})}
        handler.rfile, handler.wfile = io.BytesIO(data), io.BytesIO()
        getattr(handler, "do_" + method)()
        head, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        return int(head.splitlines()[0].split()[1]), body

    def ready(self):
        self.lab.worker = MagicMock()
        self.lab.worker.pid = 123
        self.lab.worker.poll.return_value = None
        self.lab.update(status="ready")

    def test_discovers_exact_two_selected_checkpoints_and_original_references(self):
        self.assertEqual(tuple(self.bundle["models"]), app.SELECTED)
        self.assertTrue(self.bundle["models"]["attention_mlp_r16_early"]["checkpoint"].endswith("early.ckpt"))
        self.assertEqual(set(self.bundle["references"]), set(app.SPEAKERS))
        self.assertEqual(self.bundle["references"]["wesam_ali"]["text"], "هذه الكلمات في التسجيل")

    def test_rejects_changed_checkpoint_and_reference(self):
        path = Path(self.models[0]["checkpoint"])
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            app.selected_bundle(self.campaign)
        path.write_bytes(original)
        Path(self.bundle["references"]["wesam_ali"]["prepared_audio"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "reference integrity"):
            app.selected_bundle(self.campaign)

    def test_audio_upload_resamples_and_requires_exact_transcript(self):
        ref = self.lab.upload(self.wav_bytes())
        stored = self.lab.references[ref["id"]]
        audio, rate = sf.read(stored["prepared_audio"])
        self.assertEqual(rate, 44100)
        self.assertEqual(audio.ndim, 1)
        self.assertAlmostEqual(ref["duration"], 4)
        with self.assertRaisesRegex(ValueError, "transcript"):
            self.lab.submit(self.request(reference_id=ref["id"]))
        job = self.lab.submit(self.request(reference_id=ref["id"], reference_text="أنا بحكي بهالتسجيل"))
        self.assertEqual(job["request"]["reference_text"], "أنا بحكي بهالتسجيل")
        self.assertEqual(job["reference_sha256"], file_hash(stored["prepared_audio"]))

    def test_short_long_silent_and_invalid_uploads_are_rejected(self):
        for data, message in ((self.wav_bytes(2), "3 and 30"), (self.wav_bytes(31), "3 and 30"),
                              (self.wav_bytes(silence=True), "silent"), (b"not audio", "decode")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    app.prepare_upload(data, self.lab.output, "a" * 32)
                self.assertFalse(list((self.lab.output / "uploads").glob("*.tmp.wav")))
                self.assertFalse(list((self.lab.output / "uploads").glob("*.input")))

    def test_common_upload_formats_and_browser_recording_decode(self):
        source = self.root / "source.wav"
        source.write_bytes(self.wav_bytes())
        for extension, codec in (("mp3", "libmp3lame"), ("m4a", "aac"), ("webm", "libopus"),
                                 ("flac", "flac"), ("ogg", "libvorbis")):
            with self.subTest(extension=extension):
                encoded = self.root / ("reference." + extension)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:a", codec,
                                str(encoded)], capture_output=True, check=True)
                reference = self.lab.upload(encoded.read_bytes())
                self.assertAlmostEqual(reference["duration"], 4, delta=.15)

    def test_invalid_models_tags_and_generation_settings_are_rejected(self):
        for values in ({"model": "attention_mlp_r16_final"}, {"text": "<|speaker:1|>"},
                       {"seed": True}, {"seed": 1.5}, {"temperature": float("nan")},
                       {"reference_id": "../../etc/passwd"}, {"max_new_tokens": 5000}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    app.validate_request(self.request(**values), self.bundle["models"], self.lab.references)
        request = app.validate_request(self.request(reference_text="override"), self.bundle["models"], self.lab.references)
        self.assertEqual(request["reference_text"], self.bundle["references"]["wesam_ali"]["text"])

    def test_serial_queue_result_and_worker_failure(self):
        self.ready()
        first = self.lab.submit(self.request())
        second = self.lab.submit(self.request(model="fast_r32_final"))
        self.lab.tick()
        self.assertEqual(self.lab.active, first["id"])
        sent = json.loads(self.lab.worker.stdin.write.call_args.args[0])
        self.assertEqual(sent["request"]["model"], "attention_mlp_r16_early")
        self.assertEqual(sent["reference"]["text"], self.bundle["references"]["wesam_ali"]["text"])
        self.lab.events.put((123, {"type": "result", "id": first["id"], "duration": 5, "elapsed": 10, "sample_rate": 44100}))
        self.lab.tick()
        self.assertEqual(self.lab.jobs[first["id"]]["status"], "complete")
        self.assertEqual(self.lab.active, second["id"])
        self.lab.worker.poll.return_value = 1
        self.lab.worker.returncode = 1
        self.lab.tick()
        self.assertEqual(self.lab.state["status"], "error")
        self.assertEqual(self.lab.jobs[second["id"]]["status"], "failed")
        self.assertEqual(self.lab.pending.qsize(), 0)

    def test_queue_capacity_and_restart_recovery(self):
        for _ in range(8):
            self.lab.submit(self.request())
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            self.lab.submit(self.request())
        self.lab.logfile.close()
        restored = app.VoiceLab(self.bundle, self.root / "voice")
        try:
            self.assertTrue(all(job["status"] == "failed" for job in restored.jobs.values()))
        finally:
            restored.logfile.close()

    def test_http_upload_generation_status_and_real_wav_delivery(self):
        code, raw = self.handler("/api/upload", "POST", self.wav_bytes(), {"Content-Type": "application/octet-stream"})
        self.assertEqual(code, 201)
        ref = json.loads(raw)
        code, wav = self.handler(ref["audio_url"])
        self.assertEqual(code, 200)
        audio, rate = sf.read(io.BytesIO(wav))
        self.assertEqual(rate, 44100)
        self.assertEqual(len(audio), 4 * 44100)
        request = self.request(reference_id=ref["id"], reference_text="النص الموجود في التسجيل")
        code, raw = self.handler("/api/generate", "POST", json.dumps(request).encode(), {"Content-Type": "application/json"})
        self.assertEqual(code, 202)
        self.assertEqual(json.loads(raw)["request"]["reference_id"], ref["id"])
        code, raw = self.handler("/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(len(json.loads(raw)["models"]), 2)
        code, _ = self.handler("/api/reference/../../etc/passwd.wav")
        self.assertEqual(code, 404)
        code, _ = self.handler("/api/upload", "POST", b"x", {"Origin": "https://elsewhere.invalid"})
        self.assertEqual(code, 403)

    def test_finished_dashboard_replacement_refuses_active_training(self):
        atomic_json(self.root / "monitor/status.json", {"status": "running"})
        with self.assertRaisesRegex(RuntimeError, "has not finished"):
            app.stop_finished_dashboard()

    def test_idle_voice_replacement_verifies_pid_and_refuses_unfinished_jobs(self):
        proc_root = self.root / "proc"
        process = proc_root / "9876"
        process.mkdir(parents=True)
        (process / "cwd").symlink_to(self.root)
        (process / "cmdline").write_bytes(("python\0" + str(self.root / "voice_server.py") + "\0").encode())
        (self.lab.output / "server.pid").write_text("9876\n")
        atomic_json(self.lab.output / "status.json", {"status": "ready", "jobs": []})
        with patch.object(app.os, "kill") as kill:
            self.assertEqual(app.stop_idle_voice_server(self.lab.output, proc_root), 9876)
            kill.assert_called_once_with(9876, app.signal.SIGTERM)
        (process / "cmdline").write_bytes(("python\0" + str(self.root / "sglang_server.py") + "\0").encode())
        with patch.object(app.os, "kill") as kill:
            self.assertEqual(app.stop_idle_voice_server(self.lab.output, proc_root), 9876)
            kill.assert_called_once_with(9876, app.signal.SIGTERM)
        for state in ({"status": "generating"}, {"status": "ready", "jobs": [{"status": "queued"}]}):
            atomic_json(self.lab.output / "status.json", state)
            with patch.object(app.os, "kill") as kill, self.assertRaisesRegex(RuntimeError, "unfinished"):
                app.stop_idle_voice_server(self.lab.output, proc_root)
            kill.assert_not_called()
        atomic_json(self.lab.output / "status.json", {"status": "ready"})
        (process / "cmdline").write_bytes(b"python\0/other/app/voice_server.py\0")
        with patch.object(app.os, "kill") as kill, self.assertRaisesRegex(RuntimeError, "verify"):
            app.stop_idle_voice_server(self.lab.output, proc_root)
        kill.assert_not_called()

    def test_dashboard_replacement_matches_port_and_skips_workers(self):
        atomic_json(self.root / "monitor/status.json", {"status": "complete"})
        proc = self.root / "proc/9876"
        proc.mkdir(parents=True)
        (proc / "cwd").symlink_to(self.root)
        command = proc / "cmdline"
        command.write_bytes("\0".join(["python", str(self.root / "run.py"), "--port", "7861", ""]).encode())
        with patch.object(Path, "glob", return_value=[command]), patch.object(app.os, "kill") as kill:
            self.assertEqual(app.stop_finished_dashboard(7860), [])
            kill.assert_not_called()
            self.assertEqual(app.stop_finished_dashboard(7861), [9876])
            kill.assert_called_once_with(9876, app.signal.SIGINT)
        command.write_bytes(("python\0" + str(self.root / "run.py") + "\0--worker=config.yaml\0").encode())
        with patch.object(Path, "glob", return_value=[command]), patch.object(app.os, "kill") as kill:
            self.assertEqual(app.stop_finished_dashboard(), [])
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
