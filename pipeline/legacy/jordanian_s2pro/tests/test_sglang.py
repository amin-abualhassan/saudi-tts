"""Real adapter merge/logit parity, export integrity and bounded HTTP audio checks."""
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import UPSTREAM, atomic_json, file_hash, write_rows
sys.path.insert(0, str(UPSTREAM))
from fish_speech.models.text2semantic.llama import DualARModelArgs, DualARTransformer
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora
import sglang_export as exporter
import sglang_worker as worker
import sglang_launch as launcher
import voice_server
import sglang_server


def tiny_model():
    return DualARTransformer(DualARModelArgs(vocab_size=64, n_layer=1, n_head=2,
        n_local_heads=2, head_dim=16, dim=32, intermediate_size=64, max_seq_len=32,
        codebook_size=16, num_codebooks=10, semantic_begin_id=16, semantic_end_id=31,
        n_fast_layer=1, fast_dim=32, fast_n_head=2, fast_n_local_heads=2,
        fast_head_dim=16, fast_intermediate_size=64))


class SGLangTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_loralib_bf16_weights_and_slow_fast_logits_match_export_math(self):
        for targets in (["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"],
                        ["attention", "mlp"]):
            with self.subTest(targets=targets):
                torch.manual_seed(17)
                model = tiny_model()
                setup_lora(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets))
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if "lora_" in name:
                            p.normal_(std=.1)
                model.to(dtype=torch.bfloat16)
                state = {n: p.clone() for n, p in model.state_dict().items() if "lora_" not in n}
                count = 0
                for name, module in model.named_modules():
                    if hasattr(module, "lora_A"):
                        state[name + ".weight"] = exporter.merge_weight(module.weight,
                            module.lora_A, module.lora_B, 4, isinstance(module, torch.nn.Embedding))
                        count += 1
                self.assertGreater(count, 0)
                model.eval()
                for name, p in model.state_dict().items():
                    if "lora_" not in name:
                        torch.testing.assert_close(p, state[name], atol=0, rtol=0)
                merged = tiny_model().to(torch.bfloat16).eval()
                merged.load_state_dict(state, strict=True)
                tokens = torch.randint(0, 16, (1, 11, 8))
                tokens[:, 0] += 16
                with torch.no_grad():
                    a, b = model(tokens, labels=tokens), merged(tokens, labels=tokens)
                torch.testing.assert_close(a.token_logits, b.token_logits, atol=0, rtol=0)
                torch.testing.assert_close(a.codebook_logits, b.codebook_logits, atol=0, rtol=0)

    def test_native_mapping_covers_fast_embedding_output_and_shared_codebooks(self):
        self.assertEqual(exporter.native_key("fast_embeddings.weight"), "audio_decoder.embeddings.weight")
        self.assertEqual(exporter.native_key("fast_output.weight"), "audio_decoder.output.weight")
        self.assertEqual(exporter.native_key("codebook_embeddings.weight"), "audio_decoder.codebook_embeddings.weight")
        self.assertEqual(exporter.native_key("layers.0.attention.wo.weight"), "text_model.model.layers.0.attention.wo.weight")

    def test_atomic_sharded_export_reuse_and_tamper_rejection(self):
        base = self.root / "base"
        base.mkdir()
        key = "audio_decoder.embeddings.weight"
        weight = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
        save_file({key: weight}, str(base / "model.safetensors"))
        (base / "codec.pth").write_bytes(b"original codec")
        atomic_json(base / "config.json", {"model_type": "fish_qwen3_omni"})
        atomic_json(base / "model.safetensors.index.json", {"weight_map": {key: "model.safetensors"}})
        source = {p.name: file_hash(p) for p in base.iterdir()}
        a, b = torch.ones(2, 6), torch.ones(4, 2)
        entry = {"name": "fast_r32_final", "checkpoint_sha256": "checkpoint", "config_sha256": "config"}
        with patch.object(exporter, "load_adapters", return_value=(
                {"fast_embeddings.lora_A": a, "fast_embeddings.lora_B": b},
                {"lora_alpha": 2}, 1500)) as load:
            destination = Path(exporter.export_model(entry, source, self.root / "exports", base))
            expected = exporter.merge_weight(weight, a, b, 2, True)
            torch.testing.assert_close(load_file(str(destination / "model.safetensors"))[key], expected)
            self.assertTrue((destination / "codec.pth").is_symlink())
            self.assertEqual(json.loads((destination / "config.json").read_text())["model_type"], "fish_qwen3_omni")
            self.assertEqual(str(destination), exporter.export_model(entry, source, self.root / "exports", base))
            self.assertEqual(load.call_count, 1)
            (destination / "config.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed export"):
                exporter.export_model(entry, source, self.root / "exports", base)

    def test_invalid_delta_shape_and_nonfinite_are_rejected(self):
        for a, b in ((torch.ones(2, 5), torch.ones(4, 2)),
                     (torch.full((2, 4), float("nan")), torch.ones(4, 2))):
            with self.assertRaisesRegex(ValueError, "Invalid adapter"):
                exporter.merge_weight(torch.zeros(4, 4), a, b, 2)

    def audio_call(self, data, stream=False):
        reference = self.root / "reference.wav"
        reference.write_bytes(b"verified reference")
        ref = {"prepared_audio": str(reference), "wav_sha256": file_hash(reference), "text": "نص التسجيل"}
        request = {"text": "كيفك؟", "temperature": .7, "top_p": .7, "seed": 42, "max_new_tokens": 1024}
        class Response(io.BytesIO):
            pass
        with patch.object(worker.HTTP, "open", return_value=Response(data)) as opened:
            result = worker.request_audio(3001, "/model", request, ref,
                self.root / "generated.wav", seed_supported=True, stream=stream)
        return result, json.loads(opened.call_args.args[0].data)

    def test_wav_and_streamed_pcm_cloning_requests_and_metrics(self):
        audio = (np.sin(np.arange(44100) * .02) * .1).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, 44100, format="WAV", subtype="PCM_16")
        for stream, data in ((False, buf.getvalue()), (True, (audio * 32767).astype("<i2").tobytes())):
            with self.subTest(stream=stream):
                metrics, payload = self.audio_call(data, stream)
                self.assertAlmostEqual(metrics["duration"], 1)
                self.assertEqual(payload["seed"], 42)
                self.assertEqual(payload["references"][0]["text"], "نص التسجيل")
                self.assertEqual(payload["response_format"], "pcm" if stream else "wav")
                self.assertEqual(payload["max_new_tokens"], 1024)
                self.assertEqual(metrics["streamed_pcm"], stream)
                self.assertEqual(metrics["rtf"], metrics["elapsed"])
                self.assertEqual(metrics["wav_sha256"], file_hash(self.root / "generated.wav"))

    def test_silent_wrong_rate_and_oversized_responses_do_not_write_audio(self):
        for rate in (44100, 24000):
            buf = io.BytesIO()
            sf.write(buf, np.zeros(rate), rate, format="WAV")
            with self.assertRaises(ValueError):
                self.audio_call(buf.getvalue())
        with patch.object(worker, "MAX_AUDIO", 8), self.assertRaisesRegex(ValueError, "too much"):
            self.audio_call(b"too many response bytes")
        self.assertFalse((self.root / "generated.wav").exists())

    def test_openapi_reference_resolution_detects_seed(self):
        schema = {"paths": {"/v1/audio/speech": {"post": {"requestBody": {"content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/Speech"}}}}}}},
            "components": {"schemas": {"Speech": {"properties": {"seed": {}, "references": {}}}}}}
        self.assertEqual(worker.speech_fields(schema), {"seed", "references"})

    def test_raw_request_openapi_uses_installed_protocol_fields(self):
        schema = {"paths": {"/v1/audio/speech": {"post": {}}}}
        self.assertEqual(worker.speech_fields(schema, ["seed", "references", "input"]),
                         {"seed", "references", "input"})
        self.assertEqual(worker.speech_fields(schema), set())

    def test_preflight_failure_cannot_exec_or_stop_existing_server(self):
        with patch.object(sys, "argv", ["sglang_launch.py"]), \
                patch.object(launcher, "HERE", self.root), \
                patch.object(launcher, "selected_bundle", return_value={}), \
                patch.object(launcher, "ensure_image", side_effect=RuntimeError("Docker unavailable")), \
                patch.object(launcher.os, "execv") as execute, \
                patch.object(voice_server.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                launcher.main()
            execute.assert_not_called()
            kill.assert_not_called()

    def test_comparison_handler_serves_real_wav_and_rejects_traversal(self):
        bundle = {"campaign": str(self.root), "models": {}, "references": {}, "base": {}}
        lab = voice_server.VoiceLab(bundle, self.root / "voice")
        try:
            audio_id = "a" * 32
            directory = lab.output / "sglang-benchmark"
            directory.mkdir()
            sf.write(directory / (audio_id + ".wav"), np.ones(4410) * .1, 44100)
            atomic_json(lab.output / "sglang-benchmark.json", {"rows": [{"audio_id": audio_id}]})
            def get(path):
                handler_cls = sglang_server.handler_for(lab)
                handler = handler_cls.__new__(handler_cls)
                handler.path, handler.command = path, "GET"
                handler.request_version, handler.requestline = "HTTP/1.1", "GET " + path
                handler.headers = {"Host": "localhost:3000"}
                handler.rfile, handler.wfile = io.BytesIO(), io.BytesIO()
                handler.do_GET()
                head, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
                return int(head.splitlines()[0].split()[1]), body
            code, body = get("/api/sglang-audio/" + audio_id + ".wav")
            self.assertEqual(code, 200)
            self.assertEqual(sf.read(io.BytesIO(body))[1], 44100)
            self.assertEqual(get("/api/sglang-audio/../../etc/passwd.wav")[0], 404)
            raw = lab.output / "results" / (audio_id + ".wav")
            raw.parent.mkdir(parents=True, exist_ok=True)
            sf.write(raw, np.ones(4410) * .1, 44100)
            raw_hash = file_hash(raw)
            code, body = get("/api/leveled-audio/" + audio_id + ".wav")
            self.assertEqual(code, 200)
            self.assertEqual(sf.read(io.BytesIO(body))[1], 44100)
            self.assertEqual(file_hash(raw), raw_hash)
            self.assertEqual(get("/api/leveled-audio/../../etc/passwd.wav")[0], 404)
            self.assertEqual(get("/api/leveled-audio/" + "b" * 32 + ".wav")[0], 404)
            code, body = get("/")
            self.assertEqual(code, 200)
            self.assertIn(b"/sglang-compare", body)
            self.assertIn(b'id="steady-volume"', body)
            self.assertIn(b"Original WAV", body)
            self.assertEqual(json.loads(get("/api/sglang-comparison")[1])["rows"][0]["audio_id"], audio_id)
        finally:
            lab.logfile.close()

    def test_benchmark_reuses_verified_audio_across_new_preflight_paths(self):
        name = "fast_r32_final"
        speaker = "wesam_ali"
        comparison = self.root / "campaign/comparison"
        write_rows(comparison / "cases.jsonl", [{"id": speaker + "_0", "speaker": speaker,
            "text": "كيفك؟", "generation": {"seed": 42, "temperature": .7,
                "top_p": .7, "max_new_tokens": 1024}}])
        write_rows(comparison / name / "results.jsonl", [])
        bundle = {"campaign": str(comparison.parent), "models": {name: {}},
            "references": {speaker: {"text": "نص التسجيل", "name": "Wesam", "wav_sha256": "ref"}}}
        engine = MagicMock()
        engine.name = name
        engine.seed_supported = True
        engine.manifest = {"image": "sha256:fixture", "exports": {name: "/model"},
            "checkpoint_hashes": {name: "checkpoint"}, "config_dir": "/first/preflight",
            "preflight": {"settings": {"max_running_requests": 1}}}
        def generate(request, ref, wav, stream=False):
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, np.ones(4410) * .1, 44100)
            return {"duration": .1, "elapsed": .2, "wav_sha256": file_hash(wav), "streamed_pcm": stream}
        engine.generate.side_effect = generate
        output = self.root / "voice"
        with patch.object(worker, "emit"):
            worker.benchmark(engine, bundle, output, startup=10)
            self.assertEqual(engine.generate.call_count, 4)
            engine.generate.reset_mock()
            engine.manifest["config_dir"] = "/second/preflight"
            worker.benchmark(engine, bundle, output, startup=5)
            self.assertEqual(engine.generate.call_count, 3)  # Fresh warm/stream probes only.
            report = json.loads((output / "sglang-benchmark.json").read_text())
            cached = output / "sglang-benchmark" / (report["rows"][0]["audio_id"] + ".wav")
            cached.write_bytes(b"changed")
            engine.generate.reset_mock()
            worker.benchmark(engine, bundle, output, startup=5)
            self.assertEqual(engine.generate.call_count, 4)

    def test_real_uv_override_resolves_the_reported_protobuf_conflict_offline(self):
        uv = Path(sys.executable).with_name("uv")
        if not uv.is_file():
            self.skipTest("uv is needed for the offline dependency-resolution regression")
        wheels = self.root / "wheels"
        wheels.mkdir()
        def wheel(name, version, requirement=None):
            normalized = name.replace("-", "_")
            info = f"{normalized}-{version}.dist-info"
            metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            if requirement:
                metadata += f"Requires-Dist: {requirement}\n"
            with zipfile.ZipFile(wheels / f"{normalized}-{version}-py3-none-any.whl", "w") as archive:
                archive.writestr(info + "/METADATA", metadata + "\n")
                archive.writestr(info + "/WHEEL", "Wheel-Version: 1.0\nGenerator: regression-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                archive.writestr(info + "/RECORD", "")
        # Minimal wheel metadata isolates the exact real-world conflict. No fixture
        # packages are installed and no network or production environment is used.
        wheel("descript-audiotools", "0.7.2", "protobuf>=3.9.2,<3.20")
        wheel("protobuf", "3.19.6")
        wheel("protobuf", "6.31.1")
        requirements = self.root / "requirements.in"
        requirements.write_text("descript-audiotools==0.7.2\nprotobuf>=6.31.1,<7.0.0\n")
        command = [str(uv), "pip", "compile", "--offline", "--no-index", "--no-config",
                   "--cache-dir", str(self.root / "uv-cache"), "--find-links", str(wheels),
                   "--python", sys.executable, str(requirements)]
        failed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("No solution", failed.stderr)
        fixed = subprocess.run([*command, "--override", str(exporter.HERE / "sglang-overrides.txt")],
                               capture_output=True, text=True, timeout=30)
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        self.assertIn("protobuf==6.31.1", fixed.stdout)

    def test_dependency_override_changes_invalidate_the_cached_container_image(self):
        for name in ("sglang.Dockerfile", "sglang-overrides.txt", "sglang_build_check.py", ".dockerignore"):
            (self.root / name).write_text("initial content")
        images, builds = {}, []
        def run(command, **kwargs):
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0 if command[-1] in images else 1)
            if command[:2] == ["docker", "build"]:
                tag = command[command.index("-t") + 1]
                builds.append(tag)
                images[tag] = "sha256:" + str(len(builds))
            return subprocess.CompletedProcess(command, 0)
        with patch.object(launcher, "HERE", self.root), \
                patch.object(launcher.shutil, "which", return_value="docker"), \
                patch.object(launcher.subprocess, "run", side_effect=run), \
                patch.object(launcher.subprocess, "check_output", side_effect=lambda args, **kwargs: images[args[-1]] + "\n"):
            first = launcher.ensure_image()
            self.assertEqual(launcher.ensure_image(), first)
            self.assertEqual(len(builds), 1)
            (self.root / "sglang-overrides.txt").write_text("updated protobuf override")
            self.assertNotEqual(launcher.ensure_image(), first)
            self.assertEqual(len(builds), 2)


if __name__ == "__main__":
    unittest.main()
