"""CPU checks for real LoRA checkpoint loading and reference-token provenance."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import UPSTREAM, atomic_json, file_hash
sys.path.insert(0, str(UPSTREAM))
import fish_speech.models.text2semantic.llama as llama
from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora
from fish_speech.models.text2semantic import inference
import voice_worker as worker


class VoiceWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def model(self, targets):
        torch.manual_seed(7)
        config = llama.DualARModelArgs(vocab_size=64, n_layer=1, n_head=2, n_local_heads=2,
            head_dim=16, dim=32, intermediate_size=64, max_seq_len=32,
            codebook_size=16, num_codebooks=10, semantic_begin_id=16, semantic_end_id=31,
            n_fast_layer=1, fast_dim=32, fast_n_head=2, fast_n_local_heads=2,
            fast_head_dim=16, fast_intermediate_size=64)
        model = llama.DualARTransformer(config)
        setup_lora(model, LoraConfig(r=2, lora_alpha=2, target_modules=targets))
        model.tokenizer = object()
        return model

    def checkpoint(self, targets):
        model = self.model(targets)
        adapters = {"model." + n: torch.full_like(p, .03) for n, p in model.named_parameters() if "lora_" in n}
        path = self.root / "adapter.ckpt"
        torch.save({"state_dict": adapters, "global_step": 200}, path)
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump({"model": {"model": {"lora_config": {
            "_target_": "fish_speech.models.text2semantic.lora.LoraConfig",
            "r": 2, "lora_alpha": 2, "target_modules": targets}}}}))
        return {"checkpoint": str(path), "checkpoint_sha256": file_hash(path),
                "config": str(config), "config_sha256": file_hash(config)}, adapters

    def test_loads_actual_fast_and_attention_mlp_adapters_from_exact_saved_configuration(self):
        for targets in (["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"], ["attention", "mlp"]):
            with self.subTest(targets=targets):
                entry, adapters = self.checkpoint(targets)
                with patch.object(llama.DualARTransformer, "from_pretrained", return_value=self.model(targets)) as loader:
                    loaded = worker.load_adapter_model(entry, "cpu", torch.float32)
                self.assertEqual(loader.call_args.kwargs["lora_config"].target_modules, targets)
                self.assertEqual(loader.call_args.kwargs["max_length"], 4096)
                self.assertFalse(loaded.training)
                for name, parameter in loaded.named_parameters():
                    if "lora_" in name:
                        torch.testing.assert_close(parameter, adapters["model." + name])

    def test_rejects_mismatched_lora_tensor_set_even_with_valid_file_hash(self):
        targets = ["attention", "mlp"]
        entry, adapters = self.checkpoint(targets)
        adapters.pop(next(iter(adapters)))
        torch.save({"state_dict": adapters}, entry["checkpoint"])
        entry["checkpoint_sha256"] = file_hash(entry["checkpoint"])
        with patch.object(llama.DualARTransformer, "from_pretrained", return_value=self.model(targets)):
            with self.assertRaisesRegex(ValueError, "does not match"):
                worker.load_adapter_model(entry, "cpu", torch.float32)

    def test_builtin_reference_reuses_verified_tokens_and_rejects_changed_receipt(self):
        wav = self.root / "reference.wav"
        wav.write_bytes(b"reference WAV fixture")
        tokens = wav.with_suffix(".npy")
        array = np.ones((10, 86), dtype=np.int64)
        np.save(tokens, array)
        ref = {"prepared_audio": str(wav), "wav_sha256": file_hash(wav),
               "dataset": str(self.root), "clip_id": "fixture", "duration": 4}
        receipt_path = self.root / "token-receipts/fixture.json"
        atomic_json(receipt_path, {"tokens": file_hash(tokens), "wav": file_hash(wav)})
        with patch.object(inference, "encode_audio") as encode:
            actual = worker.reference_tokens(ref, None, "cpu")
            encode.assert_not_called()
        np.testing.assert_array_equal(actual.numpy(), array)
        atomic_json(receipt_path, {"tokens": "changed", "wav": file_hash(wav)})
        with self.assertRaisesRegex(ValueError, "tokens changed"):
            worker.reference_tokens(ref, None, "cpu")

    def test_uploaded_voice_is_encoded_with_original_codec(self):
        wav = self.root / "reference.wav"
        wav.write_bytes(b"reference WAV fixture")
        ref = {"prepared_audio": str(wav), "wav_sha256": file_hash(wav), "uploaded": True}
        expected = torch.ones((10, 86), dtype=torch.int64)
        codec = object()
        with patch.object(inference, "encode_audio", return_value=expected) as encode:
            actual = worker.reference_tokens(ref, codec, "cpu")
        encode.assert_called_once_with(wav, codec, "cpu")
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
