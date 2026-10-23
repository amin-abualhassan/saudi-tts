"""Meaningful CPU checks for data isolation, bounds, provenance and run settings."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as suite_run
import corpus
from balance import probabilities
from common import HERE, read_rows, shared, write_rows, talha_validation
from compare import audio_metrics, cases_for
from dashboard import safe_asset


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = suite_run.load_config(HERE / "experiments.json")

    def tearDown(self):
        self.temp.cleanup()

    def rows(self):
        rows = []
        for family in ("talha", "podcast", "single"):
            for group in range(10):
                speakers = ["wesam_ali", "guest_"+str(group)] if family == "podcast" else [family+"_speaker"]
                for speaker in speakers:
                    for clip in range(20):
                        rows.append({"clip_id": f"{family}_{group}_{speaker}_{clip}", "family": family,
                            "speaker": speaker, "source_group": family+str(group), "duration": 10,
                            "text": "نص اختبار", "split": "test" if group == 0 else "validation" if group == 1 else "train"})
        return rows

    def test_selection_is_bounded_disjoint_and_deterministic(self):
        budgets = {f: .2 for f in ("talha", "podcast", "single")}
        first, summary = corpus.select(self.rows(), budgets, 42)
        second, _ = corpus.select(list(reversed(self.rows())), budgets, 42)
        self.assertEqual(first, second)
        corpus.talha_pipeline.assert_disjoint(first)
        for family in budgets:
            self.assertLessEqual(sum(r["duration"] for r in first if r["family"] == family), budgets[family]*3600)
        podcast = [r for r in first if r["family"] == "podcast"]
        self.assertEqual(sum(r["duration"] for r in podcast if r["speaker"] == "wesam_ali"),
                         sum(r["duration"] for r in podcast if r["speaker"] != "wesam_ali"))
        self.assertTrue(all(summary[f][s]["clips"] > 0 for f in budgets for s in corpus.SPLITS))

    def test_family_sampling_preserves_weights_and_wesam_share(self):
        names = ["talha_ahmad", "single_speaker", "wesam_ali", "guest_a", "guest_b"]
        weights = probabilities(names, [100000, 100, 200, 300, 900], self.config["family_weights"])
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(weights[0], .4)
        self.assertAlmostEqual(weights[1], .15)
        self.assertAlmostEqual(weights[2], .225)
        self.assertAlmostEqual(sum(weights[3:]), .225)
        self.assertAlmostEqual(weights[4]/weights[3], np.sqrt(3))
        with self.assertRaises(ValueError):
            probabilities(["talha_ahmad"], [10], self.config["family_weights"])

    def test_huge_budget_and_invalid_trial_rejected(self):
        path = self.root / "config.json"
        self.config["hours"]["talha"] = 100
        path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "20 total hours"):
            suite_run.load_config(path)
        self.config["hours"]["talha"] = 6
        self.config["experiments"][0]["targets"] = ["new_attention_heads"]
        path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "LoRA targets"):
            suite_run.load_config(path)

    def test_resolved_configs_share_data_and_control_rank(self):
        for e in self.config["experiments"]:
            cfg = suite_run.resolved_config(self.root / "data", self.root / e["name"], self.config, e)
            self.assertEqual(cfg["trainer"]["val_check_interval"], 400)
            self.assertEqual(cfg["data"]["train_dataset"]["family_weights"], self.config["family_weights"])
            self.assertEqual(cfg["data"]["num_workers"], 0)
            self.assertEqual(cfg["data"]["_target_"], "balanced_dataset.BalancedDataModule")
            self.assertEqual(cfg["data"]["batch_size"], 1)
            self.assertEqual(cfg["trainer"]["default_root_dir"], str(self.root / e["name"]))
            self.assertEqual(cfg["model"]["model"]["lora_config"]["r"], e["r"])
            self.assertNotEqual(cfg["train_dataset"]["proto_files"], cfg["val_dataset"]["proto_files"])
        pilot = suite_run.resolved_config(self.root, self.root / "pilot", self.config, self.config["experiments"][0], True)
        self.assertEqual(pilot["trainer"]["max_steps"], 2)
        self.assertEqual(pilot["trainer"]["val_check_interval"], 4)

    def fixture(self):
        rows = []
        for i, split in enumerate(corpus.SPLITS):
            path = self.root / f"clip{i}.wav"
            samples = np.arange(44100*6)
            sf.write(path, .1*np.sin(samples*2*np.pi*(300+i*30)/44100), 44100, subtype="PCM_16")
            path.with_suffix(".lab").write_text("نص اختبار\n", encoding="utf-8")
            rows.append({"clip_id": path.stem, "audio": str(path), "label": str(path.with_suffix(".lab")),
                        "text": "نص اختبار", "sample_rate": 44100, "duration": 6,
                        "family": "single", "speaker": "single_speaker", "origin": "single",
                        "split": split, "source_group": split})
        return rows

    def test_preparation_cache_repair_and_label_integrity(self):
        rows = self.fixture()
        dataset = self.root / "dataset"
        monitor = shared.Monitor(self.root / "monitor")
        try:
            prepared = corpus.prepare(rows, dataset, monitor, pilot=True)
            with patch.object(corpus.talha_pipeline, "normalize", side_effect=AssertionError("Must reuse")):
                corpus.prepare(rows, dataset, monitor, pilot=True)
            wav = Path(prepared[0]["prepared_audio"])
            wav.write_bytes(b"corrupt")
            np.save(wav.with_suffix(".npy"), np.zeros((10, 130), dtype=np.int64))
            corpus.prepare(rows, dataset, monitor, pilot=True)
            self.assertEqual(sf.info(wav).samplerate, 44100)
            self.assertFalse(wav.with_suffix(".npy").exists())
            Path(rows[0]["label"]).write_text("Changed transcript")
            with self.assertRaisesRegex(ValueError, "Source label changed"):
                corpus.prepare(rows, dataset, monitor, pilot=True)
        finally:
            monitor.logfile.close()

    def test_cross_split_duplicate_stops(self):
        rows = self.fixture()
        Path(rows[1]["audio"]).write_bytes(Path(rows[0]["audio"]).read_bytes())
        monitor = shared.Monitor(self.root / "monitor")
        try:
            with self.assertRaisesRegex(ValueError, "Duplicate audio across splits"):
                corpus.prepare(rows, self.root / "dataset", monitor, pilot=True)
        finally:
            monitor.logfile.close()

    def test_packed_validation_checks_speaker_as_well_as_content(self):
        # Construct expected fingerprints and fake parsed records; token shape/ranges
        # and actual file/label pairing still run through the real validators.
        rows = self.fixture()
        for r in rows:
            r["prepared_audio"] = r["audio"]
            r["prepared_label"] = r["label"]
            codes = np.zeros((10, 130), dtype=np.int64)
            np.save(Path(r["audio"]).with_suffix(".npy"), codes)
            folder = self.root / "protos" / r["split"]
            folder.mkdir(parents=True)
            (folder / "000.protos").touch()
        write_rows(self.root / "prepared.jsonl", rows)
        fingerprint = talha_validation.fingerprint(["نص اختبار "], codes)
        with patch.object(talha_validation, "packed_records", return_value=iter([])):
            with self.assertRaisesRegex(ValueError, "speaker/text/token/count"):
                corpus.validate_packed(self.root)
        with patch.object(talha_validation, "packed_records", side_effect=lambda _: iter([("single_speaker", [fingerprint])])):
            result = corpus.validate_packed(self.root)
            self.assertEqual(result["train"]["verified_sentences"], 1)
        with patch.object(talha_validation, "packed_records", side_effect=lambda _: iter([("wrong_voice", [fingerprint])])):
            with self.assertRaisesRegex(ValueError, "speaker/text/token/count"):
                corpus.validate_packed(self.root)

    def test_comparison_has_fixed_references_and_held_out_text(self):
        rows = []
        for speaker in ("talha_ahmad", "wesam_ali", "single_speaker"):
            for split in corpus.SPLITS:
                rows.append({"speaker": speaker, "split": split, "clip_id": speaker+split,
                             "duration": 12, "text": split+" نص الاختبار"})
        write_rows(self.root / "prepared.jsonl", rows)
        cases = cases_for(self.root, self.config["evaluation"])
        self.assertEqual(len(cases), 15)
        self.assertTrue(all(c["reference"]["split"] == "train" for c in cases))
        self.assertEqual(sum(c["held_out_transcript"] for c in cases), 3)

    def test_audio_metrics_expose_silence_and_paths_are_confined(self):
        silent = audio_metrics(np.zeros(16000), 16000)
        self.assertEqual(silent["warning"], "near-silent output")
        self.assertAlmostEqual(silent["trailing_quiet_seconds"], 1)
        audio = np.concatenate([np.ones(8000)*.1, np.zeros(8000)])
        self.assertAlmostEqual(audio_metrics(audio, 16000)["quiet_fraction"], .5)
        with self.assertRaises(ValueError):
            safe_asset("/etc/passwd")

    def test_balanced_dataset_never_truncates_paired_clips(self):
        from common import UPSTREAM
        sys.path.insert(0, str(UPSTREAM))
        import torch
        from balanced_dataset import BalancedDataset
        dataset = BalancedDataset(family_weights=self.config["family_weights"], proto_files=[], max_length=100,
                                  tokenizer=None, num_codebooks=10)
        sentence = SimpleNamespace(texts=["نص"], semantics=[])
        dataset.groups = [SimpleNamespace(name="voice", sentences=[sentence])]
        dataset.group_weights = [1]
        dataset.pack_sentences = lambda *a, **k: (torch.ones((11, 40), dtype=torch.long), torch.ones((11, 40), dtype=torch.long))
        example = dataset.augment()
        self.assertEqual(example["tokens"].shape[1], 80)
        self.assertEqual(example["labels"].shape, example["tokens"].shape)
        dataset.pack_sentences = lambda *a, **k: (torch.ones((11, 101)), torch.ones((11, 101)))
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            dataset.augment()

    def test_zero_worker_loaders_iterate_training_and_validation(self):
        from common import UPSTREAM
        sys.path.insert(0, str(UPSTREAM))
        import torch
        from balanced_dataset import BalancedDataModule
        examples = [{"tokens": torch.ones((11, 20), dtype=torch.long),
                     "labels": torch.ones((11, 20), dtype=torch.long)}]
        datamodule = BalancedDataModule(examples, examples, batch_size=1,
                                       max_length=2048, num_workers=0)
        for loader in (datamodule.train_dataloader(), datamodule.val_dataloader()):
            self.assertEqual(loader.num_workers, 0)
            self.assertFalse(loader.persistent_workers)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["inputs"].shape), (1, 11, 20))
            self.assertEqual(tuple(batch["labels"].shape), (1, 11, 20))

    def test_loader_preserves_pilot_collator_and_positive_worker_support(self):
        from common import UPSTREAM
        sys.path.insert(0, str(UPSTREAM))
        import torch
        import fish_speech.datasets.semantic as semantic
        from balanced_dataset import BalancedDataModule
        examples = [{"tokens": torch.ones((11, 20)), "labels": torch.ones((11, 20))}]
        datamodule = BalancedDataModule(examples, examples, batch_size=1,
                                       max_length=2048, num_workers=0)
        # The same runtime replacement is used by shared.worker's full-context pilot.
        def full_context(examples):
            return {"inputs": torch.zeros((1, 11, 2048)), "labels": torch.zeros((1, 11, 2048))}
        with patch.object(semantic, "TextDataCollator", return_value=full_context) as collator:
            for loader in (datamodule.train_dataloader(), datamodule.val_dataloader()):
                self.assertEqual(tuple(next(iter(loader))["inputs"].shape), (1, 11, 2048))
            self.assertEqual(collator.call_count, 2)
        datamodule.num_workers = 1
        # Construction only; no subprocesses are needed for this compatibility check.
        for loader in (datamodule.train_dataloader(), datamodule.val_dataloader()):
            self.assertTrue(loader.persistent_workers)
            self.assertEqual(loader.num_workers, 1)

    def test_guard_rejects_nonfinite_gradients_and_proves_updates(self):
        import torch
        from training_guard import TrainingGuard
        module = torch.nn.Module()
        module.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2)))
        trainer = SimpleNamespace(global_step=0, default_root_dir=str(self.root))
        guard = TrainingGuard()
        guard.on_train_start(trainer, module)
        with self.assertRaisesRegex(RuntimeError, "has no gradient"):
            guard.on_before_optimizer_step(trainer, module, None)
        module.lora_A.grad = torch.tensor([float("nan"), 1])
        with self.assertRaisesRegex(RuntimeError, "Nonfinite gradient"):
            guard.on_before_optimizer_step(trainer, module, None)
        with torch.no_grad():
            module.lora_A.add_(.1)
        trainer.global_step = 1
        guard.on_train_end(trainer, module)
        saved = json.loads((self.root / "adapter_updates.json").read_text())
        self.assertEqual(saved["changed_tensors"], 1)

    def test_suite_continues_after_failed_pilot_and_training(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        rows = [{"clip_id": "sample", "prepared_audio": str(self.root / "reference.wav"), "duration": 6,
                 "family": "talha", "split": "train", "speaker": "talha_ahmad"}]
        write_rows(dataset / "prepared.jsonl", rows)
        sf.write(rows[0]["prepared_audio"], np.ones(44100*6)*.1, 44100)
        args = SimpleNamespace(config=HERE / "experiments.json", steps=None, only=None, install=False,
                               check=False, plan=False, preview=False, compare_only=False, review=None, accepted_only=False)
        monitor = shared.Monitor(self.root / "monitor")
        def fake_process(command, monitor):
            if "--output-path" in command:
                path = command[command.index("--output-path")+1]
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.ones(44100*6)*.1, 44100)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(suite_run, "HERE", self.root))
                stack.enter_context(patch.object(corpus, "plan", return_value=(dataset, rows, {})))
                stack.enter_context(patch.object(corpus, "prepare", return_value=rows))
                stack.enter_context(patch.object(suite_run, "prepare_pilot", return_value=rows))
                stack.enter_context(patch.object(suite_run, "check_resources"))
                stack.enter_context(patch.object(suite_run, "code_identity", return_value={}))
                stack.enter_context(patch.object(suite_run, "encode"))
                stack.enter_context(patch.object(shared, "preflight"))
                stack.enter_context(patch.object(shared, "base_signature", return_value={"codec.pth": "hash"}))
                stack.enter_context(patch.object(shared, "artifact_hashes", return_value={}))
                stack.enter_context(patch.object(shared, "process", side_effect=fake_process))
                stack.enter_context(patch.object(suite_run, "gpu_pilot", side_effect=[RuntimeError("OOM"), None, None, None]))
                stack.enter_context(patch.object(suite_run, "cases_for", return_value=[{"speaker": "talha_ahmad", "reference": rows[0]}]))
                stack.enter_context(patch.object(suite_run, "compare_models", return_value=[]))
                train = stack.enter_context(patch.object(suite_run, "train", side_effect=[RuntimeError("nonfinite loss"), ({"step": 1500}, []), ({"step": 1500}, [])]))
                suite_run.run_suite(args, monitor)
                self.assertEqual(train.call_count, 3)
                state = monitor.snapshot()
                self.assertEqual(state["status"], "complete with failures")
                self.assertEqual(state["experiments"]["fast_r16"]["status"], "pilot failed")
                self.assertEqual(state["experiments"]["fast_r32"]["status"], "failed")
                self.assertEqual(state["experiments"]["attention_mlp_r16"]["status"], "trained")
        finally:
            monitor.logfile.close()


if __name__ == "__main__":
    unittest.main()
