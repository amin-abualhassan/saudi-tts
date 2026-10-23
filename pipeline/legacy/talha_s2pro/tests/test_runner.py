"""Regression checks for preparation reuse, repair and launcher supervision."""
from argparse import Namespace
import io
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.monitor = run.Monitor(self.root / "monitor")

    def tearDown(self):
        self.monitor.logfile.close()
        self.temp.cleanup()

    def fixture(self):
        original = self.root / "full-audios-transcript/audios-48khz"
        original.mkdir(parents=True)
        rows = []
        for i, split in enumerate(run.SPLITS):
            key = f"source{i}"
            rate = 48000
            pcm = .1 * np.sin(2*np.pi*(300+i*100)*np.arange(rate*7)/rate)
            sf.write(original/f"{key}.flac", pcm, rate, subtype="PCM_16")
            pcm, _ = sf.read(original/f"{key}.flac")
            clip = self.root / f"{key}-1.flac"
            sf.write(clip, pcm[rate:6*rate], rate, subtype="PCM_16")
            label = clip.with_suffix(".txt")
            label.write_text("نص الاختبار", encoding="utf-8")
            rows.append(dict(clip_id=clip.stem, audio=str(clip), label=str(label), text="نص الاختبار",
                             source_id=key, source_group=key, split=split, start=1, end=6,
                             duration=5, sample_rate=rate, channels=1))
        return rows

    def test_second_preparation_reuses_audio_without_normalization(self):
        rows = self.fixture()
        dataset = self.root / "dataset"
        prepared = run.prepare(rows, dataset, self.monitor, root=self.root)
        paths = [Path(r["prepared_audio"]) for r in prepared]
        mtimes = [p.stat().st_mtime_ns for p in paths]
        with patch("pipeline.normalize", side_effect=AssertionError("Must reuse cached waveforms")):
            run.prepare(rows, dataset, self.monitor, root=self.root)
        self.assertEqual(mtimes, [p.stat().st_mtime_ns for p in paths])
        self.assertEqual(self.monitor.state["reused"], 3)

    def test_corrupted_waveform_is_rebuilt_and_tokens_invalidated(self):
        rows = self.fixture()
        dataset = self.root / "dataset"
        prepared = run.prepare(rows, dataset, self.monitor, root=self.root)
        path = Path(prepared[0]["prepared_audio"])
        path.write_bytes(b"broken")
        np.save(path.with_suffix(".npy"), np.zeros((10, 108), dtype=np.int64))
        run.prepare(rows, dataset, self.monitor, root=self.root)
        self.assertEqual(sf.info(path).samplerate, 44100)
        self.assertFalse(path.with_suffix(".npy").exists())
        self.assertEqual(self.monitor.state["reused"], 2)

    def test_alignment_mismatch_stops_before_mislabeled_output(self):
        rows = self.fixture()
        rows[0]["start"] += .001
        dataset = self.root / "dataset"
        with self.assertRaisesRegex(ValueError, "Insufficient clean audio in train"):
            run.prepare(rows, dataset, self.monitor, root=self.root)
        rejection = json.loads((dataset / "rejected" / f"{rows[0]['clip_id']}.json").read_text())
        self.assertIn("SRT interval", rejection["reason"])
        self.assertFalse((dataset / "audio/train/talha_ahmad" / f"{rows[0]['clip_id']}.wav").exists())

    def test_truncated_boundaries_accept_exact_one_sample_offset(self):
        rows = self.fixture()
        row = rows[0]
        row.update(start=1.001, end=6.001)
        rate = 48000
        self.assertEqual(int(row['start']*rate), round(row['start']*rate)-1)
        pcm, _ = sf.read(self.root / 'full-audios-transcript/audios-48khz/source0.flac')
        audio = pcm[int(row['start']*rate):int(row['end']*rate)]
        sf.write(row['audio'], audio, rate, subtype='PCM_16')
        row['duration'] = len(audio)/rate
        dataset = self.root / 'dataset'
        prepared = run.prepare(rows, dataset, self.monitor, root=self.root)
        receipt = json.loads((dataset / 'receipts' / f"{row['clip_id']}.json").read_text())
        self.assertEqual(receipt['source_rounding'], 'truncate')
        self.assertEqual(receipt['source_offset_samples'], -1)
        self.assertEqual(receipt['source_max_pcm_difference'], 0)
        self.assertEqual(len(prepared), 3)

    def test_rounding_does_not_accept_a_broader_time_shift(self):
        row = self.fixture()[0]
        row.update(start=1.001, end=6.001)
        audio, rate = sf.read(row['audio'])
        with self.assertRaisesRegex(run.QualityRejection, 'SRT interval'):
            run.verify_source_interval(audio,rate,row,self.root / 'full-audios-transcript/audios-48khz/source0.flac')

    def test_compatible_cache_migration_preserves_paths_and_skips_inventory(self):
        import copy
        args = Namespace(hours=10, seed=42)
        with patch('run.HERE', self.root), patch('run.source_signature', return_value='unchanged-sources'), \
                patch('run.file_hash', return_value='unchanged-helper'), patch('run.pipeline.inventory') as inventory:
            recipe = {**run.recipe_signature(), 'source':'unchanged-sources','hours':10,'seed':42}
            legacy = {k:v for k,v in recipe.items() if k != 'preparation_revision'}
            legacy.update(runner=run.LEGACY_RUNNER_SHA256,alignment='existing_srt_plus_original_pcm_check')
            dataset = self.root / 'cache/previous-dataset-path'
            rows = [dict(clip_id='train-1',source_group='train',split='train',duration=5)]
            run.atomic_json(dataset / 'selection.json', {'recipe':legacy,'summary':{}})
            (dataset / 'manifest.jsonl').write_text(json.dumps(rows[0])+'\n')
            (dataset / 'reserves.jsonl').write_text('')
            self.assertEqual(run.make_plan(args,self.monitor)[0], dataset)
            self.assertEqual(run.make_plan(args,self.monitor)[0], dataset)
            inventory.assert_not_called()
            saved = json.loads((dataset / 'selection.json').read_text())
            self.assertEqual(saved['legacy_recipe'],legacy)
            self.assertEqual(saved['recipe'],recipe)
            for key in ('runner','source','lufs','hours'):
                incompatible = copy.deepcopy(legacy)
                incompatible[key] = 'changed'
                self.assertFalse(run.compatible_selection(incompatible,recipe))
            self.assertFalse(run.compatible_selection(legacy,{**recipe,'preparation_revision':3}))

    def test_transcript_change_stops_before_using_stale_plan(self):
        rows = self.fixture()
        Path(rows[0]["label"]).write_text("different transcript")
        with self.assertRaisesRegex(ValueError, "transcript changed"):
            run.prepare(rows, self.root / "dataset", self.monitor, root=self.root)

    def clipped_fixture_with_reserves(self):
        rows = self.fixture()
        row = rows[0]
        rate = 48000
        tt = np.arange(rate*17)/rate
        pcm = .1*np.sin(2*np.pi*(300*tt+17*tt**2))
        pcm[rate+1000:rate+2200] = 1
        original = self.root/'full-audios-transcript/audios-48khz'/f"{row['source_id']}.flac"
        sf.write(original,pcm,rate,subtype='PCM_16')
        pcm,_ = sf.read(original)
        sf.write(row['audio'],pcm[rate:11*rate],rate,subtype='PCM_16')
        row.update(duration=10,end=11)
        reserves = []
        for index,start in enumerate((2,7),2):
            clip = self.root/f"source0-{index}.flac"
            sf.write(clip,pcm[start*rate:(start+5)*rate],rate,subtype='PCM_16')
            clip.with_suffix('.txt').write_text(row['text'])
            reserves.append({**row,'clip_id':clip.stem,'audio':str(clip),'label':str(clip.with_suffix('.txt')),
                             'start':start,'end':start+5,'duration':5})
        return rows,reserves

    def test_clipped_audio_is_excluded_and_duration_refilled_from_same_split(self):
        rows,reserves = self.clipped_fixture_with_reserves()
        dataset = self.root/'dataset'
        prepared = run.prepare(rows,dataset,self.monitor,root=self.root,replacements=reserves)
        self.assertNotIn(rows[0]['clip_id'],{r['clip_id'] for r in prepared})
        train = [r for r in prepared if r['split']=='train']
        self.assertEqual(sum(r['duration'] for r in train),10)
        self.assertEqual({r['source_group'] for r in train},{rows[0]['source_group']})
        self.assertEqual(self.monitor.state['rejected'],1)
        report = json.loads((dataset/'rejected'/f"{rows[0]['clip_id']}.json").read_text())
        self.assertIn('clipping',report['reason'])
        self.assertFalse((dataset/'audio/train/talha_ahmad'/f"{rows[0]['clip_id']}.wav").exists())
        with patch('pipeline.normalize',side_effect=AssertionError('Completed clean clips must be reused')):
            repeated = run.prepare(rows,dataset,self.monitor,root=self.root,replacements=reserves)
        self.assertEqual([r['clip_id'] for r in prepared],[r['clip_id'] for r in repeated])
        self.assertEqual(self.monitor.state['reused'],4)

    def test_pilot_replaces_bad_clip_without_exceeding_clip_count(self):
        rows,reserves = self.clipped_fixture_with_reserves()
        prepared = run.prepare(rows,self.root/'pilot',self.monitor,root=self.root,replacements=reserves,target='count')
        self.assertEqual(len(prepared),3)
        self.assertEqual(sum(r['split']=='train' for r in prepared),1)

    def test_unverified_alignment_is_replaced_without_moving_audio_boundaries(self):
        rows,reserves = self.clipped_fixture_with_reserves()
        rows[0]['start'] += .001
        rows[0]['end'] += .001
        source_hash = run.file_hash(rows[0]['audio'])
        dataset = self.root / 'dataset'
        prepared = run.prepare(rows,dataset,self.monitor,root=self.root,replacements=reserves)
        self.assertNotIn(rows[0]['clip_id'],{r['clip_id'] for r in prepared})
        self.assertEqual(sum(r['duration'] for r in prepared if r['split']=='train'),10)
        self.assertEqual(source_hash,run.file_hash(rows[0]['audio']))
        report = json.loads((dataset / 'rejected' / f"{rows[0]['clip_id']}.json").read_text())
        self.assertIn('SRT interval',report['reason'])
        self.assertEqual({r['source_group'] for r in prepared if r['split']=='train'},{rows[0]['source_group']})

    def test_no_clean_replacement_fails_with_saved_rejection_report(self):
        rows,_ = self.clipped_fixture_with_reserves()
        dataset = self.root/'dataset'
        with self.assertRaisesRegex(ValueError,'Insufficient clean audio in train'):
            run.prepare(rows,dataset,self.monitor,root=self.root)
        self.assertTrue((dataset/'qc.json').exists())
        self.assertTrue((dataset/'rejected'/f"{rows[0]['clip_id']}.json").exists())

    def test_reserves_cannot_introduce_unassigned_or_leaking_videos(self):
        planned = [dict(clip_id=f'{split}-1',source_group=split,split=split,duration=10) for split in run.SPLITS]
        candidates = [dict(clip_id=f'{split}-2',source_group=split,duration=10) for split in run.SPLITS]
        candidates.append(dict(clip_id='other-1',source_group='not-assigned',duration=10))
        reserves = run.make_reserves(planned,candidates,42)
        self.assertEqual(len(reserves),3)
        self.assertTrue(all(r['split']==r['source_group'] for r in reserves))
        bad = {**reserves[1],'source_group':'validation','split':'train'}
        with self.assertRaisesRegex(ValueError,'leaks across splits'):
            run.pipeline.assert_disjoint([*planned,bad])

    def test_composed_config_uses_optimizer_step_validation_and_actual_lora(self):
        args = Namespace(seed=42,max_length=2048,steps=2000,validation_steps=100)
        cfg = run.composed_config(self.root / "dataset", self.root / "experiment", args)
        self.assertEqual(cfg["trainer"]["val_check_interval"], 400)
        self.assertEqual(cfg["trainer"]["accumulate_grad_batches"], 4)
        self.assertTrue(cfg["callbacks"]["model_checkpoint"]["save_last"])
        self.assertEqual(cfg["model"]["model"]["lora_config"]["lora_alpha"], 32)
        self.assertEqual(cfg["tokenizer"]["model_path"], str(run.BASE))
        self.assertNotEqual(cfg["train_dataset"]["proto_files"], cfg["val_dataset"]["proto_files"])

    def test_worker_events_and_failure_exit_are_not_silently_ignored(self):
        run.process([sys.executable,"-c",'print(\'TALHA_EVENT {"step": 7, "metrics": {"train/loss": 1.5}}\')'], self.monitor)
        self.assertEqual(self.monitor.state["step"], 7)
        self.assertEqual(self.monitor.state["history"][-1]["train/loss"], 1.5)
        with self.assertRaisesRegex(RuntimeError, "exit code 3"):
            run.process([sys.executable,"-c","raise SystemExit(3)"], self.monitor)

    def test_dashboard_http_routes_without_a_network_socket(self):
        # The workspace prohibits socket bind; exercise the real HTTP handlers using BytesIO.
        with patch("run.ThreadingHTTPServer") as server:
            instance = server.return_value
            instance.serve_forever.return_value = None
            run.dashboard(self.monitor,"127.0.0.1",7860)
            handler_type = server.call_args.args[1]
        self.monitor.update(message="<script>untrusted</script>", step=12)
        for route, expected in (("/api/status", b'"step": 12'), ("/", b"Talha"), ("/missing", b"404")):
            handler = object.__new__(handler_type)
            handler.path = route
            handler.request_version = "HTTP/1.1"
            handler.requestline = "GET " + route + " HTTP/1.1"
            handler.command = "GET"
            handler.wfile = io.BytesIO()
            handler.do_GET()
            response = handler.wfile.getvalue()
            self.assertIn(expected,response)
            if route == "/":
                self.assertNotIn(b"<script>untrusted</script>",response)
            if route == "/api/status":
                self.assertIn(b"Cache-Control: no-store",response)
                self.assertEqual(json.loads(response.split(b"\r\n\r\n",1)[1])["step"],12)

    def test_failed_preflight_does_not_start_preparation(self):
        with patch("sys.argv",["run.py","--check","--no-dashboard"]), \
                patch("run.HERE",self.root), \
                patch("run.preflight",side_effect=RuntimeError("No GPU")), \
                patch("run.make_plan") as plan:
            self.assertEqual(run.main(),1)
            plan.assert_not_called()
        self.assertEqual(json.loads((self.root / "monitor/status.json").read_text())["status"],"failed")

    def test_verified_tokens_and_packed_data_reuse_and_repair(self):
        from test_validation import CorpusValidationTests, record
        dataset = self.root / "dataset"
        rows = CorpusValidationTests().fixture(dataset)
        codec = "synthetic-codec-for-cache-tests"
        run.atomic_json(dataset/"token_recipe.json",{"codec":codec})
        for row in rows:
            wav = Path(row['prepared_audio'])
            run.atomic_json(dataset/'token-receipts'/f"{row['clip_id']}.json",
                            {'codec':codec,'tokens':run.file_hash(wav.with_suffix('.npy')),'wav':run.file_hash(wav)})
        calls = []
        def fake_worker(command, monitor, progress=None):
            calls.append(str(command[1]))
            if str(command[1]).endswith('extract_vq.py'):
                folder = Path(command[2])
                for row in rows:
                    wav = Path(row['prepared_audio'])
                    if wav.is_relative_to(folder) and not wav.with_suffix('.npy').exists():
                        np.save(wav.with_suffix('.npy'),np.full((10,108),7,dtype=np.int64))
                return
            self.assertTrue(str(command[1]).endswith('build_dataset.py'))
            output = Path(command[command.index('--output')+1])
            output.mkdir(parents=True)
            data = b''
            for row in rows:
                if row['split'] == output.name:
                    text = Path(row['prepared_label']).read_text()
                    text = re.sub(r'\s+',' ',re.sub(r'\{.*?\}|<.*?>',' ',text))
                    data += record(text,np.load(Path(row['prepared_audio']).with_suffix('.npy')))
            (output/'00000000.protos').write_bytes(data)
        with patch('run.process',side_effect=fake_worker):
            run.encode_and_pack(dataset,rows,self.monitor,codec)
        self.assertEqual(len(calls),3)
        with patch('run.process',side_effect=AssertionError('Unchanged data must not run any GPU/packer process')):
            run.encode_and_pack(dataset,rows,self.monitor,codec)
        # Bounds-valid corruption must still be detected through content hashes.
        token = Path(rows[0]['prepared_audio']).with_suffix('.npy')
        np.save(token,np.full((10,108),8,dtype=np.int64))
        stale = dataset/'protos/train/99999999.protos'
        stale.write_bytes(b'stale shard')
        calls.clear()
        with patch('run.process',side_effect=fake_worker):
            run.encode_and_pack(dataset,rows,self.monitor,codec)
        self.assertEqual(sum(c.endswith('extract_vq.py') for c in calls),1)
        self.assertFalse(stale.exists())
        self.assertTrue(np.all(np.load(token) == 7))

    def test_installer_uses_active_env_and_upstream_protobuf_override(self):
        with patch('run.process') as process:
            run.install_dependencies(self.monitor)
        commands = [call.args[0] for call in process.call_args_list]
        install = next(c for c in commands if '--override' in c)
        self.assertIn(sys.executable,install)
        self.assertEqual((self.monitor.directory/'dependency-overrides.txt').read_text(),'protobuf>=3.20,<6\n')
        self.assertIn('tensorboard==2.20.0\n',(self.monitor.directory/'training-requirements.in').read_text())

    def test_events_log_does_not_repeat_entire_history(self):
        for step in range(10):
            self.monitor.event({'step':step,'metrics':{'train/loss':1.5}})
        records = [json.loads(line) for line in (self.monitor.directory/'events.jsonl').read_text().splitlines()]
        self.assertTrue(all('history' not in r for r in records))
        self.assertEqual(len(self.monitor.state['history']),10)

    def test_tensorboard_protobuf_events_write_and_read(self):
        import importlib.util
        if not all(importlib.util.find_spec(name) for name in ('torch','tensorboard')):
            self.skipTest('Training dependencies not installed')
        # VersionError must fail this test rather than silently disabling TensorBoard.
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        directory = self.root/'tensorboard'
        with SummaryWriter(str(directory)) as writer:
            writer.add_scalar('train/loss',1.25,7)
        events = EventAccumulator(str(directory)).Reload().Scalars('train/loss')
        self.assertEqual((events[0].step,events[0].value),(7,1.25))

    def test_resume_removes_metrics_beyond_the_restored_checkpoint(self):
        for step in (1,2,3,4):
            self.monitor.event({'step':step,'metrics':{'train/loss':1.5}})
        self.monitor.event({'step':2,'metrics':{'train/loss':1.3}})
        self.assertEqual([p['step'] for p in self.monitor.state['history']],[1,2])
        self.assertEqual(self.monitor.state['history'][-1]['train/loss'],1.3)


if __name__ == "__main__":
    unittest.main()
