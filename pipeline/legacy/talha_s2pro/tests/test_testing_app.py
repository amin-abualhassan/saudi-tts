"""Listening-lab discovery, lifecycle, HTTP and real WAV delivery regressions."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import testing_app as app


class TestingAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / 'runs'
        self.run = self.runs / 'experiment'
        self.run.mkdir(parents=True)
        self.base = self.root / 'base'
        self.base.mkdir()
        (self.base / 'codec.pth').write_bytes(b'codec')
        self.patches = [patch('testing_app.HERE',self.root),patch('testing_app.RUNS',self.runs),
                        patch('testing_app.BASE',self.base),patch('testing_app.training_workers',return_value=[])]
        for item in self.patches:
            item.start()
        app.atomic_json(self.run / 'identity.json',{'dataset':str(self.root / 'cache/dataset'),
                                                   'base':{'codec.pth':app.file_hash(self.base / 'codec.pth')}})
        app.atomic_json(self.root / 'monitor/status.json',{'run_dir':str(self.run),'max_steps':3,'step':1,
                                                         'status':'running','stage':'training'})
        self.lab = app.Lab(self.root / 'testing')

    def tearDown(self):
        self.lab.worker = None
        self.lab.logfile.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def completed(self, publish=True):
        checkpoint = self.run / 'checkpoints/final.ckpt'
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(b'checkpoint')
        app.atomic_json(self.run / 'completed.json',{'step':3,'chosen_checkpoint':str(checkpoint)})
        merged = self.run / 'merged'
        merged.mkdir(exist_ok=True)
        for name in app.MERGED_FILES:
            (merged / name).write_bytes(b'model artifact')
        app.atomic_json(merged / 'merge_provenance.json',{'checkpoint':str(checkpoint)})
        (merged / 'codec.pth').symlink_to(self.base / 'codec.pth')
        if publish:
            app.atomic_json(self.run / 'merged.json',{'checkpoint':str(checkpoint),
                            'files':{n:app.file_hash(merged / n) for n in app.MERGED_FILES}})
        return merged

    def ready_worker(self,candidate):
        self.lab.worker = MagicMock()
        self.lab.worker.pid = 1234
        self.lab.worker.poll.return_value = None
        self.lab.model_key = candidate['key']
        self.lab.update(status='ready',model={k:candidate[k] for k in ('directory','run_dir','step','checkpoint')})

    def test_waits_for_merge_publication_and_complete_step_budget(self):
        self.completed(publish=False)
        self.assertIsNone(app.candidate_model(self.run))
        with patch.object(self.lab,'start_worker') as start:
            self.lab.tick()
            start.assert_not_called()
        self.assertEqual(self.lab.state['status'],'waiting')
        merged = self.run / 'merged'
        app.atomic_json(self.run / 'merged.json',{'checkpoint':str(self.run / 'checkpoints/final.ckpt'),
                        'files':{n:app.file_hash(merged / n) for n in app.MERGED_FILES}})
        self.assertIsNone(app.candidate_model(self.run,required_steps=4))
        self.assertIsNotNone(app.candidate_model(self.run,required_steps=3))

    def test_integrity_and_chosen_checkpoint_are_enforced(self):
        merged = self.completed()
        app.verify_model(app.candidate_model(self.run))
        (merged / 'model.pth').write_bytes(b'corrupt weights')
        with self.assertRaisesRegex(ValueError,'integrity'):
            app.verify_model(app.candidate_model(self.run))
        receipt = app.read_json(self.run / 'merged.json')
        receipt['checkpoint'] = str(self.run / 'checkpoints/other.ckpt')
        app.atomic_json(self.run / 'merged.json',receipt)
        with self.assertRaisesRegex(ValueError,'chosen completion'):
            app.candidate_model(self.run)

    def test_active_training_prevents_model_loading(self):
        self.completed()
        with patch('testing_app.training_workers',return_value=[111]),patch.object(self.lab,'start_worker') as start:
            self.lab.tick()
            start.assert_not_called()
        self.assertEqual(self.lab.state['status'],'waiting')
        self.assertEqual(self.lab.state['training_worker_pids'],[111])

    def test_low_free_gpu_memory_defers_loading(self):
        self.completed()
        self.lab.update(system={'gpu':{'free_gb':8}})
        with patch.object(self.lab,'start_worker') as start:
            self.lab.tick()
            start.assert_not_called()
        self.assertIn('16 GB',self.lab.state['message'])

    def test_completed_model_loads_without_manual_reload(self):
        self.completed()
        with patch.object(self.lab,'start_worker',side_effect=self.ready_worker) as start:
            self.lab.tick()
            self.lab.tick()
            start.assert_called_once()
        self.assertEqual(self.lab.state['status'],'ready')

    def test_generation_is_serial_and_results_are_persisted(self):
        self.completed()
        with patch.object(self.lab,'start_worker',side_effect=self.ready_worker):
            self.lab.tick()
        first = self.lab.submit({'text':'نص الاختبار'})
        second = self.lab.submit({'text':'نص آخر'})
        self.lab.tick()
        self.assertEqual(self.lab.active_job,first['id'])
        self.lab.tick()
        self.assertEqual(self.lab.worker.stdin.write.call_count,1)
        self.lab.events.put((1234,{'type':'result','id':first['id'],'duration':2,'elapsed':1,'sample_rate':44100}))
        self.lab.tick()
        self.assertEqual(self.lab.active_job,second['id'])
        saved = app.read_json(self.lab.output / 'results' / f"{first['id']}.json")
        self.assertEqual(saved['status'],'complete')
        self.assertEqual(saved['request']['text'],'نص الاختبار')
        self.assertEqual(self.lab.worker.stdin.write.call_count,2)

    def test_new_training_releases_loaded_model(self):
        self.completed()
        self.ready_worker(app.candidate_model(self.run))
        with patch('testing_app.training_workers',return_value=[111]),patch.object(self.lab,'stop_worker') as stop:
            self.lab.tick()
            stop.assert_called_once()
        self.assertEqual(self.lab.state['status'],'waiting')

    def test_shutdown_during_integrity_check_cannot_start_a_gpu_worker(self):
        self.completed()
        with patch('testing_app.verify_model',side_effect=lambda _:self.lab.stop_event.set()), \
                patch('testing_app.subprocess.Popen') as process:
            self.lab.start_worker(app.candidate_model(self.run))
            process.assert_not_called()

    def test_switching_to_new_completed_run_reloads_automatically(self):
        self.completed()
        with patch.object(self.lab,'start_worker',side_effect=self.ready_worker) as start:
            self.lab.tick()
            old = self.run
            self.run = self.runs / 'second-experiment'
            self.run.mkdir()
            app.atomic_json(self.run / 'identity.json',app.read_json(old / 'identity.json'))
            self.completed()
            app.atomic_json(self.root / 'monitor/status.json',{'run_dir':str(self.run),'max_steps':3,'step':3})
            with patch.object(self.lab,'stop_worker',side_effect=lambda _:setattr(self.lab,'worker',None)) as stop:
                self.lab.tick()
                stop.assert_called_once()
            self.assertEqual(start.call_count,2)
            self.assertEqual(self.lab.state['model']['run_dir'],str(self.run))

    def test_waiting_app_rejects_generation_and_invalid_parameters(self):
        with self.assertRaisesRegex(RuntimeError,'not ready'):
            self.lab.submit({'text':'test'})
        for payload in ({'text':' '},{'text':'<|speaker:1|>test'},{'text':'a','temperature':float('nan')},
                        {'text':'a','seed':True},{'text':'a','seed':2.5},{'text':'a','reference_id':'unavailable'}):
            with self.assertRaises(ValueError):
                app.validate_request(payload,{})

    def test_http_status_and_generation_routes_without_socket(self):
        handler_type = app.make_handler(self.lab)
        def request(method,path,body=b''):
            handler = object.__new__(handler_type)
            handler.path,handler.request_version,handler.command = path,'HTTP/1.1',method
            handler.requestline = method+' '+path+' HTTP/1.1'
            handler.headers = {'Content-Length':str(len(body)),'Content-Type':'application/json'}
            handler.wfile,handler.rfile = io.BytesIO(),io.BytesIO(body)
            getattr(handler,'do_'+method)()
            return handler.wfile.getvalue()
        status = request('GET','/api/status')
        self.assertIn(b'200 OK',status)
        self.assertEqual(json.loads(status.split(b'\r\n\r\n',1)[1])['status'],'waiting')
        self.assertIn(b'409 Conflict',request('POST','/api/generate',b'{"text":"test"}'))
        self.assertIn(b'400 Bad Request',request('POST','/api/generate',b'{"text":""}'))
        self.assertIn(b'404 Not Found',request('GET','/api/audio/../../config.json'))

    def test_reference_selection_uses_only_train_and_distinct_sources(self):
        dataset = self.root / 'cache/dataset'
        rows=[]
        for i,split in enumerate(('train','train','train','validation','test')):
            wav = dataset / 'audio' / split / f'source{i}-1.wav'
            wav.parent.mkdir(parents=True,exist_ok=True)
            wav.write_bytes(b'audio')
            rows.append({'clip_id':wav.stem,'prepared_audio':str(wav),'split':split,
                         'source_group':str(i%2) if split=='train' else f'heldout{i}',
                         'duration':10,'text':'test'})
        (dataset / 'prepared.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        refs = app.references_for(self.run)
        self.assertEqual(len(refs),2)
        self.assertTrue(all(r['split']=='train' for r in refs))

    def test_native_worker_produces_wav_with_cached_reference_using_mocked_models(self):
        # Exercise the real worker, reference hash checks, token validation, seeding,
        # generator arguments and WAV writing; replace only unavailable GPU/models.
        import numpy as np
        import soundfile as sf
        import torch
        sys.path.insert(0,str(app.UPSTREAM))
        from fish_speech.models.text2semantic import inference
        from fish_speech.models.text2semantic.llama import DualARTransformer
        from test_validation import CorpusValidationTests
        dataset = self.root / 'cache/dataset'
        row = CorpusValidationTests().fixture(dataset)[0]
        row['dataset'] = str(dataset)
        wav = Path(row['prepared_audio'])
        app.atomic_json(dataset / 'receipts' / f"{row['clip_id']}.json",{
            'wav_sha256':app.file_hash(wav),'text':row['text'],'lab_sha256':app.file_hash(wav.with_suffix('.lab'))})
        app.atomic_json(dataset / 'token-receipts' / f"{row['clip_id']}.json",{
            'tokens':app.file_hash(wav.with_suffix('.npy')),'wav':app.file_hash(wav)})
        output = self.lab.output / 'results/test.wav'
        item = {'id':'fake-job','request':app.validate_request({'text':'كلام جديد','reference_id':row['clip_id']},{row['clip_id']:row}),
                'reference':row,'output':str(output)}
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config.num_codebooks = 10
        audio = torch.from_numpy(.1*np.sin(np.arange(44100)*2*np.pi*300/44100)).float()
        responses = iter([SimpleNamespace(action='sample',codes=torch.zeros((10,20),dtype=torch.long)),SimpleNamespace(action='next')])
        events = io.StringIO()
        with patch.object(DualARTransformer,'from_pretrained',return_value=model), \
             patch.object(inference,'load_codec_model',return_value=SimpleNamespace(sample_rate=44100)), \
             patch.object(inference,'generate_long',return_value=responses) as generate, \
             patch.object(inference,'decode_to_audio',return_value=audio), \
             patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.is_bf16_supported',return_value=True), \
             patch('torch.cuda.set_device'),patch('torch.cuda.manual_seed_all'), \
             patch('torch.Tensor.to',lambda tensor,*args,**kwargs:tensor), \
             patch('torch.cuda.memory_allocated',return_value=0),patch('sys.stdin',io.StringIO(json.dumps(item)+'\n')), \
             redirect_stdout(events):
            app.inference_worker(self.base,0)
        decoded,rate = sf.read(output)
        self.assertEqual(rate,44100)
        self.assertTrue(np.isfinite(decoded).all())
        self.assertGreater(float(np.sqrt(np.mean(decoded**2))),.01)
        self.assertEqual(generate.call_args.kwargs['text'],'<|speaker:0|>كلام جديد')
        self.assertEqual(generate.call_args.kwargs['prompt_text'],[row['text']])
        self.assertEqual(tuple(generate.call_args.kwargs['prompt_tokens'][0].shape),(10,108))
        self.assertIn('"type": "result"',events.getvalue())


if __name__ == '__main__':
    unittest.main()
