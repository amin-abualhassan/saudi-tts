"""Run with the existing app Python, against an isolated copied app and NF4 API.

No app credentials or .env files are read/copied from the original app.
"""
import argparse
import asyncio
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import httpx

p=argparse.ArgumentParser();p.add_argument('--original-app',type=Path,required=True)
p.add_argument('--bundle',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
args=p.parse_args();bundle=args.bundle.resolve()
identity='bd78f331a5a1534cb8bfecc6c0a024db7c0c5267a6c9da806a2dde1e18dd29af'
report={}
with tempfile.TemporaryDirectory(prefix='aljawab-nf4-app-') as temp:
    app=Path(temp)
    for folder in ['app','tts','assets/voice']:(app/folder).mkdir(parents=True,exist_ok=True)
    for relative in ['app/config.py','app/speech.py','app/numbers.py','tts/gateway.py','tts/model_files.py','tts/model-artifacts.json']:
        shutil.copyfile(args.original_app/relative,app/relative)
    for name in ['reference.json','noura_reference.wav']:
        shutil.copyfile(bundle/'reference'/name,app/'assets/voice'/name)
    reference=json.loads((bundle/'reference/reference.json').read_text())
    subprocess.run([sys.executable,str(bundle/'integration/apply_integration.py'),'--app',str(app)],check=True)
    original=(app/'tts/model-artifacts.json').read_bytes()
    env=os.environ.copy();env.update(SGLANG_URL='http://127.0.0.1:3012',SGLANG_MODEL_PATH=str(bundle/'model'),
        TTS_DERIVATIVE_MANIFEST=str(bundle/'integration/derivative-model-artifacts.json'),
        TTS_ENGINE_MODEL_ID='attention_mlp_r16_early-nf4-bd78f331a5a1534c',TTS_CHECKPOINT_SHA256=identity,
        TTS_ENGINE_REFERENCE_MAP=json.dumps({str(app/'assets/voice'):'/references'}),
        TTS_REFERENCE=reference['reference_id'],TTS_REFERENCE_TEXT=reference['transcript'],
        TTS_REFERENCE_SHA256=reference['prepared_sha256'],TTS_VERIFY_MODEL_HASHES='true')
    log=(app/'gateway.log').open('w')
    process=subprocess.Popen([sys.executable,'-m','uvicorn','tts.gateway:app','--host','127.0.0.1','--port','3013'],cwd=app,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if process.poll() is not None:raise RuntimeError((app/'gateway.log').read_text())
            try:
                with urllib.request.urlopen('http://127.0.0.1:3013/health',timeout=1):break
            except Exception:time.sleep(.25)
        else:raise RuntimeError('Gateway not ready')
        with httpx.Client(timeout=180,trust_env=False) as client:
            status=client.get('http://127.0.0.1:3013/api/status').json()
            assert status['checkpoint_sha256']==identity
            payload=dict(model='attention_mlp_r16_early',reference_id=reference['reference_id'],reference_text=reference['transcript'],text='Hello, how can I help you?',max_new_tokens=384,temperature=.7,top_p=.7)
            events=[];arrivals=[];start=time.perf_counter();pcm=0
            with client.stream('POST','http://127.0.0.1:3013/api/stream',json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: '):
                        event=json.loads(line[6:]);kind=event['event'];events.append(kind)
                        if kind=='audio':
                            import base64
                            # TCP/HTTP boundaries may split a 16-bit sample. The
                            # app accumulates bytes before creating PCM frames.
                            raw=base64.b64decode(event['pcm'],validate=True)
                            pcm+=len(raw);arrivals.append(time.perf_counter()-start)
                        if kind=='complete':assert event['checkpoint_sha256']==identity
            assert events[0]=='accepted' and events[-1]=='complete' and len(arrivals)>1 and pcm>0 and pcm%2==0
            status=client.get('http://127.0.0.1:3013/api/status').json()
            job=status['jobs'][-1];assert job['checkpoint_sha256']==identity and job['reference_sha256']==reference['prepared_sha256']
            report.update(gateway_sse_passed=True,first_audio_seconds=arrivals[0],total_seconds=time.perf_counter()-start,
                          audio_events=len(arrivals),pcm_bytes=pcm,derivative_job_identity_verified=True,
                          original_manifest_unchanged=(app/'tts/model-artifacts.json').read_bytes()==original)
        # App configuration, responses and cached clips use the same deliberate identity.
        os.environ.update({k:v for k,v in env.items() if k.startswith('TTS_') or k.startswith('SGLANG_')})
        sys.path.insert(0,str(app));speech=importlib.import_module('app.speech')
        assert speech.MODEL_SHA==identity
        manifest=dict(greeting_en=dict(model='attention_mlp_r16_early',reference=reference['reference_id'],
                      checkpoint_sha256='9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf',
                      reference_sha256=reference['prepared_sha256']))
        (app/'assets/voice/manifest.json').write_text(json.dumps(manifest))
        shutil.copyfile(bundle/'samples/nf4-en_greeting.wav',app/'assets/voice/greeting_en.wav')
        async def cached_rejected():
            instance=speech.Speech()
            try:
                try:
                    async for frame in instance.cached('greeting_en'):pass
                except RuntimeError as e:
                    assert 'regenerate' in str(e);return True
                return False
            finally:await instance.client.aclose()
        report['bf16_cached_clip_rejected']=asyncio.run(cached_rejected());assert report['bf16_cached_clip_rejected']
        # A modified pinned derivative manifest cannot turn checking into an empty loop.
        mod=importlib.import_module('tts.model_files')
        bad=app/'bad-manifest.json';bad.write_text(json.dumps(dict(derivative_identity_sha256=identity,required_files={})))
        try:mod.verify_model(bundle/'model',bad)
        except RuntimeError:report['tampered_manifest_rejected']=True
        else:raise AssertionError('Tampered manifest was accepted')
    finally:
        process.terminate();process.wait(timeout=15);log.close()
        args.report.with_suffix('.log').write_text((app/'gateway.log').read_text())
args.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
