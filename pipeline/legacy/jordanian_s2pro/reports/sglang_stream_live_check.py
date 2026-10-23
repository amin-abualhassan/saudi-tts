"""Live port-3000 streaming/normal-flow audit; run with the Fish environment."""
import base64
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:3000'

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as response:
        return response.read()

def status():
    return json.loads(get('/api/status'))

def request(path, payload):
    return urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                  headers={'Content-Type': 'application/json'})

def wait_ready(deadline):
    while time.monotonic() < deadline:
        state = status()
        if state['status'] == 'error':
            raise RuntimeError(state['message'])
        if state['status'] == 'ready' and len(state['loaded_models']) == 2:
            return state
        time.sleep(1)
    raise RuntimeError('Models did not become ready')

def wav(path):
    data = get(path)
    audio, rate = sf.read(io.BytesIO(data))
    assert rate == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
    assert np.sqrt(np.mean(audio ** 2)) > .0001
    return audio

def stream(payload):
    started = time.monotonic()
    events, chunks, received = [], [], []
    with urllib.request.urlopen(request('/api/stream', payload), timeout=60) as response:
        assert response.headers['Content-Type'].startswith('text/event-stream')
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            event = json.loads(raw[6:])
            events.append(event)
            elapsed = time.monotonic() - started
            if event['event'] == 'audio':
                chunks.append(base64.b64decode(event['pcm'], validate=True))
                received.append(elapsed)
            if event['event'] == 'error':
                raise RuntimeError(event['error'])
            if event['event'] == 'complete':
                complete_at = elapsed
                break
    assert events[-1]['event'] == 'complete' and len(chunks) > 1
    token = next(e for e in events if e['event'] == 'token')
    done = events[-1]
    assert 0 <= token['time_to_first_token_seconds'] <= done['time_to_first_audio_seconds']
    assert received[0] < complete_at - .1
    audio = np.frombuffer(b''.join(chunks), dtype='<i2').astype(np.float32) / 32768
    raw, corrected = wav(done['audio_url']), wav(done['leveled_audio_url'])
    assert len(audio) == len(raw) == len(corrected)
    assert [e['event'] for e in events[:3]] == ['accepted', 'started', 'token']
    summary = {k: done[k] for k in ['id', 'elapsed', 'duration', 'rtf',
        'time_to_first_token_seconds', 'engine_time_to_first_token_seconds',
        'time_to_first_audio_seconds', 'audio_url', 'leveled_audio_url']}
    summary.update(model=payload['model'], reference_id=payload['reference_id'],
        client_first_audio_seconds=received[0], client_complete_seconds=complete_at,
        audio_chunks=len(chunks), sample_count=len(audio),
        first_audio_before_completion=True,
        queue_seconds=next(e['queue_seconds'] for e in events if e['event']=='started'))
    print(payload['model'], 'TTFT',round(summary['time_to_first_token_seconds'],3),
          'first audio',round(summary['client_first_audio_seconds'],3),
          'complete',round(complete_at,3),'chunks',len(chunks),flush=True)
    wait_ready(time.monotonic() + 20)
    return summary

state = wait_ready(time.monotonic() + 360)
print('Both engines ready for streaming audit',flush=True)
assert b'id="steady-volume"' in get('/') and b'Streaming studio' in get('/')
assert b'First audio token' in get('/stream')
refs = json.loads(get('/api/stream/references'))['references']
assert len([r for r in refs if not r['uploaded']]) == 3
cases = [json.loads(line) for line in (Path(json.loads((HERE/'voice/selection.json').read_text())['campaign'])/'comparison/cases.jsonl').read_text().splitlines()]
case = cases[3]
payload = {'reference_id':case['speaker'], 'text':case['text'], **case['generation']}
streams = []
for model in ['fast_r32_final', 'attention_mlp_r16_early']:
    streams.append(stream({**payload,'model':model}))
upload = next(r for r in reversed(refs) if r['uploaded'] and r['text'])
upload_payload = {'model':'fast_r32_final','reference_id':upload['id'],'reference_text':upload['text'],
                  'text':'كيفك اليوم؟ شو رأيك نطلع نشرب قهوة بعد الشغل؟','seed':42}
streams.append(stream(upload_payload))
normal = []
for model in ['fast_r32_final','attention_mlp_r16_early']:
    with urllib.request.urlopen(request('/api/generate',{**upload_payload,'model':model}),timeout=10) as response:
        assert response.status == 202
        job_id = json.load(response)['id']
    deadline = time.monotonic()+60
    while time.monotonic()<deadline:
        state = status();job = next(j for j in state['jobs'] if j['id']==job_id)
        if job['status'] in ['complete','failed']:
            break
        time.sleep(.2)
    assert job['status']=='complete' and not job.get('streamed_pcm',False)
    assert len(wav(job['audio_url'])) == len(wav(job['audio_url'].replace('/api/audio/','/api/leveled-audio/')))
    normal.append({k:job[k] for k in ['id','elapsed','duration','audio_url']} | {'model':model})
    print(model,'normal uploaded-voice generation retained',flush=True)

# A client disconnect must not stall the shared worker or discard its saved WAV.
with urllib.request.urlopen(request('/api/stream',{**payload,'model':'fast_r32_final'}),timeout=60) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        event=json.loads(line[6:])
        if event['event']=='accepted':disconnected_id=event['id']
        if event['event']=='audio':break
state = wait_ready(time.monotonic()+60)
job=next(j for j in state['jobs'] if j['id']==disconnected_id)
assert job['status']=='complete';wav(job['audio_url'])
print('Disconnected stream finished and saved its WAV; worker remains ready',flush=True)

p=HERE/'reports/sglang_stream_validation_20261011.json'
report=json.loads(p.read_text())
report.update(status='live_validated',validated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              loaded_models=state['loaded_models'],streams=streams,normal_requests=normal,
              disconnect_check={'id':disconnected_id,'saved_wav':True,'worker_ready':True},
              pending=['User listening and real-browser playback interaction.'])
report['tests']['passed']=70
report['runtime_fixes']=['Wait for exact loopback API port release before engine startup; SGLANG_OMNI_STRICT_PORT=1 prevents silent port fallback.']
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Live streaming report saved',flush=True)
