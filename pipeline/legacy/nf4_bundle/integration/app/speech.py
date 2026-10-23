"""Exact port-3000 studio contract. No alternate model or endpoint fallback."""
import asyncio
import base64
import hashlib
import json
import logging
import re
import time
import wave
from pathlib import Path
import httpx
from livekit import rtc
from .config import settings, ROOT
from .numbers import numbers_to_words

log = logging.getLogger('aljawab')
MODEL_SHA = settings.tts_checkpoint_sha256
STUDIO = settings.tts_studio_directory.resolve()
SAMPLE_RATE = 44100
OWNED = ROOT / 'logs/tts-owned'
LEGACY_OWNED = ROOT / 'logs/tts-jobs.json'

def payload(text):
    if not settings.tts_reference:raise ValueError('Configure the Noura voice reference first')
    request=dict(model=settings.tts_model, reference_id=settings.tts_reference,
                text=numbers_to_words(text,'ar' if re.search(r'[\u0600-\u06ff]',text) else 'en'),
                seed=42, max_new_tokens=1024, temperature=.7, top_p=.7)
    if settings.tts_reference.startswith('upload_'):
        if not settings.tts_reference_text:raise ValueError('Uploaded voice requires its exact reference transcript')
        request['reference_text']=settings.tts_reference_text
    return request

async def check_model():
    payload('اختبار.')
    async with httpx.AsyncClient(trust_env=False) as c:
        r = await c.get(settings.tts_url + '/api/status'); r.raise_for_status(); s = r.json()
        if s.get('status') != 'ready' or settings.tts_model not in s.get('loaded_models', []):
            raise RuntimeError('Required TTS model is not ready')
    if MODEL_SHA != '9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf' and s.get('checkpoint_sha256') != MODEL_SHA:
        raise RuntimeError('Required derivative identity is not ready')
    return True

class Speech:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), trust_env=False)
        OWNED.mkdir(parents=True,exist_ok=True)
        if LEGACY_OWNED.exists():
            for job_id in json.loads(LEGACY_OWNED.read_text()):self.own(job_id)
            LEGACY_OWNED.unlink(missing_ok=True)

    def own(self, job_id):
        if not re.fullmatch('[a-f0-9]{32}',job_id):raise ValueError('Invalid local job ID')
        # One marker per job avoids lost registrations across maintenance processes.
        (OWNED/job_id).touch(mode=0o600,exist_ok=True)

    async def cleanup(self):
        # The pre-existing studio saves every synthesis. Delete only this app's
        # registered jobs once the worker has finished, including interrupted jobs.
        for marker in OWNED.iterdir():
            job_id=marker.name
            if not re.fullmatch('[a-f0-9]{32}',job_id):continue
            path = STUDIO / 'results' / (job_id + '.json')
            try:
                job = json.loads(path.read_text())
                if job.get('status') not in ('complete', 'failed'): continue
                if job['request']['model'] != settings.tts_model: continue
                for folder in ('results', 'leveled'):
                    for suffix in ('.wav', '.json', '.tmp.wav'):
                        (STUDIO / folder / (job_id + suffix)).unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
            except FileNotFoundError:
                marker.unlink(missing_ok=True)
            except json.JSONDecodeError:
                continue

    async def stream(self, text):
        started = time.monotonic(); first = True; buf = bytearray()
        async with self.client.stream('POST', settings.tts_url + '/api/stream', json=payload(text)) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith('data: '): continue
                event = json.loads(line[6:]); kind = event['event']
                if kind == 'accepted':
                    if event.get('encoding') != 'pcm_s16le' or event.get('sample_rate') != SAMPLE_RATE:
                        raise RuntimeError('Unexpected TTS format')
                    self.own(event['id'])
                    status=await self.client.get(settings.tts_url+'/api/status');status.raise_for_status()
                    job=next((j for j in status.json()['jobs'] if j['id']==event['id']),None)
                    if not job or job['request']['model']!=settings.tts_model or job['checkpoint_sha256']!=MODEL_SHA or job['request']['reference_id']!=settings.tts_reference or (settings.tts_reference_sha256 and job['reference_sha256']!=settings.tts_reference_sha256):
                        raise RuntimeError('Streaming TTS model verification failed')
                elif kind == 'audio':
                    if first:
                        log.info('tts_first_audio_ms=%.1f model=%s', (time.monotonic()-started)*1000, settings.tts_model)
                        first = False
                    buf.extend(base64.b64decode(event['pcm'], validate=True))
                    while len(buf) >= 1764:  # 20 ms, independent of HTTP/SSE chunk boundaries
                        data = bytes(buf[:1764]); del buf[:1764]
                        yield rtc.AudioFrame(data, SAMPLE_RATE, 1, 882)
                elif kind == 'error': raise RuntimeError('Local synthesis failed')
                elif kind == 'complete':
                    if event.get('checkpoint_sha256', MODEL_SHA) != MODEL_SHA:
                        raise RuntimeError('TTS checkpoint mismatch')
                    break
        if len(buf) % 2: raise RuntimeError('Truncated PCM sample')
        if buf: yield rtc.AudioFrame(bytes(buf), SAMPLE_RATE, 1, len(buf)//2)
        if first: raise RuntimeError('No speech returned')

    async def generate(self, text):
        r = await self.client.post(settings.tts_url + '/api/generate', json=payload(text)); r.raise_for_status()
        job = r.json()
        if job['request']['model'] != settings.tts_model or job['checkpoint_sha256'] != MODEL_SHA or job['request']['reference_id']!=settings.tts_reference or (settings.tts_reference_sha256 and job['reference_sha256']!=settings.tts_reference_sha256):
            raise RuntimeError('TTS selected the wrong checkpoint')
        self.own(job['id'])
        for _ in range(240):
            await asyncio.sleep(.5)
            r = await self.client.get(settings.tts_url + '/api/status'); r.raise_for_status()
            result = next((j for j in r.json()['jobs'] if j['id'] == job['id']), None)
            if result and result['status'] == 'failed': raise RuntimeError('Clip synthesis failed')
            if result and result['status'] == 'complete':
                audio = await self.client.get(settings.tts_url + result['audio_url']); audio.raise_for_status()
                return audio.content, result
        raise TimeoutError('Clip synthesis timed out')

    async def cached(self, name):
        path = ROOT / 'assets/voice' / (name + '.wav')
        manifest = json.loads((ROOT / 'assets/voice/manifest.json').read_text())
        meta = manifest[name]
        if meta['model'] != settings.tts_model or meta['reference'] != settings.tts_reference or meta['checkpoint_sha256'] != MODEL_SHA or meta.get('reference_sha256','')!=settings.tts_reference_sha256:
            raise RuntimeError('Cached voice settings changed; regenerate assets')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise RuntimeError('Cached audio checksum mismatch')
        with wave.open(str(path)) as w:
            if w.getframerate()!=SAMPLE_RATE or w.getnchannels()!=1 or w.getsampwidth()!=2:
                raise RuntimeError('Unexpected cached format')
            while data := w.readframes(882):
                yield rtc.AudioFrame(data, SAMPLE_RATE, 1, len(data)//2)
