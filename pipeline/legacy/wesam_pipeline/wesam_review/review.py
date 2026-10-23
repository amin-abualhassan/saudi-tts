#!/usr/bin/env python3
"""Listen to aligned Wesam corpus clips and audit speaker changes with Community-1."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = 'pyannote/speaker-diarization-community-1'
MANIFEST = ROOT / 'training/wesam_exploration/inventory.jsonl'
CACHE = HERE / 'cache'
DATASET = ROOT / 'training/wesam_dataset_words'
DATASET_ROWS = {}
DATASET_MTIME = None
ROWS = {r['clip_id']: r for r in map(json.loads, MANIFEST.read_text().splitlines())}
LOCK = threading.Lock()
JOBS = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1)
PIPELINE = None
IDENTITY_JOB = {'state': 'idle'}
DEVICE = 'auto'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def result_path(clip_id):
    return CACHE / 'analysis' / (clip_id + '.json')


def load_result(clip_id):
    path = result_path(clip_id)
    return json.loads(path.read_text()) if path.exists() else None


def summarize(turns, start, end, threshold=0.3):
    """Retain real overlap; do not interpret exclusive diarization as overlap-free audio."""
    segments = []
    seconds = defaultdict(float)
    for t in turns:
        a, b = max(t['start'], start), min(t['end'], end)
        if b > a:
            segments.append({'start': a-start, 'end': b-start, 'speaker': t['speaker']})
            seconds[t['speaker']] += b-a
    events = []
    for t in segments:
        events.extend([(t['start'], 1), (t['end'], -1)])
    active, previous, overlap = 0, 0., 0.
    for at, delta in sorted(events):
        if active > 1:
            overlap += at-previous
        active += delta
        previous = at
    count = sum(s >= threshold for s in seconds.values())
    verdict = 'multiple speakers predicted' if count > 1 else ('one speaker predicted' if count == 1 else 'insufficient speech detected')
    return {'segments': segments, 'speaker_seconds': dict(seconds), 'speaker_count': count,
            'overlap_seconds': overlap, 'verdict': verdict, 'speaker_threshold_seconds': threshold}


def analyze(clip_id, device='auto'):
    global PIPELINE
    row = ROWS[clip_id]
    with LOCK:
        JOBS[clip_id] = {'state': 'running', 'message': 'Loading model / analyzing source context'}
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        import torch
        os.environ.setdefault('HF_HUB_CACHE', str(CACHE / 'huggingface' / 'hub'))
        os.environ.setdefault('PYANNOTE_METRICS_ENABLED', '0')
        from huggingface_hub import get_token
        from pyannote.audio import Pipeline
        if PIPELINE is None:
            # Login remains in its existing location; only model artifacts use workspace storage.
            PIPELINE = Pipeline.from_pretrained(MODEL, token=get_token())
            target = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
            PIPELINE.to(torch.device(target))
        source = ROOT / 'full-audios-transcript/audios-48khz' / (row['source_id'] + '.flac')
        with sf.SoundFile(source) as stream:
            rate = stream.samplerate
            first = max(0, round((row['start']-30) * rate))
            last = min(len(stream), round((row['end']+30) * rate))
            stream.seek(first)
            audio = stream.read(last-first, dtype='float32', always_2d=True).mean(axis=1)
        divisor = __import__('math').gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor).astype(np.float32)
        # In-memory input bypasses torchcodec/ffmpeg file decoding.
        output = PIPELINE({'waveform': torch.from_numpy(audio).unsqueeze(0), 'sample_rate': 16000})
        annotation = output.speaker_diarization
        turns = [{'start': float(turn.start), 'end': float(turn.end), 'speaker': speaker}
                 for turn, _, speaker in annotation.itertracks(yield_label=True)]
        offset = first/rate
        result = {'clip_id': clip_id, 'model': MODEL, 'context_start': offset,
                  'context_end': last/rate, 'context_turns': turns,
                  'audio_sha256': hashlib.sha256(Path(row['audio']).read_bytes()).hexdigest(),
                  'note': 'Automatic prediction on clip plus 30 seconds on each side. Speaker IDs are local to this analysis, not verified identities.',
                  **summarize(turns, row['start']-offset, row['end']-offset)}
        save_json(result_path(clip_id), result)
        with LOCK:
            JOBS[clip_id] = {'state': 'done', 'message': result['verdict']}
        return result
    except Exception as exc:
        message = str(exc)
        # Never expose credentials if a library includes them in an error.
        message = re.sub(r'hf_[A-Za-z0-9]+', '[redacted token]', message)
        with LOCK:
            JOBS[clip_id] = {'state': 'error', 'message': f'{type(exc).__name__}: {message}'}
        print(f'Analysis {clip_id}: {type(exc).__name__}: {message}', file=sys.stderr, flush=True)
        return None


def run_identity_task(command, backend, source=None, vad='silero', limit=None, timing=None):
    global IDENTITY_JOB
    try:
        import identity
        with LOCK:
            IDENTITY_JOB = {'state': 'running', 'message': 'Loading speaker model / processing audio'}
        if command == 'build':
            result = identity.build_profile(backend, DEVICE)
            message = f"Reference built: {result['reference_seconds']:.1f} s"
            if not result['calibration']:
                message += '. Scores only until validation examples calibrate the thresholds.'
                if result['calibration_error']:
                    message += ' ' + result['calibration_error']
        else:
            result = identity.identify_source(source, backend, vad, DEVICE, limit_seconds=limit, **(timing or {}))
            message = f"Finished {source}: {len(result['windows'])} speech windows"
        with LOCK:
            IDENTITY_JOB = {'state': 'done', 'message': message}
    except Exception as exc:
        message = re.sub(r'hf_[A-Za-z0-9]+', '[redacted token]', str(exc))
        with LOCK:
            IDENTITY_JOB = {'state': 'error', 'message': f'{type(exc).__name__}: {message}'}


def wav_path(clip_id):
    import soundfile as sf
    destination = CACHE / 'playback' / (clip_id + '.wav')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        audio, rate = sf.read(ROWS[clip_id]['audio'], dtype='float32')
        # Concurrent requests use distinct temporary paths.
        temp = destination.with_name(f'{clip_id}.{threading.get_ident()}.tmp.wav')
        sf.write(temp, audio, rate, subtype='PCM_16')
        temp.replace(destination)
    return destination


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, value, mime='application/json', status=200):
        data = json.dumps(value, ensure_ascii=False).encode() if mime == 'application/json' else value
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global DATASET_ROWS, DATASET_MTIME
        url = urlparse(self.path)
        args = parse_qs(url.query)
        if url.path == '/':
            return self.send((HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if url.path == '/dataset':
            page = DATASET / 'review.html'
            if not page.exists():
                return self.send({'error': 'Refined dataset review is not ready'}, status=404)
            return self.send(page.read_bytes(), 'text/html; charset=utf-8')
        if url.path == '/dataset-audio':
            manifest = DATASET / 'manifest.jsonl'
            if not manifest.exists():
                return self.send({'error': 'No refined manifest'}, status=404)
            mtime = manifest.stat().st_mtime_ns
            with LOCK:
                if DATASET_MTIME != mtime:
                    DATASET_ROWS = {r['clip_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
                    DATASET_MTIME = mtime
                row = DATASET_ROWS.get(args.get('id', [''])[0])
            if not row:
                return self.send({'error': 'Unknown dataset clip'}, status=404)
            path = Path(row['audio']).resolve()
            if not path.is_relative_to(DATASET.resolve()) or not path.exists():
                return self.send({'error': 'Dataset audio unavailable'}, status=404)
            return self.send_audio(path)
        if url.path == '/api/reference':
            import identity
            return self.send(identity.state())
        if url.path == '/api/identity/status':
            with LOCK:
                job = dict(IDENTITY_JOB)
            return self.send(job)
        if url.path == '/api/identity/predictions':
            import identity
            source = args.get('source', [''])[0]
            backend = args.get('backend', ['wespeaker'])[0]
            vad = args.get('vad', ['silero'])[0]
            if source not in {r['source_id'] for r in ROWS.values()} or backend not in identity.MODELS or vad not in ('silero', 'pyannote'):
                return self.send({'error': 'Unknown source/model/VAD'}, status=400)
            path = identity.DATA / 'predictions' / f'{source}_{backend}_{vad}.json'
            value = json.loads(path.read_text()) if path.exists() else {'windows': []}
            if value['windows']:
                try:
                    profile = identity.load_profile(backend)
                    value['stale'] = value.get('profile_fingerprint') != identity.digest(profile)
                except ValueError:
                    value['stale'] = True
            return self.send(value)
        if url.path == '/reference.wav':
            import identity
            path = identity.DATA / 'wesam_reference.wav'
            if not path.exists():
                return self.send({'error': 'Build the reference first'}, status=404)
            return self.send(path.read_bytes(), 'audio/wav')
        if url.path == '/api/episodes':
            episodes = {}
            for row in ROWS.values():
                key = row['source_id']
                entry = episodes.setdefault(key, {'id': key, 'title': row['source_title'], 'clips': 0})
                entry['clips'] += 1
            return self.send(list(episodes.values()))
        if url.path == '/api/clips':
            source = args.get('source', [''])[0]
            rows = [dict(r, analysis=load_result(r['clip_id'])) for r in ROWS.values() if r['source_id'] == source]
            return self.send(sorted(rows, key=lambda r: r['start']))
        clip_id = args.get('id', [''])[0]
        if clip_id not in ROWS:
            return self.send({'error': 'Unknown clip'}, status=404)
        if url.path == '/api/status':
            with LOCK:
                job = dict(JOBS.get(clip_id, {'state': 'idle'}))
            return self.send({'job': job, 'analysis': load_result(clip_id)})
        if url.path == '/audio':
            path = wav_path(clip_id)
            return self.send_audio(path)
        self.send({'error': 'Not found'}, status=404)

    def send_audio(self, path):
        size = path.stat().st_size
        start, end, status = 0, size-1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.send({'error': 'Invalid range'}, status=416)
            a, b = match.groups()
            if a:
                start, end = int(a), min(int(b), size-1) if b else size-1
            else:
                start = max(0, size-int(b))
            if start > end or start >= size:
                return self.send({'error': 'Range outside file'}, status=416)
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end-start+1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end-start+1
            while remaining:
                block = stream.read(min(65536, remaining))
                self.wfile.write(block)
                remaining -= len(block)
        return

    def do_POST(self):
        # Prevent a different website from triggering costly model work.
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.send({'error': 'Origin rejected'}, status=403)
        url = urlparse(self.path)
        if url.path.startswith('/api/reference') or url.path == '/api/identity/run':
            import identity
            global IDENTITY_JOB
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 8192:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length)) if length else {}
                if url.path in ('/api/reference', '/api/reference/remove'):
                    with LOCK:
                        if IDENTITY_JOB['state'] in ('running', 'queued'):
                            return self.send({'error': 'Wait for current identity task before editing references'}, status=409)
                        result = identity.add_selection(data) if url.path == '/api/reference' else identity.remove_selection(data['id'])
                    return self.send(result)
                if url.path not in ('/api/reference/build', '/api/identity/run'):
                    raise ValueError('Unknown operation')
                backend = data.get('backend', 'wespeaker')
                vad = data.get('vad', 'silero')
                source = data.get('source')
                if backend not in identity.MODELS or vad not in ('silero', 'pyannote'):
                    raise ValueError('Unknown embedding model/VAD')
                if url.path == '/api/identity/run' and source not in {r['source_id'] for r in ROWS.values()}:
                    raise ValueError('Unknown episode')
                limit = data.get('limit_seconds')
                if limit is not None and (not isinstance(limit, (int, float)) or not 0 < limit <= 10000):
                    raise ValueError('Invalid analysis limit')
                timing = {key: data.get(key, default) for key, default in
                          [('min_speech_seconds', .5), ('window_seconds', 3.), ('hop_seconds', 1.5)]}
                identity.validate_timing(**timing)
                with LOCK:
                    if IDENTITY_JOB['state'] in ('queued', 'running'):
                        return self.send({'error': 'Identity analysis already running'}, status=409)
                    IDENTITY_JOB = {'state': 'queued', 'message': 'Waiting for model worker'}
                command = 'build' if url.path == '/api/reference/build' else 'identify'
                EXECUTOR.submit(run_identity_task, command, backend, source, vad, limit, timing)
                return self.send({'state': 'queued'}, status=202)
            except (ValueError, KeyError, TypeError) as exc:
                return self.send({'error': str(exc)}, status=400)
        clip_id = parse_qs(url.query).get('id', [''])[0]
        if url.path != '/api/analyze'  or clip_id not in ROWS:
            return self.send({'error': 'Unknown clip'}, status=404)
        with LOCK:
            pending = JOBS.get(clip_id, {}).get('state') in ('queued', 'running')
            if not pending:
                JOBS[clip_id] = {'state': 'queued', 'message': 'Waiting for analysis'}
        if not pending:
            EXECUTOR.submit(analyze, clip_id)
        self.send({'state': 'queued'}, status=202)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--analyze', nargs='+', metavar='CLIP_ID', help='Analyze clips without opening a server')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    DEVICE = args.device
    if args.check:
        import importlib.util
        missing_audio = sum(not Path(r['audio']).is_file() for r in ROWS.values())
        print(json.dumps({'clips': len(ROWS), 'episodes': len({r['source_id'] for r in ROWS.values()}),
                          'missing_audio': missing_audio, 'port': args.port,
                          'pyannote_installed': importlib.util.find_spec('pyannote') is not None}, indent=2))
        return
    if args.analyze:
        for clip_id in args.analyze:
            if clip_id not in ROWS:
                parser.error(f'Unknown clip: {clip_id}')
            if analyze(clip_id, args.device) is None:
                raise SystemExit(1)
            print(json.dumps(load_result(clip_id), ensure_ascii=False))
        return
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Wesam listening review: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
