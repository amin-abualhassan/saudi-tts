#!/usr/bin/env python3
"""Manual reference enrollment, speaker embeddings, and speech-window identification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import uuid

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'identity'
SELECTIONS = DATA / 'selections.json'
MANIFEST = ROOT / 'training/wesam_exploration/inventory.jsonl'
ROWS = {r['clip_id']: r for r in map(json.loads, MANIFEST.read_text().splitlines())}
MODELS = {'wespeaker': 'pyannote/wespeaker-voxceleb-resnet34-LM',
          'ecapa': 'speechbrain/spkrec-ecapa-voxceleb',
          'wavlm': 'microsoft/wavlm-base-plus-sv'}
ROLES = ('enrollment', 'wesam_validation', 'other_validation')
EDIT_LOCK = threading.Lock()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def selections():
    return json.loads(SELECTIONS.read_text()) if SELECTIONS.exists() else []


def validate_selection(value, current):
    clip_id = value.get('clip_id')
    if clip_id not in ROWS:
        raise ValueError('Unknown clip')
    role = value.get('role', 'enrollment')
    if role not in ROLES:
        raise ValueError('Unknown reference role')
    start, end = float(value['start']), float(value['end'])
    row = ROWS[clip_id]
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= row['duration'] + 1e-5):
        raise ValueError('Start/end must lie inside the clip')
    if not 2 <= end-start <= 10:
        raise ValueError('Choose a clean excerpt between 2 and 10 seconds')
    absolute_start, absolute_end = row['start']+start, row['start']+end
    for old in current:
        if old['source_id'] == row['source_id'] and min(absolute_end, old['source_end'])-max(absolute_start, old['source_start']) > 1e-5:
            raise ValueError('This audio overlaps an existing selection; enrollment and validation must be separate')
    path = Path(row['audio'])
    return {'id': uuid.uuid4().hex, 'clip_id': clip_id, 'source_id': row['source_id'],
            'start': start, 'end': end, 'duration': end-start, 'source_start': absolute_start,
            'source_end': absolute_end, 'role': role,
            'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def add_selection(value):
    with EDIT_LOCK:
        current = selections()
        current.append(validate_selection(value, current))
        write_json(SELECTIONS, current)
    return state()


def remove_selection(selection_id):
    with EDIT_LOCK:
        current = selections()
        if not any(s['id'] == selection_id for s in current):
            raise ValueError('Unknown selection')
        write_json(SELECTIONS, [s for s in current if s['id'] != selection_id])
    return state()


def state():
    selected = selections()
    return {'selections': selected,
            'totals': {role: sum(s['duration'] for s in selected if s['role'] == role) for role in ROLES}}


def read16(path, start=0., end=None):
    with sf.SoundFile(path) as stream:
        rate = stream.samplerate
        first = round(start*rate)
        last = len(stream) if end is None else min(len(stream), round(end*rate))
        stream.seek(first)
        audio = stream.read(last-first, dtype='float32', always_2d=True).mean(axis=1)
    factor = math.gcd(rate, 16000)
    audio = resample_poly(audio, 16000//factor, rate//factor).astype(np.float32)
    if not len(audio) or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 1e-6:
        raise ValueError(f'Invalid or silent audio: {Path(path).name}')
    return audio


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError('Invalid embedding')
    return vector/norm


class Encoder:
    def __init__(self, backend='wespeaker', device='auto'):
        import torch
        os.environ.setdefault('HF_HUB_CACHE', str(HERE / 'cache/huggingface/hub'))
        os.environ.setdefault('PYANNOTE_METRICS_ENABLED', '0')
        self.backend = backend
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        if backend == 'wespeaker':
            from huggingface_hub import get_token
            from pyannote.audio import Model, Inference
            model = Model.from_pretrained(MODELS[backend], token=get_token())
            self.model = Inference(model, window='whole').to(torch.device(self.device))
        elif backend == 'ecapa':
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(source=MODELS[backend],
                savedir=str(HERE / 'cache/ecapa'), run_opts={'device': self.device})
        elif backend == 'wavlm':
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODELS[backend])
            self.model = WavLMForXVector.from_pretrained(MODELS[backend]).to(self.device).eval()
        else:
            raise ValueError('Unknown embedding backend')

    def __call__(self, audio):
        import torch
        with torch.inference_mode():
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            if self.backend == 'wespeaker':
                result = self.model({'waveform': waveform, 'sample_rate': 16000})
            elif self.backend == 'ecapa':
                result = self.model.encode_batch(waveform.to(self.device)).detach().cpu().numpy()
            else:
                inputs = self.processor(audio, sampling_rate=16000, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                result = self.model(**inputs).embeddings.detach().cpu().numpy()
        return unit(result)


def selection_audio(selected):
    path = Path(ROWS[selected['clip_id']]['audio'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != selected['audio_sha256']:
        raise ValueError('A selected source audio changed; remove it and select it again')
    return read16(path, selected['start'], selected['end'])


def windows(start, end, length=3., hop=1.5, minimum=1.5):
    """Overlapping evidence windows; their boundaries are not exact speaker turns."""
    if end-start < minimum:
        return []
    if end-start <= length:
        return [(start, end)]
    positions = list(np.arange(start, end-length+1e-6, hop))
    if end-length-positions[-1] > 1e-6:
        positions.append(end-length)
    return [(float(a), float(a+length)) for a in positions]


def calibrate_scores(positive, negative):
    if len(positive) < 3 or len(negative) < 3:
        raise ValueError('Select at least 3 separate Wesam validation excerpts and 3 other-speaker excerpts')
    low = float(np.quantile(negative, .9))
    high = float(np.quantile(positive, .1))
    if high-low < .05:
        raise ValueError('Validation scores overlap or have too little margin. Review examples or compare a different encoder')
    # Empirical bands, not calibrated probabilities or guaranteed error rates.
    return {'other_max': low, 'wesam_min': high, 'positive_scores': positive,
            'negative_scores': negative, 'status': 'empirical_validation_bands'}


def classify(score, calibration, short=False, overlap=False):
    if score is None or not math.isfinite(score) or short or overlap or not calibration:
        return 'uncertain'
    if score >= calibration['wesam_min']:
        return 'wesam'
    if score <= calibration['other_max']:
        return 'other_speaker'
    return 'uncertain'


def build_profile(backend='wespeaker', device='auto', encoder=None):
    selected = selections()
    enrolled = [s for s in selected if s['role'] == 'enrollment']
    total = sum(s['duration'] for s in enrolled)
    if total < 25:
        raise ValueError(f'Choose about 30 s of clean Wesam speech first; currently {total:.1f} s (minimum 25 s)')
    model = encoder or Encoder(backend, device)
    vectors, parts = [], []
    for s in enrolled:
        audio = selection_audio(s)
        parts.append(audio)
        # Similar duration windows prevent a long excerpt dominating enrollment.
        for a, b in windows(0., len(audio)/16000):
            vectors.append(model(audio[round(a*16000):round(b*16000)]))
    center = unit(np.mean(vectors, axis=0))
    validation = {'wesam_validation': [], 'other_validation': []}
    validation_counts = {role: sum(s['role'] == role for s in selected) for role in validation}
    for s in selected:
        if s['role'] in validation:
            audio = selection_audio(s)
            for a, b in windows(0., len(audio)/16000):
                validation[s['role']].append(float(model(audio[round(a*16000):round(b*16000)]) @ center))
    calibration = None
    calibration_error = None
    if any(validation.values()):
        try:
            if min(validation_counts.values()) < 3:
                raise ValueError('Select at least 3 separate Wesam validation excerpts and 3 other-speaker excerpts')
            calibration = calibrate_scores(validation['wesam_validation'], validation['other_validation'])
        except ValueError as exc:
            calibration_error = str(exc)
    DATA.mkdir(parents=True, exist_ok=True)
    # Concatenation is for listening only. Embeddings were computed per clean excerpt/window.
    sf.write(DATA / 'wesam_reference.wav', np.concatenate(parts), 16000, subtype='PCM_16')
    profile = {'backend': backend, 'model': MODELS[backend], 'embedding': center.tolist(),
               'selection_fingerprint': digest(selected), 'reference_seconds': total,
               'enrollment_sources': sorted({s['source_id'] for s in enrolled}),
               'enrollment_window_count': len(vectors), 'calibration': calibration,
               'calibration_error': calibration_error, 'validation_scores': validation,
               'validation_excerpt_counts': validation_counts,
               'note': 'Cosine scores are not probabilities. Other speaker is not a verified guest identity.'}
    write_json(DATA / f'profile_{backend}.json', profile)
    return profile


def load_profile(backend):
    path = DATA / f'profile_{backend}.json'
    if not path.exists():
        raise ValueError('Build the reference profile first')
    profile = json.loads(path.read_text())
    if profile['selection_fingerprint'] != digest(selections()):
        raise ValueError('Selections changed; rebuild the profile before identification')
    for s in selections():
        if hashlib.sha256(Path(ROWS[s['clip_id']]['audio']).read_bytes()).hexdigest() != s['audio_sha256']:
            raise ValueError('Selected source audio changed; reselect and rebuild')
    return profile


class SpeechDetector:
    def __init__(self, method='silero', device='auto'):
        self.method = method
        if method == 'silero':
            from silero_vad import load_silero_vad
            self.model = load_silero_vad()
        else:
            import torch
            from huggingface_hub import get_token
            from pyannote.audio import Pipeline
            self.model = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=get_token())
            target = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
            self.model.to(torch.device(target))

    def __call__(self, audio):
        import torch
        waveform = torch.from_numpy(audio)
        if self.method == 'silero':
            from silero_vad import get_speech_timestamps
            times = get_speech_timestamps(waveform, self.model, sampling_rate=16000,
                return_seconds=True, min_silence_duration_ms=150, speech_pad_ms=0)
            return [{'start': t['start'], 'end': t['end'], 'speaker': None} for t in times]
        output = self.model({'waveform': waveform.unsqueeze(0), 'sample_rate': 16000})
        return [{'start': float(t.start), 'end': float(t.end), 'speaker': speaker}
                for t, _, speaker in output.speaker_diarization.itertracks(yield_label=True)]


def validate_timing(min_speech_seconds, window_seconds, hop_seconds):
    values = (min_speech_seconds, window_seconds, hop_seconds)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Speech timing settings must be finite numbers')
    if not .25 <= min_speech_seconds <= window_seconds <= 10:
        raise ValueError('Require 0.25 <= minimum speech <= window length <= 10 seconds')
    if not 0 < hop_seconds <= window_seconds:
        raise ValueError('Hop must be positive and no longer than the embedding window')


def identify_source(source_id, backend='wespeaker', vad='silero', device='auto',
                    profile=None, encoder=None, detector=None, limit_seconds=None,
                    min_speech_seconds=.5, window_seconds=3., hop_seconds=1.5):
    """Process one recording in bounded blocks, retaining source timestamps."""
    validate_timing(min_speech_seconds, window_seconds, hop_seconds)
    profile = profile or load_profile(backend)
    model = encoder or Encoder(backend, device)
    detector = detector or SpeechDetector(vad, device)
    center = unit(profile['embedding'])
    source = ROOT / 'full-audios-transcript/audios-48khz' / (source_id + '.flac')
    duration = sf.info(source).duration
    if limit_seconds is not None:
        duration = min(duration, limit_seconds)
    predictions = []
    selected = selections()
    for core_start in np.arange(0., duration, 300.):
        core_end = min(duration, core_start+300.)
        block_start, block_end = max(0., core_start-4.), min(duration, core_end+4.)
        audio = read16(source, block_start, block_end)
        speech = detector(audio)
        for turn in speech:
            spans = windows(turn['start'], turn['end'], window_seconds, hop_seconds, min_speech_seconds)
            if not spans:
                spans = [(turn['start'], turn['end'])]
            for a, b in spans:
                start, end = block_start+a, block_start+b
                midpoint = (start+end)/2
                if not core_start <= midpoint < core_end or b <= a:
                    continue
                overlap = any(t is not turn and t['speaker'] != turn['speaker'] and
                              min(b, t['end']) > max(a, t['start']) for t in speech) if vad == 'pyannote' else False
                sample = audio[round(a*16000):round(b*16000)]
                short = len(sample) < round(min_speech_seconds*16000)
                low_evidence = b-a < 1.5-1e-6
                score, embedding_error = None, None
                if not short and not overlap:
                    try:
                        score = float(model(sample) @ center)
                        if not math.isfinite(score):
                            raise ValueError('Non-finite embedding score')
                    except (ValueError, RuntimeError) as exc:
                        score = None
                        embedding_error = type(exc).__name__
                # Calibration was fitted on longer windows, so short scores remain estimates.
                label = classify(score, profile['calibration'], short or low_evidence, overlap)
                used_reference = any(s['source_id'] == source_id and s['role'] == 'enrollment' and
                                     min(end, s['source_end']) > max(start, s['source_start']) for s in selected)
                predictions.append({'source_id': source_id, 'start': start, 'end': end,
                    'label': label, 'cosine_similarity': score, 'overlap': overlap if vad == 'pyannote' else None,
                    'short_speech': short, 'low_evidence': low_evidence,
                    'min_speech_seconds': min_speech_seconds, 'embedding_error': embedding_error,
                    'no_score_reason': ('short_speech_and_overlap' if short and overlap else
                                        'short_speech' if short else 'overlap' if overlap else
                                        'embedding_failed' if embedding_error else None),
                    'local_speaker': turn['speaker'],
                    'intersects_enrollment': used_reference})
        print(f'{source_id}: {core_end:.0f}/{duration:.0f} seconds processed', flush=True)
    result = {'source_id': source_id, 'model': MODELS[backend], 'vad': vad,
              'profile_fingerprint': digest(profile), 'calibration': profile['calibration'],
              'processed_seconds': duration, 'window_seconds': window_seconds, 'hop_seconds': hop_seconds,
              'min_speech_seconds': min_speech_seconds,
              'warning': 'Overlapping evidence windows, not exact speaker boundaries. Silero does not detect overlap or speaker changes. Labels require listening review.',
              'windows': predictions}
    path = DATA / 'predictions' / f'{source_id}_{backend}_{vad}.json'
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=list(MODELS), default='wespeaker')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    add = sub.add_parser('add')
    add.add_argument('clip_id')
    add.add_argument('start', type=float)
    add.add_argument('end', type=float)
    add.add_argument('--role', choices=ROLES, default='enrollment')
    sub.add_parser('build')
    identify = sub.add_parser('identify')
    identify.add_argument('--source', help='Episode ID; omit for all 23')
    identify.add_argument('--vad', choices=['silero', 'pyannote'], default='silero')
    identify.add_argument('--min-speech-seconds', type=float, default=.5, help='Minimum speech to embed (default 0.5 s)')
    identify.add_argument('--window-seconds', type=float, default=3., help='Maximum embedding window (default 3 s)')
    identify.add_argument('--hop-seconds', type=float, default=1.5, help='Window advance (default 1.5 s)')
    identify.add_argument('--limit-seconds', type=float, help='Optional bounded trial per episode')
    args = parser.parse_args()
    try:
        if args.command == 'status':
            print(json.dumps(state(), ensure_ascii=False, indent=2))
        elif args.command == 'add':
            print(json.dumps(add_selection(vars(args)), ensure_ascii=False, indent=2))
        elif args.command == 'build':
            profile = build_profile(args.backend, args.device)
            print(json.dumps({k:v for k,v in profile.items() if k != 'embedding'}, indent=2))
        else:
            sources = sorted({r['source_id'] for r in ROWS.values()})
            if args.source and args.source not in sources:
                raise ValueError('Unknown source ID')
            if args.limit_seconds is not None and args.limit_seconds <= 0:
                raise ValueError('Limit must be positive')
            validate_timing(args.min_speech_seconds, args.window_seconds, args.hop_seconds)
            profile = load_profile(args.backend)
            model = Encoder(args.backend, args.device)
            detector = SpeechDetector(args.vad, args.device)
            for source in [args.source] if args.source else sources:
                identify_source(source, args.backend, args.vad, args.device, profile, model, detector, args.limit_seconds,
                                args.min_speech_seconds, args.window_seconds, args.hop_seconds)
    except (ValueError, ImportError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
