#!/usr/bin/env python3
"""Validate refined files and generate a browser listening review."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import identity
import prepare_dataset as data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'training/wesam_dataset_words'


def main():
    if json.loads((OUT / 'status.json').read_text())['stage'] != 'complete':
        raise RuntimeError('Wait for the export to complete')
    rows = [json.loads(line) for line in (OUT / 'manifest.jsonl').read_text().splitlines()]
    failures, hashes, ids, selections = [], set(), set(), {}
    turns = {p.stem: json.loads(p.read_text())['turns']
             for p in (ROOT / 'training/wesam_dataset_gpu/diarization').glob('*.json')}
    for row in rows:
        selections.setdefault(('speaker', row['speaker']), row)
        selections.setdefault(('episode', row['source_id']), row)
        errors = []
        path = Path(row['audio'])
        info = sf.info(path)
        if row['clip_id'] in ids or row['audio_sha256'] in hashes:
            errors.append('duplicate_id_or_audio')
        ids.add(row['clip_id']); hashes.add(row['audio_sha256'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['audio_sha256']:
            errors.append('audio_checksum_mismatch')
        if path.with_suffix('.lab').read_text().strip() != row['text']:
            errors.append('label_mismatch')
        if info.channels != 1 or info.subtype != 'PCM_16' or info.samplerate != row['sample_rate']:
            errors.append('audio_format')
        if abs(info.duration - row['duration']) > 1 / info.samplerate:
            errors.append('duration_mismatch')
        if not 3 - 1 / info.samplerate <= info.duration <= 30 + 1 / info.samplerate:
            errors.append('clip_duration')
        if row['speech_seconds'] < 3 - 1e-6:
            errors.append('minimum_speech_duration')
        words = row['word_timestamps']
        if ' '.join(w['text'] for w in words) != row['text']:
            errors.append('word_text_mismatch')
        previous = -1.
        for word in words:
            if not -.000001 <= word['start'] < word['end'] <= info.duration + .00003 or word['start'] < previous - 1e-6:
                errors.append('word_boundaries')
                break
            previous = word['end']
        other = [(t['start'], t['end']) for t in turns[row['source_id']] if t['speaker'] != row['local_speaker']]
        if data.duration_union(data.intersections(other, row['start'], row['end'])) > .010001:
            errors.append('other_speaker_inside_clip')
        if errors:
            failures.append({'clip_id': row['clip_id'], 'errors': errors})
    crops = []
    for row in {r['clip_id']: r for r in selections.values()}.values():
        source = ROOT / 'full-audios-transcript/audios-48khz' / (row['source_id'] + '.flac')
        with sf.SoundFile(source) as audio:
            audio.seek(row['source_start_frame'])
            expected = audio.read(row['source_end_frame'] - row['source_start_frame'],
                                  dtype='float32', always_2d=True).mean(axis=1)
        actual, rate = sf.read(row['audio'], dtype='float32')
        if rate != row['source_sample_rate']:
            raise ValueError('This audit expects the native-rate export')
        error = float(np.max(np.abs(expected - actual))) if len(expected) == len(actual) else None
        good = error is not None and error <= 1 / 32768 + 1e-7
        crops.append({'clip_id': row['clip_id'], 'max_sample_error': error, 'passes_pcm16_tolerance': good})
        if not good:
            failures.append({'clip_id': row['clip_id'], 'errors': ['source_crop_mismatch']})
    before = json.loads((ROOT / 'training/wesam_dataset_gpu/summary.json').read_text())
    after = json.loads((OUT / 'summary.json').read_text())
    report = {'status': 'passed' if not failures else 'failed', 'clips_checked': len(rows),
              'episodes': len({r['source_id'] for r in rows}),
              'duration_range_seconds': [min(r['duration'] for r in rows), max(r['duration'] for r in rows)],
              'source_crop_checks': crops, 'failures': failures,
              'comparison': {'srt_clips': before['clips'], 'vtt_clips': after['clips'],
                             'srt_hours': before['hours'], 'vtt_hours': after['hours'],
                             'srt_wesam': before['speakers'].get('wesam_ali'),
                             'vtt_wesam': after['speakers'].get('wesam_ali')},
              'note': 'These checks do not verify spoken text or voice identity. Listening review is pending.'}
    identity.write_json(OUT / 'audit.json', report)
    compact = [{k: r[k] for k in ('clip_id', 'speaker', 'source_id', 'duration', 'start', 'end', 'text', 'identity_score', 'split')}
               for r in rows]
    payload = json.dumps(compact, ensure_ascii=False).replace('<', '\\u003c')
    (OUT / 'review.html').write_text((HERE / 'dataset_review.html').read_text().replace('__ROWS__', payload), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_crop_checks'}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
