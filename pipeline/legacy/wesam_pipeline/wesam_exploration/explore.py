#!/usr/bin/env python3
"""Read-only exploration of the non-Talha Jordan-YouTube source; writes reports only."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'training/talha_s2pro'))
import pipeline

spec = importlib.util.spec_from_file_location('s2_audit_helpers', ROOT / 'training/talha_s2pro/run.py')
s2_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2_audit)


def quantiles(values):
    if not values:
        return {}
    return {name: float(value) for name, value in zip(('min', 'p10', 'median', 'p90', 'max'),
                                                     np.quantile(values, [0, .1, .5, .9, 1]))}


def main():
    refs = pipeline.read_reference(ROOT / 'refrence.txt')
    other = {key: value for key, value in refs.items() if value['section'] == 'old_name - new_key'}
    by_source = defaultdict(list)
    all_sources = Counter()
    unmapped = []
    for audio in (ROOT / 'splitted-audio-transcript/Jordan-youtube-48khz').glob('*.flac'):
        key = audio.stem.partition('-')[0]
        all_sources[refs[key]['section'] if key in refs else 'unmapped'] += 1
        if key in other:
            by_source[key].append(audio)
        elif key not in refs:
            unmapped.append(audio.name)
    episodes, manifest, quality_samples = [], [], []
    reasons, format_counts = Counter(), Counter()
    for key, ref in other.items():
        print(f"Inspecting {key}: {len(by_source[key])} clips", flush=True)
        full_audio = ROOT / 'full-audios-transcript/audios-48khz' / (key + '.flac')
        srt = ROOT / 'full-audios-transcript/transcript' / (key + '.ar-new.srt')
        cues = pipeline.srt_cues(srt) if srt.exists() else {}
        lookup = defaultdict(list)
        for number, cue in cues.items():
            lookup[cue[2]].append((number, cue))
        clips = sorted(by_source[key], key=lambda path: int(path.stem.rpartition('-')[2]))
        local_reasons = Counter()
        rows = []
        previous_cue = 0
        for audio in clips:
            info = pipeline.flac_info(audio)
            format_counts[f"{info['sample_rate']} Hz / {info['channels']} channel(s)"] += 1
            label = ROOT / 'splitted-audio-transcript/transcript' / (audio.stem + '.txt')
            text = ' '.join(label.read_text(encoding='utf-8').split()) if label.exists() else ''
            flags = []
            if not label.exists():
                flags.append('missing_label')
            if not text:
                flags.append('empty_transcript')
            if re.search(r'<\|.*?\|>|[<>]', text):
                flags.append('control_text')
            if not 5 <= info['duration'] <= 30:
                flags.append('outside_qwen_runner_5_30_pilot_range')
            if info['sample_rate'] != 48000 or info['channels'] != 1:
                flags.append('unexpected_audio_format')
            rate = len(text.split()) / info['duration']
            if rate > 6:
                flags.append('more_than_6_words_per_second')
            matches = [(number, cue) for number, cue in lookup.get(text, [])
                       if number > previous_cue and abs(cue[1] - cue[0] - info['duration']) <= .05]
            row = {'clip_id': audio.stem, 'source_id': key, 'source_title': ref['title'],
                   'source_group': hashlib.sha256(ref['title'].encode()).hexdigest(),
                   'audio': str(audio), 'label': str(label), 'text': text, **info,
                   'words_per_second': rate, 'speaker': None,
                   'review_status': 'unassigned_speaker_and_unreviewed_transcript'}
            if matches:
                number, (start, end, _) = min(matches)
                previous_cue = number
                row.update(start=start, end=end, srt_cue_number=number,
                           alignment_matches=len(matches))
            else:
                flags.append('unmatched_srt_timing')
            row['audit_flags'] = flags
            local_reasons.update(flags)
            rows.append(row)
        durations = [row['duration'] for row in rows]
        clean = [row for row in rows if not row['audit_flags']]
        episode = {'source_id': key, 'title': ref['title'], 'clips': len(rows),
                   'clip_hours': sum(durations) / 3600, 'duration_seconds': quantiles(durations),
                   'structural_candidate_clips': len(clean),
                   'structural_candidate_hours': sum(row['duration'] for row in clean) / 3600,
                   'audit_flags': dict(local_reasons), 'full_audio_exists': full_audio.exists(),
                   'srt_exists': srt.exists(), 'srt_cues': len(cues),
                   'full_audio_hours': pipeline.flac_info(full_audio)['duration'] / 3600 if full_audio.exists() else None}
        episodes.append(episode)
        manifest.extend(rows)
        reasons.update(local_reasons)
        # A deterministic sample per episode, not a corpus-wide acoustic qualification.
        picked = []
        for index in sorted({0, len(clean) // 2, len(clean) - 1}):
            if clean:
                picked.append(clean[index])
        for row in picked:
            waveform, rate = sf.read(row['audio'], dtype='float32')
            finite = bool(np.isfinite(waveform).all())
            try:
                alignment = s2_audit.verify_source_interval(waveform, rate, row, full_audio)
            except (ValueError, s2_audit.QualityRejection) as error:
                alignment = {'error': str(error)}
            quality_samples.append({'clip_id': row['clip_id'], 'source_id': key, 'duration': row['duration'],
                'finite': finite, 'peak': float(np.max(np.abs(waveform))) if finite else None,
                'rms': float(np.sqrt(np.mean(waveform.astype('float64') ** 2))) if finite else None,
                'fullscale_fraction': float(np.mean(np.abs(waveform) >= 1 - 1 / 32768)) if finite else None,
                'decoded_pcm_sha256': hashlib.sha256(waveform.tobytes()).hexdigest(),
                'source_interval_check': alignment, 'text': row['text']})
    parquet = []
    try:
        import pyarrow.parquet as pq
        for folder in ('youtube-jordan-48khz', 'youtube-jordan-telephony'):
            for path in sorted((ROOT / folder).glob('*.parquet')):
                file = pq.ParquetFile(path)
                item = {'path': str(path), 'rows': file.metadata.num_rows,
                        'columns': file.schema.names, 'bytes': path.stat().st_size}
                name_column = next((name for name in ('audio_name', 'audio_filepath', 'name') if name in file.schema.names), None)
                if name_column:
                    counts, source_ids, names = Counter(), set(), set()
                    for batch in file.iter_batches(columns=[name_column], batch_size=4096):
                        for value in batch.column(0).to_pylist():
                            name = str(value)
                            stem = Path(name).stem
                            match = re.match(r'([a-z0-9]{10})(?:-|$)', stem)
                            key = match[1] if match else None
                            category = refs[key]['section'] if key in refs else 'unmapped_name'
                            counts[category] += 1
                            if key in other:
                                source_ids.add(key)
                                names.add(name)
                    item.update(source_counts=dict(counts), wesam_source_ids=sorted(source_ids),
                                wesam_unique_audio_names=len(names))
                parquet.append(item)
    except ImportError:
        parquet = [{'error': 'pyarrow unavailable in this Python environment'}]
    clean = [row for row in manifest if not row['audit_flags']]
    text_counts = Counter(row['text'] for row in manifest if row['text'])
    repeated_text = {text: count for text, count in text_counts.items() if count > 1}
    report = {'source': 'wesamali.official (identified by README.md and first reference section)',
        'reference_sections': dict(Counter(row['section'] for row in refs.values())),
        'local_all_source_clip_counts': dict(all_sources), 'episodes': episodes,
        'summary': {'reference_episodes': len(other), 'episodes_with_clips': len(by_source),
            'clips': len(manifest), 'clip_hours': sum(row['duration'] for row in manifest) / 3600,
            'full_audio_hours': sum(episode['full_audio_hours'] or 0 for episode in episodes),
            'structural_candidate_clips': len(clean),
            'structural_candidate_hours': sum(row['duration'] for row in clean) / 3600,
            'duration_seconds': quantiles([row['duration'] for row in manifest]),
            'words_per_second': quantiles([row['words_per_second'] for row in manifest]),
            'formats': dict(format_counts), 'audit_flags': dict(reasons),
            'repeated_transcript_strings': len(repeated_text),
            'clips_with_repeated_transcripts': sum(repeated_text.values()),
            'speaker_labels_available': False, 'acoustic_sample_clips': len(quality_samples)},
        'acoustic_samples': quality_samples, 'parquet': parquet,
        'existing_rttm_files': [str(path.relative_to(ROOT)) for path in (ROOT / 'Splitter').rglob('*.rttm')],
        'manual_review_shortlist': {
            'low_transcript_rate': [{'clip_id': row['clip_id'], 'words_per_second': row['words_per_second'],
                                     'duration': row['duration'], 'text': row['text']}
                                    for row in manifest if row['words_per_second'] < .75],
            'music_or_applause_text': [{'clip_id': row['clip_id'], 'text': row['text']}
                                      for row in manifest if re.search(r'موسيقى|تصفيق|music', row['text'], re.I)]},
        'limitations': ['No diarization or speaker identity inference was performed.',
            'SRT matches establish text/timing metadata consistency, not acoustic transcript correctness.',
            'Structural candidates are not verified single-speaker TTS examples.',
            'Acoustic statistics are sampled, not a corpus-wide audit.']}
    partitions = [item for item in parquet if '/youtube-jordan-48khz/' in item.get('path', '')]
    if len(partitions) == 2:
        report['existing_parquet_split_overlap'] = sorted(set(partitions[0].get('wesam_source_ids', [])) &
                                                          set(partitions[1].get('wesam_source_ids', [])))
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for filename, rows in (('inventory.jsonl', manifest), ('structural_candidates.jsonl', clean)):
        (OUT / filename).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(json.dumps({'summary': report['summary'], 'parquet': parquet}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
