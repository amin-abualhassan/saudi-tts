#!/usr/bin/env python3
"""Export word-timed speaker clips using existing VTT and saved diarization only."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import fcntl
import hashlib
import html
import json
from pathlib import Path
import re
import time

import identity
import prepare_dataset as data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAMP = r'\d{2}:\d{2}:\d{2}\.\d+'
INLINE = re.compile('<(' + STAMP + ')>')


def seconds(value):
    hours, minutes, tail = value.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(tail)


def plain(value):
    return ' '.join(html.unescape(re.sub(r'<[^>]*>', '', value)).replace('>>', ' ').split())


def vtt_words(path):
    """Read fresh caption lines, ignoring repeated rolling-caption history.

    Inline timestamps give onsets, not measured word ends. Each end is bounded
    by the next distinct onset or its original caption end, whichever is first.
    Words sharing an onset stay together; their durations are not invented.
    """
    events, diagnostics, seen = [], Counter(), set()
    for cue_id, block in enumerate(re.split(r'\n\s*\n', path.read_text(encoding='utf-8-sig').strip()), 1):
        lines = block.splitlines()
        timing = next((i for i, line in enumerate(lines) if '-->' in line), None)
        if timing is None:
            continue
        match = re.search('(' + STAMP + r')\s+-->\s+(' + STAMP + ')', lines[timing])
        if not match:
            diagnostics['invalid_cue_timing'] += 1
            continue
        begin, end = map(seconds, match.groups())
        if end - begin <= .02:
            diagnostics['rolling_clear_cues'] += 1
            continue
        fresh = [line for line in lines[timing + 1:] if INLINE.search(line)]
        # A non-rolling, single-line cue can contain one word and no inline mark.
        nonempty = [line for line in lines[timing + 1:] if plain(line)]
        if not fresh and len(nonempty) == 1:
            fresh = nonempty
        if not fresh:
            diagnostics['history_only_cues'] += 1
            continue
        for line in fresh:
            pieces = INLINE.split(line)
            onset = begin
            for index in range(0, len(pieces), 2):
                if index:
                    onset = seconds(pieces[index - 1])
                text = plain(pieces[index])
                if not text:
                    continue
                key = (round(onset, 6), text)
                if key in seen:
                    diagnostics['repeated_timed_fragments'] += 1
                    continue
                seen.add(key)
                if not begin <= onset <= end:
                    diagnostics['out_of_cue_onset'] += 1
                    continue
                events.append({'start': onset, 'caption_end': end, 'text': text,
                               'cue_ids': [cue_id]})
    events.sort(key=lambda row: row['start'])
    grouped = []
    for event in events:
        if grouped and abs(event['start'] - grouped[-1]['start']) < 1e-6:
            grouped[-1]['text'] += ' ' + event['text']
            grouped[-1]['cue_ids'] = sorted(set(grouped[-1]['cue_ids'] + event['cue_ids']))
            grouped[-1]['caption_end'] = min(grouped[-1]['caption_end'], event['caption_end'])
            diagnostics['shared_onset_fragments'] += 1
        else:
            grouped.append(event.copy())
    words = []
    for index, word in enumerate(grouped):
        next_onset = grouped[index + 1]['start'] if index + 1 < len(grouped) else word['caption_end']
        word['end'] = min(next_onset, word.pop('caption_end'))
        word['word_id'] = index + 1
        if word['end'] <= word['start']:
            diagnostics['zero_duration_fragments'] += 1
            continue
        word['end_provenance'] = 'next_distinct_caption_onset_or_caption_end'
        words.append(word)
    diagnostics['timed_fragments'] = len(words)
    diagnostics['lexical_words'] = sum(len(w['text'].split()) for w in words)
    return words, dict(diagnostics)


def plan_word_clips(words, turns, identities, minimum=3., maximum=30., max_gap=.5):
    overlaps = data.overlap_regions(turns)
    by_speaker = defaultdict(list)
    for turn in turns:
        by_speaker[turn['speaker']].append((turn['start'], turn['end']))
    def indexed(intervals):
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        starts, ends = [a for a, _ in merged], [b for _, b in merged]
        return lambda a, b: data.intersections(
            merged[bisect_right(ends, a):bisect_left(starts, b)], a, b)
    by_speaker = {s: indexed(spans) for s, spans in by_speaker.items()}
    overlap_intersections = indexed(overlaps)
    accepted, rejected, pending = [], [], None

    def flush():
        nonlocal pending
        if pending:
            a, b = pending['start'], pending['end']
            reason = None
            if b - a < minimum or pending['speech_seconds'] < minimum:
                reason = 'less_than_3_seconds_single_speaker_speech'
            elif pending['speech_seconds'] / (b - a) < .6:
                reason = 'low_speech_coverage'
            elif len(pending['text'].split()) / (b - a) > 6:
                reason = 'implausible_transcript_rate'
            if reason:
                rejected.append({'start': a, 'end': b, 'word_ids': pending['word_ids'], 'reason': reason})
            else:
                # Relative times allow playback and transcript-boundary inspection.
                pending['word_timestamps'] = [{**w, 'start': w['start'] - a, 'end': w['end'] - a}
                                              for w in pending['word_timestamps']]
                accepted.append(pending)
        pending = None

    for word in words:
        a, b = word['start'], word['end']
        hits = {s: data.duration_union(query(a, b)) for s, query in by_speaker.items()}
        hits = {s: duration for s, duration in hits.items() if duration > .01}
        reason = None
        if b - a > 2.5:
            reason = 'long_uncertain_word_interval'
        elif re.search(r'\[.*?\]|(?<!\S)(?:موسيقى|تصفيق)(?!\S)|[<>]', word['text']):
            reason = 'non_speech_or_control_transcript'
        elif data.duration_union(overlap_intersections(a, b)) > .01:
            reason = 'overlapping_speakers'
        elif len(hits) != 1:
            reason = 'speaker_change_within_word_interval' if hits else 'no_detected_speech'
        else:
            local = next(iter(hits))
            mapping = identities.get(local, {})
            if not mapping.get('speaker'):
                reason = mapping.get('reason', 'unidentified_speaker')
            elif hits[local] / (b - a) < .3:
                reason = 'word_interval_low_speech_coverage'
        if reason:
            flush()
            rejected.append({'start': a, 'end': b, 'word_ids': [word['word_id']], 'reason': reason})
            continue
        can_join = pending and pending['local_speaker'] == local and \
            -.000001 <= a - pending['end'] <= max_gap and b - pending['start'] <= maximum
        if can_join:
            can_join = not any(query(pending['end'], a) for speaker, query in by_speaker.items()
                               if speaker != local)
        if can_join:
            pending['end'] = b
            pending['text'] += ' ' + word['text']
            pending['word_ids'].append(word['word_id'])
            pending['cue_ids'] = sorted(set(pending['cue_ids'] + word['cue_ids']))
            pending['speech_seconds'] += hits[local]
            pending['word_timestamps'].append(word.copy())
        else:
            flush()
            pending = {'start': a, 'end': b, 'text': word['text'], 'cue_ids': word['cue_ids'][:],
                       'word_ids': [word['word_id']], 'word_timestamps': [word.copy()],
                       'local_speaker': local, 'speaker': mapping['speaker'],
                       'speech_seconds': hits[local], 'identity_score': mapping.get('cosine_similarity')}
    flush()
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'training/wesam_dataset_gpu')
    parser.add_argument('--out', type=Path, default=ROOT / 'training/wesam_dataset_words')
    parser.add_argument('--source')
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    if out == base:
        parser.error('Refinement must use a separate output directory')
    original = json.loads((base / 'recipe.json').read_text())
    fingerprint = identity.digest(original)
    if original.get('limit_seconds') is not None:
        parser.error('Use full-episode diarization, not a limited pilot')
    sources = sorted(original['inputs'])
    if args.source:
        if args.source not in sources:
            parser.error('Unknown episode')
        sources = [args.source]
    input_hashes = {}
    for source in sources:
        audio = ROOT / 'full-audios-transcript/audios-48khz' / (source + '.flac')
        stat = audio.stat()
        expected = original['inputs'][source]
        if stat.st_size != expected['audio_size'] or stat.st_mtime_ns != expected['audio_mtime_ns']:
            raise ValueError(f'Source audio changed: {source}')
        for kind in ('episodes', 'diarization'):
            path = base / kind / (source + '.json')
            row = json.loads(path.read_text())
            if row.get('recipe_fingerprint') != fingerprint:
                raise ValueError(f'Stale {kind}: {source}')
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        vtt = ROOT / 'full-audios-transcript/transcript' / (source + '.ar.vtt')
        input_hashes[str(vtt)] = hashlib.sha256(vtt.read_bytes()).hexdigest()
    recipe = {'revision': 1, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'exporter_sha256': hashlib.sha256(Path(data.__file__).read_bytes()).hexdigest(),
              'base_recipe_fingerprint': fingerprint, 'input_sha256': input_hashes,
              'profile_fingerprint': original['profile_fingerprint'], 'sample_rate': original['sample_rate'],
              'minimum_seconds': 3, 'maximum_seconds': 30, 'max_gap': .5,
              'minimum_speech_coverage': .6, 'minimum_word_speech_coverage': .3,
              'max_word_interval_seconds': 2.5,
              'timing_provenance': 'original_caption_onsets_with_inferred_interval_ends',
              'split_by_source': original['split_by_source']}
    refined_fingerprint = identity.digest(recipe)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'recipe.json').exists() and json.loads((out / 'recipe.json').read_text()) != recipe:
        parser.error('Different refinement recipe; choose a separate output directory')
    identity.write_json(out / 'recipe.json', recipe)
    (out / 'episodes').mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        for index, source in enumerate(sources, 1):
            receipt_path = out / 'episodes' / (source + '.json')
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get('recipe_fingerprint') == refined_fingerprint and all(
                    Path(r['audio']).exists() and
                    hashlib.sha256(Path(r['audio']).read_bytes()).hexdigest() == r['audio_sha256'] and
                    Path(r['audio']).with_suffix('.lab').read_text().strip() == r['text']
                    for r in receipt['clips']):
                    print(f'Reuse {source}', flush=True)
                    continue
            identity.write_json(out / 'status.json', {'stage': 'word_export', 'source_id': source,
                'episode_index': index, 'requested_episodes': len(sources), 'models_rerun': False})
            turns = json.loads((base / 'diarization' / (source + '.json')).read_text())['turns']
            mapping = json.loads((base / 'episodes' / (source + '.json')).read_text())['identities']
            words, diagnostics = vtt_words(ROOT / 'full-audios-transcript/transcript' / (source + '.ar.vtt'))
            clips, rejected = plan_word_clips(words, turns, mapping)
            audio = ROOT / 'full-audios-transcript/audios-48khz' / (source + '.flac')
            records = data.export_clips(audio, source, clips, out, original['split_by_source'][source],
                                       original['sample_rate'], original['profile_fingerprint'], rejected)
            for record in records:
                record['transcript_provenance'] = 'deduplicated_original_vtt_word_onsets'
                record['word_end_provenance'] = 'next_distinct_caption_onset_or_caption_end_not_forced_alignment'
            identity.write_json(receipt_path, {'source_id': source, 'recipe_fingerprint': refined_fingerprint,
                'identities': mapping, 'clips': records, 'rejected': rejected, 'vtt_diagnostics': diagnostics})
            print(f'Export {index}/{len(sources)} {source}: {len(records)} clips, '
                  f'{sum(r["duration"] for r in records)/60:.1f} minutes', flush=True)
            data.rebuild_manifests(out, refined_fingerprint)
        summary = data.rebuild_manifests(out, refined_fingerprint)
        summary['limitations'][-1] = ('Word starts are original caption timings; interval ends are inferred. '
                                    'Boundary clipping and transcript errors require listening review or forced alignment.')
        summary['rejection_counts'] = dict(sum((Counter(r['reason'] for r in json.loads(p.read_text())['rejected'])
                                               for p in (out / 'episodes').glob('*.json')), Counter()))
        identity.write_json(out / 'summary.json', summary)
        identity.write_json(out / 'status.json', {'stage': 'complete', 'completed_episodes': summary['completed_episodes'],
            'clips': summary['clips'], 'hours': summary['hours'], 'elapsed_seconds': time.monotonic() - started,
            'models_rerun': False})
    except (Exception, KeyboardInterrupt) as exc:
        identity.write_json(out / 'status.json', {'stage': 'error' if not isinstance(exc, KeyboardInterrupt) else 'interrupted',
                                               'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
