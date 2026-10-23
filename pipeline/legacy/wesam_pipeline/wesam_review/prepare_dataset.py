#!/usr/bin/env python3
"""Resume full-episode diarization and export transcript-paired single-speaker WAVs."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import identity
from gpu_runtime import configure_pipeline, resolve_device

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / 'training/wesam_dataset'
REVISION = 1
ACTIVE_OUT = None


def srt_cues(path):
    def seconds(value):
        h, m, s = value.replace(',', '.').split(':')
        return int(h)*3600+int(m)*60+float(s)
    result = []
    for block in re.split(r'\n\s*\n', path.read_text(encoding='utf-8-sig').strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and lines[0].strip().isdigit() and '-->' in lines[1]:
            a, b = lines[1].split('-->')
            result.append({'cue_id': int(lines[0]), 'start': seconds(a.strip()),
                           'end': seconds(b.strip().split()[0]), 'text': ' '.join(' '.join(lines[2:]).split())})
    if not result:
        raise ValueError(f'No subtitle cues: {path}')
    return sorted(result, key=lambda r: (r['start'], r['end']))


def duration_union(intervals):
    end, total = -math.inf, 0.
    for a, b in sorted(intervals):
        if b > max(a, end):
            total += b-max(a, end)
        end = max(end, b)
    return total


def overlap_regions(turns):
    events = defaultdict(list)
    for t in turns:
        events[t['start']].append((t['speaker'], 1))
        events[t['end']].append((t['speaker'], -1))
    active, previous, regions = Counter(), None, []
    for at in sorted(events):
        if previous is not None and at > previous and sum(n > 0 for n in active.values()) > 1:
            regions.append((previous, at))
        for speaker, delta in events[at]:
            active[speaker] += delta
        previous = at
    return regions


def intersections(intervals, start, end):
    return [(max(a,start), min(b,end)) for a,b in intervals if min(b,end)>max(a,start)]


def clean_spans(turns, speaker):
    spans = []
    blockers = overlap_regions(turns)
    for t in turns:
        if t['speaker'] != speaker:
            continue
        cursor = t['start']
        for a,b in intersections(blockers,t['start'],t['end']):
            if a > cursor:
                spans.append((cursor,a))
            cursor = max(cursor,b)
        if cursor < t['end']:
            spans.append((cursor,t['end']))
    return sorted(spans)


def speaker_identities(turns, audio16, encoder, profile, source_id, threshold):
    result = {}
    for speaker in sorted({t['speaker'] for t in turns}):
        # Longest clean turns first, at most 60 seconds of representative voice.
        spans = sorted(clean_spans(turns,speaker), key=lambda s: s[1]-s[0], reverse=True)
        vectors, scores, seconds = [], [], 0.
        for a,b in spans:
            if b-a < 3:
                continue
            for start,end in identity.windows(a,b,length=3,hop=3,minimum=3):
                vector = encoder(audio16[round(start*16000):round(end*16000)])
                vectors.append(vector)
                scores.append(float(vector @ identity.unit(profile['embedding'])))
                seconds += end-start
                if seconds >= 60-1e-6:
                    break
            if seconds >= 60-1e-6:
                break
        if seconds < 9:
            result[speaker] = {'speaker': None, 'reason': 'insufficient_clean_identity_evidence', 'evidence_seconds': seconds}
            continue
        score = float(identity.unit(np.mean(vectors,axis=0)) @ identity.unit(profile['embedding']))
        # Use the user-observed threshold with per-window consistency to reject unstable clusters.
        wesam_votes = sum(s > threshold for s in scores)/len(scores)
        if wesam_votes >= .8 and score > threshold:
            label, reason = 'wesam_ali', None
        elif wesam_votes <= .2 and score < threshold:
            safe = re.sub('[^a-zA-Z0-9_]', '_', speaker).lower()
            label, reason = f'guest_{source_id}_{safe}', None
        else:
            label, reason = None, 'inconsistent_identity_scores'
        result[speaker] = {'speaker': label, 'reason': reason, 'cosine_similarity': score,
                           'wesam_vote_fraction': wesam_votes, 'evidence_seconds': seconds,
                           'window_scores': scores}
    return result


def plan_clips(cues, turns, identities, minimum=3., maximum=30., max_gap=.5, min_coverage=.6):
    """Keep whole cues. Never assign a mixed cue's entire transcript to one turn."""
    overlaps = overlap_regions(turns)
    by_speaker = defaultdict(list)
    for t in turns:
        by_speaker[t['speaker']].append((t['start'],t['end']))
    accepted, rejected, pending = [], [], None

    def flush():
        nonlocal pending
        if pending:
            if pending['end']-pending['start'] >= minimum and pending['speech_seconds'] >= minimum:
                accepted.append(pending)
            else:
                rejected.append({'cue_ids': pending['cue_ids'], 'reason': 'less_than_3_seconds_single_speaker_speech'})
        pending = None

    for cue in cues:
        a,b,text = cue['start'],cue['end'],cue['text']
        hits = {s:duration_union(intersections(spans,a,b)) for s,spans in by_speaker.items()}
        hits = {s:d for s,d in hits.items() if d > .01}
        reason = None
        if b <= a or b-a > maximum:
            reason = 'invalid_or_overlong_complete_cue'
        elif not text or re.search(r'<\|.*?\|>|[<>]',text):
            reason = 'empty_or_control_transcript'
        elif duration_union(intersections(overlaps,a,b)) > .01:
            reason = 'overlapping_speakers'
        elif len(hits) != 1:
            reason = 'speaker_change_within_cue' if hits else 'no_detected_speech'
        else:
            local = next(iter(hits))
            mapping = identities.get(local,{})
            if not mapping.get('speaker'):
                reason = mapping.get('reason','unidentified_speaker')
            elif hits[local]/(b-a) < min_coverage:
                reason = 'low_speech_coverage'
            elif len(text.split())/(b-a) > 6:
                reason = 'implausible_transcript_rate'
            elif re.search(r'(?<!\S)(?:موسيقى|تصفيق)(?!\S)',text):
                reason = 'music_or_applause_text_needs_review'
        if reason:
            flush()
            rejected.append({'cue_ids':[cue['cue_id']], 'start':a,'end':b,'reason':reason})
            continue
        row = {'start':a,'end':b,'text':text,'cue_ids':[cue['cue_id']],
               'local_speaker':local,'speaker':mapping['speaker'], 'speech_seconds':hits[local],
               'identity_score':mapping.get('cosine_similarity')}
        can_join = pending and pending['local_speaker'] == local and 0 <= a-pending['end'] <= max_gap and b-pending['start'] <= maximum
        if can_join:
            gap_has_other = any(t['speaker'] != local and min(a,t['end']) > max(pending['end'],t['start']) for t in turns)
            can_join = not gap_has_other
        if can_join:
            pending['end'] = b
            pending['text'] += ' '+text
            pending['cue_ids'].append(cue['cue_id'])
            pending['speech_seconds'] += hits[local]
        else:
            flush()
            pending = row
    flush()
    return accepted,rejected


def episode_splits(sources):
    order = sorted(sources,key=lambda s:hashlib.sha256(('wesam-split-v1:'+s).encode()).hexdigest())
    holdout = max(1,round(len(order)*.1)) if len(order)>=3 else 0
    return {s:('test' if i<holdout else 'validation' if i<2*holdout else 'train') for i,s in enumerate(order)}


def write_jsonl(path, rows):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    temporary.replace(path)


def export_clips(source, source_id, rows, out, split, sample_rate, profile_fingerprint, qc_rejections=None):
    receipts = []
    with sf.SoundFile(source) as stream:
        rate = stream.samplerate
        for row in rows:
            first,last = round(row['start']*rate), round(row['end']*rate)
            stream.seek(first)
            audio = stream.read(last-first,dtype='float32',always_2d=True).mean(axis=1)
            if sample_rate != rate:
                divisor=math.gcd(sample_rate,rate)
                audio=resample_poly(audio,sample_rate//divisor,rate//divisor).astype(np.float32)
            qc_reason = None
            if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 1e-6:
                qc_reason = 'non_finite_or_silent_audio'
            elif np.mean(np.abs(audio) >= .999) > .001:
                qc_reason = 'clipped_source_audio'
            if qc_reason:
                if qc_rejections is not None:
                    qc_rejections.append({'cue_ids':row['cue_ids'],'start':row['start'],'end':row['end'],'reason':qc_reason})
                continue
            key = f'{source_id}_{first}_{last}'
            target=out/'audio'/row['speaker']/(key+'.wav')
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix('.tmp.wav')
            sf.write(temp,audio,sample_rate,subtype='PCM_16')
            temp.replace(target)
            target.with_suffix('.lab').write_text(row['text']+'\n',encoding='utf-8')
            receipts.append({**row,'clip_id':key,'source_id':source_id,'source_group':source_id,
                'split':split,'audio':str(target.resolve()),'duration':len(audio)/sample_rate,
                'sample_rate':sample_rate,'channels':1,'source_start_frame':first,'source_end_frame':last,
                'source_sample_rate':rate,'audio_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'profile_fingerprint':profile_fingerprint,'language':'ar',
                'transcript_provenance':'complete_aligned_srt_cues',
                'review_status':'automatic_speaker_candidates_pending_listening_review'})
    return receipts


def rebuild_manifests(out, fingerprint):
    rows, episodes = [],[]
    for path in sorted((out/'episodes').glob('*.json')):
        receipt=json.loads(path.read_text())
        if receipt.get('recipe_fingerprint') == fingerprint:
            rows.extend(receipt['clips'])
            episodes.append({'source_id':receipt['source_id'],'clips':len(receipt['clips']),
                             'rejected_cues':len(receipt['rejected']),'speakers':receipt['identities']})
    out.mkdir(parents=True,exist_ok=True)
    unique,seen,duplicates=[],set(),[]
    for row in rows:
        if row['audio_sha256'] in seen:
            duplicates.append(row['clip_id'])
        else:
            seen.add(row['audio_sha256']);unique.append(row)
    rows=unique
    write_jsonl(out/'manifest.jsonl',rows)
    for split in ('train','validation','test'):
        write_jsonl(out/(split+'.jsonl'),[r for r in rows if r['split']==split])
    by_speaker=defaultdict(lambda:{'clips':0,'seconds':0.,'splits':set()})
    for row in rows:
        item=by_speaker[row['speaker']];item['clips']+=1;item['seconds']+=row['duration'];item['splits'].add(row['split'])
    summary={'clips':len(rows),'hours':sum(r['duration'] for r in rows)/3600,
        'completed_episodes':len(episodes),'episodes':episodes,'exact_duplicate_clips_excluded_from_manifests':duplicates,
        'speakers':{s:{**v,'splits':sorted(v['splits'])} for s,v in by_speaker.items()},
        'status':'automatic_candidates_pending_listening_review',
        'limitations':['Guests have episode-local IDs, not verified cross-episode identities.',
            'Guests confined to one episode cannot have recording-disjoint train/validation/test data for that identity.',
            'Diarization, identity and ASR transcript errors still require listening review.',
            'Cues containing speaker changes require word-level realignment before export.']}
    identity.write_json(out/'summary.json',summary)
    return summary


def main():
    global ACTIVE_OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--source',help='Optional one-episode trial; omit for all episodes')
    parser.add_argument('--limit-seconds',type=float)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--sample-rate',type=int,default=48000)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    parser.add_argument('--segmentation-batch-size',type=int,default=64)
    parser.add_argument('--embedding-batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not -1 < args.threshold < 1 or args.sample_rate not in (16000,24000,44100,48000) or args.threads<1:
        parser.error('Invalid threshold, sample rate or CPU thread count')
    if args.limit_seconds is not None and args.limit_seconds<=0:
        parser.error('Limit must be positive')
    if min(args.segmentation_batch_size,args.embedding_batch_size)<1:
        parser.error('Batch sizes must be positive')
    sources=sorted({r['source_id'] for r in identity.ROWS.values()})
    if args.source and args.source not in sources:
        parser.error('Unknown source')
    profile=identity.load_profile('wespeaker')
    splits=episode_splits(sources)
    input_info={}
    for source in sources:
        audio=ROOT/'full-audios-transcript/audios-48khz'/(source+'.flac')
        srt=ROOT/'full-audios-transcript/transcript'/(source+'.ar-new.srt')
        stat=audio.stat()
        input_info[source]={'audio_size':stat.st_size,'audio_mtime_ns':stat.st_mtime_ns,
                           'srt_sha256':hashlib.sha256(srt.read_bytes()).hexdigest()}
    recipe={'revision':REVISION,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model':'pyannote/speaker-diarization-community-1','profile_fingerprint':identity.digest(profile),
            'inputs':input_info,'threshold':args.threshold,'sample_rate':args.sample_rate,
            'minimum_seconds':3,'maximum_seconds':30,'max_merge_gap':.5,'minimum_speech_coverage':.6,
            'split_by_source':splits,'limit_seconds':args.limit_seconds,
            'segmentation_batch_size':args.segmentation_batch_size,
            'embedding_batch_size':args.embedding_batch_size}
    fingerprint=identity.digest(recipe)
    if args.check:
        print(json.dumps({'sources':len(sources),'profile_reference_seconds':profile['reference_seconds'],
                         'out':str(args.out.resolve()),'recipe_fingerprint':fingerprint,'splits':dict(Counter(splits.values()))},indent=2))
        return
    import torch
    torch.set_num_threads(args.threads)
    device=resolve_device(args.device,torch)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=out/'recipe.json'
    if receipt.exists() and json.loads(receipt.read_text())!=recipe:
        parser.error('Output has a different recipe. Choose a separate --out directory to preserve it.')
    identity.write_json(receipt,recipe)
    ACTIVE_OUT=out
    (out/'episodes').mkdir(exist_ok=True)
    (out/'diarization').mkdir(exist_ok=True)
    os.environ.setdefault('HF_HUB_CACHE',str(HERE/'cache/huggingface/hub'))
    os.environ.setdefault('PYANNOTE_METRICS_ENABLED','0')
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline
    pipeline=configure_pipeline(Pipeline.from_pretrained(recipe['model'],token=get_token()),
        torch,device,args.segmentation_batch_size,args.embedding_batch_size)
    encoder=identity.Encoder('wespeaker',device)
    print(json.dumps({'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
        'segmentation_batch_size':pipeline.segmentation_batch_size,
        'embedding_batch_size':pipeline.embedding_batch_size}),flush=True)
    chosen=[args.source] if args.source else sources
    for index,source_id in enumerate(chosen):
        status={'stage':'diarization','source_id':source_id,'episode_index':index+1,'requested_episodes':len(chosen),
                'device':device,'output':str(out)}
        identity.write_json(out/'status.json',status)
        saved=out/'episodes'/(source_id+'.json')
        if saved.exists():
            old=json.loads(saved.read_text())
            if old.get('recipe_fingerprint')==fingerprint and all(Path(r['audio']).exists() and hashlib.sha256(Path(r['audio']).read_bytes()).hexdigest()==r['audio_sha256'] and Path(r['audio']).with_suffix('.lab').exists() and Path(r['audio']).with_suffix('.lab').read_text().strip()==r['text'] for r in old['clips']):
                print(f'Reuse {source_id}: {len(old["clips"])} clips',flush=True)
                continue
        source=ROOT/'full-audios-transcript/audios-48khz'/(source_id+'.flac')
        cache=out/'diarization'/(source_id+'.json')
        start=time.monotonic()
        audio16=identity.read16(source,end=args.limit_seconds)
        if cache.exists():
            cached=json.loads(cache.read_text())
            if cached.get('recipe_fingerprint')!=fingerprint:
                raise ValueError('Stale diarization cache')
            turns=cached['turns']
        else:
            print(f'Diarize {source_id}: {len(audio16)/16000/60:.1f} minutes on {device}',flush=True)
            def hook(step, artifact, total=None, completed=None, **kwargs):
                if completed is None or completed>=0:
                    identity.write_json(out/'status.json',{**status,'step':step,'completed':None if completed is None else int(completed),'total':None if total is None else int(total)})
            output=pipeline({'waveform':torch.from_numpy(audio16).unsqueeze(0),'sample_rate':16000},hook=hook)
            turns=[{'start':float(t.start),'end':float(t.end),'speaker':s} for t,_,s in output.speaker_diarization.itertracks(yield_label=True)]
            identity.write_json(cache,{'recipe_fingerprint':fingerprint,'turns':turns})
        identity.write_json(out/'status.json',{**status,'stage':'identity_and_export'})
        identities=speaker_identities(turns,audio16,encoder,profile,source_id,args.threshold)
        cues=srt_cues(ROOT/'full-audios-transcript/transcript'/(source_id+'.ar-new.srt'))
        if args.limit_seconds:
            cues=[c for c in cues if c['end']<=args.limit_seconds]
        clips,rejected=plan_clips(cues,turns,identities)
        records=export_clips(source,source_id,clips,out,splits[source_id],args.sample_rate,recipe['profile_fingerprint'],rejected)
        identity.write_json(saved,{'source_id':source_id,'recipe_fingerprint':fingerprint,'identities':identities,
                                  'clips':records,'rejected':rejected,'elapsed_seconds':time.monotonic()-start})
        summary=rebuild_manifests(out,fingerprint)
        print(f'Export {source_id}: {len(records)} clips / {sum(r["duration"] for r in records)/60:.1f} min; total {summary["clips"]} clips',flush=True)
        del audio16
    summary=rebuild_manifests(out,fingerprint)
    identity.write_json(out/'status.json',{'stage':'complete','completed_episodes':summary['completed_episodes'],
                                         'requested_episodes':len(chosen),'clips':summary['clips'],'hours':summary['hours']})
    print(json.dumps({'out':str(out),'clips':summary['clips'],'hours':summary['hours']},indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        if ACTIVE_OUT is not None:
            path=ACTIVE_OUT/'status.json'
            status=json.loads(path.read_text()) if path.exists() else {}
            message=re.sub(r'hf_[A-Za-z0-9]+','[redacted token]',str(exc))
            identity.write_json(path,{**status,'stage':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'error','error':type(exc).__name__+': '+message})
        raise
