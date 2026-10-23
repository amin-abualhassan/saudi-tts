#!/usr/bin/env python3
"""Measure identical real audio on CPU and CUDA, then save timings."""
import argparse
import gc
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HUB_CACHE', str(HERE / 'cache/huggingface/hub'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYANNOTE_METRICS_ENABLED', '0')

import identity
from gpu_runtime import configure_pipeline, resolve_device
from prepare_dataset import speaker_identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='utrhqcji78')
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path, default=HERE / 'reports/gpu_benchmark.json')
    args = parser.parse_args()
    if args.seconds <= 0 or args.batch_size < 1 or args.threads < 1:
        parser.error('Duration, batch size and thread count must be positive')
    if args.source not in {r['source_id'] for r in identity.ROWS.values()}:
        parser.error('Unknown episode')
    import torch
    report = {'source_id': args.source, 'requested_audio_seconds': args.seconds,
              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'cpu_threads': args.threads, 'runs': []}
    try:
        resolve_device('cuda', torch)
    except RuntimeError as exc:
        report.update(status='blocked', error=str(exc))
        identity.write_json(args.out, report)
        raise
    report['gpu'] = torch.cuda.get_device_name()
    torch.set_num_threads(args.threads)
    from huggingface_hub import get_token
    from pyannote.audio import Pipeline
    audio = identity.read16(identity.ROOT / 'full-audios-transcript/audios-48khz' /
                            (args.source + '.flac'), end=args.seconds)
    report['audio_seconds'] = len(audio) / 16000
    waveform = {'waveform': torch.from_numpy(audio).unsqueeze(0), 'sample_rate': 16000}
    profile = identity.load_profile('wespeaker')
    report['status'] = 'running'
    identity.write_json(args.out, report)
    for device, batch_size in [('cpu', 32), ('cuda', args.batch_size)]:
        print(f'Benchmark {device}: {report["audio_seconds"]:.1f}s audio, batch {batch_size}', flush=True)
        start = time.perf_counter()
        pipeline = configure_pipeline(Pipeline.from_pretrained(
            'pyannote/speaker-diarization-community-1', token=get_token()),
            torch, device, batch_size, batch_size)
        encoder = identity.Encoder('wespeaker', device)
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        loaded = time.perf_counter()
        with torch.inference_mode():
            output = pipeline(waveform)
        if device == 'cuda':
            torch.cuda.synchronize()
        diarized = time.perf_counter()
        turns = [{'start': float(t.start), 'end': float(t.end), 'speaker': s}
                 for t, _, s in output.speaker_diarization.itertracks(yield_label=True)]
        identities = speaker_identities(turns, audio, encoder, profile, args.source, .5)
        if device == 'cuda':
            torch.cuda.synchronize()
        finished = time.perf_counter()
        run = {'device': device, 'batch_size': batch_size,
               'model_load_seconds': loaded - start,
               'diarization_seconds': diarized - loaded,
               'identity_seconds': finished - diarized,
               'inference_seconds': finished - loaded,
               'real_time_factor': (finished - loaded) / report['audio_seconds'],
               'turns': len(turns), 'speakers': len(identities)}
        if device == 'cuda':
            run['peak_allocated_gpu_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        report['runs'].append(run)
        identity.write_json(args.out, report)
        print(run, flush=True)
        del pipeline, encoder, output
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    cpu, gpu = report['runs']
    report.update(status='complete', inference_speedup=cpu['inference_seconds'] / gpu['inference_seconds'],
                  estimated_full_corpus_inference_hours=31.431 * gpu['real_time_factor'],
                  note='Single cold sample; estimates exclude full-corpus I/O, export and clustering scaling. '
                       'CPU batch 32 versus configured GPU batch measures practical configuration speedup.')
    identity.write_json(args.out, report)
    print(f'Speedup: {report["inference_speedup"]:.2f}x; report: {args.out}', flush=True)


if __name__ == '__main__':
    main()
