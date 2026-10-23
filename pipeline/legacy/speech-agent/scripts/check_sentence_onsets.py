"""Check actual separately synthesized sentences against onset smoothing."""
import asyncio
import json
from pathlib import Path
import time

from livekit import rtc
import numpy as np

from app.audio import SAMPLE_RATE, StreamDynamics
from app.config import settings
from app.phrases import phrases
from app.speech import Speech
from scripts.check_audio_dynamics import peak_dbfs, save


def process(segments,onset_ms):
    processor=StreamDynamics(onset_ms=onset_ms);out=[];times=[]
    for segment in segments:
        processor.start_segment()
        for start in range(0,len(segment),1764):
            pcm=segment[start:start+1764]
            frame=rtc.AudioFrame(pcm,SAMPLE_RATE,1,len(pcm)//2)
            before=time.perf_counter_ns();out.extend(processor.process(frame))
            times.append((time.perf_counter_ns()-before)/1e6)
    out.extend(processor.finish())
    return b''.join(bytes(frame.data) for frame in out),{
        'median_cpu_ms':round(float(np.median(times)),4),'p95_cpu_ms':round(float(np.percentile(times,95)),4)}


async def main():
    target=Path('tests/results')
    source=json.loads((target/'stream-dynamics-20261018.json').read_text())
    voice=Speech();report={'passed':False,'onset_ms':settings.audio_onset_ms,
        'initial_attenuation_db':settings.audio_onset_attenuation_db,'extra_buffering_ms':0,'cases':[]}
    try:
        for previous in source['cases']:
            if 'language' not in previous:continue
            language=previous['language']
            async def text():yield previous['llm_output']
            sentences=[part async for part in phrases(text(),expect_cues=True)]
            assert len(sentences)>=2
            segments=[]
            for sentence in sentences:
                pcm=bytearray()
                async for frame in voice.stream(sentence):pcm.extend(frame.data)
                segments.append(bytes(pcm))
            baseline,old_time=process(segments,0);smoothed,new_time=process(segments,settings.audio_onset_ms)
            assert len(smoothed)==len(baseline)==sum(map(len,segments))
            assert peak_dbfs(smoothed)<=settings.audio_peak_dbfs+.002
            a=np.frombuffer(baseline,dtype='<i2').astype(np.int32)
            b=np.frombuffer(smoothed,dtype='<i2').astype(np.int32)
            starts=[];offset=0
            for segment in segments:
                stop=offset+len(segment)//2
                audible=np.flatnonzero(np.abs(a[offset:stop])>=32768*10**(-50/20))
                assert len(audible)
                first=offset+int(audible[0]);end=min(first+441,stop)
                old_peak=int(np.max(np.abs(a[first:end])));new_peak=int(np.max(np.abs(b[first:end])))
                reduction=20*np.log10(new_peak/old_peak) if new_peak else -100
                assert reduction<-6,(language,reduction)
                starts.append({'sample_index':first,'first_10_ms_peak_change_db':round(float(reduction),3)})
                offset=stop
            # Insert a listening-only gap between processed sentences. The DSP
            # itself sees only arrival frames/markers, as in the real pipeline.
            def listening(pcm):
                pieces=[];offset=0
                for index,segment in enumerate(segments):
                    if index:pieces.append(bytes(round(SAMPLE_RATE*.25)*2))
                    pieces.append(pcm[offset:offset+len(segment)]);offset+=len(segment)
                return b''.join(pieces)
            old_path=target/f'sentence-onsets-{language}-before-20261018.wav'
            new_path=target/f'sentence-onsets-{language}-after-20261018.wav'
            save(old_path,listening(baseline));save(new_path,listening(smoothed))
            report['cases'].append({'language':language,'sentence_count':len(segments),
                'sample_count_preserved':True,'starts':starts,'old_processing':old_time,'new_processing':new_time,
                'before_path':str(old_path),'after_path':str(new_path)})
            print(language,'actual sentence starts passed.',flush=True)
        report['passed']=True
    finally:
        await voice.cleanup();await voice.client.aclose()
        (target/'sentence-onsets-20261018.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
