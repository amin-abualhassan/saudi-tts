"""Measure gradual-fade correction and save paired real-speech examples."""
import json
from pathlib import Path
import time
import wave

from livekit import rtc
import numpy as np

from app.audio import SAMPLE_RATE, StreamDynamics
from app.config import settings
from scripts.check_audio_dynamics import peak_dbfs, save


def process(samples,enabled):
    processor=StreamDynamics(leveler_enabled=enabled);output=[];times=[];first_count=0
    for start in range(0,len(samples),882):
        block=samples[start:start+882]
        frame=rtc.AudioFrame(block.tobytes(),SAMPLE_RATE,1,len(block))
        began=time.perf_counter_ns();parts=processor.process(frame)
        times.append((time.perf_counter_ns()-began)/1e6)
        if start==0:first_count=sum(part.samples_per_channel for part in parts)
        output.extend(parts)
    output.extend(processor.finish())
    pcm=b''.join(bytes(part.data) for part in output)
    assert len(pcm)==len(samples)*2
    assert first_count>=838 and peak_dbfs(pcm)<=settings.audio_peak_dbfs+.002
    return pcm,{'median_cpu_ms':round(float(np.median(times)),4),
                'p95_cpu_ms':round(float(np.percentile(times,95)),4),
                'first_frame_output_samples':first_count,
                'sample_count_preserved':True,'peak_dbfs':peak_dbfs(pcm)}


def main():
    target=Path('tests/results')
    report={'passed':False,'history_ms':settings.audio_leveler_window_ms,
            'target_rms_dbfs':settings.audio_leveler_target_dbfs,
            'max_boost_db':settings.audio_leveler_max_boost_db,
            'additional_lookahead_ms':0,'cases':[]}
    time_axis=np.arange(SAMPLE_RATE*4)/SAMPLE_RATE
    fixture=(np.sin(time_axis*2*np.pi*220)*10**((-20-12*time_axis/4)/20)*32767).astype('<i2')
    drift={}
    for name,enabled in [('previous',False),('rolling',True)]:
        pcm,_=process(fixture,enabled)
        samples=np.frombuffer(pcm,dtype='<i2').astype(float)
        rms=lambda block:np.sqrt(np.mean(block**2))
        drift[name]=round(float(20*np.log10(rms(samples[SAMPLE_RATE//2:SAMPLE_RATE])/rms(samples[-SAMPLE_RATE//2:]))),3)
    assert drift['rolling']<drift['previous']-4
    report['synthetic_gradual_fade_drop_db']=drift
    for language in ('ar','en'):
        with wave.open(str(target/f'stream-dynamics-{language}-raw-20261018.wav')) as audio:
            assert (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())==(SAMPLE_RATE,1,2)
            source=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2')
        for variant in ('normal','gradual-fade'):
            # Controlled 12 dB fade over existing generated speech; not a claim
            # that the original model naturally produces this exact envelope.
            scale=1 if variant=='normal' else 10**(np.linspace(0,-12,len(source))/20)
            samples=(source.astype(float)*scale).astype('<i2')
            case={'language':language,'variant':variant}
            for name,enabled in [('before',False),('after',True)]:
                pcm,metrics=process(samples,enabled)
                path=target/f'rolling-leveler-{language}-{variant}-{name}-20261018.wav'
                save(path,pcm);case[name]={'path':str(path),**metrics}
            report['cases'].append(case)
    report['passed']=True
    (target/'rolling-leveler-20261018.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
