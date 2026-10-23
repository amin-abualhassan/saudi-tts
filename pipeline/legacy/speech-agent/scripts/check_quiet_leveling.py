"""Compare bounded quiet lifting against the previous compressor/limiter."""
import json
from pathlib import Path
import time
import wave

from livekit import rtc
import numpy as np

from app.audio import SAMPLE_RATE, StreamDynamics
from app.config import settings
from scripts.check_audio_dynamics import peak_dbfs, save


def process(samples,boost):
    processor=StreamDynamics(quiet_boost_db=boost,leveler_enabled=False);output=[];times=[]
    for start in range(0,len(samples),882):
        block=samples[start:start+882]
        frame=rtc.AudioFrame(block.tobytes(),SAMPLE_RATE,1,len(block))
        started=time.perf_counter_ns();output.extend(processor.process(frame))
        times.append((time.perf_counter_ns()-started)/1e6)
    output.extend(processor.finish())
    pcm=b''.join(bytes(frame.data) for frame in output)
    return pcm,{'median_cpu_ms':round(float(np.median(times)),4),
                'p95_cpu_ms':round(float(np.percentile(times,95)),4),
                'first_frame_cpu_ms':round(times[0],4)}


def window_levels(pcm):
    samples=np.frombuffer(pcm,dtype='<i2').astype(np.float64)/32768
    size=4410;length=len(samples)//size*size
    rms=np.sqrt(np.mean(samples[:length].reshape(-1,size)**2,axis=1))
    return 20*np.log10(np.maximum(rms,1e-10))


def main():
    target=Path('tests/results')
    report={'passed':False,'maximum_additional_quiet_boost_db':settings.audio_quiet_boost_db,
            'peak_ceiling_dbfs':settings.audio_peak_dbfs,'added_lookahead_ms':0,'cases':[]}
    for language in ('ar','en'):
        source=target/f'stream-dynamics-{language}-raw-20261018.wav'
        with wave.open(str(source)) as audio:
            assert (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())==(SAMPLE_RATE,1,2)
            samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2')
        for variant,scale in [('normal',1),('quiet',10**(-10/20))]:
            data=(samples.astype(np.float64)*scale).astype('<i2')
            baseline,baseline_times=process(data,0)
            lifted,lifted_times=process(data,settings.audio_quiet_boost_db)
            assert len(baseline)==len(lifted)==len(data)*2
            assert peak_dbfs(lifted)<=settings.audio_peak_dbfs+.002
            before=window_levels(baseline);after=window_levels(lifted)
            # Exclude near-silence; focus on quiet audible windows.
            selected=(before>-45)&(before<-26)
            gain=after[selected]-before[selected]
            assert np.any(selected)
            if variant=='quiet':assert float(np.median(gain))>1
            assert np.max(gain)<settings.audio_quiet_boost_db+.1
            base_path=target/f'quiet-leveling-{language}-{variant}-before-20261018.wav'
            new_path=target/f'quiet-leveling-{language}-{variant}-after-20261018.wav'
            save(base_path,baseline);save(new_path,lifted)
            report['cases'].append({'language':language,'variant':variant,
                'median_quiet_window_lift_db':round(float(np.median(gain)),3),
                'max_quiet_window_lift_db':round(float(np.max(gain)),3),
                'processed_peak_dbfs':peak_dbfs(lifted),'sample_count_preserved':True,
                'old_processing':baseline_times,'new_processing':lifted_times,
                'before_path':str(base_path),'after_path':str(new_path)})
    report['passed']=True
    (target/'quiet-leveling-20261018.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
