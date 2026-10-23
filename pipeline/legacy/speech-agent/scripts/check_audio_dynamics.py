"""Measure streaming dynamics, real TTS peaks/tails, and revised breath prompting."""
import asyncio
import json
from pathlib import Path
import time
import wave

from livekit import rtc
import numpy as np

from app.agent import AljawabAgent
from app.audio import SAMPLE_RATE, LOOKAHEAD_MS, StreamDynamics, controlled_audio
from app.config import settings
from app.speech import Speech


def peak_dbfs(pcm):
    samples=np.frombuffer(pcm,dtype='<i2').astype(np.float64)
    peak=np.max(np.abs(samples)) if len(samples) else 0
    return round(float(20*np.log10(peak/32768)),4) if peak else None


def save(path,pcm):
    with wave.open(str(path),'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm)


def measure_processor():
    tone=(np.sin(np.arange(882)*.1)*30000).astype('<i2')
    frame=rtc.AudioFrame(tone.tobytes(),SAMPLE_RATE,1,882)
    processor=StreamDynamics();times=[]
    for _ in range(1000):
        start=time.perf_counter_ns();processor.process(frame)
        times.append((time.perf_counter_ns()-start)/1e6)
    processor.finish()
    return {'frame_duration_ms':20,'lookahead_ms':LOOKAHEAD_MS,'first_frame_cpu_ms':round(times[0],4),
        'median_cpu_ms':round(float(np.median(times)),4),'p95_cpu_ms':round(float(np.percentile(times,95)),4)}


async def main():
    target=Path('tests/results');target.mkdir(exist_ok=True)
    report={'passed':False,'ceiling_dbfs':settings.audio_peak_dbfs,'processing':measure_processor(),'cases':[]}
    voice=Speech();agent=AljawabAgent(None,None)
    try:
        for language,prompt in [('ar','وش يسوي الجواب؟ جاوب بجملتين قصيرتين.'),
                                ('en','What does Aljawab do? Answer in two short sentences.')]:
            response=await agent.client.responses.create(model=settings.llm_model,reasoning={'effort':'none'},
                instructions=agent.instructions_text+'\nCurrent preferred language: '+language+'.',
                input=[{'role':'user','content':prompt}],store=False,max_output_tokens=450)
            text=response.output_text
            assert text.count('[short breath]')==1 and text.rstrip().endswith('[short breath]'),text
            raw=bytearray();processed=bytearray();first_raw=None;first_processed=None
            async def source():
                nonlocal first_raw
                async for frame in voice.stream(text):
                    if first_raw is None:first_raw=time.perf_counter()
                    raw.extend(frame.data);yield frame
            async for frame in controlled_audio(source()):
                if first_processed is None:first_processed=time.perf_counter()
                processed.extend(frame.data)
            assert len(raw)==len(processed) and len(raw)>0
            assert peak_dbfs(processed)<=settings.audio_peak_dbfs+.002
            raw_path=target/f'stream-dynamics-{language}-raw-20261018.wav'
            processed_path=target/f'stream-dynamics-{language}-limited-20261018.wav'
            save(raw_path,raw);save(processed_path,processed)
            report['cases'].append({'language':language,'llm_output':text,'breath_count':text.count('[short breath]'),
                'raw_peak_dbfs':peak_dbfs(raw),'processed_peak_dbfs':peak_dbfs(processed),
                'pcm_bytes':len(raw),'same_sample_count':True,
                'first_frame_processing_ms':round((first_processed-first_raw)*1000,4),
                'raw_path':str(raw_path),'processed_path':str(processed_path)})
            print(language,'single final breath and native streaming limiter passed.',flush=True)
        for name in ('greeting_ar','greeting_en'):
            raw=bytearray();processed=bytearray()
            async def source():
                async for frame in voice.cached(name):raw.extend(frame.data);yield frame
            async for frame in controlled_audio(source()):processed.extend(frame.data)
            assert len(raw)==len(processed) and len(raw)>0
            assert peak_dbfs(processed)<=settings.audio_peak_dbfs+.002
            report['cases'].append({'cached_clip':name,'raw_peak_dbfs':peak_dbfs(raw),
                'processed_peak_dbfs':peak_dbfs(processed),'same_sample_count':True})
        report['passed']=True
    finally:
        await agent.aclose();await voice.cleanup();await voice.client.aclose()
        (target/'stream-dynamics-20261018.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
