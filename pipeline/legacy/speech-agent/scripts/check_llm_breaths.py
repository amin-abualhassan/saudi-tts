"""Check LLM-authored cues and compare real streamed PCM with completed WAVs."""
import asyncio
import base64
import io
import json
from pathlib import Path
import re
import wave

import httpx

from app.agent import AljawabAgent
from app.config import settings
from app.cues import CUE, transcript_chunks, without_delivery_cues
from app.numbers import spoken_chunks
from app.phrases import phrases
from app.speech import Speech, payload


async def main():
    target=Path('tests/results');target.mkdir(exist_ok=True)
    agent=AljawabAgent(None,None);owner=Speech()
    report={'passed':False,'cases':[]}
    try:
        async with httpx.AsyncClient(trust_env=False,timeout=120) as client:
            for language,prompt in [('ar','وش يسوي الجواب؟ جاوب بجملتين قصيرتين.'),
                                    ('en','What does Aljawab do? Answer in two short sentences.')]:
                chunks=[]
                stream=await agent.client.responses.create(model=settings.llm_model,reasoning={'effort':'none'},
                    instructions=agent.instructions_text+'\nCurrent preferred language: '+language+'.',
                    input=[{'role':'user','content':prompt}],stream=True,store=False,max_output_tokens=450)
                try:
                    async for event in stream:
                        if event.type=='response.output_text.delta':chunks.append(event.delta)
                        elif event.type in ('error','response.failed'):raise RuntimeError('LLM check failed')
                finally:await stream.close()
                text=''.join(chunks)
                assert text.rstrip().endswith('[short breath]') and text.count('[short breath]')==1,text
                async def source():
                    for chunk in chunks:yield chunk
                sentences=[part async for part in phrases(spoken_chunks(source(),language),expect_cues=True)]
                assert len(sentences)>=2 and sentences[-1].endswith('[short breath]'),sentences
                assert all('[short breath]' not in part for part in sentences[:-1]),sentences
                assert sum(part.count('[short breath]') for part in sentences)==text.count('[short breath]')
                visible=''.join([part async for part in transcript_chunks(source())])
                assert not CUE.search(visible) and without_delivery_cues(visible)==without_delivery_cues(text)
                case={'language':language,'llm_output':text,'phrases':sentences,
                      'llm_breath_count':text.count('[short breath]'),'transcript_has_cues':bool(CUE.search(visible)),'audio':[]}
                for variant,tts_text in [('tagged',text),('plain',without_delivery_cues(text))]:
                    pcm=bytearray();job_id=None;complete=False
                    try:
                        async with client.stream('POST',settings.tts_url+'/api/stream',json=payload(tts_text)) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line.startswith('data: '):continue
                                event=json.loads(line[6:])
                                if event['event']=='accepted':job_id=event['id']
                                elif event['event']=='audio':pcm.extend(base64.b64decode(event['pcm'],validate=True))
                                elif event['event']=='error':raise RuntimeError('TTS check failed')
                                elif event['event']=='complete':complete=True
                        assert complete and pcm and len(pcm)%2==0
                        audio=await client.get(settings.tts_url+'/api/audio/'+job_id+'.wav');audio.raise_for_status()
                        with wave.open(io.BytesIO(audio.content)) as wav:
                            assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(44100,1,2)
                            saved_pcm=wav.readframes(wav.getnframes())
                        assert saved_pcm==pcm,'Stream omitted or changed samples from completed WAV'
                        path=target/f'final-breaths-{language}-{variant}-20261018.wav'
                        path.write_bytes(audio.content)
                        case['audio'].append({'variant':variant,'path':str(path),'seconds':round(len(pcm)/88200,3),
                            'stream_matches_completed_wav':True,'pcm_bytes':len(pcm)})
                    finally:
                        if job_id:owner.own(job_id)
                report['cases'].append(case)
                print(language,'LLM cues and streamed/WAV sample comparison passed.',flush=True)
        report['passed']=True
    finally:
        await agent.aclose();await owner.cleanup();await owner.client.aclose()
        (target/'final-breaths-20261018.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
