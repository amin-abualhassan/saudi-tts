"""Verify real tagged TTS with a clean browser transcript using generated input."""
import asyncio
import json
import os
from pathlib import Path
import re
import time
import wave

import httpx
from playwright.async_api import async_playwright


async def main():
    target=Path('tests/results');target.mkdir(exist_ok=True)
    fixture=(target/'delivery-tags-mic-20261018.wav').resolve()
    with wave.open('/tmp/aljawab-ar.wav') as source:
        assert (source.getframerate(),source.getnchannels(),source.getsampwidth())==(44100,1,2)
        speech=source.readframes(source.getnframes())
    with wave.open(str(fixture),'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(44100)
        output.writeframes(bytes(44100*2*6)+speech+bytes(44100*2*45))
    url=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',Path('logs/tunnel.log').read_text())[-1]
    result={'passed':False,'page_errors':[]}
    prefix=os.environ.get('CHECK_ARTIFACT_PREFIX','stream-dynamics')
    events=[];tagged_jobs={};tts_jobs={}
    cues=re.compile(r'\[(?:friendly|calm|empathetic|short\s+breath)\]',re.I)
    async with httpx.AsyncClient(trust_env=False,timeout=5) as client:
        baseline={job['id'] for job in (await client.get('http://127.0.0.1:3000/api/status')).json()['jobs']}
        async with async_playwright() as playwright:
            browser=await playwright.chromium.launch(headless=True,args=['--no-sandbox',
                '--autoplay-policy=no-user-gesture-required','--use-fake-ui-for-media-stream',
                '--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+str(fixture)+'%noloop'])
            context=await browser.new_context(permissions=['microphone'])
            if os.environ.get('FORCE_TURN_RELAY')=='1':
                await context.add_init_script("""const Native=window.RTCPeerConnection;
                    window.RTCPeerConnection=new Proxy(Native,{construct(target,args){
                        const pc=Reflect.construct(target,[{...args[0],iceTransportPolicy:'relay'}]);
                        const configure=pc.setConfiguration.bind(pc);
                        pc.setConfiguration=(config)=>configure({...config,iceTransportPolicy:'relay'});
                        return pc;
                    }});""")
                result['forced_turn_relay']=True
            page=await context.new_page()
            page.on('pageerror',lambda error:result['page_errors'].append(str(error)))
            def websocket(socket):
                if not socket.url.endswith('/api/events'):return
                def received(raw):
                    event=json.loads(raw)
                    if event.get('type')=='transcript':events.append(event)
                socket.on('framereceived',received)
            page.on('websocket',websocket)
            try:
                await page.goto(url,wait_until='networkidle')
                await page.locator('#transcript-toggle').check()
                await page.locator('#start').click()
                await page.locator('#end').wait_for(state='visible',timeout=30000)
                started=time.monotonic()
                while time.monotonic()-started<45:
                    status=(await client.get('http://127.0.0.1:3000/api/status')).json()
                    for job in status['jobs']:
                        text=job['request']['text']
                        if job['id'] not in baseline and job.get('streaming'):
                            tts_jobs[job['id']]={'one_final_breath':text.count('[short breath]')==1 and text.rstrip().endswith('[short breath]')}
                        if job['id'] not in baseline and '[short breath]' in text:
                            tagged_jobs[job['id']]={'ends_with_breath':text.rstrip().endswith('[short breath]')}
                    if tagged_jobs and any(event.get('role')=='assistant' and event.get('final') for event in events):break
                    await asyncio.sleep(.5)
                transcript=await page.locator('#transcript').inner_text()
                result.update(tagged_tts_jobs=len(tagged_jobs),
                    tts_jobs=len(tts_jobs),
                    all_tts_phrases_end_with_one_breath=bool(tts_jobs) and all(job['one_final_breath'] for job in tts_jobs.values()),
                    tagged_phrases_end_with_breath=bool(tagged_jobs) and all(job['ends_with_breath'] for job in tagged_jobs.values()),
                    user_final_events=sum(event.get('role')=='user' and bool(event.get('final')) for event in events),
                    assistant_final_events=sum(event.get('role')=='assistant' and bool(event.get('final')) for event in events),
                    transcript_events_have_tags=any(bool(cues.search(event.get('text',''))) for event in events),
                    visible_transcript_has_tags=bool(cues.search(transcript)),
                    transcript_visible=bool(transcript.strip()),
                    audio=await page.locator('audio').evaluate_all('(elements)=>elements.map(audio=>({paused:audio.paused,currentTime:audio.currentTime,readyState:audio.readyState}))'),
                    error_visible=await page.locator('#error').is_visible())
                await page.screenshot(path=str(target/(prefix+'-ui-20261018.png')),full_page=True)
                assert result['tagged_tts_jobs']>=1 and result['user_final_events']>=1 and result['assistant_final_events']>=1,result
                assert not result['transcript_events_have_tags'] and not result['visible_transcript_has_tags'],result
                assert result['tagged_phrases_end_with_breath'],result
                assert result['all_tts_phrases_end_with_one_breath'],result
                assert result['transcript_visible'] and not result['error_visible'] and not result['page_errors'],result
                assert any(audio['currentTime']>0 and audio['readyState']>=2 for audio in result['audio']),result
                result['passed']=True
            finally:
                if await page.locator('#end').is_visible():await page.locator('#end').click()
                await browser.close()
                (target/(prefix+'-ui-20261018.json')).write_text(json.dumps(result,indent=2))
                print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
