"""Check both current cached welcomes through playback processing and Chromium."""
import asyncio
import json
import os
from pathlib import Path
import re
import time
import wave

import numpy as np
from playwright.async_api import async_playwright

from app.audio import controlled_audio
from app.config import settings
from app.speech import Speech


async def main():
    target=Path('tests/results');target.mkdir(parents=True,exist_ok=True)
    prefix=os.environ.get('CHECK_ARTIFACT_PREFIX','cached-welcomes')
    manifest=json.loads(Path('assets/voice/manifest.json').read_text())
    result={'passed':False,'page_errors':[],'greetings':[]};voice=Speech()
    try:
        for language in ('ar','en'):
            name='greeting_'+language
            frames=[part async for part in controlled_audio(voice.cached(name))]
            data=np.frombuffer(b''.join(bytes(part.data) for part in frames),dtype='<i2').astype(float)
            assert len(data)==manifest[name]['postprocess']['output_samples']
            peak=float(20*np.log10(max(1,np.max(np.abs(data)))/32768))
            assert peak<=settings.audio_peak_dbfs+.002
            result['greetings'].append({'language':language,'samples_preserved':True,'peak_dbfs':round(peak,3)})
    finally:await voice.client.aclose()
    url=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',Path('logs/tunnel.log').read_text())[-1]
    fixture=(target/'greeting-silent-mic-20261018.wav').resolve()
    with wave.open(str(fixture),'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(48000)
        audio.writeframes(bytes(48000*2*25))
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox',
            '--autoplay-policy=no-user-gesture-required','--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+str(fixture)])
        try:
            context=await browser.new_context(permissions=['microphone'])
            for case in result['greetings']:
                page=await context.new_page();events=[]
                page.on('pageerror',lambda error:result['page_errors'].append(str(error)))
                def socket(ws):
                    if not ws.url.endswith('/api/events'):return
                    def received(raw):
                        event=json.loads(raw)
                        if event.get('type')=='state':events.append((event['state'],time.monotonic()))
                    ws.on('framereceived',received)
                page.on('websocket',socket)
                try:
                    await page.goto(url,wait_until='networkidle')
                    if case['language']=='en':await page.locator('#language').click()
                    await page.locator('#transcript-toggle').check();await page.locator('#start').click()
                    await page.locator('#end').wait_for(state='visible',timeout=30000)
                    deadline=time.monotonic()+15
                    while time.monotonic()<deadline:
                        if any(s=='speaking' for s,t in events) and events[-1][0]=='listening':break
                        await asyncio.sleep(.1)
                    start=next((t for s,t in events if s=='speaking'),None)
                    finish=next((t for s,t in events if s=='listening' and start and t>start),None)
                    assert start and finish
                    audio=await page.locator('#audio audio').evaluate_all('(els)=>els.map(a=>({currentTime:a.currentTime,readyState:a.readyState}))')
                    text=await page.locator('body').inner_text()
                    assert '[friendly]' not in text and '[short breath]' not in text
                    assert not await page.locator('#error').is_visible()
                    assert any(a['currentTime']>0 and a['readyState']>=2 for a in audio)
                    case.update(speaking_seconds=round(finish-start,3),audio=audio,tags_hidden=True)
                finally:
                    if await page.locator('#end').is_visible():await page.locator('#end').click()
                    await page.close()
            assert not result['page_errors'];result['passed']=True
        finally:
            await browser.close()
            (target/(prefix+'-playback-20261018.json')).write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
