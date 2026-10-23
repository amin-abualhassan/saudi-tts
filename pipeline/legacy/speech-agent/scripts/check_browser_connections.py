"""Check two tabs sharing cookies and silent-microphone idle cleanup."""
import asyncio
import json
from pathlib import Path
import re
import time
import wave

from playwright.async_api import async_playwright


async def main():
    target=Path('tests/results');target.mkdir(exist_ok=True)
    fixture=(target/'silence-20261018.wav').resolve()
    with wave.open(str(fixture),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(48000)
        wav.writeframes(bytes(48000*2*85))
    url=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',Path('logs/tunnel.log').read_text())[-1]
    result={'passed':False,'page_errors':[],'tabs':[]}
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(headless=True,args=['--no-sandbox',
            '--autoplay-policy=no-user-gesture-required','--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+str(fixture)])
        context=await browser.new_context(permissions=['microphone'])
        pages=[await context.new_page(),await context.new_page()]
        try:
            for page in pages:
                page.on('pageerror',lambda error:result['page_errors'].append(str(error)))
                await page.goto(url,wait_until='networkidle')
                await page.locator('#language').click()
            start=time.monotonic()
            await asyncio.gather(*(page.locator('#start').click() for page in pages))
            await asyncio.gather(*(page.locator('#end').wait_for(state='visible',timeout=30000) for page in pages))
            result['both_connected']=True
            print('Both browser tabs connected in one shared-cookie context.',flush=True)
            await asyncio.gather(*(page.get_by_text('Conversation closed after one minute without speech. You can start again.',exact=True).wait_for(timeout=75000) for page in pages))
            result['idle_elapsed_s']=round(time.monotonic()-start,2)
            for index,page in enumerate(pages):
                result['tabs'].append({'start_visible':await page.locator('#start').is_visible(),
                    'end_hidden':await page.locator('#end').is_hidden(),
                    'status':await page.locator('#status').inner_text(),
                    'audio_elements':await page.locator('audio').count(),
                    'error_visible':await page.locator('#error').is_visible()})
                await page.screenshot(path=str(target/f'connections-idle-tab-{index+1}.png'),full_page=True)
            assert not result['page_errors'],result
            assert all(tab['start_visible'] and tab['end_hidden'] and not tab['error_visible'] and tab['audio_elements']==0 for tab in result['tabs']),result
            result['passed']=True
        finally:
            await browser.close()
            (target/'browser-connections-20261018.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
