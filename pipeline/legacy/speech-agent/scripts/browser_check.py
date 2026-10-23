"""Synthetic browser QA; does not claim human microphone or remote-network QA."""
import asyncio,json,os,time
from pathlib import Path
from playwright.async_api import async_playwright
import re
URL=os.environ.get('TEST_URL') or re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',Path('logs/tunnel.log').read_text())[-1]

async def main():
 target=Path('tests/results');target.mkdir(parents=True,exist_ok=True)
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required',
    '--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',
    '--use-file-for-fake-audio-capture='+os.environ.get('FAKE_FIXTURE','/tmp/aljawab-browser-fixture.wav')+'%noloop'])
  context=await browser.new_context(permissions=['microphone'])
  tcp_only=os.environ.get('ICE_TCP_ONLY')=='1'
  await context.add_init_script('''window.__pcs=[]; const Native=window.RTCPeerConnection; window.RTCPeerConnection=class extends Native{
   constructor(...a){super(...a);window.__pcs.push(this);super.addEventListener('icecandidate',e=>{if(TCP_ONLY&&e.candidate?.candidate?.toLowerCase().includes(' udp '))e.stopImmediatePropagation()},true)}
   get localDescription(){const d=super.localDescription;return TCP_ONLY&&d?new RTCSessionDescription({type:d.type,sdp:d.sdp.replace(/^a=candidate:.* udp .*\\r?\\n/gmi,'')}):d}
   setRemoteDescription(d){if(TCP_ONLY&&d.sdp)d={type:d.type,sdp:d.sdp.replace(/^a=candidate:.* udp .*\\r?\\n/gmi,'')};return super.setRemoteDescription(d)}
   addIceCandidate(c){if(TCP_ONLY&&c?.candidate?.toLowerCase().includes(' udp '))return Promise.resolve();return super.addIceCandidate(c)}
  };'''.replace('TCP_ONLY','true' if tcp_only else 'false'))
  page=await context.new_page();errors=[];responses=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('console',lambda e:errors.append(re.sub(r'access_token=[^&\s\'"<>]+','access_token=[redacted]',e.text)) if e.type=='error' else None)
  page.on('response',lambda r:responses.append((r.url.split('?')[0],r.status)) if '/api/' in r.url else None)
  for attempt in range(6):
   await page.goto(URL,wait_until='networkidle')
   if await page.locator('#start').count():break
   await asyncio.sleep(2)
  print('page_loaded',await page.title(),flush=True)
  await page.screenshot(path=str(target/'arabic.png'),full_page=True)
  if os.environ.get('UI_LANGUAGE','en')=='en':await page.locator('#language').click()
  await page.locator('#transcript-toggle').check()
  await page.locator('#start').click()
  parallel=None
  for i in range(int(os.environ.get('CHECK_SECONDS','45'))):
   await asyncio.sleep(1)
   if i==5 and await page.locator('#end').is_visible():
    parallel=await page.evaluate("""async()=> {
      const response=await fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({language:'en'})});
      if(response.ok){const data=await response.json();await fetch('/api/end',{method:'POST',headers:{'X-Session-Token':data.session_token}});}
      return response.status;
    }""")
   if i%5==0:print('browser',i,await page.locator('#status').inner_text(),await page.locator('#transcript').inner_text(),await page.locator('#error').inner_text(),flush=True)
   if not await page.locator('#end').is_visible() and i>15:break
  await page.screenshot(path=str(target/'english-call.png'),full_page=True)
  result={'url':URL,'errors':errors,'api_responses':responses,'transcript':await page.locator('#transcript').inner_text(),
          'audio_elements':await page.locator('audio').count(),'status':await page.locator('#status').inner_text(),'parallel_status':parallel}
  result['media_stats']=await page.evaluate('''async()=>{const result=[];for(const pc of window.__pcs){for(const s of (await pc.getStats()).values()){if((s.type==='inbound-rtp'&&s.kind==='audio')||(s.type==='candidate-pair'&&s.nominated)||s.type==='local-candidate'||s.type==='remote-candidate')result.push(JSON.parse(JSON.stringify(s)))}}return result}''')
  result['audio']=await page.locator('audio').evaluate_all('(es)=>es.map(e=>({paused:e.paused,readyState:e.readyState,currentTime:e.currentTime}))')
  if await page.locator('#mute').is_visible():
   await page.locator('#mute').click()
   result['mute_label']=await page.locator('#mute').inner_text()
  if await page.locator('#end').is_visible():await page.locator('#end').click()
  await asyncio.sleep(1);result['ended']=await page.locator('#start').is_visible()
  (target/'browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  print(json.dumps(result,ensure_ascii=False),flush=True)
  await browser.close()
if __name__=='__main__':asyncio.run(main())
