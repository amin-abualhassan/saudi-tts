"""Concurrent Chromium clients and simulated mobile browser failure recovery.

Emulation runs from this server's network, not physical phones or a carrier.
No session credentials, spoken content or client IPs are recorded.
"""
import asyncio
import json
import os
from pathlib import Path
import re
import wave

from playwright.async_api import async_playwright


async def main():
    force_relay=os.environ.get('FORCE_TURN_RELAY')=='1'
    target=Path('tests/results');fixture=(target/'mobile-silent-mic-20261018.wav').resolve()
    with wave.open(str(fixture),'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(48000)
        audio.writeframes(bytes(48000*2*35))
    url=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',Path('logs/tunnel.log').read_text())[-1]
    result={'passed':False,'limitation':'Chromium emulation on the server network; not physical Safari or carrier testing.',
            'forced_turn_relay':force_relay,
            'page_errors':[],'concurrent_clients':[],'checks':{}}
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox',
            '--autoplay-policy=no-user-gesture-required','--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+str(fixture)])
        contexts=[];pages=[]
        async def page_for(device=None):
            options={**(p.devices[device] if device else {}),'permissions':['microphone']}
            # All profiles deliberately use Chromium, including iPhone viewport.
            options.pop('default_browser_type',None)
            context=await browser.new_context(**options);contexts.append(context)
            page=await context.new_page();pages.append(page)
            if force_relay:
                await page.add_init_script("""window.__rtcPeers=[];
                    const Native=window.RTCPeerConnection;
                    window.RTCPeerConnection=new Proxy(Native,{construct(target,args){
                        const pc=Reflect.construct(target,[{...args[0],iceTransportPolicy:'relay'}]);
                        const configure=pc.setConfiguration.bind(pc);
                        pc.setConfiguration=(config)=>configure({...config,iceTransportPolicy:'relay'});
                        window.__rtcPeers.push(pc);return pc;
                    }});""")
            page.on('pageerror',lambda error:result['page_errors'].append(str(error)))
            return page
        async def open_page(page):
            await page.goto(url,wait_until='networkidle');await page.locator('#language').click()
        async def connect(page):
            await page.locator('#start').click()
            await page.locator('#end').wait_for(state='visible',timeout=30000)
        async def end(page):
            if await page.locator('#end').is_visible():await page.locator('#end').click()
        try:
            clients=[await page_for(device) for device in (None,'Pixel 7','iPhone 13')]
            await asyncio.gather(*(open_page(page) for page in clients))
            await asyncio.gather(*(connect(page) for page in clients))
            if force_relay:
                # Exercise the same graceful reload used for certificate renewal
                # while all three browser calls have active TLS connections.
                reload=await asyncio.create_subprocess_exec('sudo','systemctl','reload','aljawab-turn-proxy')
                assert await reload.wait()==0
                result['checks']['tls_proxy_reload_with_three_active_calls']=True
            await asyncio.sleep(5)
            for label,page in zip(('desktop','android-profile','iphone-profile'),clients):
                audio=await page.locator('#audio audio').evaluate_all('(els)=>els.map(a=>({currentTime:a.currentTime,readyState:a.readyState}))')
                item={'profile':label,'connected':await page.locator('#end').is_visible(),
                      'error_visible':await page.locator('#error').is_visible(),'audio':audio}
                assert item['connected'] and not item['error_visible']
                assert any(a['currentTime']>0 and a['readyState']>=2 for a in audio)
                if force_relay:
                    pairs=await page.evaluate("""async()=>{
                        const chosen=[];
                        for(const pc of window.__rtcPeers){
                            const stats=await pc.getStats();
                            for(const transport of stats.values())if(transport.type==='transport'&&transport.selectedCandidatePairId){
                                const pair=stats.get(transport.selectedCandidatePairId),local=stats.get(pair.localCandidateId);
                                chosen.push({candidateType:local.candidateType,relayProtocol:local.relayProtocol,url:local.url});
                            }
                        }
                        return chosen;
                    }""")
                    assert pairs and all(pair['candidateType']=='relay' and pair.get('relayProtocol')=='tls' for pair in pairs),pairs
                    item['selected_transports']=pairs
                result['concurrent_clients'].append(item)
            await asyncio.gather(*(end(page) for page in clients))

            blocked=await page_for('iPhone 13')
            await blocked.add_init_script("""window.__blockAudio=true;
                const nativePlay=HTMLMediaElement.prototype.play;
                HTMLMediaElement.prototype.play=function(){
                    if(window.__blockAudio)return Promise.reject(new DOMException('Simulated autoplay denial','NotAllowedError'));
                    return nativePlay.call(this);
                };""")
            await open_page(blocked);await connect(blocked)
            await blocked.locator('#enable-audio').wait_for(state='visible',timeout=10000)
            await blocked.screenshot(path=str(target/'mobile-enable-audio-20261018.png'),full_page=True)
            await blocked.evaluate('window.__blockAudio=false')
            await blocked.locator('#enable-audio').click()
            await blocked.locator('#enable-audio').wait_for(state='hidden',timeout=10000)
            assert await blocked.locator('#end').is_visible()
            result['checks']['autoplay_denial_keeps_call_connected_and_recovers']=True
            await end(blocked)

            limited=await page_for('Pixel 7')
            await limited.route('**/api/session',lambda route:route.fulfill(status=429,content_type='application/json',body='{}'))
            await open_page(limited);await limited.locator('#start').click()
            await limited.get_by_text('Too many connection attempts. Wait one minute, then try again.',exact=True).wait_for(timeout=10000)
            assert await limited.locator('#start').is_enabled()
            result['checks']['rate_limit_has_distinct_retry_message']=True

            failed=await page_for('iPhone 13')
            await failed.add_init_script("""const Native=window.RTCPeerConnection;
                window.RTCPeerConnection=new Proxy(Native,{construct(){throw new DOMException('Simulated media failure','NetworkError')}});""")
            reports=[]
            failed.on('request',lambda request:reports.append(request.post_data_json) if request.url.endswith('/api/connection-report') else None)
            await open_page(failed);await failed.locator('#start').click()
            await failed.get_by_text('We couldn’t establish the audio connection. Try switching between Wi-Fi and mobile data, or open this link in Safari or Chrome.',exact=True).wait_for(timeout=30000)
            assert reports and reports[-1]=={'stage':'media','reason':'network','platform':'ios'},reports
            assert await failed.locator('#start').is_enabled()
            result['checks']['media_failure_has_actionable_message_and_private_diagnostic']=True
            assert not result['page_errors'],result['page_errors']
            result['passed']=True
        finally:
            for page in pages:
                try:await end(page)
                except Exception:pass
            await browser.close()
            name='turn-relay-connections-20261018.json' if force_relay else 'mobile-connections-20261018.json'
            (target/name).write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
