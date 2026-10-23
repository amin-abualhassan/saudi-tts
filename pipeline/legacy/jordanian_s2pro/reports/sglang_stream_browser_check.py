"""Check live streaming playback controls and the preserved normal UI in Chromium."""
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]
campaign=Path(json.loads((root/'voice/selection.json').read_text())['campaign'])
case=json.loads((campaign/'comparison/cases.jsonl').read_text().splitlines()[3])
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required','--mute-audio'])
    page=browser.new_page(viewport={'width':1120,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:3000/stream',wait_until='domcontentloaded')
    page.wait_for_function("!document.getElementById('start').disabled",timeout=20000)
    page.select_option('#model','fast_r32_final')
    page.select_option('#voice',case['speaker'])
    page.fill('#speech',case['text'])
    page.click('#start')
    page.wait_for_function("document.getElementById('audio-time').textContent!=='—'",timeout=15000)
    first=page.evaluate("() => ({token:document.getElementById('token-time').textContent,firstAudio:document.getElementById('audio-time').textContent,contextState:session.context.state,sources:session.sources.size,complete:session.complete,audioSeconds:session.audioSeconds})")
    assert first['contextState']=='running' and first['sources']>0 and not first['complete'],first
    page.wait_for_function("!document.getElementById('downloads').hidden",timeout=20000)
    assert page.locator('#start').is_disabled() and page.locator('#stop').is_enabled()
    page.screenshot(path=str(root/'reports/sglang_stream_browser.png'),full_page=True)
    page.wait_for_function("document.getElementById('progress').textContent==='Playback complete.'",timeout=45000)
    assert page.locator('#start').is_enabled() and page.locator('#error').inner_text()==''
    page.fill('#speech','كيفك اليوم؟ شو رأيك نطلع نشرب قهوة بعد الشغل؟')
    page.click('#start')
    page.wait_for_function("document.getElementById('audio-time').textContent!=='—'",timeout=15000)
    page.click('#stop')
    page.wait_for_function('session===null',timeout=10000)
    assert page.locator('#error').inner_text()==''
    stopped=page.locator('#progress').inner_text()
    page.goto('http://127.0.0.1:3000/',wait_until='domcontentloaded')
    page.wait_for_function("!document.getElementById('generate').disabled",timeout=15000)
    assert page.locator('#steady-volume').is_checked()
    page.fill('#speech','كيفك اليوم؟ شو رأيك نطلع نشرب قهوة بعد الشغل؟')
    with page.expect_response(lambda r:r.url.endswith('/api/generate') and r.request.method=='POST') as posted:
        page.click('#generate')
    job=posted.value.json();assert posted.value.status==202
    page.wait_for_selector('audio[src="/api/leveled-audio/'+job['id']+'.wav"]',timeout=20000)
    assert page.get_by_role('link',name='Original WAV',exact=True).count()>0
    assert not errors,errors
    report={'status':'passed','browser':browser.version,'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'first_audio_observation':first,'playback_completed':True,'stop_playback':stopped,'normal_ui_generation_id':job['id'],'ending_volume_default_on':True,'page_errors':errors,'screenshot':'reports/sglang_stream_browser.png','note':'Headless audio output muted; AudioContext scheduling/playback lifecycle checked. Human listening quality remains pending.'}
    (root/'reports/sglang_stream_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    browser.close()
