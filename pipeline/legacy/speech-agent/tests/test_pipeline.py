import asyncio
import json
import time
import pytest
from app.phrases import boundary,phrases,punctuate
from app.config import Settings,ROOT
from app.agent import AljawabAgent

@pytest.mark.parametrize('text',['Hello?','كيف؟','Great!','Value 3.14.','Use gpt-6-luna.'])
def test_punctuation(text):assert punctuate(text)==text

def test_decimals_urls_abbreviations():
    text='Dr. Ahmad uses 3.14 at https://aljawab.ai. Next sentence.'
    end=boundary(text)
    assert '3.14' in text[:end] and 'https://aljawab.ai.' in text[:end]

@pytest.mark.asyncio
async def test_no_punctuation_time_flush():
    async def source():
        yield 'A long phrase without punctuation'
        await asyncio.sleep(.1)
        yield ' and more'
    gen=phrases(source(),max_wait=.02)
    start=time.monotonic()
    assert await anext(gen)=='A long phrase without punctuation.'
    assert time.monotonic()-start<.08
    await gen.aclose()

@pytest.mark.asyncio
async def test_token_fragments_not_words():
    async def source():
        for x in ['Value ', '3.', '14', ' is useful.', ' Next question?']:yield x
    result=[x async for x in phrases(source())]
    assert result==['Value 3.14 is useful.','Next question?']

def test_model_cannot_change():
    with pytest.raises(ValueError):Settings(tts_model='fast_r32_final')

def test_verified_knowledge_and_assets():
    data=json.loads((ROOT/'knowledge/snapshot.json').read_text())
    assert len(data['pages'])>=2
    for p in data['pages']:
        assert p['url'].startswith('https://aljawab.ai/') and p['retrieved_at'] and len(p['text'])>1800
    manifest=json.loads((ROOT/'assets/voice/manifest.json').read_text())
    assert len(manifest)==8
    assert all(x['model']=='attention_mlp_r16_early' and x['duration']>.1 for x in manifest.values())

@pytest.mark.asyncio
async def test_slow_decisions_fails_open(monkeypatch):
    from types import SimpleNamespace
    agent=object.__new__(AljawabAgent);agent.last_ack=0
    from app.config import settings
    monkeypatch.setattr(settings,'decisions_deadline_ms',50)
    async def slow(**kwargs):await asyncio.sleep(10)
    agent.client=SimpleNamespace(decisions=SimpleNamespace(create=slow))
    start=time.monotonic()
    assert await agent.choose_ack('What does Aljawab offer?','en') is None
    assert time.monotonic()-start<.2

@pytest.mark.asyncio
async def test_answer_never_waits_for_decisions():
    from types import SimpleNamespace
    from livekit.agents import llm
    agent=object.__new__(AljawabAgent)
    agent.generation=0;agent.preference='auto';agent.language='ar';agent.instructions_text='Test';agent.ack_task=None
    async def slow(*args):await asyncio.sleep(10)
    agent.choose_ack=slow
    class Stream:
        def __aiter__(self):return self
        async def __anext__(self):return SimpleNamespace(type='response.output_text.delta',delta='Substantive answer.')
        async def close(self):pass
    async def create(**kwargs):return Stream()
    agent.client=SimpleNamespace(responses=SimpleNamespace(create=create))
    context=llm.ChatContext();context.add_message(role='user',content='What does Aljawab do?')
    iterator=agent.llm_node(context,[],None);start=time.monotonic()
    assert await anext(iterator)=='Substantive answer.'
    assert time.monotonic()-start<.05 and not agent.ack_task.done()
    await iterator.aclose();agent.ack_task.cancel();await asyncio.gather(agent.ack_task,return_exceptions=True)

@pytest.mark.asyncio
async def test_late_tts_frame_cannot_revive_interruption():
    from types import SimpleNamespace
    from livekit import rtc
    agent=object.__new__(AljawabAgent);agent.generation=0;agent.ack_task=None
    async def late(text):
        await asyncio.sleep(.03)
        yield rtc.AudioFrame(bytes(1764),44100,1,882)
    agent.speech=SimpleNamespace(stream=late)
    async def source():yield 'Complete sentence.'
    async def interrupt():await asyncio.sleep(.01);agent.generation+=1
    task=asyncio.create_task(interrupt())
    assert [x async for x in agent.tts_node(source(),None)]==[]
    await task

@pytest.mark.asyncio
async def test_tts_failure_returns_plain_recoverable_event():
    from types import SimpleNamespace
    agent=object.__new__(AljawabAgent);agent.generation=0;agent.ack_task=None
    async def refused(text):
        raise RuntimeError('Private service failure details')
        yield
    messages=[]
    async def notify(message):messages.append(message)
    agent.on_failure=notify;agent.speech=SimpleNamespace(stream=refused)
    async def source():yield 'Complete sentence.'
    assert [x async for x in agent.tts_node(source(),None)]==[]
    assert messages==[{'type':'error'}]

@pytest.mark.asyncio
async def test_missing_credentials_is_recoverable(monkeypatch):
    import httpx
    from app.server import app
    from app.config import settings
    monkeypatch.setattr(settings,'openai_api_key','')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='https://test') as client:
        response=await client.post('/api/session',json={'language':'ar'})
    assert response.status_code==503
    assert response.json()=={'detail':'Please try again shortly'}

@pytest.mark.asyncio
async def test_ready_substantive_audio_suppresses_cached_ack(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings,'audio_dynamics_enabled',False)
    from types import SimpleNamespace
    from livekit import rtc
    agent=object.__new__(AljawabAgent);agent.generation=0;agent.turn_started=time.monotonic()-1
    agent.last_ack=0;called=[]
    async def choice():return 'ack_en_aha'
    agent.ack_task=asyncio.create_task(choice());await agent.ack_task
    async def audio(text):yield rtc.AudioFrame(bytes(1764),44100,1,882)
    async def cached(name):called.append(name);yield rtc.AudioFrame(bytes(1764),44100,1,882)
    agent.speech=SimpleNamespace(stream=audio,cached=cached)
    # Property used only for timing metadata, not synthesis.
    agent._stt=SimpleNamespace(last_speech_end=time.monotonic())
    async def text():yield 'Answer.'
    assert len([x async for x in agent.tts_node(text(),None)])==1
    assert called==[]

@pytest.mark.asyncio
async def test_cleanup_is_scoped_and_survives_multiple_clients(tmp_path,monkeypatch):
    import app.speech as speech
    studio=tmp_path/'studio';(studio/'results').mkdir(parents=True)
    monkeypatch.setattr(speech,'STUDIO',studio)
    monkeypatch.setattr(speech,'OWNED',tmp_path/'owned')
    monkeypatch.setattr(speech,'LEGACY_OWNED',tmp_path/'legacy.json')
    first=speech.Speech();second=speech.Speech();one='a'*32;two='b'*32;other='c'*32
    first.own(one);second.own(two)
    for name in (one,two,other):
        (studio/'results'/(name+'.json')).write_text(json.dumps({'status':'complete','request':{'model':'attention_mlp_r16_early'}}))
        (studio/'results'/(name+'.wav')).write_bytes(b'generated fixture')
    await first.cleanup()
    assert not (studio/'results'/(one+'.wav')).exists()
    assert not (studio/'results'/(two+'.wav')).exists()
    assert (studio/'results'/(other+'.wav')).exists()
    assert list(speech.OWNED.iterdir())==[]
    await first.client.aclose();await second.client.aclose()

@pytest.mark.asyncio
async def test_selection_after_eligibility_timer_can_play_without_holding_synthesis(monkeypatch):
    from types import SimpleNamespace
    from livekit import rtc
    from app.config import settings
    monkeypatch.setattr(settings,'audio_dynamics_enabled',False)
    monkeypatch.setattr(settings,'acknowledgment_eligibility_delay_ms',10)
    agent=object.__new__(AljawabAgent);agent.generation=0;agent.turn_started=time.monotonic();agent.last_ack=0
    started=[]
    async def choose():await asyncio.sleep(.03);return 'ack_ar_tamam'
    agent.ack_task=asyncio.create_task(choose())
    async def audio(text):
        started.append(time.monotonic())
        await asyncio.sleep(.08)
        yield rtc.AudioFrame(bytes([2,0])*882,44100,1,882)
    async def cached(name):yield rtc.AudioFrame(bytes([1,0])*882,44100,1,882)
    agent.speech=SimpleNamespace(stream=audio,cached=cached)
    agent._stt=SimpleNamespace(last_speech_end=time.monotonic())
    async def text():yield 'جواب.'
    frames=[x async for x in agent.tts_node(text(),None)]
    assert [bytes(x.data)[:2] for x in frames]==[b'\x01\x00',b'\x02\x00']
    assert started[0]-agent.turn_started<.02

@pytest.mark.asyncio
async def test_similar_ack_variants_do_not_suppress_a_clear_ack_decision():
    from types import SimpleNamespace as NS
    agent=object.__new__(AljawabAgent);agent.last_ack=0
    async def create(**kwargs):
        return NS(answers=[NS(choice='ack_ar_understood',confidence=.25,
            probabilities=[NS(value='none',probability=.05),NS(value='ack_ar_tamam',probability=.45),NS(value='ack_ar_understood',probability=.5)])])
    agent.client=NS(decisions=NS(create=create))
    assert await agent.choose_ack('أبي أعرف أكثر عن خدماتكم.','ar')=='ack_ar_understood'
