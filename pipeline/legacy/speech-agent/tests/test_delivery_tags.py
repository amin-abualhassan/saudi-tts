import base64
import json

import httpx
import pytest

from app import speech
from app.cues import transcript_chunks, without_delivery_cues, with_final_breath
from app.phrases import phrases, punctuate


@pytest.fixture
def voice(monkeypatch,tmp_path):
    monkeypatch.setattr(speech.settings,'tts_reference','upload_fixture')
    monkeypatch.setattr(speech.settings,'tts_reference_text','Exact reference transcript.')
    monkeypatch.setattr(speech.settings,'tts_reference_sha256','fixture-hash')
    monkeypatch.setattr(speech,'OWNED',tmp_path/'owned')
    monkeypatch.setattr(speech,'LEGACY_OWNED',tmp_path/'legacy.json')


@pytest.mark.parametrize('text,expected',[
    ('We have 3 options.','We have three options.'),
    ('عندنا ٣ خيارات.','عندنا ثلاثة خيارات.'),
])
def test_payload_preserves_cues_without_changing_reference_or_sampling(voice,text,expected):
    plain=speech.payload(text)
    tagged=speech.payload('[calm] '+text+' [short breath]')
    assert plain['text']==expected
    assert tagged['text']=='[calm] '+expected+' [short breath]'
    assert plain['reference_text']==tagged['reference_text']=='Exact reference transcript.'
    assert {k:v for k,v in plain.items() if k!='text'}=={k:v for k,v in tagged.items() if k!='text'}
    assert '[' not in text


@pytest.mark.parametrize('text,expected',[
    ('[friendly] Hello. [short breath]','Hello.'),
    ('[empathetic] فهمت عليك. [short breath] خليني أوضح. [short breath]','فهمت عليك. خليني أوضح.'),
    ('Hello. [short bre','Hello.'),
    ('A list [one, two] is data.','A list [one, two] is data.'),
])
def test_complete_and_interrupted_cues_stay_out_of_transcripts(text,expected):
    assert without_delivery_cues(text)==expected


@pytest.mark.asyncio
async def test_split_cues_never_leak_into_transcription():
    text='[friendly] Hello. [short breath] Next. [short breath]'
    # Exercise every possible two-delta boundary, including within tag names.
    for index in range(len(text)+1):
        async def source():
            yield text[:index]
            yield text[index:]
        clean=''.join([chunk async for chunk in transcript_chunks(source())])
        assert without_delivery_cues(clean)=='Hello. Next.'
        assert '[' not in clean and 'breath' not in clean


@pytest.mark.asyncio
async def test_breath_stays_with_previous_sentence_across_deltas():
    import asyncio
    async def source():
        for chunk in ['[friendly] Hello. ','[short',' breath]',' Next.',' [short ','breath]']:
            yield chunk
            await asyncio.sleep(.005)
    result=[chunk async for chunk in phrases(source(),max_wait=.001,expect_cues=True)]
    assert result==['[friendly] Hello. [short breath]','Next. [short breath]']


@pytest.mark.asyncio
async def test_long_sentence_is_not_cut_before_its_llm_breath():
    import asyncio
    sentence='[calm] '+('A longer explanation continues naturally, '*6)+'and finishes. [short breath]'
    async def source():
        for start in range(0,len(sentence),13):
            yield sentence[start:start+13]
            await asyncio.sleep(.002)
    assert [chunk async for chunk in phrases(source(),max_wait=.001,expect_cues=True)]==[sentence]


@pytest.mark.asyncio
async def test_untagged_sentence_then_final_breath_never_inserts_a_cue():
    async def source():
        for chunk in ['First sentence. ','Second sentence. ','[short breath]']:yield chunk
    assert [chunk async for chunk in phrases(source(),expect_cues=True)]==[
        'First sentence.','Second sentence. [short breath]']


def test_punctuation_precedes_trailing_cue():
    assert punctuate('Hello [short breath]')=='Hello. [short breath]'
    assert punctuate('Hello. [short bre')=='Hello.'


@pytest.mark.asyncio
async def test_cue_only_input_does_not_generate_audio():
    async def source():yield '[short breath]'
    assert [chunk async for chunk in phrases(source(),expect_cues=True)]==[]


@pytest.mark.asyncio
async def test_live_stream_retains_llm_cues_and_complete_pcm_tail(voice):
    requests=[];job_id='a'*32
    pcm=bytes(range(256))*9  # Includes a final frame shorter than 20 ms.
    def handle(request):
        if request.method=='POST':
            requests.append(json.loads(request.content))
            events=[{'event':'accepted','id':job_id,'encoding':'pcm_s16le','sample_rate':44100},
                    {'event':'audio','pcm':base64.b64encode(pcm).decode()},
                    {'event':'complete','checkpoint_sha256':speech.MODEL_SHA}]
            return httpx.Response(200,text=''.join('data: '+json.dumps(event)+'\n\n' for event in events))
        return httpx.Response(200,json={'jobs':[{'id':job_id,'request':requests[0],
            'checkpoint_sha256':speech.MODEL_SHA,'reference_sha256':'fixture-hash'}]})
    client=speech.Speech()
    await client.client.aclose()
    client.client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    llm_text='[empathetic] Hello, how can I help? [short breath]'
    try:
        frames=[frame async for frame in client.stream(llm_text)]
        assert len(frames)==2
        assert b''.join(bytes(frame.data) for frame in frames)==pcm
        assert requests[0]['text']==llm_text
        assert without_delivery_cues(llm_text)=='Hello, how can I help?'
        assert requests[0]['model']=='attention_mlp_r16_early'
    finally:await client.client.aclose()


@pytest.mark.parametrize('text,expected',[
    ('First sentence.','First sentence. [short breath]'),
    ('[calm] مرحبًا.','[calm] مرحبًا. [short breath]'),
    ('Hello. [short breath] [SHORT  BREATH]','Hello. [short breath]'),
    ('Hello [short breath] there.','Hello there. [short breath]'),
    ('Hello. [short bre','Hello. [short breath]'),
    ('[calm] [short breath]',''),
])
def test_private_final_breath_is_added_once_and_is_idempotent(text,expected):
    assert with_final_breath(text)==expected
    assert with_final_breath(expected)==expected


@pytest.mark.asyncio
async def test_each_actual_tts_request_gets_a_breath_even_when_llm_emits_none(monkeypatch):
    import time
    from types import SimpleNamespace
    from livekit import rtc
    from app.agent import AljawabAgent
    from app.config import settings
    monkeypatch.setattr(settings,'audio_dynamics_enabled',False)
    requests=[]
    async def stream(text):
        requests.append(text)
        yield rtc.AudioFrame(bytes(1764),44100,1,882)
    agent=object.__new__(AljawabAgent);agent.generation=0;agent.ack_task=None
    agent.speech=SimpleNamespace(stream=stream)
    agent._stt=SimpleNamespace(last_speech_end=time.monotonic())
    async def source():
        for chunk in ['[calm] First sentence. ','ثاني جملة؟ ','Final sentence.']:yield chunk
    frames=[frame async for frame in agent.tts_node(source(),None)]
    assert len(frames)==3
    assert requests==['[calm] First sentence. [short breath]',
                      'ثاني جملة؟ [short breath]','Final sentence. [short breath]']
    assert 'breath' not in without_delivery_cues(' '.join(requests))
