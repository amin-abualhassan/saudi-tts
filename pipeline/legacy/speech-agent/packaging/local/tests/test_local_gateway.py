import asyncio,base64,hashlib,io,json,wave
from pathlib import Path
import httpx
import pytest
from tts import gateway

@pytest.fixture
def engine(monkeypatch,tmp_path):
    calls=[];model='/models/attention_mlp_r16_early'
    output=io.BytesIO()
    with wave.open(output,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(bytes([1,0])*4410)
    wav=output.getvalue();pcm=bytes([1,0])*882
    def handle(request):
        if request.url.path=='/v1/models':return httpx.Response(200,json={'data':[{'id':model}]})
        data=json.loads(request.content);calls.append(data)
        assert data['model']==model and data['references'][0]['text']
        assert Path(data['references'][0]['audio_path']).is_file()
        return httpx.Response(200,content=pcm if data['stream'] else wav)
    original=httpx.AsyncClient
    monkeypatch.setattr(gateway.configuration,'sglang_model_path',model)
    monkeypatch.setattr(gateway,'data_root',tmp_path)
    monkeypatch.setattr(gateway,'jobs',{})
    monkeypatch.setattr(gateway,'references',{})
    monkeypatch.setattr(gateway,'lock',asyncio.Lock())
    monkeypatch.setattr(gateway,'verify_model',lambda path:None)
    monkeypatch.setattr(gateway.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handle)))
    return original,calls,wav,pcm

def payload():
    reference=json.loads((gateway.ROOT/'assets/voice/reference.json').read_text())
    return {'model':gateway.MODEL,'reference_id':reference['reference_id'],'reference_text':reference['transcript'],
            'text':'حيّاك الله، معك نورة.','seed':42,'temperature':.7,'top_p':.7,'max_new_tokens':1024}

@pytest.mark.asyncio
async def test_stream_contract_and_exact_model_voice(engine):
    original,calls,wav,pcm=engine
    async with gateway.lifespan(gateway.app):
        async with original(transport=httpx.ASGITransport(app=gateway.app),base_url='http://test') as client:
            response=await client.post('/api/stream',json=payload());assert response.status_code==200
            events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            assert events[0]['event']=='accepted' and events[-1]['event']=='complete'
            assert b''.join(base64.b64decode(e['pcm']) for e in events if e['event']=='audio')==pcm
            job=(await client.get('/api/status')).json()['jobs'][0]
            reference=json.loads((gateway.ROOT/'assets/voice/reference.json').read_text())
            assert job['checkpoint_sha256']==gateway.CHECKPOINT and job['reference_sha256']==reference['prepared_sha256']
            assert calls[0]['stream'] is True and calls[0]['response_format']=='pcm'

@pytest.mark.asyncio
async def test_wrong_model_is_rejected(engine):
    original,calls,_,_=engine
    async with gateway.lifespan(gateway.app):
        async with original(transport=httpx.ASGITransport(app=gateway.app),base_url='http://test') as client:
            request=payload();request['model']='fast_r32_final'
            assert (await client.post('/api/generate',json=request)).status_code==400
            assert calls==[]

@pytest.mark.asyncio
async def test_cached_wav_generation_contract(engine):
    original,calls,wav,_=engine
    async with gateway.lifespan(gateway.app):
        async with original(transport=httpx.ASGITransport(app=gateway.app),base_url='http://test') as client:
            response=await client.post('/api/generate',json=payload());assert response.status_code==202
            for _ in range(20):
                job=(await client.get('/api/status')).json()['jobs'][0]
                if job['status']=='complete':break
                await asyncio.sleep(.01)
            assert job['status']=='complete'
            response=await client.get(job['audio_url']);assert response.content==wav
            assert calls[0]['stream'] is False

@pytest.mark.asyncio
async def test_supplied_noura_reference_is_available(engine):
    original,_,_,_=engine
    async with gateway.lifespan(gateway.app):
        async with original(transport=httpx.ASGITransport(app=gateway.app),base_url='http://test') as client:
            refs=(await client.get('/api/stream/references')).json()['references'];assert len(refs)==1
            audio=(await client.get('/api/reference/'+refs[0]['id']+'.wav')).content
            reference=json.loads((gateway.ROOT/'assets/voice/reference.json').read_text())
            assert hashlib.sha256(audio).hexdigest()==reference['prepared_sha256']

@pytest.mark.asyncio
async def test_real_app_adapter_consumes_gateway_stream(engine,monkeypatch,tmp_path):
    import app.speech as speech
    original,_,_,pcm=engine
    reference=json.loads((gateway.ROOT/'assets/voice/reference.json').read_text())
    monkeypatch.setattr(speech.settings,'tts_reference',reference['reference_id'])
    monkeypatch.setattr(speech.settings,'tts_reference_text',reference['transcript'])
    monkeypatch.setattr(speech.settings,'tts_reference_sha256',reference['prepared_sha256'])
    monkeypatch.setattr(speech,'OWNED',tmp_path/'owned');speech.OWNED.mkdir()
    monkeypatch.setattr(speech,'STUDIO',gateway.data_root)
    async with gateway.lifespan(gateway.app):
        async with original(transport=httpx.ASGITransport(app=gateway.app),base_url='http://test') as client:
            adapter=object.__new__(speech.Speech);adapter.client=client
            frames=[frame async for frame in adapter.stream('حيّاك الله، معك نورة.')]
            assert b''.join(bytes(frame.data) for frame in frames)==pcm
            await adapter.cleanup()
            assert list(speech.OWNED.iterdir())==[]

def test_local_livekit_config_has_no_aws_dependency(monkeypatch,tmp_path):
    import sys,yaml
    from scripts import configure
    monkeypatch.setattr(configure,'ROOT',tmp_path)
    (tmp_path/'.env.example').write_text('OPENAI_API_KEY=\nLIVEKIT_API_KEY=\nLIVEKIT_API_SECRET=\n')
    monkeypatch.setattr(sys,'argv',['configure','--node-ip','127.0.0.1'])
    configure.main()
    config=yaml.safe_load((tmp_path/'deployment/livekit.yaml').read_text())
    assert config['rtc']['node_ip']=='127.0.0.1' and config['bind_addresses']==['127.0.0.1']
    assert all(len(secret)>32 for secret in config['keys'].values())

def test_silent_reference_is_rejected():
    output=io.BytesIO()
    with wave.open(output,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(bytes(44100*2*3))
    with pytest.raises(ValueError,match='silent'):gateway.validate_audio(output.getvalue())
