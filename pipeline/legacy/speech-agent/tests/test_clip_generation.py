import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import wave

import pytest

from scripts import generate_clips


@pytest.mark.asyncio
@pytest.mark.parametrize('cached_breath',[False,True])
async def test_existing_greeting_is_regenerated_once_with_friendly_tone_without_changing_ui_or_other_clips(tmp_path,monkeypatch,cached_breath):
    generator=generate_clips
    monkeypatch.setattr(generator,'ROOT',tmp_path)
    monkeypatch.setattr(generator.settings,'tts_reference','upload_fixture')
    monkeypatch.setattr(generator.settings,'tts_reference_text','Reference fixture.')
    monkeypatch.setattr(generator.settings,'tts_reference_sha256','reference-hash')
    monkeypatch.setattr(generator,'CLIPS',{'greeting_ar':'حيّاك الله.','ack_ar_tamam':'تمام.'})
    target=tmp_path/'assets/voice';target.mkdir(parents=True)
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(44100)
        audio.writeframes(bytes([100,0])*4410)
    raw=buffer.getvalue();manifest={}
    for name,text in generator.CLIPS.items():
        (target/(name+'.wav')).write_bytes(raw)
        manifest[name]={'text':text,'model':generator.settings.tts_model,
            'reference':generator.settings.tts_reference,'reference_sha256':'reference-hash',
            'checkpoint_sha256':generator.MODEL_SHA,'sha256':hashlib.sha256(raw).hexdigest(),
            'settings':generator.payload(text+' [short breath]' if name.startswith('greeting_') and cached_breath else text)}
    (target/'manifest.json').write_text(json.dumps(manifest))
    requests=[]
    async def generate(text):
        requests.append(text)
        return raw,{'reference_sha256':'reference-hash','request':generator.payload(text)}
    monkeypatch.setattr(generator,'Speech',lambda:SimpleNamespace(generate=generate,
        cleanup=AsyncMock(),client=SimpleNamespace(aclose=AsyncMock())))
    await generator.main()
    assert requests==['[friendly] حيّاك الله. [short breath]']
    updated=json.loads((target/'manifest.json').read_text())
    assert updated['greeting_ar']['text']=='حيّاك الله.'
    assert updated['greeting_ar']['settings']['text']=='[friendly] حيّاك الله. [short breath]'
    assert updated['greeting_ar']['postprocess']['profile']=='greeting-pauses-v1'
    assert updated['ack_ar_tamam']==manifest['ack_ar_tamam']
    assert (target/'ack_ar_tamam.wav').read_bytes()==raw
    await generator.main()
    assert len(requests)==1
