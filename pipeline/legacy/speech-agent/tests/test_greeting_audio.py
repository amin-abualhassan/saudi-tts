import io
import wave

import numpy as np
import pytest

from app.greeting_audio import tighten_greeting


def wav(samples,rate=44100):
    output=io.BytesIO()
    with wave.open(output,'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(rate)
        audio.writeframes(np.asarray(samples,dtype='<i2').tobytes())
    return output.getvalue()


def read(data):
    with wave.open(io.BytesIO(data)) as audio:
        return np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2')


def test_only_long_quiet_gaps_are_shortened_and_speech_samples_are_untouched():
    speech=(np.sin(np.arange(22050)*.1)*12000).astype('<i2')
    samples=np.concatenate((speech,np.zeros(22050),speech,np.zeros(4410),speech))
    output,meta=tighten_greeting(wav(samples));result=read(output)
    assert len(meta['gaps'])==1
    assert meta['removed_seconds']==pytest.approx(.3)
    assert len(result)==len(samples)-13230
    assert np.array_equal(result[:len(speech)],speech)
    assert np.array_equal(result[-len(speech)*2-4410:],np.concatenate((speech,np.zeros(4410),speech)))
    assert np.max(np.abs(result.astype(int)))<=12000


def test_short_pauses_and_quiet_speech_above_the_threshold_are_unchanged():
    samples=np.concatenate((np.ones(22050)*150,np.zeros(8820),np.ones(22050)*150))
    data=wav(samples);result,meta=tighten_greeting(data)
    assert result==data and not meta['gaps']


def test_repeated_processing_is_idempotent_and_preserves_final_sound():
    samples=np.r_[np.ones(22050)*1000,np.zeros(26460),np.ones(12345)*777]
    first,_=tighten_greeting(wav(samples));second,meta=tighten_greeting(first)
    assert first==second and not meta['gaps']
    assert read(first)[-1]==777


def test_other_sample_rates_are_rejected():
    with pytest.raises(ValueError):tighten_greeting(wav([0,100,0],rate=24000))
