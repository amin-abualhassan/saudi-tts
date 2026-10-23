import asyncio

from livekit import rtc
import numpy as np
import pytest

from app.audio import SAMPLE_RATE, SEGMENT_START, RollingLeveler, StreamDynamics, controlled_audio
from app.config import settings


def frame(samples):
    samples=np.asarray(samples,dtype='<i2')
    return rtc.AudioFrame(samples.tobytes(),SAMPLE_RATE,1,len(samples))


def run(samples,chunks=(882,),quiet_boost_db=None,onset_ms=None,leveler_enabled=False):
    # Explicitly isolate the legacy quiet curve for its existing component tests.
    processor=StreamDynamics(quiet_boost_db=quiet_boost_db,onset_ms=onset_ms,leveler_enabled=leveler_enabled);output=[];offset=0;index=0
    while offset<len(samples):
        size=chunks[index%len(chunks)];index+=1
        output.extend(processor.process(frame(samples[offset:offset+size])))
        offset+=size
    output.extend(processor.finish())
    return np.frombuffer(b''.join(bytes(item.data) for item in output),dtype='<i2')


def test_peak_ceiling_handles_sudden_spikes_including_negative_full_scale():
    samples=np.zeros(8820,dtype=np.int16)
    samples[::73]=32767;samples[1::101]=-32768
    output=run(samples)
    peak=np.max(np.abs(output.astype(np.int32)))
    assert peak<=32768*10**(settings.audio_peak_dbfs/20)+1
    assert len(output)==len(samples)


def test_compression_reduces_quiet_to_loud_level_change():
    tone=np.sin(np.arange(SAMPLE_RATE)*2*np.pi*440/SAMPLE_RATE)
    quiet=(tone*.04*32767).astype(np.int16)
    loud=(tone*.9*32767).astype(np.int16)
    output=run(np.concatenate((quiet,loud))).astype(np.float64)
    rms=lambda data:np.sqrt(np.mean(data.astype(np.float64)**2))
    original_difference=20*np.log10(rms(loud)/rms(quiet))
    output_difference=20*np.log10(rms(output[-22050:])/rms(output[22050:44100]))
    assert output_difference<original_difference-8


@pytest.mark.parametrize('size',[1,43,44,882,1001])
def test_delay_compensation_preserves_sample_count_and_final_sound(size):
    samples=np.zeros(size,dtype=np.int16);samples[-1]=5000
    output=run(samples)
    assert len(output)==size
    assert output[-1]!=0


def test_state_is_continuous_across_input_chunk_boundaries():
    samples=(np.sin(np.arange(15001)*.1)*25000).astype(np.int16)
    assert np.array_equal(run(samples,(len(samples),)),run(samples,(1,43,882,317)))


def test_first_twenty_ms_frame_produces_output_without_waiting_for_next_frame():
    processor=StreamDynamics()
    output=processor.process(frame(np.ones(882)*1000))
    assert output and sum(part.samples_per_channel for part in output)>=838
    assert sum(part.samples_per_channel for part in output+processor.finish())==882


@pytest.mark.asyncio
async def test_generator_bypass_is_exact(monkeypatch):
    monkeypatch.setattr(settings,'audio_dynamics_enabled',False)
    original=frame([0,100,-100,32767,-32768])
    async def source():
        yield SEGMENT_START
        yield original
    result=[part async for part in controlled_audio(source())]
    assert result==[original]


@pytest.mark.asyncio
async def test_cancel_closes_source_without_playing_delayed_tail(monkeypatch):
    monkeypatch.setattr(settings,'audio_dynamics_enabled',True)
    closed=asyncio.Event()
    async def source():
        try:
            yield frame(np.ones(882)*30000)
            await asyncio.Future()
        finally:closed.set()
    output=controlled_audio(source())
    first=await asyncio.wait_for(anext(output),.5)
    assert first.samples_per_channel<882
    await output.aclose()
    assert closed.is_set()


def test_quiet_voiced_audio_receives_only_a_gentle_bounded_boost():
    samples=(np.sin(np.arange(SAMPLE_RATE)*2*np.pi*440/SAMPLE_RATE)*32767*10**(-30/20)).astype(np.int16)
    baseline=run(samples,quiet_boost_db=0).astype(np.float64)
    lifted=run(samples,quiet_boost_db=3).astype(np.float64)
    rms=lambda audio:np.sqrt(np.mean(audio[-22050:]**2))
    gain=20*np.log10(rms(lifted)/rms(baseline))
    assert 2.8<=gain<=3.1
    # Scalar gain leaves the pitch and zero crossings intact.
    assert np.array_equal(np.signbit(lifted[-22050:]),np.signbit(baseline[-22050:]))


def test_silence_and_settled_noise_floor_are_not_lifted():
    assert not np.any(run(np.zeros(SAMPLE_RATE,dtype=np.int16)))
    noise=(np.sin(np.arange(SAMPLE_RATE)*.1)*32767*10**(-60/20)).astype(np.int16)
    assert np.array_equal(run(noise,quiet_boost_db=0),run(noise,quiet_boost_db=3))


def test_normal_loud_voice_is_not_lifted_and_can_return_after_quiet_speech():
    tone=np.sin(np.arange(SAMPLE_RATE)*2*np.pi*440/SAMPLE_RATE)
    quiet=(tone*32767*10**(-30/20)).astype(np.int16)
    loud=(tone*32767*10**(-6/20)).astype(np.int16)
    samples=np.concatenate((quiet,loud))
    baseline=run(samples,quiet_boost_db=0).astype(np.int32)
    lifted=run(samples,quiet_boost_db=3).astype(np.int32)
    assert np.max(np.abs(lifted[-22050:]-baseline[-22050:]))<=2
    assert np.max(np.abs(lifted))<=32768*10**(settings.audio_peak_dbfs/20)+1


def test_quiet_boost_rises_smoothly_without_a_gain_jump():
    samples=np.ones(SAMPLE_RATE,dtype=np.int16)*1000
    baseline=run(samples,quiet_boost_db=0).astype(np.float64)
    lifted=run(samples,quiet_boost_db=3).astype(np.float64)
    gain=20*np.log10(lifted/baseline)
    assert np.max(np.abs(np.diff(gain)))<.05
    assert abs(gain[0])<.05 and gain[-1]>2.8


def test_restart_after_a_pause_has_a_soft_attack_then_returns_to_normal():
    tone=(np.sin(np.arange(8820)*2*np.pi*440/SAMPLE_RATE)*30000).astype(np.int16)
    pause=np.zeros(round(SAMPLE_RATE*.25),dtype=np.int16)
    samples=np.concatenate((tone,pause,tone))
    baseline=run(samples,onset_ms=0).astype(np.int32)
    smooth=run(samples,onset_ms=40).astype(np.int32)
    restart=len(tone)+len(pause)
    assert np.max(np.abs(smooth[restart:restart+441]))<np.max(np.abs(baseline[restart:restart+441]))*.45
    assert np.array_equal(smooth[restart+2205:],baseline[restart+2205:])
    assert len(smooth)==len(samples)


def test_short_word_gaps_do_not_restart_the_volume_ramp():
    tone=(np.sin(np.arange(8820)*.1)*12000).astype(np.int16)
    pause=np.zeros(round(SAMPLE_RATE*.08),dtype=np.int16)
    samples=np.concatenate((tone,pause,tone))
    baseline=run(samples,onset_ms=0)
    smooth=run(samples,onset_ms=40)
    restart=len(tone)+len(pause)
    assert np.array_equal(smooth[restart:],baseline[restart:])


def test_leading_silence_does_not_consume_the_onset_ramp():
    tone=(np.sin(np.arange(8820)*.1)*12000).astype(np.int16)
    silence=np.zeros(SAMPLE_RATE//2,dtype=np.int16)
    samples=np.concatenate((silence,tone))
    baseline=run(samples,onset_ms=0).astype(np.int32)
    smooth=run(samples,onset_ms=40).astype(np.int32)
    assert np.max(np.abs(smooth[len(silence):len(silence)+441]))<np.max(np.abs(baseline[len(silence):len(silence)+441]))*.45
    assert np.any(smooth[len(silence):len(silence)+441])


def test_phrase_marker_handles_network_gap_without_fading_previous_delayed_tail():
    tone=(np.sin(np.arange(8820)*.1)*12000).astype(np.int16)
    def marked(onset_ms):
        processor=StreamDynamics(onset_ms=onset_ms);output=[]
        for _ in range(2):
            processor.start_segment()
            for start in range(0,len(tone),882):output.extend(processor.process(frame(tone[start:start+882])))
        output.extend(processor.finish())
        return np.frombuffer(b''.join(bytes(part.data) for part in output),dtype='<i2').astype(np.int32)
    baseline=marked(0);smooth=marked(40);restart=len(tone)
    assert np.array_equal(smooth[restart-80:restart],baseline[restart-80:restart])
    assert np.max(np.abs(smooth[restart:restart+441]))<np.max(np.abs(baseline[restart:restart+441]))*.45
    assert np.array_equal(smooth[restart+2205:],baseline[restart+2205:])


def test_pause_detection_is_independent_of_frame_boundaries():
    tone=(np.sin(np.arange(8820)*.1)*12000).astype(np.int16)
    samples=np.concatenate((tone,np.zeros(10000,dtype=np.int16),tone))
    assert np.array_equal(run(samples,(len(samples),)),run(samples,(1,43,317,882)))


def test_rolling_window_corrects_a_sustained_fade_without_flattening_it():
    time=np.arange(SAMPLE_RATE*4)/SAMPLE_RATE
    samples=(np.sin(time*2*np.pi*220)*10**((-20-12*time/4)/20)*32767).astype(np.int16)
    def drop(output):
        output=output.astype(np.float64)
        rms=lambda part:np.sqrt(np.mean(part**2))
        return 20*np.log10(rms(output[SAMPLE_RATE//2:SAMPLE_RATE])/rms(output[-SAMPLE_RATE//2:]))
    previous=drop(run(samples));rolling=drop(run(samples,leveler_enabled=True))
    assert previous>7
    assert 1<rolling<previous-4


def test_rolling_gain_is_bounded_smooth_and_does_not_raise_new_attacks():
    leveler=RollingLeveler(250,-24,6)
    samples=np.ones(SAMPLE_RATE)*.01
    output=leveler.process(samples)
    gain=20*np.log10(output/samples)
    assert np.max(gain)<=6.000001 and gain[-1]>5.9
    assert np.max(np.abs(np.diff(gain)))<=40/SAMPLE_RATE+1e-10
    assert np.max(gain[:int(SAMPLE_RATE*.115)])<=0
    leveler.start_segment()
    assert leveler.gain_db==0
    attack=leveler.process(np.ones(882)*.5)
    assert np.max(attack)<=.5


def test_rolling_window_does_not_amplify_settled_noise_or_silence():
    leveler=RollingLeveler(250,-24,6)
    silence=np.zeros(SAMPLE_RATE)
    assert not np.any(leveler.process(silence))
    noise=np.sin(np.arange(SAMPLE_RATE)*.1)*10**(-60/20)
    assert np.array_equal(leveler.process(noise),noise)
    leveler.process(np.ones(SAMPLE_RATE)*.01)
    released=leveler.process(silence)
    assert not np.any(released) and abs(leveler.gain_db)<1e-10
    assert leveler.voiced_samples==0


def test_rolling_profile_is_chunk_independent_and_keeps_sample_tails():
    samples=(np.sin(np.arange(25001)*.1)*25000).astype(np.int16)
    whole=run(samples,(len(samples),),leveler_enabled=True)
    parts=run(samples,(1,43,882,317),leveler_enabled=True)
    assert np.array_equal(whole,parts)
    assert len(whole)==len(samples) and whole[-1]!=0


def test_rolling_boost_remains_float_until_limiter_and_handles_a_hot_restart():
    leveler=RollingLeveler(250,-24,6)
    leveler.process(np.ones(SAMPLE_RATE)*.01)
    assert leveler.process(np.ones(1)*.9)[0]>1
    # A sudden loud sample following boosted speech reaches the limiter without
    # int16 wraparound; a sentence marker also removes the carried boost.
    samples=np.concatenate((np.ones(SAMPLE_RATE)*327,np.ones(882)*32767)).astype(np.int16)
    output=run(samples,leveler_enabled=True)
    assert np.min(output)>=0
    assert np.max(output)<=32768*10**(settings.audio_peak_dbfs/20)+1


def test_short_quiet_phoneme_gaps_hold_gain_then_long_silence_releases_it():
    leveler=RollingLeveler(250,-24,9)
    leveler.process(np.ones(SAMPLE_RATE)*.01)
    before=leveler.gain_db
    assert before>8.9
    leveler.process(np.zeros(round(SAMPLE_RATE*.06)))
    assert leveler.gain_db==pytest.approx(before)
    leveler.process(np.ones(round(SAMPLE_RATE*.06))*.01)
    assert leveler.gain_db>=before-.1
    leveler.process(np.zeros(SAMPLE_RATE))
    assert abs(leveler.gain_db)<1e-10 and leveler.voiced_samples==0


def test_fast_fade_recovery_remains_bounded_with_the_default_headroom():
    leveler=RollingLeveler(250,-24,settings.audio_leveler_max_boost_db)
    leveler.process(np.ones(SAMPLE_RATE)*.05)
    quiet=np.ones(SAMPLE_RATE)*.005
    out=leveler.process(quiet)
    gain=20*np.log10(out/quiet)
    assert 8.9<gain[-1]<=9.000001
    assert np.max(np.diff(gain))<=24/SAMPLE_RATE+1e-10
    assert np.min(np.diff(gain))>=-40/SAMPLE_RATE-1e-10
