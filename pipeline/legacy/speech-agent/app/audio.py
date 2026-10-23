"""Low-latency, per-reply dynamics using in-process FFmpeg audio filters."""
from fractions import Fraction
from collections import deque

import av
from av.error import BlockingIOError, EOFError
from livekit import rtc
import numpy as np

from .config import settings

SAMPLE_RATE=44100
LOOKAHEAD_MS=1
SEGMENT_START=object()


class RollingPower:
    def __init__(self,length):
        self.history=np.zeros(length,dtype=np.float64)
        self.cursor=0;self.filled=0;self.total=0.0

    def clear(self):
        self.history.fill(0);self.cursor=0;self.filled=0;self.total=0.0

    def push(self,samples):
        offset=0
        while offset<len(samples):
            count=min(len(samples)-offset,len(self.history)-self.cursor)
            values=np.square(samples[offset:offset+count])
            old=self.history[self.cursor:self.cursor+count]
            self.total+=float(values.sum()-old.sum());old[:]=values
            self.cursor=(self.cursor+count)%len(self.history)
            self.filled=min(len(self.history),self.filled+count);offset+=count

    @property
    def dbfs(self):
        power=max(self.total/max(1,self.filled),1e-12)
        return float(10*np.log10(power))


class RollingLeveler:
    """Past-only RMS level tracking; no audio buffering or spectral processing."""
    def __init__(self,window_ms,target_dbfs,max_boost_db):
        self.window=RollingPower(round(SAMPLE_RATE*window_ms/1000))
        self.activity=RollingPower(round(SAMPLE_RATE*.02))
        self.target_dbfs=target_dbfs;self.max_boost_db=max_boost_db
        self.hop=round(SAMPLE_RATE*.005)
        self.gain_db=0.0;self.remaining=0;self.step=0.0
        self.quiet_samples=0;self.voiced_samples=0

    def start_segment(self):
        # Do not carry an ending's positive gain into a new sentence's attack.
        self.window.clear();self.activity.clear()
        self.gain_db=0.0;self.remaining=0;self.step=0.0
        self.quiet_samples=0;self.voiced_samples=0

    def _control(self):
        if self.activity.dbfs<-48:
            self.quiet_samples+=self.hop
            # Brief unvoiced consonants/word gaps must not shed the gain that
            # the following quiet vowel needs. Longer pauses still release it.
            target=self.gain_db if self.quiet_samples<SAMPLE_RATE*.08 else 0.0
            if self.quiet_samples>=SAMPLE_RATE*.18:
                self.window.clear();self.voiced_samples=0
        else:
            self.quiet_samples=0;self.voiced_samples+=self.hop
            error=self.target_dbfs-self.window.dbfs
            # Leave small inflections alone; correct larger, sustained fades.
            target=np.sign(error)*max(0,abs(error)-2)*.8
            target=float(np.clip(target,-6,self.max_boost_db))
            if self.voiced_samples<SAMPLE_RATE*.12:target=min(0,target)
        seconds=self.hop/SAMPLE_RATE
        change=float(np.clip(target-self.gain_db,-40*seconds,24*seconds))
        self.step=change/self.hop;self.remaining=self.hop

    def process(self,samples):
        output=np.empty(len(samples),dtype=np.float64);offset=0
        while offset<len(samples):
            if self.remaining==0:self._control()
            count=min(len(samples)-offset,self.remaining)
            block=samples[offset:offset+count]
            gains=self.gain_db+self.step*np.arange(1,count+1)
            output[offset:offset+count]=block*np.power(10,gains/20)
            self.window.push(block);self.activity.push(block)
            self.gain_db+=self.step*count;self.remaining-=count;offset+=count
        return output


class OnsetSmoother:
    def __init__(self,duration_ms,attenuation_db):
        length=max(1,round(SAMPLE_RATE*duration_ms/1000))
        phase=np.linspace(0,np.pi,length)
        floor=10**(attenuation_db/20)
        self.curve=floor+(1-floor)*(.5-.5*np.cos(phase))
        self.position=0;self.starts=deque([0]);self.armed=True;self.ramp_index=None
        self.silence=0;self.hold=round(SAMPLE_RATE*.18)
        self.threshold=32768*10**(-50/20)

    def schedule(self,sample_index):
        self.starts.append(sample_index)

    def _block(self,samples):
        gain=np.ones(len(samples),dtype=np.float64)
        if self.ramp_index is not None:
            count=min(len(samples),len(self.curve)-self.ramp_index)
            gain[:count]=self.curve[self.ramp_index:self.ramp_index+count]
            self.ramp_index+=count
            if self.ramp_index==len(self.curve):self.ramp_index=None
        active=np.flatnonzero(np.abs(samples.astype(np.int32))>=self.threshold)
        starts=[]
        if len(active):
            if self.armed or self.silence+active[0]>=self.hold:starts.append(int(active[0]))
            starts.extend(int(active[index+1]) for index in np.flatnonzero(np.diff(active)-1>=self.hold))
            self.silence=len(samples)-1-int(active[-1]);self.armed=self.silence>=self.hold
        else:
            self.silence+=len(samples)
            if self.silence>=self.hold:self.armed=True;self.ramp_index=None
        for start in starts:
            count=min(len(samples)-start,len(self.curve))
            gain[start:start+count]=self.curve[:count]
            self.ramp_index=count if count<len(self.curve) else None
        return np.rint(samples*gain).astype('<i2')

    def process(self,samples):
        result=[];offset=0
        while offset<len(samples):
            while self.starts and self.starts[0]<=self.position:
                self.starts.popleft();self.armed=True;self.ramp_index=None
            count=len(samples)-offset
            if self.starts:count=min(count,self.starts[0]-self.position)
            result.append(self._block(samples[offset:offset+count]))
            offset+=count;self.position+=count
        return np.concatenate(result) if result else samples


class StreamDynamics:
    def __init__(self,peak_dbfs=None,quiet_boost_db=None,onset_ms=None,leveler_enabled=None):
        peak_dbfs=settings.audio_peak_dbfs if peak_dbfs is None else peak_dbfs
        quiet_boost_db=settings.audio_quiet_boost_db if quiet_boost_db is None else quiet_boost_db
        onset_ms=settings.audio_onset_ms if onset_ms is None else onset_ms
        leveler_enabled=settings.audio_leveler_enabled if leveler_enabled is None else leveler_enabled
        self.leveler=RollingLeveler(settings.audio_leveler_window_ms,settings.audio_leveler_target_dbfs,
                                   settings.audio_leveler_max_boost_db) if leveler_enabled else None
        self.onset=OnsetSmoother(onset_ms,settings.audio_onset_attenuation_db) if onset_ms>0 else None
        self.ceiling=10**(peak_dbfs/20)
        self.samples=0;self.closed=False
        self.graph=av.filter.Graph()
        self.source=self.graph.add_abuffer(sample_rate=SAMPLE_RATE,format='dbl' if self.leveler else 's16',layout='mono',
                                           time_base=Fraction(1,SAMPLE_RATE))
        compressor=self.graph.add('acompressor',
            'threshold=0.125892541:ratio=3:attack=1:release=100:makeup=1.258925412:knee=2.828427:detection=peak')
        limiter=self.graph.add('alimiter',
            f'limit={self.ceiling:.12f}:attack={LOOKAHEAD_MS}:release=60:level=false:latency=true')
        format_filter=self.graph.add('aformat',f'sample_fmts=s16:sample_rates={SAMPLE_RATE}:channel_layouts=mono')
        self.sink=self.graph.add('abuffersink')
        if self.leveler:self.source.link_to(compressor)
        elif quiet_boost_db>0:
            # A bounded gain curve lifts quiet speech without chasing the noise
            # floor or flattening normal expression. No added analysis delay.
            points=(f'-90/-90|-50/-50|-45/-45|-36/{-36+quiet_boost_db:.8f}'
                    f'|-24/{-24+quiet_boost_db:.8f}|-18/-18|0/0')
            quiet=self.graph.add('compand',
                f'attacks=0.02:decays=0.18:points={points}:soft-knee=3:gain=0:volume=-90:delay=0')
            self.source.link_to(quiet);quiet.link_to(compressor)
        else:self.source.link_to(compressor)
        compressor.link_to(limiter)
        limiter.link_to(format_filter);format_filter.link_to(self.sink)
        self.graph.configure()

    def _pull(self):
        result=[]
        while True:
            try:frame=self.sink.pull()
            except (BlockingIOError,EOFError):return result
            # ndarray excludes plane alignment padding, including in the tail.
            samples=frame.to_ndarray().reshape(-1).astype('<i2',copy=False)
            if self.onset:samples=self.onset.process(samples)
            pcm=samples.tobytes()
            if frame.samples:result.append(rtc.AudioFrame(pcm,SAMPLE_RATE,1,frame.samples))

    def process(self,frame):
        if self.closed:raise RuntimeError('Audio processor is finished')
        if frame.sample_rate!=SAMPLE_RATE or frame.num_channels!=1:
            raise ValueError('Dynamics requires mono 44100 Hz PCM')
        data=np.frombuffer(frame.data,dtype='<i2').reshape(1,-1)
        if self.leveler:data=self.leveler.process(data[0].astype(np.float64)/32768).reshape(1,-1)
        audio=av.AudioFrame.from_ndarray(data,format='dbl' if self.leveler else 's16',layout='mono')
        audio.sample_rate=SAMPLE_RATE;audio.time_base=Fraction(1,SAMPLE_RATE);audio.pts=self.samples
        self.samples+=frame.samples_per_channel
        self.source.push(audio)
        return self._pull()

    def start_segment(self):
        if self.leveler:self.leveler.start_segment()
        if self.onset:self.onset.schedule(self.samples)

    def finish(self):
        if self.closed:return []
        self.closed=True;self.source.push(None)
        return self._pull()


async def controlled_audio(source):
    processor=None
    try:
        async for frame in source:
            if frame is SEGMENT_START:
                if processor:processor.start_segment()
                continue
            if not settings.audio_dynamics_enabled:
                yield frame
                continue
            if processor is None:processor=StreamDynamics()
            for output in processor.process(frame):yield output
        if processor:
            for output in processor.finish():yield output
    finally:
        if hasattr(source,'aclose'):await source.aclose()
