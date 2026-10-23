"""Realtime transcription with local Silero turn ownership and 24 kHz PCM."""
import asyncio
import base64
from collections import deque
import json
import time
import logging
import websockets
from livekit import rtc
from livekit.agents import stt, vad, APIConnectOptions
from .config import settings

SESSION = {'type':'session.update','session':{'type':'transcription','audio':{'input':{
    'format':{'type':'audio/pcm','rate':24000},
    'transcription':{'model':'gpt-live-transcribe','languages':['ar','en'],
                     'keywords':['Aljawab','الجواب','AliveArchive','ASR'],'delay':'low'},
    'turn_detection':None}}}}
log=logging.getLogger('aljawab')

class LiveTranscription(stt.STT):
    def __init__(self, detector, on_speech=None, on_activity=None):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True,interim_results=True,offline_recognize=False))
        self.detector=detector; self.on_speech=on_speech
        self.on_activity=on_activity
    @property
    def model(self): return settings.stt_model
    @property
    def provider(self): return 'openai'
    async def _recognize_impl(self,*args,**kwargs): raise NotImplementedError('Live transcription only')
    def stream(self,*,language=None,conn_options=APIConnectOptions(max_retry=0)):
        return LiveStream(stt=self,conn_options=conn_options,sample_rate=24000)

class LiveStream(stt.SpeechStream):
    async def _run(self):
        async with websockets.connect('wss://api.openai.com/v1/realtime?intent=transcription',
            additional_headers={'Authorization':'Bearer '+settings.openai_api_key},max_size=2**22,
            open_timeout=10,ping_interval=20) as ws:
            await ws.send(json.dumps(SESSION))
            # Fail immediately on a refused/unsupported session, before accepting audio.
            while True:
                event=json.loads(await asyncio.wait_for(ws.recv(),10))
                if event['type']=='error': raise RuntimeError('Transcription session refused')
                if event['type'] in ('session.updated','transcription_session.updated'): break
            detector=self._stt.detector.stream()
            speaking=False; prefix=deque(maxlen=8); epoch=0
            commits=deque(); items={}; partials={}; completed=set()
            send_lock=asyncio.Lock()
            async def append(frame):
                await ws.send(json.dumps({'type':'input_audio_buffer.append','audio':base64.b64encode(bytes(frame.data)).decode()}))
            async def feed():
                async for frame in self._input_ch:
                    if not isinstance(frame,rtc.AudioFrame): continue
                    detector.push_frame(frame)
                    async with send_lock:
                        if speaking: await append(frame)
                        else: prefix.append(frame)
                detector.end_input()
            async def detect():
                nonlocal speaking,epoch
                async for event in detector:
                    if self._stt.on_activity and (speaking or event.type==vad.VADEventType.START_OF_SPEECH):
                        self._stt.on_activity()
                    if event.type==vad.VADEventType.START_OF_SPEECH:
                        async with send_lock:
                            epoch+=1; speaking=True
                            if self._stt.on_speech: self._stt.on_speech()
                            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
                            for frame in prefix: await append(frame)
                            prefix.clear()
                    elif event.type==vad.VADEventType.END_OF_SPEECH:
                        async with send_lock:
                            speaking=False
                            self._stt.last_speech_end=time.monotonic()-settings.end_of_turn_silence_ms/1000
                            commits.append((epoch,time.time()-settings.end_of_turn_silence_ms/1000))
                            log.info('speech_end generation=%d',epoch)
                            await ws.send(json.dumps({'type':'input_audio_buffer.commit'}))
            async def receive():
                async for raw in ws:
                    e=json.loads(raw); kind=e['type']; item=e.get('item_id','')
                    if kind=='error': raise RuntimeError('Transcription stream failed')
                    if kind=='input_audio_buffer.committed':
                        if commits: items[item]=commits.popleft()
                    elif kind=='conversation.item.input_audio_transcription.delta':
                        # delta appends; explicit transcript is a replacement revision.
                        partials[item]=e.get('transcript',partials.get(item,'')+e.get('delta',''))
                        if item in items and items[item][0]!=epoch: continue
                        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=item,alternatives=[stt.SpeechData(language='',text=partials[item])]))
                    elif kind=='conversation.item.input_audio_transcription.completed':
                        if item in completed: continue
                        completed.add(item); record=items.pop(item,None); partials.pop(item,None)
                        if record is None or record[0]!=epoch: continue
                        text=e.get('transcript','').strip()
                        log.info('final_transcript_after_speech_end_ms=%.1f',(time.time()-record[1])*1000)
                        if text:
                            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                request_id=item,alternatives=[stt.SpeechData(language='',text=text)]))
                        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH,
                                                                 speech_end_time=record[1]))
                    elif kind=='conversation.item.input_audio_transcription.failed':
                        raise RuntimeError('Transcription failed')
            tasks=[asyncio.create_task(fn()) for fn in (feed,detect,receive)]
            try:
                done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                for task in done: task.result()
            finally:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                await detector.aclose()
