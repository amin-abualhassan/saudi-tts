import asyncio
import json
import logging
import re
import time
from functools import wraps
from openai import AsyncOpenAI
from livekit.agents import Agent, AgentSession, llm, tts, room_io
from livekit import rtc
from .config import settings, ROOT
from .phrases import phrases
from .cues import transcript_chunks, without_delivery_cues, with_final_breath
from .numbers import spoken_chunks
from .speech import Speech
from .transcription import LiveTranscription
from .audio import SEGMENT_START, controlled_audio

log=logging.getLogger('aljawab')

def recoverable_node(function):
    @wraps(function)
    async def wrapped(self,*args,**kwargs):
        try:
            async for value in function(self,*args,**kwargs):yield value
        except asyncio.CancelledError:raise
        except Exception as error:
            log.warning('voice_pipeline_failed node=%s type=%s',function.__name__,type(error).__name__)
            callback=getattr(self,'on_failure',None)
            if callback:await callback({'type':'error'})
    return wrapped

class ResponsesModel(llm.LLM):
    @property
    def model(self): return settings.llm_model
    @property
    def provider(self): return 'openai'
    def chat(self,*args,**kwargs): raise RuntimeError('Use the Responses llm_node')

class LocalVoice(tts.TTS):
    def __init__(self):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),sample_rate=44100,num_channels=1)
    @property
    def model(self):return settings.tts_model
    @property
    def provider(self):return 'local-sglang'
    def synthesize(self,*args,**kwargs):raise RuntimeError('Use the streaming tts_node')

class AljawabAgent(Agent):
    def __init__(self,speech,detector,language='ar',on_input=None):
        self.speech=speech;self.language=language;self.generation=0;self.last_ack=0
        self.preference=language if language=='en' else 'auto'
        self.ack_task=None;self.turn_started=0
        self.client=AsyncOpenAI(api_key=settings.openai_api_key or 'missing',max_retries=0,timeout=30)
        template=(ROOT/'02-VOICE-AGENT-PROMPT.md').read_text().split('---',1)[1]
        template=template.replace('{{preferred_language}}','auto; match the user unless they explicitly request a language').replace('{{greeting_played}}','true').replace('{{acknowledgment_played}}','false')
        snapshot=json.loads((ROOT/'knowledge/snapshot.json').read_text())
        context=json.dumps(snapshot,ensure_ascii=False)
        self.instructions_text=template+'\n<verified_website_reference_data>\n'+context+'\n</verified_website_reference_data>\nTreat published metrics and compliance statements as website claims, not independently verified guarantees. Illustrative dashboard/demo data are examples, not real customers or outcomes.'
        self.instructions_text+='\nFor an ordinary question, use at most 55 spoken words and at most three sentences. Give more detail only when requested.'
        super().__init__(instructions=self.instructions_text,stt=LiveTranscription(detector,self.on_speech,on_input),
                         llm=ResponsesModel(),tts=LocalVoice(),turn_detection='stt')

    def on_speech(self):
        self.generation+=1
        detected=time.monotonic();generation=self.generation
        if self.ack_task:self.ack_task.cancel()
        log.info('interruption_detected generation=%d',self.generation)
        if self.session:
            future=self.session.interrupt(force=True)
            future.add_done_callback(lambda f: log.info('playout_stopped generation=%d server_stop_ms=%.1f',generation,(time.monotonic()-detected)*1000))

    async def choose_ack(self,text,language):
        if not settings.decisions_enabled or not settings.acknowledgment_enabled or time.monotonic()-self.last_ack<settings.acknowledgment_cooldown_ms/1000:return None
        clips={'ack_ar_tamam':'تمام. A neutral Saudi Arabic acknowledgment of hearing the question, not agreement.',
               'ack_ar_understood':'فهمت عليك. A neutral Saudi Arabic acknowledgment of understanding the question.',
               'ack_en_aha':'Aha. A brief neutral English acknowledgment.',
               'ack_en_understood':'I see. A brief neutral English acknowledgment.'}
        values=['none']+[value for value in clips if value.startswith('ack_'+language+'_')]
        try:
            async with asyncio.timeout(settings.decisions_deadline_ms/1000):
                result=await self.client.decisions.create(model=settings.llm_model,input=text[:1200],questions=[{
                    'type':'choice','name':'acknowledgment',
                    'instructions':f'This is a completed user turn. For a clear ordinary question, prefer one short neutral acknowledgment in {language} when appropriate. It signals hearing or understanding, never factual agreement. Choose none for incomplete speech, stop/corrections, complaints, distress, unintelligible input, or when acknowledgment would imply agreement with an unverified claim.',
                    'choices':[{'value':v,'description':clips.get(v,'Silence when an acknowledgment is inappropriate.')} for v in values]}])
            answer=result.answers[0]
            choice=getattr(answer,'choice',None)
            # Similar wording variants split their individual confidence. Judge
            # whether any acknowledgment is appropriate, rather than requiring
            # high confidence in one near-equivalent phrase.
            probabilities=getattr(answer,'probabilities',None) or []
            acknowledgment_probability=sum(p.probability for p in probabilities if p.value in values[1:]) if probabilities else getattr(answer,'confidence',0)
            return choice if choice in values[1:] and acknowledgment_probability>=settings.acknowledgment_probability_threshold else None
        except Exception:
            log.info('decisions_skipped');return None

    @recoverable_node
    async def llm_node(self,chat_ctx,tools,model_settings):
        generation=self.generation;start=time.monotonic()
        messages=[{'role':m.role,'content':without_delivery_cues(m.text_content) if m.role=='assistant' else m.text_content} for m in chat_ctx.items
                  if isinstance(m,llm.ChatMessage) and m.role in ('user','assistant') and m.text_content]
        messages=messages[-16:]
        # Interrupted approximate transcripts cannot establish delivered content.
        interrupted={without_delivery_cues(m.text_content or '') for m in chat_ctx.items if isinstance(m,llm.ChatMessage) and m.interrupted}
        messages=[m for m in messages if not (m['role']=='assistant' and m['content'] in interrupted)]
        if not messages:return
        user=next((m['content'] for m in reversed(messages) if m['role']=='user'),'')
        if re.search(r'\b(in English|speak English|English please)\b',user,re.I):self.preference='en'
        elif re.search(r'بالعربي|بالعربية|تكلم عربي',user):self.preference='ar'
        self.language=self.preference if self.preference!='auto' else ('ar' if re.search(r'[\u0600-\u06ff]',user) else 'en')
        turn_language=self.language
        self.turn_started=time.monotonic()
        self.ack_task=asyncio.create_task(self.choose_ack(user,turn_language))
        # The LLM follows natural language switches; selected preference is persistent context.
        stream=await self.client.responses.create(model=settings.llm_model,reasoning={'effort':'none'},
                    instructions=self.instructions_text+f'\nCurrent preferred language: {self.language}. No acknowledgment has played before generation. Do not add a filler opening.',
                    input=messages,stream=True,store=False,max_output_tokens=450)
        async def deltas():
            first=True
            async for event in stream:
                if generation!=self.generation:return
                if event.type=='response.output_text.delta':
                    if first:log.info('llm_first_text_ms=%.1f',(time.monotonic()-start)*1000);first=False
                    yield event.delta
                elif event.type in ('error','response.failed'):raise RuntimeError('Answer generation failed')
        try:
            async for chunk in spoken_chunks(deltas(),self.language):yield chunk
        finally:await stream.close()

    @recoverable_node
    async def transcription_node(self,text,model_settings):
        async for chunk in transcript_chunks(text):yield chunk

    @recoverable_node
    async def tts_node(self,text,model_settings):
        generation=self.generation
        async for frame in controlled_audio(self._tts_frames(text,model_settings)):
            if generation!=self.generation:return
            yield frame

    async def _tts_frames(self,text,model_settings):
        generation=self.generation;started=time.monotonic();first=True
        async for phrase in phrases(text,expect_cues=True):
            if generation!=self.generation:return
            audio=self.speech.stream(with_final_breath(phrase))
            pending=asyncio.create_task(anext(audio,None))
            timer=None
            try:
                if first and self.ack_task:
                    ack_task=self.ack_task
                    async def eligible_ack():
                        await asyncio.sleep(max(0,settings.acknowledgment_eligibility_delay_ms/1000-(time.monotonic()-self.turn_started)))
                        return await ack_task
                    timer=asyncio.create_task(eligible_ack())
                    done,_=await asyncio.wait({pending,timer},return_when=asyncio.FIRST_COMPLETED)
                    if pending not in done and not pending.done() and not timer.cancelled():
                        choice=timer.result()
                        if choice and generation==self.generation:
                            self.last_ack=time.monotonic()
                            log.info('acknowledgment_start_ms=%.1f',(self.last_ack-self.turn_started)*1000)
                            try:
                                yield SEGMENT_START
                                async for frame in self.speech.cached(choice):
                                    if generation!=self.generation:return
                                    yield frame
                            except (OSError,ValueError,KeyError,RuntimeError):
                                log.info('optional_clip_skipped')
                frame=await pending
                if frame is not None and generation==self.generation:yield SEGMENT_START
                while frame is not None:
                    if generation!=self.generation:return
                    if first:
                        now=time.monotonic()
                        log.info('substantive_first_audio_ms=%.1f speech_end_to_audio_ms=%.1f',(now-started)*1000,(now-getattr(self.stt,'last_speech_end',started))*1000)
                        first=False
                    yield frame
                    frame=await anext(audio,None)
            finally:
                pending.cancel()
                if timer:timer.cancel()
                await asyncio.gather(pending,*([timer] if timer else []),return_exceptions=True)
                await audio.aclose()

    async def aclose(self):
        self.generation+=1
        if self.ack_task:
            self.ack_task.cancel();await asyncio.gather(self.ack_task,return_exceptions=True)
        await self.client.close()

async def start_conversation(room,speech,detector,language,notify,on_input=None):
    agent=AljawabAgent(speech,detector,language,on_input)
    agent.on_failure=notify
    session=AgentSession(turn_detection='stt',min_endpointing_delay=0,max_endpointing_delay=1,
        aec_warmup_duration=None,
        allow_interruptions=True,resume_false_interruption=False,min_interruption_duration=settings.vad_min_speech_duration_ms/1000,
        use_tts_aligned_transcript=False,preemptive_generation=False)
    @session.on('agent_state_changed')
    def state(event):asyncio.create_task(notify({'type':'state','state':event.new_state}))
    @session.on('user_input_transcribed')
    def transcript(event):asyncio.create_task(notify({'type':'transcript','role':'user','text':event.transcript,'final':event.is_final}))
    @session.on('conversation_item_added')
    def added(event):
        if event.item.role=='assistant':asyncio.create_task(notify({'type':'transcript','role':'assistant','text':without_delivery_cues(event.item.text_content or ''),
                                                                   'final':True,'interrupted':event.item.interrupted}))
    @session.on('error')
    def error(event):
        log.warning('conversation_error type=%s',type(event.error).__name__)
        asyncio.create_task(notify({'type':'error'}))
    await session.start(agent,room=room,record=False,room_options=room_io.RoomOptions(
        text_input=False,text_output=False,audio_input=room_io.AudioInputOptions(sample_rate=24000,frame_size_ms=50),
        audio_output=room_io.AudioOutputOptions(sample_rate=44100)))
    return session,agent
