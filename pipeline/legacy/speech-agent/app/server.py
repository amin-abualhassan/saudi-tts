import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import datetime
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit
from typing import Literal
import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from livekit import api, rtc
from livekit.plugins import silero
from .config import ROOT, settings
from .speech import Speech, check_model
from .agent import start_conversation
from .cues import without_delivery_cues
from .audio import controlled_audio

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
for name in ('livekit','openai','httpx','httpx2','httpcore','websockets'):
    logging.getLogger(name).setLevel(logging.CRITICAL)  # SDK diagnostics may include spoken content.
log=logging.getLogger('aljawab')

@dataclass
class Call:
    id:str
    secret:str
    language:str
    last_seen:float=field(default_factory=time.monotonic)
    last_input:float=field(default_factory=time.monotonic)
    lifecycle_lock:asyncio.Lock=field(default_factory=asyncio.Lock)
    sockets:set=field(default_factory=set)
    room:rtc.Room|None=None
    session:object=None
    agent:object=None
    started:bool=False
    messages:deque=field(default_factory=lambda:deque(maxlen=50))
    connection_reports:set=field(default_factory=set)

class Runtime:
    def __init__(self):
        self.calls={};self.speech=None;self.detector=None
        self.rates=defaultdict(deque)
runtime=Runtime()

def token(room,identity,agent=False):
    grants=api.VideoGrants(room_join=True,room=room,can_publish=True,can_subscribe=True,
                           can_publish_data=agent,can_update_own_metadata=False)
    return api.AccessToken(settings.livekit_api_key,settings.livekit_api_secret).with_identity(identity).with_ttl(
        datetime.timedelta(minutes=5)).with_grants(grants).to_jwt()

def origin_ok(request):
    origin=request.headers.get('origin')
    return not origin or urlsplit(origin).netloc==request.headers.get('host')

def current(request):
    secret=request.headers.get('x-session-token')
    if secret is None:
        protocols=request.headers.get('sec-websocket-protocol','').split(',')
        secret=next((p.strip()[8:] for p in protocols if p.strip().startswith('session.')),None)
    if secret is None:secret=request.cookies.get('aljawab_session','')
    call=runtime.calls.get(secret)
    if not call:raise HTTPException(401,'Session ended')
    if not origin_ok(request):raise HTTPException(403,'Please reopen the page')
    return call

async def notify(call,message):
    if message.get('type')=='transcript' and message.get('role')=='assistant':
        message={**message,'text':without_delivery_cues(message.get('text',''))}
    if runtime.calls.get(call.secret) is call:call.messages.append(message)

def input_activity(call):
    if runtime.calls.get(call.secret) is call:call.last_input=time.monotonic()

async def close_call(call,reason='ended'):
    async with call.lifecycle_lock:
        if runtime.calls.get(call.secret) is not call:return
        runtime.calls.pop(call.secret)
        for ws in list(call.sockets):
            try:await asyncio.wait_for(ws.send_json({'type':'ended','reason':reason}),1)
            except Exception:pass
        try:
            if call.agent:call.agent.on_speech()
        except Exception:log.warning('interruption_cleanup_failed')
        for resource,method in ((call.session,'aclose'),(call.agent,'aclose'),(call.room,'disconnect')):
            if resource:
                try:await asyncio.wait_for(getattr(resource,method)(),5)
                except Exception:log.warning('conversation_cleanup_failed resource=%s',method)
        try:
            async with api.LiveKitAPI(settings.livekit_url,settings.livekit_api_key,settings.livekit_api_secret) as lk:
                await lk.room.delete_room(api.DeleteRoomRequest(room=call.id))
        except Exception:log.info('room_cleanup_retry_not_needed')
        call.messages.clear();call.sockets.clear();call.session=None;call.agent=None;call.room=None

async def expire_calls():
    now=time.monotonic()
    expired=[call for call in list(runtime.calls.values())
             if now-call.last_input>=settings.session_idle_timeout_seconds or now-call.last_seen>=60]
    await asyncio.gather(*(close_call(call,'idle' if now-call.last_input>=settings.session_idle_timeout_seconds else 'disconnected')
                           for call in expired))

@asynccontextmanager
async def lifespan(app):
    runtime.speech=Speech()
    runtime.detector=silero.VAD.load(min_silence_duration=settings.end_of_turn_silence_ms/1000,
                                    min_speech_duration=settings.vad_min_speech_duration_ms/1000,
                                    activation_threshold=settings.vad_activation_threshold,force_cpu=True)
    async def maintenance():
        while True:
            await asyncio.sleep(1)
            try:
                await expire_calls()
                await runtime.speech.cleanup()
                for key in list(runtime.rates):
                    if not runtime.rates[key] or time.monotonic()-runtime.rates[key][-1]>60:del runtime.rates[key]
            except Exception:log.warning('maintenance_error')
    task=asyncio.create_task(maintenance())
    try:yield
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        await asyncio.gather(*(close_call(call) for call in list(runtime.calls.values())))
        await runtime.speech.cleanup();await runtime.speech.client.aclose()

app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)

@app.exception_handler(Exception)
async def unavailable(request,exception):
    log.warning('request_failed type=%s',type(exception).__name__)
    return Response('{"detail":"Please try again shortly"}',status_code=503,media_type='application/json')

@app.middleware('http')
async def headers(request,call_next):
    if request.method=='POST' and not origin_ok(request):return Response(status_code=403)
    response=await call_next(request)
    response.headers.update({'Cache-Control':'no-store','Referrer-Policy':'no-referrer',
        'X-Content-Type-Options':'nosniff','Permissions-Policy':'microphone=(self)',
        'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self' wss:; media-src 'self' blob:; worker-src 'self' blob:; frame-ancestors 'none'"})
    return response

@app.get('/health')
async def health():return {'ok':True}

@app.get('/api/readiness')
async def readiness():
    ready=bool(settings.openai_api_key and (ROOT/'knowledge/snapshot.json').exists() and (ROOT/'assets/voice/manifest.json').exists())
    return {'ready':ready}

@app.post('/api/session')
async def create(request:Request,response:Response):
    # Forwarded client address only from Cloudflare's header; direct origin is loopback.
    address=request.headers.get('cf-connecting-ip',request.client.host)
    queue=runtime.rates[address];now=time.monotonic()
    while queue and now-queue[0]>60:queue.popleft()
    if len(queue)>=settings.session_start_limit_per_minute:
        log.info('session_admission_denied reason=rate_limit')
        raise HTTPException(429,'Please wait a moment and try again')
    queue.append(now)
    if not settings.openai_api_key:raise HTTPException(503,'Please try again shortly')
    data=await request.json()
    language=data.get('language','ar')
    if language not in ('ar','en'):raise HTTPException(400,'Please choose a language')
    await check_model()
    call=Call('aljawab-'+secrets.token_hex(12),secrets.token_urlsafe(32),language)
    runtime.calls[call.secret]=call
    try:
        async with call.lifecycle_lock:
            call.room=rtc.Room()
            await call.room.connect(settings.livekit_url,token(call.id,'assistant',True))
            call.session,call.agent=await start_conversation(call.room,runtime.speech,runtime.detector,language,
                                                           lambda message:notify(call,message),
                                                           on_input=lambda:input_activity(call))
    except Exception as e:
        log.warning('session_start_failed type=%s',type(e).__name__)
        await close_call(call)
        raise HTTPException(503,'Please try again shortly')
    response.set_cookie('aljawab_session',call.secret,httponly=True,secure=True,samesite='strict',max_age=3600)
    log.info('session_admitted active=%d',len(runtime.calls))
    return {'token':token(call.id,'visitor'),'room':call.id,'session_token':call.secret}

class ConnectionReport(BaseModel):
    stage:Literal['media','publish','ready']
    reason:Literal['timeout','network','permission','other']
    platform:Literal['ios','android','other']

@app.post('/api/connection-report')
async def connection_report(request:Request,report:ConnectionReport):
    call=current(request)
    if report.stage not in call.connection_reports:
        call.connection_reports.add(report.stage)
        # Coarse, enumerated diagnostics only; never error strings, addresses,
        # browser fingerprints, session credentials or spoken content.
        log.info('browser_connection_failed stage=%s reason=%s platform=%s',
                 report.stage,report.reason,report.platform)
    return {'ok':True}

@app.post('/api/ready')
async def ready(request:Request):
    call=current(request);call.last_seen=time.monotonic()
    if not call.started:
        call.started=True
        name='greeting_'+call.language
        from scripts.generate_clips import CLIPS
        call.session.say(CLIPS[name],audio=controlled_audio(runtime.speech.cached(name)),allow_interruptions=True,add_to_chat_ctx=False)
    return {'ok':True}

@app.post('/api/heartbeat')
async def heartbeat(request:Request):
    call=current(request);call.last_seen=time.monotonic();return {'ok':True}

@app.post('/api/language')
async def language(request:Request):
    call=current(request);data=await request.json()
    if data.get('language') not in ('ar','en'):raise HTTPException(400)
    call.language=data['language'];call.agent.language=call.language
    call.agent.preference=call.language
    return {'ok':True}

@app.post('/api/end')
async def end(request:Request,response:Response):
    call=current(request);await close_call(call)
    response.delete_cookie('aljawab_session',secure=True,httponly=True,samesite='strict')
    return {'ok':True}

@app.websocket('/api/events')
async def events(ws:WebSocket):
    try:call=current(ws)
    except HTTPException:await ws.close(code=1008);return
    protocols=[p.strip() for p in ws.headers.get('sec-websocket-protocol','').split(',')]
    await ws.accept(subprotocol='aljawab' if 'aljawab' in protocols else None)
    call.sockets.add(ws)
    try:
        while runtime.calls.get(call.secret) is call:
            while call.messages:await ws.send_json(call.messages.popleft())
            await asyncio.sleep(.1)
    except (WebSocketDisconnect,RuntimeError):pass
    finally:
        call.sockets.discard(ws)
        try:await ws.close()
        except (RuntimeError,WebSocketDisconnect):pass

@app.get('/rtc/validate')
async def validate(request:Request):
    async with aiohttp.ClientSession() as client:
        async with client.get('http://127.0.0.1:7880/rtc/validate',params=list(request.query_params.multi_items())) as r:
            return Response(await r.read(),status_code=r.status,media_type=r.headers.get('content-type'))

@app.websocket('/rtc/v1')
@app.websocket('/rtc')
async def signaling(ws:WebSocket):
    if not origin_ok(ws):await ws.close(code=1008);return
    await ws.accept()
    try:
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect('http://127.0.0.1:7880'+ws.url.path,params=list(ws.query_params.multi_items()),autoping=True,max_msg_size=4*1024*1024) as upstream:
                async def browser():
                    while True:
                        message=await ws.receive()
                        if message['type']=='websocket.disconnect':return
                        if message.get('bytes') is not None:await upstream.send_bytes(message['bytes'])
                        elif message.get('text') is not None:await upstream.send_str(message['text'])
                async def server():
                    async for message in upstream:
                        if message.type==aiohttp.WSMsgType.BINARY:await ws.send_bytes(message.data)
                        elif message.type==aiohttp.WSMsgType.TEXT:await ws.send_text(message.data)
                        elif message.type in (aiohttp.WSMsgType.CLOSED,aiohttp.WSMsgType.ERROR):return
                tasks=[asyncio.create_task(browser()),asyncio.create_task(server())]
                try:await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
    except (WebSocketDisconnect,RuntimeError,aiohttp.ClientError):pass
    finally:
        try:await ws.close()
        except (RuntimeError,WebSocketDisconnect):pass

app.mount('/',StaticFiles(directory=ROOT/'web/dist',html=True),name='website')
