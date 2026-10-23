"""Port-3000 API compatible with the app; forwards only the required model.

This gateway does not install/start/stop SGLang or select another model.
"""
import asyncio,base64,hashlib,io,json,secrets,shutil,wave
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
import numpy as np
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,StreamingResponse
from pydantic_settings import BaseSettings,SettingsConfigDict
from .model_files import ROOT,verify_model

MODEL='attention_mlp_r16_early'
CHECKPOINT='9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf'
class Configuration(BaseSettings):
    model_config=SettingsConfigDict(env_file=ROOT/'.env',extra='ignore')
    sglang_url:str='http://127.0.0.1:3002'
    sglang_model_path:str=''
    tts_derivative_manifest:str=''
    tts_engine_model_id:str=''
    tts_engine_reference_map:str='{}'
    tts_verify_model_hashes:bool=True
    tts_studio_directory:Path=ROOT/'tts-data'

configuration=Configuration()
jobs={};references={};background=set();lock=asyncio.Lock();client=None
data_root=configuration.tts_studio_directory.resolve()

def persist(job):
    path=data_root/'results'/(job['id']+'.json');temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(job,ensure_ascii=False));temp.replace(path)

def public(ref):return {k:v for k,v in ref.items() if k!='prepared_audio'}

def validate_audio(raw):
    with wave.open(io.BytesIO(raw)) as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()!=44100:raise ValueError('Expected mono 44.1 kHz PCM16 WAV')
        duration=w.getnframes()/44100
        if not 3<=duration<=30:raise ValueError('Voice reference must be between three and thirty seconds')
        samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)/32768
        if np.sqrt(np.mean(samples*samples))<.0001:raise ValueError('Voice reference is silent')
    return duration

@asynccontextmanager
async def lifespan(app):
    global client,CHECKPOINT
    if not configuration.sglang_model_path:raise RuntimeError('Set SGLANG_MODEL_PATH to your local attention_mlp_r16_early export')
    if not configuration.tts_verify_model_hashes:raise RuntimeError('Hash checking is required by the derivative integration')
    verified=await asyncio.to_thread(verify_model,configuration.sglang_model_path,configuration.tts_derivative_manifest)
    CHECKPOINT=verified.get('derivative_identity_sha256',CHECKPOINT)
    client=httpx.AsyncClient(trust_env=False,timeout=httpx.Timeout(120,connect=5))
    try:
        response=await client.get(configuration.sglang_url+'/v1/models');response.raise_for_status()
        if (configuration.tts_engine_model_id or configuration.sglang_model_path) not in {m['id'] for m in response.json().get('data',[])}:
            raise RuntimeError('SGLang does not advertise the configured model path; refusing fallback')
        for folder in ('results','uploads'): (data_root/folder).mkdir(parents=True,exist_ok=True)
        reference=json.loads((ROOT/'assets/voice/reference.json').read_text())
        path=ROOT/'assets/voice/noura_reference.wav';raw=path.read_bytes();duration=validate_audio(raw)
        if hashlib.sha256(raw).hexdigest()!=reference['prepared_sha256']:raise RuntimeError('Noura reference hash mismatch')
        references[reference['reference_id']]=dict(id=reference['reference_id'],name='Noura نورة',duration=duration,
            text=reference['transcript'],wav_sha256=reference['prepared_sha256'],prepared_audio=str(path),uploaded=True)
        for path in (data_root/'uploads').glob('*.json'):
            ref=json.loads(path.read_text());audio=data_root/'uploads'/(ref['id']+'.wav')
            if audio.is_file():ref['prepared_audio']=str(audio);references[ref['id']]=ref
        yield
    finally:
        for task in background:task.cancel()
        await asyncio.gather(*background,return_exceptions=True)
        await client.aclose()

app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)

@app.get('/health')
async def health():return {'ok':True}
@app.get('/api/status')
async def status():return {'checkpoint_sha256':CHECKPOINT,'status':'ready','loaded_models':[MODEL],'models':[{'id':MODEL,'name':MODEL}],
    'references':[public(r) for r in references.values()],'jobs':list(jobs.values())[-32:]}
@app.get('/api/stream/references')
async def list_references():return {'references':[public(r) for r in references.values()]}
@app.get('/api/reference/{reference_id}.wav')
async def reference_audio(reference_id:str):
    if reference_id not in references:raise HTTPException(404)
    return FileResponse(references[reference_id]['prepared_audio'],media_type='audio/wav')
@app.get('/api/audio/{job_id}.wav')
async def result_audio(job_id:str):
    if job_id not in jobs or jobs[job_id]['status']!='complete':raise HTTPException(404)
    return FileResponse(data_root/'results'/(job_id+'.wav'),media_type='audio/wav')

@app.post('/api/upload',status_code=201)
async def upload(request:Request):
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>25*1024*1024:raise HTTPException(413)
    reference_id='upload_'+secrets.token_hex(16)
    source=data_root/'uploads'/(reference_id+'.input');output=data_root/'uploads'/(reference_id+'.wav')
    source.write_bytes(raw)
    try:
        if not shutil.which('ffmpeg'):raise ValueError('Install ffmpeg for voice uploads')
        process=await asyncio.create_subprocess_exec('ffmpeg','-nostdin','-v','error','-y','-protocol_whitelist','file,pipe',
            '-format_whitelist','wav,mp3,flac,ogg,mov,matroska,webm,aac','-i',str(source),'-t','31','-vn','-ac','1','-ar','44100','-c:a','pcm_s16le',str(output),
            stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
        try:await asyncio.wait_for(process.wait(),45)
        except TimeoutError:
            process.kill();await process.wait();raise ValueError('Reference decoding timed out')
        if process.returncode:raise ValueError('Could not decode reference audio')
        prepared=output.read_bytes();duration=validate_audio(prepared)
        ref=dict(id=reference_id,name='Noura نورة',duration=duration,text='',wav_sha256=hashlib.sha256(prepared).hexdigest(),prepared_audio=str(output),uploaded=True)
        references[reference_id]=ref;(data_root/'uploads'/(reference_id+'.json')).write_text(json.dumps(ref,ensure_ascii=False))
        return public(ref)
    except (ValueError,wave.Error) as error:
        output.unlink(missing_ok=True);raise HTTPException(400,str(error))
    finally:source.unlink(missing_ok=True)

async def accept(request):
    if int(request.headers.get('content-length','0'))>16384:raise HTTPException(413)
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>16384:raise HTTPException(413)
    try:payload=json.loads(raw)
    except (json.JSONDecodeError,UnicodeDecodeError):raise HTTPException(400,'Invalid JSON')
    if not isinstance(payload,dict):raise HTTPException(400,'Send a JSON object')
    if payload.get('model')!=MODEL:raise HTTPException(400,'Only attention_mlp_r16_early is permitted')
    reference=references.get(payload.get('reference_id'))
    if not reference:raise HTTPException(400,'Unknown voice reference')
    text=payload.get('text');ref_text=payload.get('reference_text')
    if not isinstance(text,str) or not 1<=len(text)<=1000:raise HTTPException(400,'Invalid speech text')
    if not isinstance(ref_text,str) or not 1<=len(ref_text)<=1000:raise HTTPException(400,'Exact reference transcript is required')
    for name,default,low,high in [('temperature',.7,.1,1.),('top_p',.7,.1,1.),('max_new_tokens',1024,128,2048)]:
        value=payload.get(name,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not low<=value<=high:raise HTTPException(400,'Invalid speech settings')
        payload[name]=value
    if sum(j['status'] in ('queued','generating') for j in jobs.values())>=8:raise HTTPException(409,'Synthesis queue is full')
    job=dict(id=secrets.token_hex(16),status='queued',request=payload,checkpoint_sha256=CHECKPOINT,reference_sha256=reference['wav_sha256'])
    jobs[job['id']]=job;persist(job)
    for key in list(jobs):
        if len(jobs)<=64:break
        if jobs[key]['status'] in ('complete','failed'):
            jobs.pop(key);(data_root/'results'/(key+'.wav')).unlink(missing_ok=True);(data_root/'results'/(key+'.json')).unlink(missing_ok=True)
    return job,reference

def engine_payload(job,reference,stream):
    request=job['request']
    reference_path=Path(reference['prepared_audio']).resolve()
    mapping=json.loads(configuration.tts_engine_reference_map)
    for local,remote in mapping.items():
        try:relative=reference_path.relative_to(Path(local).resolve())
        except ValueError:continue
        reference_path=Path(remote)/relative;break
    else:
        if mapping:raise RuntimeError('Reference is outside the explicit shared path mapping')
    return dict(model=configuration.tts_engine_model_id or configuration.sglang_model_path,input=request['text'],voice='default',
        references=[{'audio_path':str(reference_path).replace(chr(92),'/'),'text':request['reference_text']}],
        response_format='pcm' if stream else 'wav',stream=stream,temperature=request['temperature'],
        top_p=request['top_p'],top_k=30,max_new_tokens=request['max_new_tokens'])

def event(kind,**payload):return ('event: '+kind+'\ndata: '+json.dumps({'event':kind,**payload},ensure_ascii=False)+'\n\n').encode()

async def generate(job,reference):
    try:
        async with lock:
            job['status']='generating';persist(job)
            response=await client.post(configuration.sglang_url+'/v1/audio/speech',json=engine_payload(job,reference,False));response.raise_for_status()
            with wave.open(io.BytesIO(response.content)) as w:
                if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()!=44100 or not 0<w.getnframes()/44100<=120:raise ValueError('Invalid engine audio')
            (data_root/'results'/(job['id']+'.wav')).write_bytes(response.content)
            job.update(status='complete',audio_url='/api/audio/'+job['id']+'.wav');persist(job)
    except Exception:job.update(status='failed',error='Synthesis failed');persist(job)

@app.post('/api/generate',status_code=202)
async def nonstream(request:Request):
    job,reference=await accept(request);task=asyncio.create_task(generate(job,reference));background.add(task);task.add_done_callback(background.discard)
    return dict(job)

@app.post('/api/stream')
async def stream(request:Request):
    job,reference=await accept(request)
    async def audio():
        completed=False
        try:
            yield event('accepted',id=job['id'],sample_rate=44100,channels=1,encoding='pcm_s16le')
            async with lock:
                job['status']='generating';persist(job);yield event('started',id=job['id'])
                total=0
                async with client.stream('POST',configuration.sglang_url+'/v1/audio/speech',json=engine_payload(job,reference,True)) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        total+=len(chunk)
                        if total>44100*2*120:raise ValueError('Engine audio exceeded duration bound')
                        if chunk:yield event('audio',pcm=base64.b64encode(chunk).decode(),sample_rate=44100)
                if not total or total%2:raise ValueError('Invalid PCM output')
                job['status']='complete';persist(job);completed=True
                yield event('complete',id=job['id'],checkpoint_sha256=CHECKPOINT)
        except asyncio.CancelledError:raise
        except Exception:yield event('error',error='Synthesis failed')
        finally:
            if not completed:job['status']='failed';persist(job)
    return StreamingResponse(audio(),media_type='text/event-stream',headers={'Cache-Control':'no-store'})
