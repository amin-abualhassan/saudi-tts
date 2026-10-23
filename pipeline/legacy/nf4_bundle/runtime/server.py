"""OpenAI-compatible loopback TTS adapter for the separate Fish NF4 runtime."""
import argparse
import io
import json
import queue
import threading
import wave
from pathlib import Path

import uvicorn
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from pipeline import Pipeline
from weights import sha

def create_app(model, codec, allowed_reference_root, codec_device='cuda'):
    pipeline = Pipeline(model, codec, codec_device=codec_device)
    root = Path(allowed_reference_root).resolve()
    identity = 'attention_mlp_r16_early-nf4-' + sha(Path(model)/'model.nf4.safetensors')[:16]
    app = FastAPI(docs_url=None, redoc_url=None)
    lock = threading.Lock()

    @app.get('/health')
    def health():
        return dict(ok=True, model=identity, runtime='fish-speech-bnb-nf4', quantized_modules=len(pipeline.quantized_modules),
                    load_seconds=pipeline.load_seconds, cuda_allocated=torch.cuda.memory_allocated(),
                    cuda_reserved=torch.cuda.memory_reserved(),cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved=torch.cuda.max_memory_reserved())

    @app.get('/v1/models')
    def models():
        return dict(object='list',data=[dict(id=identity,object='model',owned_by='local')])

    @app.post('/v1/audio/speech')
    async def speech(request: Request):
        if int(request.headers.get('content-length','0')) > 16384:
            raise HTTPException(413)
        body=await request.body()
        if len(body)>16384:raise HTTPException(413)
        try:
            payload=json.loads(body)
            if payload['model'] != identity:raise ValueError('Derivative model identity required')
            refs=payload['references']
            if len(refs)!=1:raise ValueError('Exactly one reference required')
            ref=Path(refs[0]['audio_path'])
            if not ref.is_absolute() or not ref.resolve().is_relative_to(root) or not ref.is_file():
                raise ValueError('Reference must be an absolute path under the shared reference root')
            text=payload['input'];transcript=refs[0]['text']
            if not isinstance(text,str) or not 1<=len(text)<=1000 or not isinstance(transcript,str) or not 1<=len(transcript)<=1000:
                raise ValueError('Text and exact reference transcript required')
            stream=payload.get('stream',False)
            if payload.get('response_format','wav') != ('pcm' if stream else 'wav'):
                raise ValueError('Use pcm for streaming, wav otherwise')
            settings={k:payload.get(k,v) for k,v in dict(seed=42,temperature=.7,top_p=.7,top_k=30,max_new_tokens=768).items()}
            if not (type(settings['seed']) is int and type(settings['max_new_tokens']) is int and 1<=settings['max_new_tokens']<=1024
                    and type(settings['top_k']) is int and 1<=settings['top_k']<=30
                    and .1<=settings['temperature']<=1 and .1<=settings['top_p']<=1):
                raise ValueError('Sampling settings out of bounds')
        except (KeyError,TypeError,ValueError) as e:raise HTTPException(400,str(e))
        if not lock.acquire(blocking=False):raise HTTPException(409,'One active request allowed')
        output=queue.Queue(maxsize=4)
        cancelled=threading.Event()
        def put(value):
            while not cancelled.is_set():
                try:output.put(value,timeout=.2);return
                except queue.Full:pass
            raise RuntimeError('Client disconnected')
        def worker():
            try:
                metrics=pipeline.synthesize(text,str(ref),transcript,put,**settings)
                print(json.dumps(dict(model=identity,metrics=metrics)),flush=True)
                put(None)
            except Exception as e:
                if not cancelled.is_set():
                    try:put(e)
                    except RuntimeError:pass
            finally:lock.release()
        threading.Thread(target=worker,daemon=True).start()
        def chunks():
            try:
                while True:
                    part=output.get()
                    if part is None:break
                    if isinstance(part,Exception):raise part
                    yield part
            finally:cancelled.set()
        headers={'Cache-Control':'no-store','X-Model-Identity':identity,'X-Audio-Sample-Rate':'44100'}
        if stream:return StreamingResponse(chunks(),media_type='audio/pcm',headers=headers)
        import asyncio
        def wav():
            buffer=io.BytesIO()
            with wave.open(buffer,'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(44100)
                for part in chunks():f.writeframes(part)
            return buffer.getvalue()
        try:return Response(await asyncio.to_thread(wav),media_type='audio/wav',headers=headers)
        except Exception as e:raise HTTPException(500,'Synthesis failed; inspect runtime log') from e
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--codec',required=True)
    p.add_argument('--reference-root',required=True)
    p.add_argument('--port',type=int,default=3012)
    p.add_argument('--container',action='store_true',help='Bind container network; publish host port on 127.0.0.1 only')
    p.add_argument('--codec-device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    uvicorn.run(create_app(args.model,args.codec,args.reference_root,args.codec_device),
                host='0.0.0.0' if args.container else '127.0.0.1',port=args.port,workers=1)
