"""Exercise actual WAV and incremental PCM responses; record network arrival times."""
import argparse
import io
import json
import time
import wave
from pathlib import Path

import httpx

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:3012')
p.add_argument('--reference',required=True,help='Absolute path as visible INSIDE the runtime')
p.add_argument('--reference-json',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
args=p.parse_args()
reference=json.loads(args.reference_json.read_text())
report={}
with httpx.Client(timeout=300,trust_env=False) as client:
    r=client.get(args.url+'/health');r.raise_for_status();report['health']=r.json()
    r=client.get(args.url+'/v1/models');r.raise_for_status();model=r.json()['data'][0]['id']
    payload=dict(model=model,input="Hello, I'm Noura, Aljawab's voice assistant. How can I help you?",
                 references=[dict(audio_path=args.reference,text=reference['transcript'])],
                 temperature=.7,top_p=.7,top_k=30,seed=42,max_new_tokens=384,response_format='wav',stream=False)
    start=time.perf_counter();r=client.post(args.url+'/v1/audio/speech',json=payload);r.raise_for_status()
    report['wav_seconds']=time.perf_counter()-start
    with wave.open(io.BytesIO(r.content)) as f:
        assert (f.getnchannels(),f.getsampwidth(),f.getframerate())==(1,2,44100)
        report['wav_duration']=f.getnframes()/44100
        wav_pcm=f.readframes(f.getnframes())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'http-nf4-greeting.wav').write_bytes(r.content)
    payload.update(stream=True,response_format='pcm')
    arrivals=[];pcm=bytearray();start=time.perf_counter()
    with client.stream('POST',args.url+'/v1/audio/speech',json=payload) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            if chunk:
                arrivals.append(dict(at_seconds=time.perf_counter()-start,bytes=len(chunk)))
                pcm.extend(chunk)
    report.update(pcm_first_seconds=arrivals[0]['at_seconds'],pcm_total_seconds=time.perf_counter()-start,
                  pcm_duration=len(pcm)/88200,pcm_arrivals=arrivals,pcm_matches_wav=bytes(pcm)==wav_pcm,
                  first_pcm_before_response_completed=arrivals[0]['at_seconds']<arrivals[-1]['at_seconds'])
    assert len(pcm)>0 and len(pcm)%2==0 and report['first_pcm_before_response_completed'] and report['pcm_matches_wav']
    (args.output/'http-nf4-greeting.pcm').write_bytes(pcm)
    bad=dict(payload,model='attention_mlp_r16_early')
    assert client.post(args.url+'/v1/audio/speech',json=bad).status_code==400
    report['original_identity_rejected']=True
    bad=dict(payload,references=[dict(audio_path='/etc/passwd',text=reference['transcript'])])
    assert client.post(args.url+'/v1/audio/speech',json=bad).status_code==400
    report['outside_reference_root_rejected']=True
    report['health_after']=client.get(args.url+'/health').json()
(args.output/'http-smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
