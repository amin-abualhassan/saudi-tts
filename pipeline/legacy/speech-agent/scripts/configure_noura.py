"""Register the owner's Noura reference using the existing upload API."""
import asyncio,datetime,hashlib,json
from pathlib import Path
import httpx
from dotenv import set_key
from app.config import ROOT

TRANSCRIPT='هذا هو السؤال، فأحنا عندنا أول نظام للشركات كان بعهد الملك فيصل عشان بس نحطه في سياق من خمسين سنة بعدين صدر نظام'
SOURCE=ROOT/'saudi_female_1minute-trimmed2.wav'

async def main():
 raw=SOURCE.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 target=ROOT/'assets/voice/reference.json'
 old=json.loads(target.read_text()) if target.exists() else {}
 async with httpx.AsyncClient(trust_env=False,timeout=60) as client:
  response=await client.get('http://127.0.0.1:3000/api/stream/references');response.raise_for_status()
  ids={r['id'] for r in response.json()['references']}
  if old.get('source_sha256')==digest and old.get('transcript')==TRANSCRIPT and old.get('reference_id') in ids:
   reference_id=old['reference_id'];duration=old['duration']
  else:
   response=await client.post('http://127.0.0.1:3000/api/upload',content=raw,headers={'Content-Type':'audio/wav'})
   response.raise_for_status();ref=response.json();reference_id=ref['id'];duration=ref['duration']
  audio=await client.get('http://127.0.0.1:3000/api/reference/'+reference_id+'.wav');audio.raise_for_status()
  prepared_digest=hashlib.sha256(audio.content).hexdigest()
 for name,value in {'TTS_REFERENCE':reference_id,'TTS_REFERENCE_TEXT':TRANSCRIPT,'TTS_REFERENCE_SHA256':prepared_digest,'TTS_VOICE_NAME':'Noura نورة',
                    'VAD_ACTIVATION_THRESHOLD':'0.7','VAD_MIN_SPEECH_DURATION_MS':'220','END_OF_TURN_SILENCE_MS':'550'}.items():
  set_key(str(ROOT/'.env'),name,value)
 (ROOT/'.env').chmod(0o600)
 target.write_text(json.dumps(dict(name='Noura',arabic_name='نورة',source=str(SOURCE),source_sha256=digest,
   reference_id=reference_id,transcript=TRANSCRIPT,duration=duration,prepared_sha256=prepared_digest,
   configured_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),ensure_ascii=False,indent=2))
 print('Noura reference configured:',reference_id,'duration:',duration)
if __name__=='__main__':asyncio.run(main())
