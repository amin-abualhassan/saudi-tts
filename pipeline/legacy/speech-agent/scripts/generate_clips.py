import asyncio, hashlib, io, json, wave
from app.config import ROOT, settings
from app.speech import Speech, MODEL_SHA, payload
from app.greeting_audio import tighten_greeting
from app.cues import with_final_breath

CLIPS = {
 'greeting_ar': 'حيّاك الله، معك نورة، المساعدة الصوتية للجواب. كيف أقدر أساعدك؟',
 'greeting_en': "Hello, I'm Noura, Aljawab's voice assistant. How can I help you?",
 'ack_ar_tamam': 'تمام.', 'ack_ar_understood': 'فهمت عليك.',
 'ack_en_aha': 'Aha.', 'ack_en_understood': 'I see.',
 'retry_ar': 'ما سمعتك زين. ممكن تعيد سؤالك؟',
 'retry_en': "I didn't catch that clearly. Could you repeat your question?",
}
async def main():
 speech=Speech(); target=ROOT/'assets/voice'; target.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((target/'manifest.json').read_text()) if (target/'manifest.json').exists() else {}
 try:
  for name,text in CLIPS.items():
   synthesis_text='[friendly] '+with_final_breath(text) if name.startswith('greeting_') else text
   old=manifest.get(name,{})
   path=target/(name+'.wav')
   if path.exists() and old.get('model')==settings.tts_model and old.get('reference')==settings.tts_reference and old.get('reference_sha256','')==settings.tts_reference_sha256 and old.get('checkpoint_sha256')==MODEL_SHA and old.get('sha256')==hashlib.sha256(path.read_bytes()).hexdigest() and all(old.get('settings',{}).get(k)==v for k,v in payload(synthesis_text).items()):continue
   data,job=await speech.generate(synthesis_text)
   postprocess={}
   if name.startswith('greeting_'):data,postprocess=tighten_greeting(data)
   with wave.open(io.BytesIO(data)) as w:
    meta=dict(sample_rate=w.getframerate(),channels=w.getnchannels(),duration=w.getnframes()/w.getframerate(),format='wav_pcm_s16le')
   (target/(name+'.wav')).write_bytes(data)
   manifest[name]=dict(text=text,model=settings.tts_model,reference=settings.tts_reference,reference_sha256=job['reference_sha256'],voice_name=settings.tts_voice_name,checkpoint_sha256=MODEL_SHA,
                       settings=job['request'],sha256=hashlib.sha256(data).hexdigest(),postprocess=postprocess,**meta)
   (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
   await speech.cleanup();print(name,meta,flush=True)
 finally: await speech.client.aclose()
if __name__=='__main__': asyncio.run(main())
