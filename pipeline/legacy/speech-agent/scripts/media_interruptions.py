"""Inject generated speech over real LiveKit media at controlled times."""
import asyncio,json,time,wave,os
from pathlib import Path
import httpx,websockets,numpy as np
from livekit import rtc

async def main():
 rows=[];events=[];energies=[];pending=bytearray();quit=asyncio.Event();tasks=[]
 source=rtc.AudioSource(44100,1,queue_size_ms=40);room=rtc.Room()
 async def collect(track):
  stream=rtc.AudioStream(track,sample_rate=44100,num_channels=1)
  try:
   async for packet in stream:
    pcm=np.frombuffer(packet.frame.data,dtype=np.int16).astype(float)
    energies.append({'time':time.monotonic(),'rms':float(np.sqrt(np.mean(pcm*pcm)))/32768})
  finally:await stream.aclose()
 @room.on('track_subscribed')
 def subscribed(track,publication,participant):tasks.append(asyncio.create_task(collect(track)))
 async def pump():
  while not quit.is_set():
   data=bytes(pending[:1764]);del pending[:1764]
   data+=bytes(1764-len(data))
   await source.capture_frame(rtc.AudioFrame(data,44100,1,882))
   await asyncio.sleep(.02)
 def inject(path,label):
  with wave.open(path) as w:pending.extend(w.readframes(w.getnframes()))
  rows.append({'event':label,'time':time.monotonic()});print(label,flush=True)
 async with httpx.AsyncClient(trust_env=False,timeout=30) as client:
  r=await client.post('http://[::1]:8002/api/session',json={'language':'ar'});r.raise_for_status()
  result=r.json();headers={'Cookie':'aljawab_session='+r.cookies['aljawab_session']}
  async def renew():
   while not quit.is_set():
    await asyncio.sleep(15)
    await client.post('http://[::1]:8002/api/heartbeat',headers=headers)
  tasks.append(asyncio.create_task(renew()))
  try:
   await room.connect('ws://127.0.0.1:7880',result['token'])
   track=rtc.LocalAudioTrack.create_audio_track('synthetic-fixture',source)
   await room.local_participant.publish_track(track,rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
   tasks.append(asyncio.create_task(pump()));await asyncio.sleep(2)
   async with websockets.connect('ws://[::1]:8002/api/events',additional_headers=headers) as ws:
    queue=asyncio.Queue()
    async def receive():
     async for raw in ws:
      e=json.loads(raw);e['time']=time.monotonic();events.append(e);await queue.put(e)
    tasks.append(asyncio.create_task(receive()))
    async def wait_for(predicate,timeout=45):
     async with asyncio.timeout(timeout):
      while True:
       e=await queue.get()
       if predicate(e):return e
    await client.post('http://[::1]:8002/api/ready',headers=headers)
    await wait_for(lambda e:e.get('state')=='speaking')
    await asyncio.sleep(1)
    inject('/tmp/aljawab-ar.wav','interrupt_greeting')
    await wait_for(lambda e:e.get('role')=='user' and e.get('final'))
    await wait_for(lambda e:e.get('state')=='speaking')
    await asyncio.sleep(.6)
    inject('/tmp/aljawab-ar-correction.wav','interrupt_answer')
    await wait_for(lambda e:e.get('role')=='user' and e.get('final') and 'البصمة' in e.get('text',''))
    await wait_for(lambda e:e.get('role')=='assistant' and e.get('final') and not e.get('interrupted'))
    if os.environ.get('LANGUAGE_SWITCH_CHECK')=='1':
     inject('/tmp/aljawab-en-switch.wav','switch_to_english')
     await wait_for(lambda e:e.get('role')=='user' and e.get('final'))
     answer=await wait_for(lambda e:e.get('role')=='assistant' and e.get('final') and not e.get('interrupted'),timeout=65)
     assert any(c.isascii() and c.isalpha() for c in answer['text']), 'English switch did not produce English'
    await asyncio.sleep(.5)
  finally:
   quit.set();await client.post('http://[::1]:8002/api/end',headers=headers)
   for task in tasks:task.cancel()
   await asyncio.gather(*tasks,return_exceptions=True);await room.disconnect();await source.aclose()
   evidence={'injections':rows,'events':events,'audio_rms':energies}
   Path('tests/results/media-interruptions.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
   print('saved',len(events),'events',len(energies),'audio frames',flush=True)
if __name__=='__main__':asyncio.run(main())
