"""Exercise concurrent real calls and the 60-second speech-idle timeout.

Uses an existing generated Arabic test fixture; reports contain timings/counts,
never session credentials, audio, or transcript text.
"""
import asyncio
import json
import time
import wave
from pathlib import Path

import httpx
import websockets
from livekit import rtc

BASE='http://[::1]:8002'


async def main():
    started=time.monotonic()
    tasks=[];rooms=[];sources=[];sockets=[];calls=[]
    result={'passed':False,'sessions':[]}
    with wave.open('/tmp/aljawab-ar.wav') as wav:
        assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(44100,1,2)
        fixture=wav.readframes(wav.getnframes())
    async with httpx.AsyncClient(trust_env=False,timeout=30) as client:
        async def admit(index):
            response=await client.post(BASE+'/api/session',json={'language':'ar'})
            response.raise_for_status()
            data=response.json()
            return {'headers':{'X-Session-Token':data['session_token']},'data':data,
                    'pending':bytearray(),'events':[],'audio_frames':0,'index':index}
        try:
            calls=await asyncio.gather(*(admit(i) for i in range(3)))
            result['concurrent_sessions']=len(calls)
            for call in calls:
                socket=await websockets.connect('ws://[::1]:8002/api/events',subprotocols=['aljawab','session.'+call['data']['session_token']])
                sockets.append(socket)
                async def receive(call=call,socket=socket):
                    async for raw in socket:
                        event=json.loads(raw)
                        call['events'].append({'type':event.get('type'),'role':event.get('role'),
                            'final':event.get('final'),'reason':event.get('reason'),
                            'elapsed_s':round(time.monotonic()-started,3)})
                tasks.append(asyncio.create_task(receive()))
                async def heartbeat(call=call):
                    while True:
                        response=await client.post(BASE+'/api/heartbeat',headers=call['headers'])
                        if response.status_code==401:return
                        response.raise_for_status()
                        await asyncio.sleep(10)
                tasks.append(asyncio.create_task(heartbeat()))
            for call in calls[:2]:
                room=rtc.Room();rooms.append(room)
                source=rtc.AudioSource(44100,1,queue_size_ms=40);sources.append(source)
                async def collect(track,call):
                    stream=rtc.AudioStream(track,sample_rate=44100,num_channels=1)
                    try:
                        async for packet in stream:call['audio_frames']+=1
                    finally:await stream.aclose()
                def subscribed(track,publication,participant,call=call):
                    tasks.append(asyncio.create_task(collect(track,call)))
                room.on('track_subscribed',subscribed)
                await room.connect('ws://127.0.0.1:7880',call['data']['token'])
                track=rtc.LocalAudioTrack.create_audio_track('generated-fixture',source)
                await room.local_participant.publish_track(track,rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
                async def pump(call=call,source=source):
                    while True:
                        data=bytes(call['pending'][:1764]);del call['pending'][:1764]
                        await source.capture_frame(rtc.AudioFrame(data+bytes(1764-len(data)),44100,1,882))
                        await asyncio.sleep(.02)
                tasks.append(asyncio.create_task(pump()))
                response=await client.post(BASE+'/api/ready',headers=call['headers']);response.raise_for_status()
            await asyncio.sleep(4)
            for call in calls[:2]:call['pending'].extend(fixture)
            print('Three simultaneous sessions admitted; two synthetic voice turns sent.',flush=True)
            while time.monotonic()-started<42:await asyncio.sleep(1)
            calls[1]['pending'].extend(fixture)
            print('Additional speech sent to the second call to extend its idle deadline.',flush=True)
            while time.monotonic()-started<76:await asyncio.sleep(1)
            statuses=[]
            for call in calls:
                response=await client.post(BASE+'/api/heartbeat',headers=call['headers'])
                statuses.append(response.status_code)
                events=call['events']
                result['sessions'].append({'index':call['index'],'heartbeat_status':response.status_code,
                    'user_turns':sum(e['type']=='transcript' and e['role']=='user' and bool(e['final']) for e in events),
                    'assistant_turns':sum(e['type']=='transcript' and e['role']=='assistant' and bool(e['final']) for e in events),
                    'audio_frames':call['audio_frames'],'ended':[e for e in events if e['type']=='ended'],
                    'error_events':sum(e['type']=='error' for e in events)})
            assert statuses==[401,200,401],statuses
            for summary in result['sessions'][:2]:
                assert summary['user_turns']>=1 and summary['assistant_turns']>=1 and summary['audio_frames']>0,summary
                assert summary['error_events']==0,summary
            assert result['sessions'][1]['user_turns']>=2
            assert result['sessions'][2]['ended'][0]['reason']=='idle'
            assert 59<=result['sessions'][2]['ended'][0]['elapsed_s']<=64
            result['passed']=True
        finally:
            for call in calls:await client.post(BASE+'/api/end',headers=call['headers'])
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            await asyncio.gather(*(socket.close() for socket in sockets))
            await asyncio.gather(*(room.disconnect() for room in rooms))
            await asyncio.gather(*(source.aclose() for source in sources))
            result['elapsed_s']=round(time.monotonic()-started,2)
            Path('tests/results').mkdir(exist_ok=True)
            Path('tests/results/connections-20261018.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':asyncio.run(main())
