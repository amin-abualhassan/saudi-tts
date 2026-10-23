import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException

from app import server


@pytest.fixture
def runtime(monkeypatch):
    instance=server.Runtime()
    monkeypatch.setattr(server,'runtime',instance)
    monkeypatch.setattr(server.settings,'openai_api_key','fixture-key')
    monkeypatch.setattr(server,'check_model',AsyncMock())
    monkeypatch.setattr(server,'token',lambda room,identity,agent=False:room+'-'+identity)
    monkeypatch.setattr(server.rtc,'Room',lambda:SimpleNamespace(connect=AsyncMock(),disconnect=AsyncMock()))
    async def start(*args,**kwargs):
        return SimpleNamespace(aclose=AsyncMock()),SimpleNamespace(on_speech=Mock(),aclose=AsyncMock())
    monkeypatch.setattr(server,'start_conversation',start)
    room_api=SimpleNamespace(delete_room=AsyncMock())
    class API:
        async def __aenter__(self):return SimpleNamespace(room=room_api)
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(server.api,'LiveKitAPI',lambda *args:API())
    return instance


@pytest.mark.asyncio
async def test_concurrent_sessions_are_isolated_in_one_browser(runtime):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='https://test') as client:
        first,second=await asyncio.gather(*(client.post('/api/session',json={'language':lang}) for lang in ('ar','en')))
        assert first.status_code==second.status_code==200
        one,two=first.json(),second.json()
        assert one['room']!=two['room'] and len(runtime.calls)==2
        headers_one={'X-Session-Token':one['session_token']}
        headers_two={'X-Session-Token':two['session_token']}
        call_one=runtime.calls[one['session_token']]
        call_two=runtime.calls[two['session_token']]
        await server.notify(call_one,{'type':'fixture','owner':1})
        assert not call_two.messages
        response=await client.post('/api/language',headers=headers_one,json={'language':'en'})
        assert response.status_code==200 and call_one.agent.preference=='en'
        response=await client.post('/api/end',headers=headers_one)
        assert response.status_code==200
        assert (await client.post('/api/heartbeat',headers=headers_one)).status_code==401
        assert (await client.post('/api/heartbeat',headers=headers_two)).status_code==200
        assert call_one.session is None and len(runtime.calls)==1
        assert list(call_two.messages)==[]
        await client.post('/api/end',headers=headers_two)


@pytest.mark.asyncio
async def test_shared_public_ip_can_admit_more_than_eight_users(runtime):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='https://test') as client:
        responses=await asyncio.gather(*(client.post('/api/session',json={'language':'ar'},
            headers={'cf-connecting-ip':'192.0.2.1'}) for _ in range(9)))
    assert all(response.status_code==200 for response in responses)
    assert len({response.json()['room'] for response in responses})==9


@pytest.mark.asyncio
async def test_configured_admission_limit_remains_per_ip_and_keeps_existing_calls(runtime,monkeypatch):
    monkeypatch.setattr(server.settings,'session_start_limit_per_minute',2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='https://test') as client:
        headers={'cf-connecting-ip':'192.0.2.1'}
        for _ in range(2):assert (await client.post('/api/session',json={'language':'ar'},headers=headers)).status_code==200
        assert (await client.post('/api/session',json={'language':'ar'},headers=headers)).status_code==429
        assert len(runtime.calls)==2
        assert (await client.post('/api/session',json={'language':'ar'},headers={'cf-connecting-ip':'192.0.2.2'})).status_code==200


@pytest.mark.asyncio
async def test_connection_reports_are_authenticated_bounded_and_do_not_log_free_text(runtime,caplog):
    import logging
    caplog.set_level(logging.INFO,logger='aljawab')
    call=server.Call('room','private-session-token','ar');runtime.calls[call.secret]=call
    report={'stage':'media','reason':'timeout','platform':'ios'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='https://test') as client:
        assert (await client.post('/api/connection-report',json=report)).status_code==401
        headers={'X-Session-Token':call.secret}
        for _ in range(2):assert (await client.post('/api/connection-report',json=report,headers=headers)).status_code==200
        assert (await client.post('/api/connection-report',json={**report,'reason':'arbitrary private text'},headers=headers)).status_code==422
    failures=[r.message for r in caplog.records if r.message.startswith('browser_connection_failed')]
    assert failures==['browser_connection_failed stage=media reason=timeout platform=ios']
    assert call.secret not in caplog.text and 'arbitrary private text' not in caplog.text


@pytest.mark.asyncio
async def test_heartbeats_do_not_prevent_idle_expiry(runtime,monkeypatch):
    clock=Mock(return_value=1000)
    # Replace server's clock reference without changing asyncio's own clock.
    monkeypatch.setattr(server,'time',SimpleNamespace(monotonic=clock))
    call=server.Call('room','secret','ar',last_input=941,last_seen=941)
    class Socket:
        send_json=AsyncMock()
    socket=Socket();call.sockets.add(socket)
    runtime.calls[call.secret]=call
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='https://test') as client:
        headers={'X-Session-Token':'secret'}
        assert (await client.post('/api/heartbeat',headers=headers)).status_code==200
        assert call.last_input==941 and call.last_seen==1000
        await server.expire_calls()
        assert 'secret' in runtime.calls
        clock.return_value=1001
        await server.expire_calls()
        assert 'secret' not in runtime.calls
        socket.send_json.assert_awaited_once_with({'type':'ended','reason':'idle'})
        assert (await client.post('/api/heartbeat',headers=headers)).status_code==401


@pytest.mark.asyncio
async def test_speech_extends_only_its_own_session(runtime,monkeypatch):
    monkeypatch.setattr(server,'time',SimpleNamespace(monotonic=lambda:1000))
    active=server.Call('active','one','ar',last_input=930,last_seen=999)
    silent=server.Call('silent','two','en',last_input=940,last_seen=999)
    runtime.calls.update(one=active,two=silent)
    server.input_activity(active)
    await server.expire_calls()
    assert active.last_input==1000 and list(runtime.calls)==['one']


@pytest.mark.asyncio
async def test_lost_browser_expires_even_with_recent_speech(runtime,monkeypatch):
    monkeypatch.setattr(server,'time',SimpleNamespace(monotonic=lambda:1000))
    call=server.Call('lost','secret','ar',last_input=999,last_seen=939)
    runtime.calls['secret']=call
    await server.expire_calls()
    assert not runtime.calls


@pytest.mark.asyncio
async def test_failed_start_does_not_end_existing_session(runtime,monkeypatch):
    existing=server.Call('existing','secret','ar')
    runtime.calls['secret']=existing
    monkeypatch.setattr(server,'start_conversation',AsyncMock(side_effect=RuntimeError('fixture')))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='https://test') as client:
        assert (await client.post('/api/session',json={'language':'en'})).status_code==503
    assert list(runtime.calls.values())==[existing]


@pytest.mark.asyncio
async def test_cleanup_is_idempotent_and_continues_after_failure(runtime):
    call=server.Call('room','secret','ar')
    call.session=SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError('fixture')))
    agent=SimpleNamespace(on_speech=Mock(),aclose=AsyncMock())
    room=SimpleNamespace(disconnect=AsyncMock())
    call.agent=agent;call.room=room
    runtime.calls['secret']=call
    await asyncio.gather(server.close_call(call),server.close_call(call))
    agent.aclose.assert_awaited_once();room.disconnect.assert_awaited_once()
    assert not runtime.calls and call.agent is None


def test_websocket_tab_token_and_origin(runtime):
    call=server.Call('room','secret','ar');runtime.calls['secret']=call
    request=SimpleNamespace(headers={'sec-websocket-protocol':'aljawab, session.secret','host':'test','origin':'https://test'},cookies={})
    assert server.current(request) is call
    request.headers['origin']='https://other'
    with pytest.raises(HTTPException) as error:server.current(request)
    assert error.value.status_code==403


@pytest.mark.asyncio
@pytest.mark.parametrize('status,models,accepted',[
    ('ready',['attention_mlp_r16_early'],True),
    ('generating',['attention_mlp_r16_early'],True),
    ('ready',['fast_r32_final'],False),
    ('loading',['attention_mlp_r16_early'],False),
    ('error',[],False),
])
async def test_model_accepts_busy_queue_but_requires_exact_loaded_model(monkeypatch,status,models,accepted):
    from app import speech
    monkeypatch.setattr(speech,'payload',lambda text:{})
    transport=httpx.MockTransport(lambda request:httpx.Response(200,json={'status':status,'loaded_models':models}))
    original=httpx.AsyncClient
    monkeypatch.setattr(speech.httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    if accepted:assert await speech.check_model()
    else:
        with pytest.raises(RuntimeError):await speech.check_model()
