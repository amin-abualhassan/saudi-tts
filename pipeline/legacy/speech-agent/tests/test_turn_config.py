import pytest
import yaml

from scripts import configure


def fixture_config(tmp_path,monkeypatch,ip):
    monkeypatch.setattr(configure,'ROOT',tmp_path)
    monkeypatch.setenv('ALJAWAB_PUBLIC_IP',ip)
    (tmp_path/'deployment').mkdir()
    (tmp_path/'.env').write_text('LIVEKIT_API_KEY=REDACTED\nLIVEKIT_API_SECRET=REDACTED\n')
    path=tmp_path/'deployment/livekit.yaml'
    turn={'enabled':True,'domain':'192.0.2.1','tls_port':5349,'external_tls':True,
          'proxy_protocol':True,'proxy_protocol_trusted_cidrs':['127.0.0.1/32']}
    path.write_text(yaml.safe_dump({'rtc':{'node_ip':'192.0.2.1'},'turn':turn}))
    return path,turn


def test_reconfiguration_preserves_the_installed_authenticated_turn_relay(tmp_path,monkeypatch):
    path,turn=fixture_config(tmp_path,monkeypatch,'192.0.2.1')
    configure.main()
    result=yaml.safe_load(path.read_text())
    assert result['turn']==turn
    assert result['keys']=={'fixture':'fixture-secret'}
    assert path.stat().st_mode&0o777==0o600


def test_ip_change_does_not_silently_publish_a_turn_address_with_the_wrong_certificate(tmp_path,monkeypatch):
    path,_=fixture_config(tmp_path,monkeypatch,'192.0.2.2');before=path.read_bytes()
    with pytest.raises(ValueError,match='reissue the TURN IP certificate'):configure.main()
    assert path.read_bytes()==before
