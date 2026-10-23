"""Generate local-only LiveKit credentials; no AWS/IMDS calls."""
import argparse,os,secrets
from pathlib import Path
import yaml
from dotenv import dotenv_values,set_key

ROOT=Path(__file__).resolve().parent.parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--node-ip',default=os.environ.get('LIVEKIT_NODE_IP','127.0.0.1'));args=parser.parse_args()
    path=ROOT/'.env'
    if not path.exists():path.write_text((ROOT/'.env.example').read_text())
    values=dotenv_values(path)
    for name,value in [('LIVEKIT_API_KEY','AJ'+secrets.token_hex(8)),('LIVEKIT_API_SECRET',secrets.token_urlsafe(36))]:
        if not values.get(name):set_key(str(path),name,value);values[name]=value
    path.chmod(0o600)
    config={'port':7880,'bind_addresses':['127.0.0.1'],'rtc':{'tcp_port':7881,'udp_port':7882,'use_external_ip':False,
        'node_ip':args.node_ip,'advertise_internal_ip':True,'allow_tcp_fallback':True},
        'keys':{values['LIVEKIT_API_KEY']:values['LIVEKIT_API_SECRET']},'logging':{'level':'warn'},
        'room':{'empty_timeout':60,'departure_timeout':20,'max_participants':2}}
    target=ROOT/'deployment/livekit.yaml';target.parent.mkdir(exist_ok=True);target.write_text(yaml.safe_dump(config));target.chmod(0o600)
    print('Local LiveKit configured. Add your OpenAI key to .env. No credentials were printed.')
if __name__=='__main__':main()
