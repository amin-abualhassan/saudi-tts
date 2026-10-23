"""Generate local LiveKit secrets/config without touching model services."""
import json,os,secrets,urllib.request
from pathlib import Path
import yaml
from dotenv import dotenv_values
from app.config import ROOT

def main():
 p=ROOT/'.env';p.touch(mode=0o600,exist_ok=True);values=dotenv_values(p)
 with p.open('a') as f:
  for name,value in [('LIVEKIT_API_KEY','AJ'+secrets.token_hex(8)),('LIVEKIT_API_SECRET',secrets.token_urlsafe(36))]:
   if not values.get(name):f.write('\n'+name+'='+value+'\n');values[name]=value
 p.chmod(0o600)
 address=os.environ.get('ALJAWAB_PUBLIC_IP')
 if not address:
  req=urllib.request.Request('http://169.254.169.254/latest/api/token',method='PUT',headers={'X-aws-ec2-metadata-token-ttl-seconds':'60'})
  imds=urllib.request.urlopen(req,timeout=3).read().decode()
  req=urllib.request.Request('http://169.254.169.254/latest/meta-data/public-ipv4',headers={'X-aws-ec2-metadata-token':imds})
  address=urllib.request.urlopen(req,timeout=3).read().decode()
 config={'port':7880,'bind_addresses':['127.0.0.1'],'rtc':{'tcp_port':7881,'udp_port':7882,
   'use_external_ip':False,'node_ip':address,'advertise_internal_ip':True,'allow_tcp_fallback':True},
   'keys':{values['LIVEKIT_API_KEY']:values['LIVEKIT_API_SECRET']},'logging':{'level':'warn'},
   'room':{'empty_timeout':60,'departure_timeout':20,'max_participants':2}}
 path=ROOT/'deployment/livekit.yaml'
 old=yaml.safe_load(path.read_text()) if path.exists() else {}
 if old.get('turn',{}).get('enabled'):
  if old['turn'].get('domain')==old.get('rtc',{}).get('node_ip') and address!=old['turn']['domain']:
   raise ValueError('Public IP changed: reissue the TURN IP certificate before changing its advertised address')
  config['turn']=old['turn']
 path.write_text(yaml.safe_dump(config));path.chmod(0o600)
 print('Configured self-hosted LiveKit; advertised public IP:',address)
if __name__=='__main__':main()
