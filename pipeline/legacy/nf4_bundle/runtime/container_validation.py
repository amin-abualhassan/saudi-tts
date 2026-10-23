"""Cold launch a restored release with ONLY derivative/reference/codec mounts.

Run from a Linux validation environment with psutil/httpx installed. Stops only
the uniquely named container it creates. Existing production containers persist.
"""
import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import psutil

p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path)
p.add_argument('--codec',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
args=p.parse_args();root=args.bundle.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
name='aljawab-nf4-restore-'+str(int(time.time()))
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1',3012))==0:raise RuntimeError('3012 occupied; no existing service will be stopped')

def device_memory():
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    return int(row.strip().splitlines()[0])*1024**2

report=dict(hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap','--format=csv,noheader'],text=True).strip(),
            baseline_device_bytes=device_memory(),laptop_verified=False,
            image_id=subprocess.check_output(['docker','image','inspect','aljawab-fish-nf4:20261017','--format','{{.Id}}'],text=True).strip(),
            model_read_only=True,bf16_shards_mounted=False)
command=['docker','run','-d','--name',name,'--gpus','all','--read-only','--tmpfs','/tmp:rw,nosuid,size=512m',
         '--shm-size=256m','--security-opt','no-new-privileges','--cap-drop','ALL','-p','127.0.0.1:3012:3012',
         '--mount',f'type=bind,source={root},target=/data,readonly',
         '--mount',f'type=bind,source={args.codec.resolve()},target=/original/codec.pth,readonly',
         '--mount',f'type=bind,source={root/"reference"},target=/references,readonly','aljawab-fish-nf4:20261017']
report['command']=command
telemetry=[];stopped=threading.Event();pid=None;start=time.perf_counter()
def monitor():
    while not stopped.is_set():
        sample=dict(at_seconds=time.perf_counter()-start,device_bytes=device_memory(),host_available=psutil.virtual_memory().available)
        if pid:
            try:sample['container_host_rss']=psutil.Process(pid).memory_info().rss
            except psutil.Error:pass
        telemetry.append(sample);stopped.wait(.25)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
    inspect=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    pid=inspect['State']['Pid']
    report['mounts']=inspect['Mounts'];report['ports']=inspect['HostConfig']['PortBindings']
    with httpx.Client(timeout=2,trust_env=False) as client:
        for _ in range(180):
            try:
                r=client.get('http://127.0.0.1:3012/health');r.raise_for_status()
                report['health_at_ready']=r.json();break
            except (httpx.HTTPError,ValueError):time.sleep(.5)
        else:raise RuntimeError('Cold container readiness timeout')
        report['cold_launch_to_ready_seconds']=time.perf_counter()-start
    subprocess.run([sys.executable,str(root/'runtime/http_smoke.py'),'--reference','/references/noura_reference.wav',
                    '--reference-json',str(root/'reference/reference.json'),'--output',str(out/'http')],check=True,
                   stdout=(out/'http-smoke.log').open('w'),stderr=subprocess.STDOUT)
    report['http']=json.loads((out/'http/http-smoke.json').read_text())
    report['success']=True
finally:
    subprocess.run(['docker','logs',name],stdout=(out/'server.log').open('w'),stderr=subprocess.STDOUT)
    subprocess.run(['docker','stop','-t','15',name],stdout=subprocess.DEVNULL)
    # Keep the stopped validation container for inspection; no production changes.
    stopped.set();thread.join()
    report['telemetry']=telemetry
    report['peak_device_bytes']=max(x['device_bytes'] for x in telemetry)
    report['peak_device_increment_bytes']=report['peak_device_bytes']-report['baseline_device_bytes']
    report['peak_container_host_rss_bytes']=max((x.get('container_host_rss',0) for x in telemetry),default=0)
    report['device_bytes_after_stop']=device_memory()
    (out/'cold-restore.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['telemetry','http','command','mounts']},indent=2))
