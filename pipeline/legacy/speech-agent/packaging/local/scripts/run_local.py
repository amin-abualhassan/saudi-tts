"""Run only the local app/LiveKit/optional gateway; never manage the GPU engine."""
import argparse,os,shutil,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
def ready(url,process):
    for _ in range(960):
        if process.poll() is not None:raise RuntimeError('A local service exited; inspect its console output')
        try:
            with urllib.request.urlopen(url,timeout=1) as response:
                if response.status==200:return
        except Exception:pass
        time.sleep(.25)
    raise RuntimeError('Service readiness timeout')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--with-tts',action='store_true');parser.add_argument('--external-livekit',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    if not (ROOT/'.env').exists() or not (ROOT/'deployment/livekit.yaml').exists():raise SystemExit('Run python -m scripts.configure and fill in .env first')
    if not (ROOT/'web/dist/index.html').exists():raise SystemExit('Build the frontend with npm --prefix web ci && npm --prefix web run build')
    from app.config import settings
    if not settings.openai_api_key:raise SystemExit('Add OPENAI_API_KEY to your local .env')
    processes=[]
    def start(command,port):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} is occupied; no existing process will be stopped')
        process=subprocess.Popen(command,cwd=ROOT);processes.append(process);return process
    def stop(*args):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        if not args.external_livekit:
            executable=os.environ.get('LIVEKIT_SERVER_BIN') or shutil.which('livekit-server')
            if not executable:raise RuntimeError('Install LiveKit Server or set LIVEKIT_SERVER_BIN; no Cloud account is needed')
            start([executable,'--config',str(ROOT/'deployment/livekit.yaml')],7880);time.sleep(.5)
        if args.with_tts:
            process=start([sys.executable,'-m','uvicorn','tts.gateway:app','--host','127.0.0.1','--port','3000','--no-access-log','--log-level','warning'],3000)
            ready('http://127.0.0.1:3000/health',process)
        process=start([sys.executable,'-m','uvicorn','app.server:app','--host','127.0.0.1','--port','8002','--no-access-log','--no-server-header','--log-level','warning'],8002)
        ready('http://127.0.0.1:8002/health',process)
        print('Open http://localhost:8002 . Press Ctrl+C to stop only these local services.',flush=True)
        while True:
            if any(p.poll() is not None for p in processes):raise RuntimeError('A local service stopped')
            time.sleep(.5)
    except KeyboardInterrupt:pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:process.terminate()
        for process in reversed(processes):
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
