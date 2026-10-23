import argparse
import json
import os
import subprocess
import threading
import time
import wave
from pathlib import Path

import psutil
import torch

from pipeline import Pipeline

TESTS = {
    'ar_greeting': 'حيّاك الله، معك نورة، المساعدة الصوتية للجواب. وش أقدر أساعدك فيه؟',
    'ar_services': 'أكيد، في الجواب نساعد الشركات في خدمة العملاء والرد على الاستفسارات وتنظيم المواعيد. نقدر نفهم احتياجك ونوضح لك الخيارات المناسبة، وإذا احتجت تفاصيل أكثر أحولك للفريق المختص. وش الخدمة اللي تهمك؟',
    'en_greeting': "Hello, I'm Noura, Aljawab's voice assistant. How can I help you?",
    'en_services': 'At Aljawab, we help companies answer customer questions, organize appointments, and provide clear information about their services. Tell me what your company needs, and I can explain the available options or connect you with the right team. Which service would you like to discuss first?',
}

def device_memory():
    row = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    return int(row.strip().splitlines()[0])*1024**2

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--codec', required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--parent', action='store_true')
    p.add_argument('--only', choices=list(TESTS))
    p.add_argument('--codec-device', default='cuda', choices=['cuda','cpu'])
    args=p.parse_args()
    root=args.root
    kind='bf16' if args.parent else 'nf4'
    name=kind+('-'+args.only if args.only else '')+('-cpu-codec' if args.codec_device=='cpu' else '')
    reference=json.loads((root/'reference/reference.json').read_text())
    report=dict(runtime='Fish Speech bitsandbytes fallback, not SGLang', hardware=torch.cuda.get_device_name(),
                compute_capability=list(torch.cuda.get_device_capability()), laptop_verified=False,
                baseline_device_bytes=device_memory(), samples={}, codec_device=args.codec_device,
                settings=dict(seed=42, temperature=.7, top_p=.7, top_k=30, max_new_tokens=768,
                              context=2048, compile=False, concurrency=1))
    telemetry=[]
    stop=threading.Event()
    process=psutil.Process()
    start=time.perf_counter()
    def monitor():
        while not stop.is_set():
            telemetry.append(dict(at=time.perf_counter()-start, device_bytes=device_memory(),
                                  host_rss=process.memory_info().rss,
                                  host_available=psutil.virtual_memory().available))
            stop.wait(.5)
    thread=threading.Thread(target=monitor, daemon=True);thread.start()
    try:
        pipeline=Pipeline(args.model,args.codec,parent=args.parent,codec_device=args.codec_device)
        report['load_seconds']=pipeline.load_seconds
        report['quantized_module_count']=len(pipeline.quantized_modules)
        report['quantized_modules']=pipeline.quantized_modules
        for case,text in TESTS.items():
            if args.only and args.only!=case:continue
            target=root/'samples'/f'{kind}-{case}{"-cpu-codec" if args.codec_device=="cpu" else ""}.wav'
            with wave.open(str(target),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(44100)
                metrics=pipeline.synthesize(text,str((root/'reference/noura_reference.wav').resolve()),
                                            reference['transcript'], f.writeframes)
            metrics['text']=text;metrics['file']=str(target.relative_to(root))
            report['samples'][case]=metrics
            print(json.dumps(dict(case=case,**metrics)),flush=True)
        report['success']=True
    except Exception as e:
        report['success']=False;report['error']=repr(e)
        raise
    finally:
        stop.set();thread.join()
        report['telemetry']=telemetry
        report['peak_device_bytes']=max(x['device_bytes'] for x in telemetry)
        report['peak_device_increment_bytes']=report['peak_device_bytes']-report['baseline_device_bytes']
        report['peak_host_rss_bytes']=max(x['host_rss'] for x in telemetry)
        report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        (root/'reports'/f'{name}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
