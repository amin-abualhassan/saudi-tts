"""Offline CPU ASR comparison. These are diagnostics, not listening judgments."""
import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

def normalize(text):
    text=''.join(c for c in unicodedata.normalize('NFKD',text.lower()) if not unicodedata.combining(c))
    text=text.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ـ':''}))
    return re.sub(r'[^\w\s]',' ',text).split()

def wer(reference,hypothesis):
    prev=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        row=[i]
        for j,b in enumerate(hypothesis,1):row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(a!=b)))
        prev=row
    return prev[-1]/max(1,len(reference))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    tree=ast.parse((args.root/'runtime/validate.py').read_text())
    texts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TESTS' for t in n.targets))
    model=WhisperModel(args.model,device='cpu',compute_type='int8',cpu_threads=8,num_workers=1)
    results={}
    for path in sorted((args.root/'samples').glob('*.wav')):
        if not path.stem.startswith(('bf16-','nf4-')):continue
        kind,case=path.stem.split('-',1)
        if case not in texts:continue
        language=case[:2]
        segments,info=model.transcribe(str(path),language=language,beam_size=5,vad_filter=False,condition_on_previous_text=False)
        transcript=' '.join(x.text.strip() for x in segments)
        wav,sr=sf.read(path,dtype='float32')
        words=normalize(transcript)
        adjacent_repeated_trigrams=sum(words[i:i+3]==words[i+3:i+6] for i in range(max(0,len(words)-5)))
        result=dict(text=texts[case],asr_transcript=transcript,normalized_wer=wer(normalize(texts[case]),words),
                    adjacent_repeated_trigrams=adjacent_repeated_trigrams,duration=len(wav)/sr,
                    rms=float(np.sqrt(np.mean(wav**2))),peak=float(np.max(np.abs(wav))),
                    clipped_fraction=float(np.mean(np.abs(wav)>=.999)),
                    finite=bool(np.isfinite(wav).all()))
        results[path.name]=result
        (args.root/'reports/quality-asr.json').write_text(json.dumps(dict(method='CPU int8 Whisper large-v3-turbo; automatic only',samples=results),indent=2,ensure_ascii=False))
        print(json.dumps(dict(file=path.name,**result),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
