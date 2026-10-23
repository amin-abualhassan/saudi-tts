"""Run with the separate, already-installed review environment; no credentials."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from pyannote.audio import Model, Inference

p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--root',required=True,type=Path)
args=p.parse_args()
torch.set_num_threads(8)
model=Model.from_pretrained(str(args.model),map_location='cpu')
inference=Inference(model,window='whole').to(torch.device('cpu'))

def embed(path):
    audio,rate=sf.read(path,dtype='float32',always_2d=True);audio=audio.mean(axis=1)
    gcd=math.gcd(rate,16000);audio=resample_poly(audio,16000//gcd,rate//gcd).astype('float32')
    with torch.inference_mode():value=np.asarray(inference(dict(waveform=torch.from_numpy(audio)[None],sample_rate=16000))).reshape(-1)
    return value/np.linalg.norm(value)

reference=embed(args.root/'reference/noura_reference.wav')
vectors={path.name:embed(path) for path in sorted((args.root/'samples').glob('*.wav'))
         if path.name.startswith(('bf16-','nf4-'))}
report=dict(model='pyannote/wespeaker-voxceleb-resnet34-LM',revision='837717ddb9ff5507820346191109dc79c958d614',
            checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
            note='Whole-clip cosine diagnostics; not human speaker-similarity judgments or calibrated acceptance thresholds',
            reference_cosines={name:float(vec@reference) for name,vec in vectors.items()},
            paired_cosines={name:float(vec@vectors[name.replace('bf16-','nf4-')]) for name,vec in vectors.items()
                            if name.startswith('bf16-') and name.replace('bf16-','nf4-') in vectors})
(args.root/'reports/quality-speaker.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
