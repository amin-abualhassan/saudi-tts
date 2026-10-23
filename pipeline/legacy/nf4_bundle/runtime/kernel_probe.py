"""Fresh packed-only reload; fail if bitsandbytes attempts quantization."""
import argparse
import collections
import json
from pathlib import Path

import bitsandbytes as bnb
import torch

from weights import load, audit
from fish_speech.models.text2semantic.inference import generate_long,decode_one_token_ar

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--report',required=True)
args=p.parse_args()

def forbidden(*args,**kwargs):raise RuntimeError('Requantization forbidden during packed reload')
bnb.nn.Params4bit._quantize=forbidden
torch.set_num_threads(8)
model=load(args.model)
activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
with torch.profiler.profile(activities=activities) as prof:
    for response in generate_long(model=model,device='cuda',decode_one_token=decode_one_token_ar,
                                  text='Hello.',max_new_tokens=8,compile=False):pass
torch.cuda.synchronize()
kernels=collections.Counter(e.name for e in prof.events() if str(e.device_type).endswith('CUDA'))
ops=collections.Counter(e.name for e in prof.events() if '4bit' in e.name.lower() or 'dequant' in e.name.lower())
result=dict(hardware=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
            requantization_forbidden_and_reload_passed=True,quantized_modules=audit(model),
            cuda_kernel_counts=dict(kernels),quantized_operation_counts=dict(ops),
            pytorch_arch_list=torch.cuda.get_arch_list(),bitsandbytes_version=bnb.__version__)
Path(args.report).write_text(json.dumps(result,indent=2))
assert any('4bit' in k.lower() or 'dequant' in k.lower() for k in kernels),kernels
print(json.dumps(dict(success=True,kernels=dict(kernels))))
