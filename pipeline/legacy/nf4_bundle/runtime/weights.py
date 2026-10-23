"""Strict, tensor-at-a-time HF -> Fish NF4 conversion and packed-only loader.

Reviewed against groxaxo/fish-speech-int4-patch fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb.
No hub API; no permissive load_state_dict; no implicit BF16 fallback.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import bitsandbytes as bnb
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fish_speech.models.text2semantic.llama import (
    BaseModelArgs, DualARTransformer, precompute_freqs_cis,
)
from fish_speech.tokenizer import FishTokenizer

REVISION = '392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4'
DERIVATIVE_SHA = 'bd78f331a5a1534cb8bfecc6c0a024db7c0c5267a6c9da806a2dde1e18dd29af'
AUXILIARY_FILES = {
    'config.json': (1863, '261b519a2a9576710fc8533a77297fae007f0e7b3aa28a217f8352b7f32fe993'),
    'tokenizer.json': (12217872, 'f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b'),
    'tokenizer_config.json': (860832, 'b8d149343ae425b0da67e6708686aceb51be7815d9792f265fc12ff04d5e9856'),
    'special_tokens_map.json': (101864, 'c2ff18fde6e43b7408435bc8ed079af74531befba549358a97cbc59ce606bc6b'),
}
PARENT = {
    'model-00001-of-00002.safetensors': (4986872984, '1f4256e6271c8aa1788190cce887414d127af594f6d7d534ccce3bae2bd85105'),
    'model-00002-of-00002.safetensors': (4136876104, '004f22a69ba9d1ab3f94570d1d76788b165173d886be5e3e42039a90ad983879'),
    'codec.pth': (1871099728, '74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f'),
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def verify_parent(path):
    for name, (size, digest) in {**PARENT, **AUXILIARY_FILES}.items():
        f = path / name
        if f.stat().st_size != size or sha(f) != digest:
            raise ValueError(f'Parent verification failed: {name}')

def map_key(key):
    if key.startswith('text_model.model.'):
        return key[len('text_model.model.'):]
    if key.startswith('audio_decoder.'):
        suffix = key[len('audio_decoder.'):]
        return suffix if suffix.startswith('codebook_embeddings.') else 'fast_' + suffix
    raise ValueError(f'Unrecognized parent key: {key}')

def skeleton(path, context=2048, device='cuda'):
    config = BaseModelArgs.from_pretrained(path)
    config.max_seq_len = context
    with torch.device('meta'):
        model = DualARTransformer(config)
    # These derived buffers are not learned weights and must be regenerated.
    model.freqs_cis = precompute_freqs_cis(context, config.head_dim, config.rope_base).to(device)
    model.fast_freqs_cis = precompute_freqs_cis(config.num_codebooks, config.fast_head_dim, config.rope_base).to(device)
    model.causal_mask = torch.ones(context, context, dtype=torch.bool, device=device).tril_()
    model.tokenizer = FishTokenizer.from_pretrained(path)
    assert (model.tokenizer.semantic_begin_id, model.tokenizer.semantic_end_id) == (151678, 155773)
    assert config.tie_word_embeddings and not hasattr(model, 'output')
    return model.eval()

def install(model, key, tensor, quantized=False, stats=None, device='cuda'):
    module_name, field = key.rsplit('.', 1)
    module = model.get_submodule(module_name)
    if quantized:
        assert isinstance(module, torch.nn.Linear) and field == 'weight'
        parent_name, child_name = module_name.rsplit('.', 1) if '.' in module_name else ('', module_name)
        replacement = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=False,
                                        compute_dtype=torch.bfloat16, quant_type='nf4',
                                        compress_statistics=True, device='meta')
        if stats is None:
            replacement.weight = bnb.nn.Params4bit(tensor.to(dtype=torch.bfloat16), requires_grad=False,
                                                   quant_type='nf4', compress_statistics=True).to(device)
        else:
            replacement.weight = bnb.nn.Params4bit.from_prequantized(
                tensor.to(device), stats, requires_grad=False, device=device, module=replacement)
        setattr(model.get_submodule(parent_name), child_name, replacement)
    else:
        setattr(module, field, torch.nn.Parameter(tensor.to(device=device, dtype=torch.bfloat16), requires_grad=False))

def audit(model):
    modules = {}
    for name, child in model.named_modules():
        if isinstance(child, bnb.nn.Linear4bit):
            w = child.weight
            assert isinstance(w, bnb.nn.Params4bit) and w.dtype == torch.uint8
            assert w.quant_state.quant_type == 'nf4' and w.quant_state.nested
            modules[name] = dict(type=type(child).__name__, weight_type=type(w).__name__,
                                 dtype=str(w.dtype), packed_shape=list(w.shape),
                                 original_shape=list(w.quant_state.shape), blocksize=w.quant_state.blocksize,
                                 double_quantization=w.quant_state.nested)
    assert all(not p.is_meta for p in model.parameters())
    return modules

@torch.inference_mode()
def convert(parent, output):
    verify_parent(parent)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        shutil.copyfile(parent / name, output / name)
    model = skeleton(output)
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    index = json.loads((parent / 'model.safetensors.index.json').read_text())['weight_map']
    mapped = {map_key(k): k for k in index}
    if len(mapped) != len(index) or set(mapped) != set(expected):
        raise ValueError(f'Coverage mismatch: missing={set(expected)-set(mapped)} unexpected={set(mapped)-set(expected)}')
    linear = {name + '.weight' for name, child in model.named_modules() if isinstance(child, torch.nn.Linear)}
    coverage = []
    for target, source in mapped.items():
        with safe_open(str(parent / index[source]), framework='pt', device='cpu') as f:
            tensor = f.get_tensor(source)
        if list(tensor.shape) != expected[target]:
            raise ValueError(f'Shape mismatch: {source}')
        install(model, target, tensor, quantized=target in linear)
        coverage.append(dict(source=source, target=target, shape=expected[target],
                             storage='NF4-double-quant' if target in linear else 'BF16'))
        del tensor
    modules = audit(model)
    assert len(modules) == 201, len(modules)
    # Save true packed uint8 tensors plus the complete bnb quantization state.
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(output / 'model.nf4.safetensors'))
    metadata = dict(format='fish-nf4-packed-v1', parent_revision=REVISION,
        adapter_provenance_sha256='9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf',
        parent_files={k: dict(bytes=v[0], sha256=v[1]) for k,v in PARENT.items()},
        tied_text_embedding_output=True, learned_tensor_count=len(coverage), coverage=coverage,
        quantized_modules=modules, compute_dtype='bfloat16', codec='external unchanged codec.pth')
    (output / 'quantization.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(success=True, learned_tensors=len(coverage), quantized_modules=len(modules),
                         bytes=(output/'model.nf4.safetensors').stat().st_size)), flush=True)

@torch.inference_mode()
def load(path, context=2048, device='cuda', parent=False):
    path = Path(path)
    for name, (size, digest) in AUXILIARY_FILES.items():
        if (path/name).stat().st_size != size or sha(path/name) != digest:
            raise ValueError(f'Configuration/tokenizer checksum mismatch: {name}')
    if not parent and sha(path/'quantization.json') != '64346a4d6b9b80fd49c552775d055841c30f79ea9636d87159cbf68989dd4a5a':
        raise ValueError('Quantization metadata checksum mismatch')
    model = skeleton(path, context, device)
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    if parent:
        index = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        mapped = {map_key(k): k for k in index}
        assert set(mapped) == set(expected) and len(mapped) == len(index)
        for target, source in mapped.items():
            with safe_open(str(path/index[source]), framework='pt', device='cpu') as f:
                tensor = f.get_tensor(source)
            assert list(tensor.shape) == expected[target]
            install(model, target, tensor, device=device)
    else:
        if sha(path/'model.nf4.safetensors') != DERIVATIVE_SHA:
            raise ValueError('Packed derivative SHA-256 mismatch')
        meta = json.loads((path/'quantization.json').read_text())
        assert meta['format'] == 'fish-nf4-packed-v1' and meta['parent_revision'] == REVISION
        quantized = meta['quantized_modules']
        consumed = set()
        with safe_open(str(path/'model.nf4.safetensors'), framework='pt', device='cpu') as f:
            keys = set(f.keys())
            for key, shape in expected.items():
                tensor = f.get_tensor(key)
                module_name = key.rsplit('.', 1)[0]
                stats = {k[len(key)+1:]: f.get_tensor(k) for k in keys if k.startswith(key+'.')}
                is_quantized = module_name in quantized
                if is_quantized:
                    assert tensor.dtype == torch.uint8 and stats
                    assert set(stats) == {'absmax', 'quant_map', 'nested_absmax', 'nested_quant_map',
                                          'quant_state.bitsandbytes__nf4'}
                    assert quantized[module_name]['original_shape'] == shape
                else:
                    assert list(tensor.shape) == shape and not stats and tensor.dtype == torch.bfloat16
                consumed.add(key)
                consumed.update(key+'.'+k for k in stats)
                # QuantState.from_dict consumes its packed metadata key in-place.
                install(model, key, tensor, quantized=is_quantized, stats=dict(stats), device=device)
            if consumed != keys:
                raise ValueError(f'Unexpected packed keys: {keys-consumed}')
        assert len(audit(model)) == len(quantized) == 201
    assert all(not p.is_meta for p in model.parameters())
    for layer in list(model.layers) + list(model.fast_layers):
        layer.attention.use_sdpa = True
    model._cache_setup_done = False
    return model.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    convert(args.parent, args.output)
