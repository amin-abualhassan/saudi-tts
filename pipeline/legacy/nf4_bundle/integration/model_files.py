import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def verify_model(directory, derivative_manifest=None):
    directory=Path(directory).resolve()
    derivative_manifest=derivative_manifest or os.environ.get('TTS_DERIVATIVE_MANIFEST')
    if derivative_manifest:
        manifest_path=Path(derivative_manifest)
        raw=manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!='0afbe95ae65a2411a8d927c599b3516fb7f464bcbe6dc0372f064b91c2dadca4':raise RuntimeError('Derivative verification manifest checksum mismatch')
        manifest=json.loads(raw)
        if manifest.get('derivative_identity_sha256')!='bd78f331a5a1534cb8bfecc6c0a024db7c0c5267a6c9da806a2dde1e18dd29af':raise RuntimeError('Unrecognized NF4 derivative identity')
    else:
        manifest=json.loads((ROOT/'tts/model-artifacts.json').read_text())
    for name,entry in manifest['required_files'].items():
        path=directory/name
        if not path.is_file():raise RuntimeError(f'Missing required model artifact: {name}')
        if 'bytes' in entry and path.stat().st_size!=entry['bytes']:raise RuntimeError('Wrong artifact size: '+name)
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=entry['sha256']:raise RuntimeError(f'Wrong model artifact: {name}')
    return manifest
