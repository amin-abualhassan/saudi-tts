import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def verify_model(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((ROOT/'tts/model-artifacts.json').read_text())
    for name,entry in manifest['required_files'].items():
        path=directory/name
        if not path.is_file():raise RuntimeError(f'Missing required model artifact: {name}')
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=entry['sha256']:raise RuntimeError(f'Wrong model artifact: {name}')
    return manifest
