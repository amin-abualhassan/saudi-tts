"""Apply only to an explicitly chosen local app copy, preserving original files."""
import argparse
import hashlib
import shutil
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--app',type=Path,required=True,help='Aljawab app root containing tts/gateway.py')
args=p.parse_args()
here=Path(__file__).resolve().parent
for relative in ['gateway.py','model_files.py','app/config.py','app/speech.py']:
    name=Path(relative).name
    target=args.app/(relative if relative.startswith('app/') else 'tts/'+relative)
    expected=(here/(relative+'.original.sha256')).read_text().strip()
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        raise SystemExit('App preimage differs; refusing to patch '+str(target))
    if target.with_suffix('.py.bf16-original').exists():
        raise SystemExit('Backup already exists; refusing overwrite')
for relative in ['gateway.py','model_files.py','app/config.py','app/speech.py']:
    target=args.app/(relative if relative.startswith('app/') else 'tts/'+relative)
    shutil.copyfile(target,target.with_suffix('.py.bf16-original'))
    shutil.copyfile(here/relative,target)
print('Patched selected app copy. Original BF16 model-artifacts.json is unchanged.')
print('Set the explicit derivative environment variables documented in README.md.')
