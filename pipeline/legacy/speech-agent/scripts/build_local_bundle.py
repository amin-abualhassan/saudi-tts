"""Build a source/configuration handoff without credentials or environments."""
import hashlib,json,os,shutil,subprocess,sys,tarfile,zipfile
from pathlib import Path
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'exports/aljawab-local'
SUITE=Path('/opt/dlami/nvme/training/jordanian_s2pro')
EXPORT=SUITE/'sglang/models/attention_mlp_r16_early/392cfb1cf6824476e616df3d0459986efdd268a13991737fdbb8260d200a5cd4'
TEMPLATES=ROOT/'packaging/local'

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def tree(source,target,extensions=None):
    for path in source.rglob('*'):
        relative=path.relative_to(source)
        if any(x in relative.parts for x in ('__pycache__','.git','node_modules','.venv','archive')):continue
        if path.is_file() and (extensions is None or path.suffix in extensions or path.name.startswith('LICENSE')):copy(path,target/relative)

def main():
    if '--repack-only' in sys.argv:
        package();return
    if DEST.exists():raise SystemExit('Bundle staging directory exists; choose a new export name or review/remove only the previous generated staging tree.')
    DEST.mkdir(parents=True)
    for folder in ('app','tests'):tree(ROOT/folder,DEST/folder,{'.py'})
    tree(ROOT/'web/src',DEST/'web/src')
    tree(ROOT/'web/dist',DEST/'web/dist')
    for name in ('index.html','package.json','package-lock.json','tsconfig.json'):copy(ROOT/'web'/name,DEST/'web'/name)
    for name in ('requirements.in','requirements.lock','pyproject.toml','.gitignore','02-VOICE-AGENT-PROMPT.md','saudi_female_1minute-trimmed2.wav'):copy(ROOT/name,DEST/name)
    copy(ROOT/'requirements.lock',DEST/'requirements.txt')
    for name in ('generate_clips.py','configure_noura.py','refresh_knowledge.py'):copy(ROOT/'scripts'/name,DEST/'scripts'/name)
    for name in ('snapshot.json',):copy(ROOT/'knowledge'/name,DEST/'knowledge'/name)
    tree(ROOT/'assets/voice',DEST/'assets/voice',{'.wav','.json'})
    reference=json.loads((DEST/'assets/voice/reference.json').read_text());reference['source']='saudi_female_1minute-trimmed2.wav'
    (DEST/'assets/voice/reference.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2))
    prepared=SUITE/'voice/uploads'/(reference['reference_id'].removeprefix('upload_')+'.wav')
    copy(prepared,DEST/'assets/voice/noura_reference.wav')
    # Preserve the exact serving sources, isolated from the portable entrypoints.
    sources=DEST/'reference_aws/training/jordanian_s2pro'
    for path in SUITE.iterdir():
        if path.is_file() and (path.suffix in ('.py','.sh','.html') or path.name in ('sglang.Dockerfile','sglang-overrides.txt','.dockerignore')):copy(path,sources/path.name)
    talha=SUITE.parent/'talha_s2pro'
    for path in talha.glob('*.py'):copy(path,DEST/'reference_aws/training/talha_s2pro'/path.name)
    upstream=SUITE.parent/'fish-speech-upstream'
    tree(upstream/'fish_speech',DEST/'reference_aws/training/fish-speech-upstream/fish_speech',{'.py','.yaml','.yml','.json','.toml','.md'})
    for name in ('LICENSE','LICENSE.md','README.md','pyproject.toml'):
        if (upstream/name).is_file():copy(upstream/name,DEST/'reference_aws/training/fish-speech-upstream'/name)
    # Source only from the actual running engine package, not its virtualenv.
    container='jordanian-s2-486b828449d84c35909d4c6388d15f26'
    command=['docker','exec',container,'tar','-C','/opt/sglang-venv/lib/python3.12/site-packages','-cf','-','sglang_omni']
    process=subprocess.Popen(command,stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
        for member in archive:
            relative=Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts or '__pycache__' in relative.parts:continue
            if relative.suffix not in ('.py','.yaml','.yml','.json','.toml','.md'):continue
            target=DEST/'reference_aws/sglang_omni_source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    if process.wait()!=0:raise RuntimeError('Could not snapshot running SGLang-Omni source')
    license_text=subprocess.check_output(['docker','exec',container,'cat','/opt/sglang-venv/lib/python3.12/site-packages/sglang_omni-0.1.3.dist-info/licenses/LICENSE'])
    (DEST/'reference_aws/sglang_omni_source/LICENSE').write_bytes(license_text)
    for name in ('LICENSE.md','README.md','chat_template.jinja'):copy(SUITE.parent/'checkpoints/s2-pro'/name,DEST/'tts/base_model_reference'/name)
    receipt=json.loads((EXPORT/'export.json').read_text())
    files={name:dict(sha256=digest,bytes=(EXPORT/name).stat().st_size,included_in_zip=not name.endswith(('.safetensors','.pth'))) for name,digest in receipt['files'].items()}
    for name,entry in files.items():
        if entry['included_in_zip']:copy(EXPORT/name,DEST/'tts/model_config'/name)
    copy(EXPORT/'export.json',DEST/'tts/model_config/export.json')
    selected=json.loads((SUITE/'voice/selection.json').read_text())['models']['attention_mlp_r16_early']
    copy(Path(selected['config']),DEST/'tts/training_config.yaml')
    artifacts=dict(model='attention_mlp_r16_early',checkpoint_sha256=selected['checkpoint_sha256'],
        original_checkpoint_path=selected['checkpoint'],original_export_path=str(EXPORT),
        export_revision=EXPORT.name,required_files=files,excluded_weights_bytes=sum(x['bytes'] for x in files.values() if not x['included_in_zip']),
        base_model='Fish Audio S2 Pro; see included upstream model README and LICENSE',runtime=json.loads((SUITE/'sglang/manifest.json').read_text())['preflight']['versions'])
    (DEST/'tts/model-artifacts.json').write_text(json.dumps(artifacts,indent=2))
    config=json.loads((SUITE/'sglang/preflight/9d821b4e51da4afea5b0a3bff76bb08d/config/attention_mlp_r16_early.json').read_text())
    config['model_path']='SET_TO_YOUR_LOCAL_EXPORTED_MODEL_DIRECTORY';config['name']=config['model_path']
    (DEST/'tts/sglang.template.json').write_text(json.dumps(config,indent=2))
    copy(SUITE/'sglang_telemetry.py',DEST/'tts/sglang_telemetry.py')
    tree(TEMPLATES,DEST)
    # Relocate only the copied application; the running AWS source stays unchanged.
    path=DEST/'app/config.py';text=path.read_text().replace("tts_url: Literal['http://127.0.0.1:3000']",'tts_url: str')
    text=text.replace('    livekit_api_key:',"    app_secure_cookies: bool = False\n    tts_studio_directory: Path = ROOT/'tts-data'\n    livekit_api_key:")
    path.write_text(text)
    path=DEST/'app/speech.py';path.write_text(path.read_text().replace("STUDIO = Path('/opt/dlami/nvme/training/jordanian_s2pro/voice')",'STUDIO = settings.tts_studio_directory.resolve()'))
    path=DEST/'app/server.py';path.write_text(path.read_text().replace('secure=True','secure=settings.app_secure_cookies'))
    path=DEST/'scripts/configure_noura.py';path.write_text(path.read_text().replace("'http://127.0.0.1:3000", "settings.tts_url+'" ).replace('from app.config import ROOT','from app.config import ROOT,settings'))
    # Do not copy any production .env values other than nonsecret voice settings.
    example=(ROOT/'.env.example').read_text()
    replacements={'TTS_REFERENCE':reference['reference_id'],'TTS_REFERENCE_TEXT':reference['transcript'],'TTS_REFERENCE_SHA256':reference['prepared_sha256']}
    lines=[line.split('=',1)[0]+'='+replacements[line.split('=',1)[0]] if '=' in line and line.split('=',1)[0] in replacements else line for line in example.splitlines()]
    example='\n'.join(lines)+'\nAPP_SECURE_COOKIES=false\nTTS_STUDIO_DIRECTORY=./tts-data\nSGLANG_URL=http://127.0.0.1:3002\nSGLANG_MODEL_PATH=\nTTS_VERIFY_MODEL_HASHES=true\n'
    (DEST/'.env.example').write_text(example)
    (DEST/'.gitignore').write_text('.env\n.venv/\n**/__pycache__/\n.pytest_cache/\nweb/node_modules/\nlogs/\ntts-data/\nmodels/\ndeployment/livekit.yaml\n')
    (DEST/'logs').mkdir(exist_ok=True)
    (DEST/'tests/results').mkdir(exist_ok=True)
    package()

def package():
    artifacts=json.loads((DEST/'tts/model-artifacts.json').read_text())
    secrets=[v for k,v in dotenv_values(ROOT/'.env').items() if k in ('OPENAI_API_KEY','LIVEKIT_API_KEY','LIVEKIT_API_SECRET') and v]
    entries=[]
    for path in sorted(DEST.rglob('*')):
        if not path.is_file():continue
        if any(x in path.relative_to(DEST).parts for x in ('__pycache__','.pytest_cache')) or path.name=='MANIFEST.json':continue
        raw=path.read_bytes()
        if any(secret.encode() in raw for secret in secrets):raise RuntimeError('Credential detected; archive not produced')
        if path.name=='.env' or any(x in path.parts for x in ('.venv','node_modules','__pycache__')):raise RuntimeError('Excluded runtime file found')
        entries.append(dict(path=path.relative_to(DEST).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    (DEST/'MANIFEST.json').write_text(json.dumps({'scope':'source/configuration/voice assets; large model weights excluded','files':entries},indent=2))
    archive=ROOT/'exports/aljawab-local.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for entry in entries+ [{'path':'MANIFEST.json'}]:
            z.write(DEST/entry['path'],Path('aljawab-local')/entry['path'])
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP integrity check failed')
    (archive.with_suffix('.zip.sha256')).write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(entries)+1,'excluded_weights_bytes':artifacts['excluded_weights_bytes']},indent=2))
if __name__=='__main__':main()
