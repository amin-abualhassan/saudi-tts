"""Back up and tighten cached welcomes without regenerating the voice."""
import hashlib
import json
from pathlib import Path
import wave
import io

from app.config import ROOT
from app.greeting_audio import PROFILE, tighten_greeting


def main():
    target=ROOT/'assets/voice';path=target/'manifest.json'
    manifest=json.loads(path.read_text())
    archive=target/'archive'/PROFILE;archive.mkdir(parents=True,exist_ok=True)
    original=archive/'manifest.json'
    if not original.exists():original.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report={}
    for name in ('greeting_ar','greeting_en'):
        entry=manifest[name];source=target/(name+'.wav');raw=source.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        if entry.get('postprocess',{}).get('profile')==PROFILE:
            print(name,'already tightened');continue
        backup=archive/(name+'.wav')
        if backup.exists() and backup.read_bytes()!=raw:raise ValueError('Greeting backup differs')
        backup.write_bytes(raw)
        output,meta=tighten_greeting(raw)
        with wave.open(io.BytesIO(output)) as audio:duration=audio.getnframes()/audio.getframerate()
        report[name]={'old_duration':entry['duration'],'new_duration':duration,**meta}
        entry.update(duration=duration,sha256=hashlib.sha256(output).hexdigest(),
                     source_sha256=hashlib.sha256(raw).hexdigest(),postprocess=meta)
        temporary=source.with_suffix('.tmp.wav');temporary.write_bytes(output);temporary.replace(source)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temporary.replace(path)
    if report:
        result=ROOT/'tests/results/greeting-pauses-20261018.json'
        result.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
