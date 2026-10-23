"""Recheck captured inference audio against the saved pre-fix playback PCM.

Requires the diagnostic snapshots in tests/results/latest-inference-20261018.
The original last-call WAVs had expired; provenance records which inputs were
regenerated. No new model inference or changes to source WAVs occur here.
"""
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

from app.audio import SAMPLE_RATE
from app.config import settings
from scripts.check_audio_dynamics import save
from scripts.check_rolling_leveling import process


def read(path):
    with wave.open(str(path)) as audio:
        assert (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())==(SAMPLE_RATE,1,2)
        return np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(float)


def rms_db(samples):
    return float(20*np.log10(max(1,np.sqrt(np.mean(samples**2)))/32768))


def main():
    root=Path('tests/results/latest-inference-20261018')
    path=root/'analysis.json';report=json.loads(path.read_text())
    report['new_audio_py_sha256']=hashlib.sha256(Path('app/audio.py').read_bytes()).hexdigest()
    report['new_profile']={'window_ms':settings.audio_leveler_window_ms,
                           'max_boost_db':settings.audio_leveler_max_boost_db,
                           'gain_rise_db_per_second':24,'quiet_hold_ms':80}
    report['passed']=False
    for case in report['cases']:
        name=case['name'];raw=read(root/(name+'-raw.wav'))
        pcm,metrics=process(raw.astype('<i2'),True)
        save(root/(name+'-revised.wav'),pcm)
        case['revised_processing']=metrics
        revised=np.frombuffer(pcm,dtype='<i2').astype(float);hop=SAMPLE_RATE//4
        case['levels']['revised']=[round(rms_db(revised[i:i+hop]),1)
                                  for i in range(0,len(revised)-hop+1,hop)]
        case['audio_sha256']={}
        for label in ('raw','before','revised'):
            audio_path=root/(name+'-'+label+'.wav')
            case['audio_sha256'][label]=hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if 'comparison_windows_seconds' in case:
            drops={}
            for label in ('raw','before','revised'):
                data=read(root/(name+'-'+label+'.wav'))
                windows=case['comparison_windows_seconds']
                level=lambda span:rms_db(data[round(span[0]*SAMPLE_RATE):round(span[1]*SAMPLE_RATE)])
                drops[label]=round(level(windows['early'])-level(windows['late']),3)
            assert drops['revised']<drops['before']-1.5
            case['early_to_late_drop_db']=drops
    report['passed']=True
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'cases':[
        {'name':case['name'],'drop_db':case.get('early_to_late_drop_db'),
         **case['revised_processing']} for case in report['cases']]},indent=2))


if __name__=='__main__':main()
