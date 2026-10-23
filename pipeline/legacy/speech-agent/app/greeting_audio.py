"""Shorten long quiet gaps in pre-generated greetings, before caching."""
import io
import wave

import numpy as np

PROFILE='greeting-pauses-v1'


def tighten_greeting(data):
    with wave.open(io.BytesIO(data)) as audio:
        rate=audio.getframerate()
        if (rate,audio.getnchannels(),audio.getsampwidth())!=(44100,1,2):
            raise ValueError('Expected mono 44100 Hz int16 greeting')
        samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2')
    hop=round(rate*.01);length=len(samples)//hop*hop
    blocks=samples[:length].astype(np.float64).reshape(-1,hop)
    quiet=np.mean((blocks/32768)**2,axis=1)<10**(-50/10)
    edges=np.diff(np.r_[False,quiet,False].astype(np.int8))
    spans=[(int(start*hop),int(end*hop)) for start,end in
           zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
           if (end-start)*hop>=round(rate*.3)]
    keep=round(rate*.2);blend=round(rate*.005);result=samples.copy();cuts=[]
    for start,end in reversed(spans):
        # Touch only the middle of a long near-silent region, leaving 100 ms
        # beside each speech edge. Blend the join within that quiet region.
        left=start+keep//2;right=end-(keep-keep//2)-blend
        weight=np.linspace(0,1,blend)
        join=np.rint(result[left-blend:left]*(1-weight)+result[right:right+blend]*weight).astype('<i2')
        result=np.concatenate((result[:left-blend],join,result[right+blend:]))
        cuts.append({'start_seconds':start/rate,'end_seconds':end/rate,
                     'before_seconds':(end-start)/rate,'after_seconds':keep/rate})
    meta={'profile':PROFILE,'threshold_dbfs':-50,'minimum_gap_ms':300,
          'retained_gap_ms':200,'join_crossfade_ms':5,
          'source_samples':len(samples),'output_samples':len(result),
          'removed_seconds':(len(samples)-len(result))/rate,
          'gaps':list(reversed(cuts))}
    if not spans:return data,meta
    output=io.BytesIO()
    with wave.open(output,'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(rate)
        audio.writeframes(result.astype('<i2').tobytes())
    return output.getvalue(),meta
