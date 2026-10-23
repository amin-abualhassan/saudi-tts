"""Private TTS delivery cues and transcription cleanup."""
import re

CUE_NAMES=('friendly','calm','empathetic','short breath')
CUE=re.compile(r'\[(?:friendly|calm|empathetic|short\s+breath)\]',re.I)
TRAILING_CUES=re.compile(r'(?:\s*\[(?:friendly|calm|empathetic|short\s+breath)\])+\s*$',re.I)
BREATH=re.compile(r'\[short\s+breath\]',re.I)


def drop_partial_cue(text):
    # An interrupted answer may end in a partially generated cue.
    start=text.rfind('[')
    if start>=0 and ']' not in text[start:]:
        fragment=' '.join(text[start+1:].lower().split())
        if any(name.startswith(fragment) for name in CUE_NAMES):text=text[:start]
    return text


def without_delivery_cues(text):
    return re.sub(r'\s+',' ',drop_partial_cue(CUE.sub('',text))).strip()


def with_final_breath(text):
    # Apply after phrase splitting, only on the synthesis branch. Normalize
    # legacy/model-authored breaths so each request has one final cue.
    body=re.sub(r'\s+',' ',drop_partial_cue(BREATH.sub('',text))).strip()
    return body+' [short breath]' if without_delivery_cues(body) else ''


async def transcript_chunks(source):
    pending=''
    async for chunk in source:
        pending+=str(chunk)
        while pending:
            start=pending.find('[')
            if start<0:
                yield pending;pending='';break
            if start:
                yield pending[:start];pending=pending[start:]
            end=pending.find(']')
            if end<0:break
            bracket=pending[:end+1]
            if not CUE.fullmatch(bracket):yield bracket
            pending=pending[end+1:]
    if pending:
        clean=without_delivery_cues(pending)
        if clean:yield clean
