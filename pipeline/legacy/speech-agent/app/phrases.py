"""Incremental linguistic phrase boundaries, with a time/length escape hatch."""
import asyncio
import re
from .cues import TRAILING_CUES, drop_partial_cue, without_delivery_cues

def clean(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'[*_`#<>]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def punctuate(text):
    text = clean(drop_partial_cue(text))
    trailing=TRAILING_CUES.search(text)
    if trailing:
        body=text[:trailing.start()].rstrip()
        return body+('.' if body and body[-1] not in '.!?؟…' else '')+' '+text[trailing.start():].strip()
    return text + ('.' if text and text[-1] not in '.!?؟…' else '')

def boundary(text, force=False, expect_cues=False):
    for m in re.finditer(r'[.!?؟]+(?:\s+|$)', text):
        end = m.end()
        if end == len(text) and not force:
            continue  # wait one delta: a period might be the beginning of a decimal or URL
        token = text[:m.start()+1].rsplit(' ', 1)[-1]
        if token.lower() in {'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'e.g.', 'i.e.', 'vs.'}:
            continue
        # Keep the LLM's breath with the sentence it follows, even when the
        # bracket cue arrives in later deltas. Never synthesize it on its own.
        cursor=end
        while True:
            rest=text[cursor:]
            breath=re.match(r'\s*\[short\s+breath\]\s*',rest,re.I)
            if breath:
                cursor+=breath.end()
                continue
            if rest.lstrip().startswith('[') and ']' not in rest:
                return 0
            break
        if cursor>end:return cursor
        if expect_cues and not text[end:].strip():return 0
        return end
    if expect_cues:return 0  # Let the LLM finish its sentence and trailing cue.
    if len(text) >= 150 or force:
        if force and len(text) <= 150:
            return len(text)
        cut = text.rfind(' ', 0, 150)
        opening=text.rfind('[',0,cut)
        if opening>text.rfind(']',0,cut):cut=opening-1
        return cut + 1 if cut > 30 else 0
    return 0

async def phrases(source, max_wait=.65, expect_cues=False):
    iterator = source.__aiter__()
    pending = asyncio.create_task(anext(iterator, None))
    buf = ''
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=max_wait if buf else None)
            if not done:
                end = boundary(buf, force=True,expect_cues=expect_cues)
                if end:
                    if without_delivery_cues(buf[:end]):yield punctuate(buf[:end])
                    buf = buf[end:]
                continue
            value = pending.result()
            if value is None:
                if without_delivery_cues(buf): yield punctuate(buf)
                break
            buf += value
            pending = asyncio.create_task(anext(iterator, None))
            while (end := boundary(buf,expect_cues=expect_cues)):
                if without_delivery_cues(buf[:end]):yield punctuate(buf[:end])
                buf = buf[end:]
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        if hasattr(iterator, 'aclose'): await iterator.aclose()
