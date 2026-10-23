"""Bounded, reversible volume recovery for quieter generated speech endings."""
import hashlib
import io
from pathlib import Path
import threading

import numpy as np
import soundfile as sf

from common import atomic_json, file_hash

PROFILE = "ending-volume-v1"
IMPLEMENTATION_SHA256 = file_hash(Path(__file__))
_cache_lock = threading.Lock()


def steady_ending(audio, rate):
    """Raise quiet final speech by at most 9 dB, leaving earlier audio intact."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError("Expected finite mono audio and a positive sample rate")
    result = audio.copy()
    meta = {"profile": PROFILE, "applied": False, "max_gain_db": 0.0}
    if len(audio) < rate or np.max(np.abs(audio), initial=0) < .0001:
        return result, meta

    # Estimate speech energy over 200ms; control points every 20ms avoid
    # following the pitch cycle or equalizing individual phonemes.
    hop = max(1, int(rate * .02))
    centers = np.arange(0, len(audio), hop)
    half = max(1, int(rate * .1))
    lo, hi = np.maximum(centers - half, 0), np.minimum(centers + half, len(audio))
    energy = np.concatenate(([0.0], np.cumsum(audio * audio)))
    rms = np.sqrt((energy[hi] - energy[lo]) / np.maximum(1, hi - lo))
    gate = max(.0001, float(np.percentile(rms, 85)) * .04)
    active = np.flatnonzero(rms > gate)
    if len(active) < 10:
        return result, meta
    first, last = int(active[0]), int(active[-1])
    span = centers[last] - centers[first]
    if span < rate:
        return result, meta
    # Compare to the same utterance's earlier voiced frames, not a fixed
    # loudness target or a reference recording with different recording gain.
    reference_end = centers[first] + span * .65
    reference = rms[(centers <= reference_end) & (rms > gate)]
    target = float(np.percentile(reference, 60))
    gain_db = np.clip(20 * np.log10(target / np.maximum(rms, 1e-12)) - 2, 0, 9)
    # Silence/room noise gets no correction. A soft gate preserves continuity.
    weight = np.clip((rms - gate) / max(gate, 1e-12), 0, 1)
    gain_db *= weight
    smooth = 25  # 500ms: gradual gain, no per-word pumping.
    gain_db = np.convolve(np.pad(gain_db, (smooth // 2, smooth // 2), mode="edge"),
                          np.ones(smooth) / smooth, mode="valid")
    start = max(centers[first], centers[last] - min(3 * rate, span * .4))
    samples = np.arange(len(audio))
    ramp = np.clip((samples - start) / (.75 * rate), 0, 1)
    sample_gain = 10 ** (np.interp(samples, centers, gain_db) * ramp / 20)
    # Gate individual very quiet samples using the smoothed speech envelope;
    # retain exact zero samples, pauses and an existing natural quiet tail.
    envelope = np.interp(np.arange(len(audio)), centers, rms)
    sample_gain = 1 + (sample_gain - 1) * np.clip((envelope - gate) / gate, 0, 1)
    # Reserve peak headroom without attenuating the whole utterance.
    sample_gain = np.minimum(sample_gain, np.maximum(1, .98 / np.maximum(np.abs(audio), 1e-12)))
    result *= sample_gain
    max_gain = float(20 * np.log10(sample_gain.max()))
    meta.update(applied=max_gain > .05, max_gain_db=max_gain,
                correction_start_seconds=float(start / rate),
                duration=len(audio) / rate, sample_rate=int(rate))
    return result, meta


def leveled_wav(source, cache):
    """Cache a separate WAV by source contents and algorithm; preserve source."""
    with _cache_lock:
        raw_hash = file_hash(source)
        key = hashlib.sha256((PROFILE + IMPLEMENTATION_SHA256 + raw_hash).encode()).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        output = cache / (key + ".wav")
        if output.is_file():
            return output
        audio, rate = sf.read(source, dtype="float64")
        processed, meta = steady_ending(audio, rate)
        buffer = io.BytesIO()
        sf.write(buffer, processed, rate, format="WAV", subtype="PCM_16")
        temporary = output.with_suffix(".tmp.wav")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(output)
        atomic_json(output.with_suffix(".json"), {**meta, "source_sha256": raw_hash,
                                                 "implementation_sha256": IMPLEMENTATION_SHA256,
                                                 "wav_sha256": file_hash(output)})
        return output
