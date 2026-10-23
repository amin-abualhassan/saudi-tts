"""Ending-volume recovery, silence/peak protection and source preservation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_finish import leveled_wav, steady_ending
from common import file_hash


class EndingVolumeTests(unittest.TestCase):
    def tone(self, seconds=8, rate=16000):
        t = np.arange(seconds * rate) / rate
        return .15 * np.sin(2 * np.pi * 190 * t), rate

    def test_recovers_fading_final_speech_preserving_prefix_and_duration(self):
        audio, rate = self.tone()
        audio[5 * rate:] *= np.linspace(1, .25, 3 * rate)
        result, meta = steady_ending(audio, rate)
        np.testing.assert_array_equal(result[:4 * rate], audio[:4 * rate])
        self.assertEqual(len(result), len(audio))
        self.assertTrue(meta["applied"])
        self.assertLessEqual(meta["max_gain_db"], 9.000001)
        ratio = np.sqrt(np.mean(result[-rate:] ** 2) / np.mean(audio[-rate:] ** 2))
        self.assertGreater(ratio, 1.5)
        self.assertTrue(np.isfinite(result).all())

    def test_constant_speech_short_audio_and_silence_are_unchanged(self):
        audio, rate = self.tone()
        for value in (audio, audio[:rate // 2], np.zeros(rate * 4), np.ones(rate * 4) * .00001):
            result, meta = steady_ending(value, rate)
            np.testing.assert_array_equal(result, value)
            self.assertFalse(meta["applied"])

    def test_does_not_raise_trailing_room_noise_or_clip_peaks(self):
        audio, rate = self.tone()
        audio[5 * rate:7 * rate] *= .3
        audio[6 * rate] = .95
        audio[7 * rate:] = .00005
        result, _ = steady_ending(audio, rate)
        self.assertLessEqual(float(np.max(np.abs(result))), .980001)
        np.testing.assert_array_equal(result[-rate // 2:], audio[-rate // 2:])

    def test_cache_preserves_raw_file_and_invalidates_when_audio_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.wav"
            audio, rate = self.tone()
            audio[5 * rate:] *= .3
            sf.write(source, audio, rate, subtype="PCM_16")
            original_hash = file_hash(source)
            output = leveled_wav(source, root / "cache")
            self.assertEqual(file_hash(source), original_hash)
            self.assertEqual(leveled_wav(source, root / "cache"), output)
            self.assertTrue(output.with_suffix(".json").is_file())
            sf.write(source, audio * .5, rate, subtype="PCM_16")
            self.assertNotEqual(leveled_wav(source, root / "cache"), output)

    def test_rejects_invalid_audio(self):
        for value in (np.array([np.nan]), np.zeros((10, 2))):
            with self.assertRaises(ValueError):
                steady_ending(value, 44100)


if __name__ == "__main__":
    unittest.main()
