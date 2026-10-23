import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location("pipeline", Path(__file__).parents[1] / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def rows(self):
        return [dict(clip_id=f"{g}-{i}", source_group=str(g), duration=10)
                for g in range(20) for i in range(20)]

    def test_selection_is_deterministic_and_disjoint(self):
        a, summary = pipeline.select(self.rows(), hours=0.1)
        b, _ = pipeline.select(list(reversed(self.rows())), hours=0.1)
        self.assertEqual(a, b)
        pipeline.assert_disjoint(a)
        self.assertGreaterEqual(summary["train"]["hours"], 0.09)

    def test_leaking_video_is_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.assert_disjoint([dict(source_group="same", split="train", clip_id="a"),
                                      dict(source_group="same", split="test", clip_id="b")])

    def test_too_little_data_fails_instead_of_silent_underfill(self):
        with self.assertRaises(ValueError):
            pipeline.select(self.rows(), hours=10)

    def test_resampling_preserves_duration_and_peak_ceiling(self):
        rate = 48000
        audio = 0.95 * np.sin(2 * np.pi * 997 * np.arange(rate * 2) / rate)
        output, _ = pipeline.normalize(audio, rate)
        self.assertEqual(len(output), 88200)
        self.assertLessEqual(np.abs(output).max(), 10 ** (-1 / 20) + 1e-6)
        self.assertTrue(np.isfinite(output).all())

    def test_silent_audio_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.normalize(np.zeros(48000), 48000)

    def test_smoke_refuses_bulk_or_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                pipeline.smoke([], Path(d), 13)
            with self.assertRaises(ValueError):
                pipeline.smoke([], Path(d), 6)

    def test_flac_metadata_matches_decoder(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.flac"
            sf.write(path, np.zeros(48000), 48000)
            info = pipeline.flac_info(path)
            self.assertEqual(info["duration"], 1)
            self.assertEqual(info["sample_rate"], 48000)
            self.assertEqual(info["channels"], 1)

    def test_reference_keeps_duplicate_titles_for_canonical_grouping(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ref.txt"
            path.write_text("talha-screenshot-2025\nvideo - abcdefghij\n"
                            "talha-screenshot-2024\nvideo - klmnopqrst\n")
            refs = pipeline.read_reference(path)
            self.assertEqual(refs["abcdefghij"]["title"], refs["klmnopqrst"]["title"])

    def test_inventory_handles_filtered_and_renumbered_cues(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "refrence.txt").write_text("talha-screenshot-2025\nvideo - abcdefghij\n"
                                              "talha-screenshot-2024\nvideo - klmnopqrst\n")
            audio = root / "splitted-audio-transcript/Jordan-youtube-48khz"
            labels = root / "splitted-audio-transcript/transcript"
            subtitles = root / "full-audios-transcript/transcript"
            for directory in (audio, labels, subtitles):
                directory.mkdir(parents=True)
            (subtitles / "abcdefghij.ar-new.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nshort dropped cue\n\n"
                "2\n00:00:02,000 --> 00:00:08,000\nmatching transcript\n")
            sf.write(audio / "abcdefghij-1.flac", np.zeros(48000 * 6), 48000)
            (labels / "abcdefghij-1.txt").write_text("matching transcript")
            rows, rejected = pipeline.inventory(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["srt_cue_number"], 2)
            self.assertEqual(rows[0]["start"], 2)

    def test_token_validation_rejects_wrong_codec_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "codes.npy"
            np.save(path, np.zeros((10, 5), dtype=np.int64))
            self.assertEqual(pipeline.validate_tokens(path), 5)
            for invalid in (np.zeros((9, 5), dtype=np.int64),
                            np.zeros((10, 5), dtype=np.float32),
                            np.full((10, 5), 4096, dtype=np.int64)):
                np.save(path, invalid)
                with self.assertRaises(ValueError):
                    pipeline.validate_tokens(path)


if __name__ == "__main__":
    unittest.main()
