from pathlib import Path
import sys
import tempfile
import unittest
import json
import struct

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[1]))
import validation
import merge_model


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def record(text, codes):
    sentence = field(1, text.encode())
    for row in codes:
        sentence += field(3, field(1, b"".join(varint(int(value)) for value in row)))
    data = field(1, b"folder") + field(2, b"talha_ahmad") + field(4, sentence)
    return struct.pack("<I", len(data)) + data


class CorpusValidationTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for i, split in enumerate(("train", "validation", "test")):
            folder = root / "audio" / split / "talha_ahmad"
            folder.mkdir(parents=True)
            audio = folder / f"clip-{i}.wav"
            text = f"transcript {i}"
            sf.write(audio, np.zeros(44100 * 5), 44100)
            label = audio.with_suffix(".lab")
            label.write_text(text)
            codes = np.full((10, 108), i, dtype=np.int64)
            np.save(audio.with_suffix(".npy"), codes)
            rows.append(dict(clip_id=f"clip-{i}", source_group=str(i), split=split,
                             duration=5, prepared_audio=str(audio), prepared_label=str(label), text=text))
            proto = root / "protos" / split
            proto.mkdir(parents=True)
            (proto / "00000000.protos").write_bytes(record(text, codes))
        (root / "prepared.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        return rows

    def test_packed_records_exactly_match_prepared_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            counts = validation.validate_packed(root)
            self.assertEqual(sum(v['verified_sentences'] for v in counts.values()), 3)

    def test_duplicate_packed_sentence_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            path = root / "protos/train/00000000.protos"
            path.write_bytes(path.read_bytes() * 2)
            with self.assertRaisesRegex(ValueError, "extra/repeated=1"):
                validation.validate_packed(root)

    def test_packer_preserves_a_trailing_space_from_label_newline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            for row in rows:
                Path(row['prepared_label']).write_text(row['text']+'\n')
                codes = np.load(Path(row['prepared_audio']).with_suffix('.npy'))
                (root/'protos'/row['split']/'00000000.protos').write_bytes(record(row['text']+' ',codes))
            self.assertEqual(len(validation.validate_packed(root)),3)

    def test_modified_labels_fail_before_packing_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            Path(rows[0]['prepared_label']).write_text('changed text')
            with self.assertRaisesRegex(ValueError, "label changed"):
                validation.validate_packed(root)

    def test_truncated_packed_stream_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "broken.protos"
            path.write_bytes(struct.pack('<I', 100) + b'abc')
            with self.assertRaisesRegex(ValueError, "Truncated"):
                list(validation.packed_records(path))

    def test_wrong_token_frame_count_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            token = Path(rows[0]['prepared_audio']).with_suffix('.npy')
            np.save(token, np.zeros((10, 1), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "Token length"):
                validation.validate_packed(root)

    def test_merge_reads_actual_alpha_not_upstream_named_default(self):
        path = Path(__file__).parents[1] / 'resolved_config.yaml'
        base, lora = merge_model.lora_settings(path)
        self.assertTrue(base.is_absolute())
        self.assertEqual(lora['r'], 32)
        self.assertEqual(lora['lora_alpha'], 32)
        self.assertEqual(lora['lora_dropout'], 0)


if __name__ == '__main__':
    unittest.main()
