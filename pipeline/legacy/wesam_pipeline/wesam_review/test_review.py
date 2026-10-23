import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review


class SpeakerReviewTests(unittest.TestCase):
    def test_change_inside_clip(self):
        turns = [{'start': 0, 'end': 10, 'speaker': 'A'},
                 {'start': 10, 'end': 20, 'speaker': 'B'}]
        result = review.summarize(turns, 5, 15)
        self.assertEqual(result['speaker_count'], 2)
        self.assertEqual(result['overlap_seconds'], 0)
        self.assertEqual(result['segments'][1]['start'], 5)

    def test_context_speaker_outside_clip_is_not_counted(self):
        turns = [{'start': 0, 'end': 5, 'speaker': 'A'},
                 {'start': 5, 'end': 20, 'speaker': 'B'}]
        self.assertEqual(review.summarize(turns, 8, 15)['speaker_count'], 1)

    def test_overlaps_use_union_duration(self):
        turns = [{'start': 0, 'end': 10, 'speaker': 'A'},
                 {'start': 2, 'end': 7, 'speaker': 'B'},
                 {'start': 4, 'end': 8, 'speaker': 'C'}]
        self.assertEqual(review.summarize(turns, 0, 10)['overlap_seconds'], 6)

    def test_brief_label_requires_review_but_not_counted(self):
        turns = [{'start': 0, 'end': 10, 'speaker': 'A'},
                 {'start': 5, 'end': 5.1, 'speaker': 'B'}]
        result = review.summarize(turns, 0, 10)
        self.assertEqual(result['speaker_count'], 1)
        self.assertEqual(len(result['segments']), 2)

    def request(self, path, headers=None):
        handler = object.__new__(review.Handler)
        handler.path = path
        handler.headers = headers or {}
        handler.wfile = io.BytesIO()
        handler.send_response = lambda status: setattr(handler, 'response_status', status)
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.do_GET()
        return handler.response_status, handler.wfile.getvalue()

    def test_inventory_and_unknown_audio(self):
        status, body = self.request('/api/episodes')
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)), 23)
        self.assertEqual(self.request('/audio?id=../../refrence.txt')[0], 404)

    def test_real_audio_range(self):
        clip_id = next(iter(review.ROWS))
        path = review.wav_path(clip_id)
        status, data = self.request('/audio?id='+clip_id, {'Range': 'bytes=0-43'})
        self.assertEqual(status, 206)
        self.assertEqual(len(data), 44)
        self.assertEqual(data[:4], b'RIFF')
        self.assertGreater(path.stat().st_size, 44)

    def test_refined_dataset_audio_is_manifest_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'sample.wav'
            audio.write_bytes(b'RIFF' + b'\0' * 96)
            (root / 'manifest.jsonl').write_text(json.dumps({'clip_id': 'sample', 'audio': str(audio)}) + '\n')
            with patch.object(review, 'DATASET', root), patch.object(review, 'DATASET_MTIME', None), patch.object(review, 'DATASET_ROWS', {}):
                status, body = self.request('/dataset-audio?id=sample', {'Range': 'bytes=0-3'})
                self.assertEqual((status, body), (206, b'RIFF'))
                self.assertEqual(self.request('/dataset-audio?id=../../refrence.txt')[0], 404)

    def test_dataset_manifest_cannot_serve_outside_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.jsonl').write_text(json.dumps({'clip_id': 'bad', 'audio': '/etc/passwd'}) + '\n')
            with patch.object(review, 'DATASET', root), patch.object(review, 'DATASET_MTIME', None), patch.object(review, 'DATASET_ROWS', {}):
                self.assertEqual(self.request('/dataset-audio?id=bad')[0], 404)


if __name__ == '__main__':
    unittest.main()
