import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import identity
import review


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        self.patches = [patch.object(identity, 'DATA', self.data),
                        patch.object(identity, 'SELECTIONS', self.data/'selections.json')]
        for p in self.patches:
            p.start()
        self.row = next(iter(identity.ROWS.values()))

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def add(self, start=0, end=5, role='enrollment'):
        return identity.add_selection({'clip_id': self.row['clip_id'], 'start': start,
                                       'end': end, 'role': role})

    def test_duration_accumulates_and_overlap_cannot_inflate_total(self):
        self.add()
        self.add(5, 10)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.add(1, 4)
        self.assertEqual(identity.state()['totals']['enrollment'], 10)
        self.assertEqual(len(identity.selections()), 2)

    def test_validation_cannot_reuse_enrollment(self):
        self.add()
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.add(0, 5, 'wesam_validation')
        self.add(5, 8, 'other_validation')
        self.assertEqual(identity.state()['totals']['other_validation'], 3)

    def test_invalid_excerpt_limits(self):
        for a, b in [(0, 1), (-1, 5), (0, 40), (float('nan'), 4)]:
            with self.assertRaises(ValueError):
                self.add(a, b)

    def test_neutral_without_calibration_or_for_overlap(self):
        bands = {'wesam_min': .7, 'other_max': .3}
        self.assertEqual(identity.classify(.99, None), 'uncertain')
        self.assertEqual(identity.classify(.99, bands, overlap=True), 'uncertain')
        self.assertEqual(identity.classify(.99, bands, short=True), 'uncertain')
        self.assertEqual(identity.classify(.8, bands), 'wesam')
        self.assertEqual(identity.classify(.2, bands), 'other_speaker')
        self.assertEqual(identity.classify(.5, bands), 'uncertain')

    def test_calibration_rejects_overlapping_distributions(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            identity.calibrate_scores([.5, .6, .7], [.4, .6, .8])
        bands = identity.calibrate_scores([.8, .85, .9], [.1, .2, .3])
        self.assertLess(bands['other_max'], bands['wesam_min'])

    def test_windows_do_not_skip_tail_or_extend_speech(self):
        windows = identity.windows(10, 20)
        self.assertEqual(windows[0], (10., 13.))
        self.assertEqual(windows[-1], (17., 20.))
        self.assertTrue(all(10 <= a < b <= 20 for a, b in windows))
        self.assertEqual(identity.windows(0, 1), [])

    def test_reference_build_and_changed_selections(self):
        rows = list(identity.ROWS.values())[:6]
        for r in rows:
            identity.add_selection({'clip_id': r['clip_id'], 'start': 0, 'end': 5, 'role': 'enrollment'})
        model = lambda audio: np.array([1., 0.], dtype=np.float32)
        profile = identity.build_profile(encoder=model)
        self.assertAlmostEqual(profile['reference_seconds'], 30.)
        self.assertEqual(profile['embedding'], [1., 0.])
        self.assertIsNone(profile['calibration'])
        self.assertAlmostEqual(sf.info(self.data/'wesam_reference.wav').duration, 30.)
        identity.load_profile('wespeaker')
        identity.remove_selection(identity.selections()[0]['id'])
        with self.assertRaisesRegex(ValueError, 'Selections changed'):
            identity.load_profile('wespeaker')

    def test_profile_requires_enough_reference(self):
        self.add()
        with self.assertRaisesRegex(ValueError, 'minimum 25'):
            identity.build_profile(encoder=lambda a: np.array([1., 0.]))

    def test_bounded_source_runner_flags_short_and_overlap(self):
        profile = {'embedding': [1., 0.], 'calibration': {'wesam_min': .8, 'other_max': .2}}
        detector = lambda audio: [{'start': 0., 'end': 4., 'speaker': 'A'},
                                 {'start': 2., 'end': 3., 'speaker': 'B'},
                                 {'start': 7., 'end': 8., 'speaker': 'A'}]
        model = lambda audio: np.array([1., 0.])
        result = identity.identify_source(self.row['source_id'], vad='pyannote', profile=profile,
                     encoder=model, detector=detector, limit_seconds=10, min_speech_seconds=1.5)
        self.assertEqual(result['processed_seconds'], 10)
        self.assertTrue(all(w['label'] == 'uncertain' for w in result['windows']))
        self.assertTrue(any(w['overlap'] for w in result['windows']))
        self.assertTrue(any(w['short_speech'] for w in result['windows']))
        self.assertTrue(all(w['no_score_reason'] for w in result['windows']))
        self.assertTrue(any(w['no_score_reason'] == 'short_speech' for w in result['windows']))
        self.assertTrue(any(w['no_score_reason'] == 'overlap' for w in result['windows']))

    def test_half_second_minimum_scores_short_speech_as_estimate(self):
        profile = {'embedding': [1., 0.], 'calibration': {'wesam_min': .8, 'other_max': .2}}
        detector = lambda audio: [{'start': 0., 'end': .75, 'speaker': None},
                                 {'start': 2., 'end': 2.25, 'speaker': None}]
        result = identity.identify_source(self.row['source_id'], profile=profile,
                     encoder=lambda a: np.array([1., 0.]), detector=detector, limit_seconds=4)
        first, second = result['windows']
        self.assertEqual(first['cosine_similarity'], 1.)
        self.assertTrue(first['low_evidence'])
        self.assertFalse(first['short_speech'])
        self.assertEqual(first['label'], 'uncertain')
        self.assertIsNone(second['cosine_similarity'])
        self.assertEqual(second['no_score_reason'], 'short_speech')
        self.assertEqual(result['min_speech_seconds'], .5)

    def test_window_length_and_hop_are_independent_of_minimum(self):
        profile = {'embedding': [1., 0.], 'calibration': None}
        result = identity.identify_source(self.row['source_id'], profile=profile,
            encoder=lambda a: np.array([1., 0.]),
            detector=lambda a: [{'start': 0., 'end': 4., 'speaker': None}],
            limit_seconds=4, min_speech_seconds=.5, window_seconds=1., hop_seconds=.5)
        self.assertEqual(len(result['windows']), 7)
        self.assertTrue(all(abs(w['end']-w['start']-1.)<1e-6 for w in result['windows']))
        for values in [(0, 3, 1.5), (.5, 3, 4), (.5, float('nan'), 1), (1, .5, .25)]:
            with self.assertRaises(ValueError):
                identity.validate_timing(*values)

    def test_short_embedding_error_preserves_other_predictions(self):
        def encoder(audio):
            if len(audio) < 16000:
                raise RuntimeError('Model cannot embed this short signal')
            return np.array([1., 0.])
        result = identity.identify_source(self.row['source_id'],
            profile={'embedding': [1., 0.], 'calibration': None}, encoder=encoder,
            detector=lambda a: [{'start': 0., 'end': .5, 'speaker': None},
                                {'start': 1., 'end': 3., 'speaker': None}], limit_seconds=4)
        self.assertEqual(result['windows'][0]['no_score_reason'], 'embedding_failed')
        self.assertIsNone(result['windows'][0]['cosine_similarity'])
        self.assertEqual(result['windows'][1]['cosine_similarity'], 1.)

    def test_post_enrollment_validation_and_origin(self):
        handler = object.__new__(review.Handler)
        payload = json.dumps({'clip_id': self.row['clip_id'], 'start': 0, 'end': 5}).encode()
        handler.path = '/api/reference'
        handler.headers = {'Content-Length': str(len(payload)), 'Host': 'localhost:3000'}
        handler.rfile = io.BytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda status: setattr(handler, 'status', status)
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        with patch.object(review, 'IDENTITY_JOB', {'state': 'idle'}):
            handler.do_POST()
        self.assertEqual(handler.status, 200)
        self.assertEqual(json.loads(handler.wfile.getvalue())['totals']['enrollment'], 5)
        handler.headers['Origin'] = 'https://different.example'
        handler.wfile = io.BytesIO()
        handler.do_POST()
        self.assertEqual(handler.status, 403)


if __name__ == '__main__':
    unittest.main()
