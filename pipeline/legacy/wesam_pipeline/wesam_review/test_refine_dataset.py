from pathlib import Path
import tempfile
import unittest

import refine_dataset as refine


class RefinementTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'captions.vtt'
            path.write_text('WEBVTT\n\n' + content)
            return refine.vtt_words(path)

    def test_rolling_history_is_not_repeated(self):
        words, _ = self.parse('00:00:00.000 --> 00:00:02.000\n'
                             'اهلا<00:00:01.000><c> وسهلا</c>\n\n'
                             '00:00:02.000 --> 00:00:02.010\nاهلا وسهلا\n\n'
                             '00:00:02.010 --> 00:00:04.000\nاهلا وسهلا\n'
                             'فيكم<00:00:03.000><c> اليوم</c>\n')
        self.assertEqual([w['text'] for w in words], ['اهلا', 'وسهلا', 'فيكم', 'اليوم'])

    def test_shared_onset_keeps_words_together(self):
        words, _ = self.parse('00:00:00.000 --> 00:00:02.000\n'
                             'ان<00:00:00.000><c> شاء</c><00:00:01.000><c> الله</c>\n')
        self.assertEqual(words[0]['text'], 'ان شاء')
        self.assertEqual(words[0]['end'], 1.)

    def test_end_is_bounded_by_own_caption(self):
        words, _ = self.parse('00:00:00.000 --> 00:00:01.000\nكلمة\n\n'
                             '00:00:04.000 --> 00:00:05.000\nثانية\n')
        self.assertEqual(words[0]['end'], 1.)

    def words(self, count):
        return [{'start': i * .5, 'end': (i + 1) * .5, 'text': f'كلمة{i}',
                 'word_id': i + 1, 'cue_ids': [1], 'end_provenance': 'caption'} for i in range(count)]

    def mapping(self):
        return {'A': {'speaker': 'wesam_ali'}, 'B': {'speaker': 'guest_ep_b'}}

    def test_recovers_two_speakers_inside_same_caption(self):
        turns = [{'start': 0., 'end': 4., 'speaker': 'A'}, {'start': 4., 'end': 8., 'speaker': 'B'}]
        accepted, _ = refine.plan_word_clips(self.words(16), turns, self.mapping())
        self.assertEqual([r['speaker'] for r in accepted], ['wesam_ali', 'guest_ep_b'])
        self.assertEqual(accepted[1]['word_timestamps'][0]['start'], 0.)
        self.assertNotIn('كلمة8', accepted[0]['text'])

    def test_straddling_word_is_not_attached_to_either_voice(self):
        turns = [{'start': 0., 'end': 3.75, 'speaker': 'A'}, {'start': 3.75, 'end': 8., 'speaker': 'B'}]
        accepted, rejected = refine.plan_word_clips(self.words(16), turns, self.mapping())
        self.assertEqual(len(accepted), 2)
        self.assertTrue(any(r['reason'] == 'speaker_change_within_word_interval' for r in rejected))
        self.assertTrue(all('كلمة7' not in r['text'] for r in accepted))

    def test_other_speaker_in_caption_gap_blocks_join(self):
        words = self.words(4) + [{**w, 'start': w['start'] + 2.4, 'end': w['end'] + 2.4} for w in self.words(4)]
        turns = [{'start': 0., 'end': 2., 'speaker': 'A'}, {'start': 2.1, 'end': 2.2, 'speaker': 'B'},
                 {'start': 2.4, 'end': 4.4, 'speaker': 'A'}]
        accepted, _ = refine.plan_word_clips(words, turns, self.mapping())
        self.assertFalse(accepted)

    def test_ambiguous_cluster_remains_excluded(self):
        accepted, rejected = refine.plan_word_clips(self.words(8), [{'start': 0., 'end': 4., 'speaker': 'A'}],
                                                   {'A': {'speaker': None, 'reason': 'inconsistent_identity_scores'}})
        self.assertFalse(accepted)
        self.assertTrue(all(r['reason'] == 'inconsistent_identity_scores' for r in rejected))

    def test_detected_overlap_is_excluded(self):
        turns = [{'start': 0., 'end': 8., 'speaker': 'A'}, {'start': 3., 'end': 4., 'speaker': 'B'}]
        accepted, rejected = refine.plan_word_clips(self.words(16), turns, self.mapping())
        self.assertTrue(any(r['reason'] == 'overlapping_speakers' for r in rejected))
        self.assertTrue(all(r['end'] <= 3. or r['start'] >= 4. for r in accepted))


if __name__ == '__main__':
    unittest.main()
