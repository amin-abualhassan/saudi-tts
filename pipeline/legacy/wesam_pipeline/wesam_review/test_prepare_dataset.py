import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

import prepare_dataset as data


class DatasetTests(unittest.TestCase):
    def mapping(self):
        return {'A':{'speaker':'wesam_ali','cosine_similarity':.8},
                'B':{'speaker':'guest_ep_speaker_01','cosine_similarity':.2}}

    def cue(self,a,b,text='نص عربي',index=1):
        return {'start':a,'end':b,'text':text,'cue_id':index}

    def test_whole_mixed_cue_is_rejected(self):
        turns=[{'start':0,'end':4,'speaker':'A'},{'start':4,'end':8,'speaker':'B'}]
        accepted,rejected=data.plan_clips([self.cue(0,8)],turns,self.mapping())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'],'speaker_change_within_cue')

    def test_overlap_is_not_exported(self):
        turns=[{'start':0,'end':8,'speaker':'A'},{'start':3,'end':4,'speaker':'B'}]
        accepted,rejected=data.plan_clips([self.cue(0,8)],turns,self.mapping())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'],'overlapping_speakers')
        self.assertEqual(data.clean_spans(turns,'A'),[(0,3),(4,8)])

    def test_merge_complete_cues_preserves_contiguous_audio_and_text(self):
        turns=[{'start':0,'end':5,'speaker':'A'}]
        cues=[self.cue(0,2,'اول',1),self.cue(2.2,4.5,'ثاني',2)]
        accepted,rejected=data.plan_clips(cues,turns,self.mapping())
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]['text'],'اول ثاني')
        self.assertEqual(accepted[0]['cue_ids'],[1,2])
        self.assertEqual(accepted[0]['end'],4.5)
        self.assertFalse(rejected)

    def test_other_speaker_in_gap_blocks_join(self):
        turns=[{'start':0,'end':2,'speaker':'A'},{'start':2.05,'end':2.1,'speaker':'B'},
               {'start':2.2,'end':4.5,'speaker':'A'}]
        accepted,rejected=data.plan_clips([self.cue(0,2,index=1),self.cue(2.2,4.5,index=2)],turns,self.mapping())
        self.assertFalse(accepted)
        self.assertEqual(len(rejected),2)

    def test_less_than_three_seconds_of_speech_rejected(self):
        turns=[{'start':0,'end':2,'speaker':'A'}]
        accepted,rejected=data.plan_clips([self.cue(0,3)],turns,self.mapping())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'],'less_than_3_seconds_single_speaker_speech')

    def test_unstable_identity_cluster_not_assigned(self):
        turns=[{'start':0,'end':12,'speaker':'A'}]
        calls=iter([np.array([1.,0.]),np.array([0.,1.]),np.array([1.,0.]),np.array([0.,1.])])
        result=data.speaker_identities(turns,np.ones(12*16000,dtype=np.float32),lambda _:next(calls),
                                      {'embedding':[1.,0.]},'ep',.5)
        self.assertIsNone(result['A']['speaker'])
        self.assertEqual(result['A']['reason'],'inconsistent_identity_scores')

    def test_guest_ids_separate_across_episodes(self):
        turns=[{'start':0,'end':12,'speaker':'B'}]
        encoder=lambda a:np.array([0.,1.])
        args=(turns,np.ones(12*16000,dtype=np.float32),encoder,{'embedding':[1.,0.]})
        a=data.speaker_identities(*args,'episode_a',.5)
        b=data.speaker_identities(*args,'episode_b',.5)
        self.assertNotEqual(a['B']['speaker'],b['B']['speaker'])

    def test_recording_splits_are_deterministic_and_disjoint(self):
        sources=[f'ep{i}' for i in range(23)]
        splits=data.episode_splits(sources)
        self.assertEqual(splits,data.episode_splits(list(reversed(sources))))
        self.assertEqual(set(splits.values()),{'train','validation','test'})
        self.assertEqual(len(splits),23)

    def test_export_audio_matches_manifest_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.wav'
            audio=.1*np.sin(np.arange(6*16000,dtype=np.float32)*.1)
            sf.write(source,audio,16000,subtype='PCM_16')
            rows=[{'start':1.,'end':5.,'speaker':'wesam_ali','text':'نص كامل','cue_ids':[1],
                   'local_speaker':'A','speech_seconds':4.,'identity_score':.8}]
            records=data.export_clips(source,'ep',rows,root,'train',24000,'profile')
            self.assertEqual(len(records),1)
            info=sf.info(records[0]['audio'])
            self.assertEqual(info.samplerate,24000)
            self.assertEqual(info.channels,1)
            self.assertAlmostEqual(info.duration,records[0]['duration'])
            self.assertEqual(Path(records[0]['audio']).with_suffix('.lab').read_text().strip(),'نص كامل')


if __name__=='__main__':
    unittest.main()
