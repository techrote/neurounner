"""Focused boundary and host-legality regressions for multi-topology TRAIN v3."""
import json
from pathlib import Path
import tempfile
import unittest

import domain_recovery_v3 as d
from recovery_curriculum import legal_state, oracle, distance_field


class CorpusMultiTopologyTests(unittest.TestCase):
    def test_new_slalom_deterministic_and_reachable(self):
        for seed in (9100, 9101, 9115):
            a,b=d.make_slalom_world(seed),d.make_slalom_world(seed)
            self.assertEqual(d.world_digest(a), d.world_digest(b))
            self.assertEqual(a['generator'],'vertical_slalom_v3')
            self.assertNotIn(a['start'],a['protected']|a['obstacles'])
            self.assertNotIn(a['goal'],a['protected']|a['obstacles'])
            cost=distance_field(a)[(*a['start'],0)]
            self.assertLessEqual(cost,192)
            self.assertGreaterEqual(cost,a['shortest_goal'])

    def test_world_family_and_generator_separation(self):
        a=d.make_world(1000);b=d.make_cross_world(7300);c=d.make_slalom_world(9100)
        self.assertEqual(d.validate_partitions({'TRAIN':[a,b],'EVAL':[c]}),
                         {'TRAIN':2,'EVAL':1})
        with self.assertRaisesRegex(ValueError,'reuse'):
            d.validate_partitions({'TRAIN':[a],'EVAL':[d.make_world(1000)]})
        with self.assertRaisesRegex(ValueError,'reuse'):
            d.validate_partitions({'TRAIN':[a],'EVAL':[dict(a,seed=90001)]})

    def test_seed_overlap_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'split seed overlap'):
                d.run(Path(t)/'not_created',source_seeds=(1000,),
                    original_train=(1000,),barrier_train=(7300,),
                    original_eval=(4100,),barrier_eval=(6100,),
                    slalom_eval=(9100,),budget=48)
            self.assertFalse((Path(t)/'not_created').exists())

    def test_teacher_action_is_host_legal_on_adversarial_slalom_states(self):
        for seed in (9100,9101):
            w=d.make_slalom_world(seed);dist=distance_field(w)
            pos=w['start'];head=0;visited={pos};history=[]
            for _ in range(64):
                _,_,legal=legal_state(w,pos,head,visited,history)
                action,negative=oracle(w,pos,head,visited,history,dist,legal)
                self.assertIn(action,legal)
                self.assertNotIn(action,negative)
                if action=='STOP':break
                pos,head=d.apply(w,pos,head,visited,history,action)
                self.assertNotIn(pos,w['obstacles']|w['protected'])
                visited.add(pos)

    def test_export_rejects_unknown_training_seed(self):
        w=d.make_world(2300)
        moves,fresh,legal=legal_state(w,w['start'],0,{w['start']},[])
        obs=d.observation(w,w['start'],0,{w['start']},[],legal,0,32,'greebles')
        normalized=json.dumps({k:v for k,v in obs.items() if k!='family'},
                              sort_keys=True,separators=(',',':'))
        row={'observation':obs,'label':'STOP','seed':4100,
             'world_group_id':'original_v1:4100',
             'observable_sha256':d.sha(normalized.encode()),
             'student_action':'STOP','negative_actions':{},
             'was_student_teacher_disagreement':False,
             'evidence_class':'synthetic-train-host-applied-multi-topology-correction'}
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'non-TRAIN seed'):
                d.export_corrections([row],[{'seed':2300}],Path(t)/'train.zip')


if __name__=='__main__':unittest.main()
