"""Cheap synthetic-host controls for cue discrimination and its blind spot."""
import copy
import unittest
import numpy as np
import torch

from experiment import ARMS, sha
from hard_recovery import rotate
from positive_correction import warning_world, candidates_fingerprint
from adversarial_cues import (
    decoy_true_world, near_wall_false_world, pair_view, source_hash,
    generate, tune, load_prior, evaluate, summarize,
)


class AdversarialCueTests(unittest.TestCase):
    def test_all_arms_near_wall_forward_legal(self):
        for arm in ARMS:
            for shape in ('straight','corner'):
                w,_,tip,_=near_wall_false_world(63,1,3,1,True,shape)
                self.assertEqual(w.material_at(tip),'AIR')
                self.assertEqual(len([c for c in w.candidates(arm) if c.legal and c.action[0]=='INK' and c.target==tip]),1)

    def test_cue_does_not_change_world_material_or_legal_menu(self):
        for arm in ARMS:
            w=near_wall_false_world(71,-1,4,3,True)[0]
            a,b=pair_view(w,arm)
            self.assertEqual(source_hash(a),source_hash(b))
            self.assertEqual(candidates_fingerprint(a,arm),candidates_fingerprint(b,arm))
            self.assertEqual([(c.action,c.legal) for c in a.candidates(arm)],
                             [(c.action,c.legal) for c in b.candidates(arm)])

    def test_cue_only_changes_observation_not_host_authority(self):
        w=warning_world(91,1,3,0,False)[0]
        a,b=pair_view(w,0)
        ca=a.candidates(0);cb=b.candidates(0)
        self.assertTrue(all(x.legal==y.legal for x,y in zip(ca,cb)))
        self.assertTrue(all(all(x.features[i]==y.features[i] for i in range(25) if i not in (16,23))
                            for x,y in zip(ca,cb)))
        self.assertTrue(any(x.features[23]!=y.features[23] for x,y in zip(ca,cb)))
        self.assertEqual(a.inked,b.inked)

    def test_near_wall_false_reaches_goal_under_host_legal_actions(self):
        for shape in ('straight','corner'):
            for rot in range(4):
                w,j,tip,route=near_wall_false_world(19,1,4,rot,False,shape)
                probe=copy.deepcopy(w)
                for p in route:
                    c=next(z for z in probe.candidates(0) if z.legal and z.action[0]=='INK' and z.target==p)
                    self.assertTrue(probe.apply(c))
                self.assertEqual(probe.pos,probe.goal)
                self.assertEqual(probe.invalid_attempts,0)

    def test_deceptive_deadend_and_valid_path_have_same_features(self):
        for rot in range(4):
            for mirror in (False,True):
                for shape in ('straight','corner'):
                    viable=near_wall_false_world(171,1,4,rot,mirror,shape)[0]
                    decoy=decoy_true_world(171,1,4,rot,mirror,shape)[0]
                    self.assertNotEqual(source_hash(viable),source_hash(decoy))
                    for arm in ARMS:
                        for cue in (False,True):
                            a=copy.deepcopy(viable);b=copy.deepcopy(decoy)
                            if cue:a.issue_recovery_cue(9);b.issue_recovery_cue(9)
                            self.assertEqual(candidates_fingerprint(a,arm),candidates_fingerprint(b,arm))
                            self.assertTrue(np.array_equal(np.asarray([c.features for c in a.candidates(arm)]),
                                                            np.asarray([c.features for c in b.candidates(arm)])))

    def test_hiding_gap_outside_features_changes_world_not_candidate_menu(self):
        a=near_wall_false_world(10,1,4,0,False,'straight')[0]
        b=decoy_true_world(10,1,4,0,False,'straight')[0]
        self.assertNotEqual(sha(sorted(a.protected)),sha(sorted(b.protected)))
        self.assertEqual(candidates_fingerprint(a,0),candidates_fingerprint(b,0))

    def test_out_of_bounds_model_action_cannot_be_host_accepted(self):
        w=near_wall_false_world(128,1,3,0,False)[0]
        c=next(c for c in w.candidates(0) if c.action[0]=='INK' and c.legal)
        forged=copy.deepcopy(c);forged.action=('INK',50,0);forged.target=(w.pos[0]+50,w.pos[1]);forged.legal=True
        self.assertFalse(w.apply(forged))
        self.assertEqual(len(w.inked),0)

    def test_dataset_pair_source_disjoint_and_cue_condition(self):
        x,y,labels,sources=generate(seed=404221,instances=12)
        self.assertEqual(x.shape,(96,10,25))
        self.assertEqual(len(sources),48)
        self.assertEqual(len({(z['source'],z['arm']) for z in sources}),48)
        self.assertEqual(len({z['source'] for z in sources}),48)
        self.assertTrue(all(abs(float(yy.sum())-1)<1e-5 for yy in y))
        self.assertTrue(all(x[i,:,23].min()==int(r['cue']) for i,r in enumerate(labels)))
        # Evaluation seeds live in 800001+offset, never the training streams.
        self.assertFalse({z['seed'] for z in sources}&{800001+i*977 for i in range(32)})

    def test_repeated_cpu_gradient_runs_are_byte_identical(self):
        from experiment import Scorer
        torch.manual_seed(19)
        base=Scorer();base.eval()
        from positive_correction import CueGatedPolicy
        x,y,labels,_=generate(seed=1453,instances=8)
        torch.manual_seed(120)
        a=CueGatedPolicy(base)
        torch.manual_seed(120)
        b=CueGatedPolicy(copy.deepcopy(base))
        a,steps,examples,loss=tune(a,x,y,labels,epochs=2,seed=555)
        b,steps2,examples2,loss2=tune(b,x,y,labels,epochs=2,seed=555)
        self.assertEqual((steps,examples,loss),(steps2,examples2,loss2))
        for key,value in a.adapter.state_dict().items():
            self.assertTrue(torch.equal(value,b.adapter.state_dict()[key]))

    def test_unguided_frozen_base_logits_not_modified_by_adapter_training(self):
        from experiment import Scorer
        from positive_correction import CueGatedPolicy
        torch.manual_seed(42)
        base=Scorer();base.eval()
        original=CueGatedPolicy(base)
        updated=CueGatedPolicy(copy.deepcopy(base))
        x,y,labels,_=generate(seed=1453,instances=8)
        updated,*_=tune(updated,x,y,labels,epochs=2,seed=8)
        w=near_wall_false_world(123,1,4,2)[0]
        for arm in ARMS:
            xx=torch.tensor([c.features for c in w.candidates(arm)],dtype=torch.float32)
            with torch.no_grad():
                self.assertTrue(torch.equal(original(xx),updated(xx)))
                self.assertTrue(torch.equal(original(xx),original.base(xx)))

    def test_matched_heldout_records_include_negative_decoys(self):
        from experiment import Scorer
        from positive_correction import CueGatedPolicy
        model=CueGatedPolicy(Scorer());model.eval()
        records=evaluate(model,seeds=2)
        self.assertEqual(len(records),2*3*6*2)
        metrics=summarize(records)
        self.assertEqual(len(metrics),18)
        self.assertTrue(all(v['paired']==2 for v in metrics.values()))
        self.assertTrue(all(v['raw_illegal_top1']>=0 for v in metrics.values()))


if __name__=='__main__':unittest.main()
