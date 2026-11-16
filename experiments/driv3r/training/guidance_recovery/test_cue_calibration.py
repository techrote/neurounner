"""Offline controls: paired cue states and held-out worm decisions."""
import unittest
import numpy as np
import torch

from experiment import ARMS, Scorer
from hard_recovery import trap_world, false_cue_world
from cue_calibration import paired_curriculum, train, matched_first_step


class CueCalibrationTests(unittest.TestCase):
    def test_deterministic_paired_curriculum(self):
        a,b,labels=paired_curriculum(seed=44,instances=6)
        c,d,labels2=paired_curriculum(seed=44,instances=6)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
        self.assertEqual(labels,labels2)

    def test_each_training_example_has_ten_candidates_25_features(self):
        x,y,_=paired_curriculum(instances=3)
        self.assertEqual(x.shape[1:],(10,25))
        self.assertEqual(y.shape[1:],(10,))
        np.testing.assert_allclose(y.sum(axis=1),1,rtol=0,atol=1e-6)

    def test_labels_contain_each_of_six_cue_strength_pairs(self):
        _,_,labels=paired_curriculum(instances=12)
        pairs={(label[2],label[-1] if label[0]=='false_cue' else label[4]) for label in labels if label[0]=='trap'}
        self.assertEqual(pairs,{(cue,arm) for cue in (False,True) for arm in ARMS})

    def test_cue_never_changes_world_or_legality(self):
        for arm in ARMS:
            a,_,_=trap_world(901,-1,depth=4,rotation=3,mirror=True)
            b,_,_=trap_world(901,-1,depth=4,rotation=3,mirror=True)
            b.issue_recovery_cue(5)
            left=a.candidates(arm);right=b.candidates(arm)
            self.assertEqual([(c.action,c.target,c.legal) for c in left],[(c.action,c.target,c.legal) for c in right])
            self.assertEqual(a.inked,b.inked)
            self.assertEqual(a.protected,b.protected)

    def test_false_cue_does_not_grant_any_permissions(self):
        a=false_cue_world(221,1)
        b=false_cue_world(221,1);b.issue_recovery_cue(7)
        for arm in ARMS:
            self.assertEqual([c.legal for c in a.candidates(arm)],[c.legal for c in b.candidates(arm)])
        self.assertEqual(a.material_at((6,7)),b.material_at((6,7)))

    def test_counterfactual_pair_features_differ_but_materials_do_not(self):
        a,_,_=trap_world(18,1,3,0,False)
        b,_,_=trap_world(18,1,3,0,False)
        b.issue_recovery_cue(3)
        ca=a.candidates(1.0);cb=b.candidates(1.0)
        self.assertEqual([c.target for c in ca],[c.target for c in cb])
        self.assertNotEqual([c.features for c in ca],[c.features for c in cb])
        self.assertEqual(a.inked,b.inked)

    def test_out_of_distribution_first_step_both_cues(self):
        torch.manual_seed(7)
        stats=matched_first_step(Scorer(),seeds=2)
        for arm in ARMS:
            self.assertEqual(stats[f'{arm:.2f}']['pairs'],2)
            self.assertLessEqual(stats[f'{arm:.2f}']['false_cued_backtrack'],2)

    def test_tiny_optimizer_run_deterministically_repeats(self):
        x,y,_=paired_curriculum(seed=12,instances=3)
        torch.manual_seed(14);init=Scorer()
        a,steps1,loss1,n1=train(init,x,y,epochs=1,style_episodes=2)
        b,steps2,loss2,n2=train(init,x,y,epochs=1,style_episodes=2)
        self.assertGreater(steps1,0)
        self.assertEqual((steps1,loss1,n1),(steps2,loss2,n2))
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key],b.state_dict()[key]))


if __name__=='__main__': unittest.main()
