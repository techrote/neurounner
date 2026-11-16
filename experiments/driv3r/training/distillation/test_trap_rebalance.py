"""D004 source-only adversarial tests for legal synthetic game-policy curriculum."""
import unittest
import numpy as np
from experiments.driv3r.training.distillation.lab import ACTIONS, STYLES, Game
from experiments.driv3r.training.distillation.repair import adverse
from experiments.driv3r.training.distillation.trap_rebalance import TRAIN, SELECT, TEST, masked_softmax, novelty_scores, targets, rollout, selection_score

class TestTrapRebalance(unittest.TestCase):
    def test_seed_splits(self):
        assert not (set(TRAIN)&set(SELECT) or set(TRAIN)&set(TEST) or set(SELECT)&set(TEST))
    def test_missing_legals_rejected(self):
        with self.assertRaises(ValueError):
            masked_softmax(np.zeros((1,6)),np.zeros((1,6),bool))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            masked_softmax(np.full((1,6),np.nan),np.ones((1,6),bool))
    def test_legal_mask_zeroes_excluded(self):
        legal=np.array([[True,False,True,False,False,True]],bool)
        p=masked_softmax(np.ones((1,6),np.float32),legal)
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertEqual(float(p[0,1]+p[0,3]+p[0,4]),0.)
    def test_teacher_mixture_normalises(self):
        p=np.array([[.25,.25,.25,.25,0,0],[0,0,0,0,.6,.4]],np.float32)
        target=targets(p,p,p,np.array([1,2],np.uint8),.30)
        self.assertTrue(np.allclose(target.sum(1),1))
        self.assertGreaterEqual(float(target.min()),0.)
    def test_invalid_mixture_rejected(self):
        p=np.ones((1,6),np.float32)/6.
        with self.assertRaises(ValueError): targets(p,p,p,np.zeros(1,np.uint8),1.1)
    def test_obstacle_and_protected_legality(self):
        game=Game(91,'branchwitch')
        for _ in range(70):
            adverse(game,'trap');ob=game.obs();score=novelty_scores(game)
            for i,a in enumerate(ACTIONS):
                if a not in ob['legal']: self.assertLess(score[i],-100)
            action=next(a for a in ('BRANCH','LEFT','STRAIGHT','RIGHT','RETRACE','STOP') if a in ob['legal'])
            game.step(action)
            self.assertNotIn(game.pos,game.protected)
    def test_selection_rejects_branch_collapse(self):
        base={c:{'cells':60,'branches':20,'decisions':80} for c in ('trap','normal','squeeze')}
        restored={c:{'cells':65,'branches':22,'decisions':80} for c in base}
        safe={c:{'cells':68,'branches':21,'decisions':80} for c in base}
        collapsed={c:{'cells':68,'branches':4,'decisions':80} for c in base}
        self.assertGreater(selection_score(safe,restored,base)['score'],selection_score(collapsed,restored,base)['score'])
    def test_flat_control_deterministic(self):
        a,ra=rollout(3011,STYLES[2],'trap','flat_continue',{},budget=40)
        b,rb=rollout(3011,STYLES[2],'trap','flat_continue',{},budget=40)
        self.assertEqual(ra,rb);self.assertEqual(a.painted,b.painted);self.assertEqual(a.invalid,0)

if __name__=='__main__':unittest.main()
