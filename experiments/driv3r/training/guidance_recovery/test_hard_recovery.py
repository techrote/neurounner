"""Safety, provenance, and falsifiable curriculum checks for hard recovery."""
import unittest
import numpy as np
from experiment import Candidate, ARMS
from hard_recovery import (trap_world, false_cue_world, build_curriculum,
                           target_distribution, paired_cue_response, summarize)

class HardRecoveryTests(unittest.TestCase):
    def test_seed_rotation_and_handedness_deterministic(self):
        for rot in range(4):
            for mirror in (False,True):
                a,j,route=trap_world(110,1,3,rot,mirror)
                b,k,route2=trap_world(110,1,3,rot,mirror)
                self.assertEqual((a.pos,a.goal,a.protected,j,route),(b.pos,b.goal,b.protected,k,route2))

    def test_all_strong_hints_have_identical_legality(self):
        a,j,route=trap_world(111,-1,3,1,True)
        self.assertEqual([[v.legal for v in a.candidates(s)] for s in ARMS],
                         [[v.legal for v in a.candidates(0)]]*3)

    def test_trap_requires_backtracking_not_illegal_shortcuts(self):
        a,j,route=trap_world(112,1,4,0,False)
        self.assertNotEqual(a.pos,j)
        self.assertEqual(len(a.history),4)
        before=a.protected.copy()
        self.assertFalse(a.apply(Candidate(('INK',9,0),(13,7),True,[])))
        self.assertEqual(a.protected,before)
        self.assertEqual(a.invalid_attempts,1)
        self.assertEqual(a.inked,set())

    def test_backtrack_cannot_rollback_ink(self):
        a,j,route=trap_world(113,-1,2,2,False)
        a.inked.add(a.pos)
        before=a.inked.copy()
        for _ in range(2):
            choice=next(v for v in a.candidates(0) if v.action[0]=='BACKTRACK')
            self.assertTrue(a.apply(choice))
        self.assertEqual(a.pos,j)
        self.assertFalse(a.history)
        self.assertEqual(a.inked,before)

    def test_open_false_cue_does_not_change_material_access(self):
        w=false_cue_world(121,1)
        before=[(c.action,c.legal) for c in w.candidates(.35)]
        w.issue_recovery_cue(3)
        self.assertEqual(before,[(c.action,c.legal) for c in w.candidates(.35)])

    def test_curriculum_reproducible_and_probabilities_valid(self):
        x,y,groups=build_curriculum(instances=12)
        xx,yy,gg=build_curriculum(instances=12)
        self.assertTrue(np.array_equal(x,xx))
        self.assertTrue(np.array_equal(y,yy))
        self.assertEqual(groups,gg)
        self.assertEqual(x.shape[1:],(10,25))
        self.assertTrue(np.allclose(y.sum(axis=1),1))
        self.assertTrue(np.isfinite(x).all())

    def test_holdout_scenarios_not_in_training(self):
        x,y,groups=build_curriculum(instances=24)
        # Each trapped prefix in the generator uses depth 2 or 3, never 4.
        self.assertFalse(any(kind=='trap' and t>=3 for kind,i,t,arm,p in groups))

    def test_teacher_legal_mask_strict(self):
        w,j,route=trap_world(125,1,3,0,False)
        candidates=w.candidates(1)
        i=next(i for i,c in enumerate(candidates) if c.action[0]=='BACKTRACK')
        dist=target_distribution(w,candidates,i)
        self.assertAlmostEqual(float(dist.sum()),1,places=5)
        for j,c in enumerate(candidates):
            if not c.legal:self.assertEqual(float(dist[j]),0.)

    def test_pairing_counts_matched(self):
        tiny=[{'arm':0.0,'cue':False,'seed':1,'personality':1,'depth':2,'rotation':0,'mirror':False,'clean_backtrack':None},
              {'arm':0.0,'cue':True,'seed':1,'personality':1,'depth':2,'rotation':0,'mirror':False,'clean_backtrack':2}]
        res=paired_cue_response(tiny)['0.00']
        self.assertEqual((res['pairs'],res['cue_only_successes'],res['uncued_only_successes']),(1,1,0))

if __name__=='__main__':unittest.main()
