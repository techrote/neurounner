import unittest
from experiment import World, Candidate, make_data, Scorer, policy_index, FAMILIES_FINAL, ARMS, blocked
import numpy as np

class ContractTests(unittest.TestCase):
    def test_seeded_geometry(self):
        self.assertEqual(blocked(123,"switchback"),blocked(123,"switchback"))
        self.assertNotEqual(blocked(123,"islands"),blocked(124,"islands"))
    def test_no_illegal_native_write_even_if_forged(self):
        w=World(2,"bars",1)
        initial=w.pos
        self.assertFalse(w.apply(Candidate(("INK",9,9),(10,10),True,[])))
        self.assertEqual(w.pos,initial)
        self.assertEqual(w.inked,set())
    def test_world_permits_only_legal_catalog(self):
        for family in FAMILIES_FINAL:
            w=World(22,family,-1)
            for c in w.candidates(1.):
                if c.action[0] == "STOP":self.assertTrue(c.legal)
                if c.action[0] == "BACKTRACK":self.assertFalse(c.legal)
    def test_arms_same_source_and_candidate_legality(self):
        w=World(123,"forks",1)
        views=[w.candidates(a) for a in ARMS]
        self.assertEqual([[v.action for v in c if v.legal] for c in views], [[v.action for v in views[0] if v.legal]]*3)
        self.assertEqual(views[0][0].features[11],0.)
    def test_matched_hint_does_not_change_permissions(self):
        w=World(100,"switchback",1)
        a=w.candidates(0.)
        b=w.candidates(1.)
        self.assertEqual([v.legal for v in a],[v.legal for v in b])
        self.assertEqual([v.target for v in a],[v.target for v in b])
    def test_train_data_shapes_and_finite(self):
        x,y,groups=make_data(episodes=2,horizon=12)
        self.assertEqual(x.shape[1:],(10,25))
        self.assertEqual(x.shape[0],y.shape[0])
        self.assertEqual(len(groups),len(x))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.allclose(y.sum(axis=1),1))
    def test_recovery_is_material_preserving(self):
        w=World(2,"bars",1)
        c=w.candidates(0)
        move=next(x for x in c if x.legal and x.action[0]=="INK")
        self.assertTrue(w.apply(move))
        before=set(w.inked)
        c=w.candidates(1)
        back=next(x for x in c if x.action[0]=="BACKTRACK")
        self.assertTrue(w.apply(back))
        self.assertEqual(w.inked,before)
    def test_u16_hint_bounds_and_determinism(self):
        w=World(7,"bars",1)
        self.assertEqual(w.hint_u16((6,4)),w.hint_u16((6,4)))
        self.assertGreaterEqual(w.hint_u16((6,4)),0)
        self.assertLessEqual(w.hint_u16((6,4)),65535)
    def test_forbidden_material_is_host_immutable(self):
        w=World(7,"bars",1)
        target=next(iter(w.protected))
        self.assertIn(w.material_at(target),("FORBIDDEN_GLASS","SOLID_ROCK"))
        oldmaterial=w.material_at(target)
        forged=Candidate(("INK",target[0]-w.pos[0],target[1]-w.pos[1]),target,True,[])
        self.assertFalse(w.apply(forged))
        self.assertEqual(w.material_at(target),oldmaterial)
    def test_host_cue_not_a_material_permission(self):
        w=World(7,"bars",-1)
        before=w.candidates(0.35)
        w.issue_recovery_cue(3)
        after=w.candidates(0.35)
        self.assertEqual([v.legal for v in before],[v.legal for v in after])
        self.assertEqual([v.target for v in before],[v.target for v in after])
        self.assertEqual(w.inked,set())
    def test_reject_diagonal_and_long_step(self):
        w=World(7,"islands",1)
        self.assertFalse(w.apply(Candidate(("LIFT",1,1),(2,2),True,[])))
        self.assertFalse(w.apply(Candidate(("INK",0,4),(1,5),True,[])))
        self.assertEqual(w.inked,set())
        self.assertEqual(w.pos,(1,1))
    def test_training_and_eval_topologies_disjoint(self):
        from experiment import FAMILIES_TRAIN
        self.assertFalse(set(FAMILIES_TRAIN)&set(FAMILIES_FINAL))

if __name__=="__main__":unittest.main()
