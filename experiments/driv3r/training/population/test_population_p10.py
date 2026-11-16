"""Focused P10 student-distillation guards (synthetic only)."""
import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch=None

from population_lab import NICHES, World, stable_seed
from population_p10 import (ACTIONS, HISTORY, build_model, observe, play_student,
                            scenario, shuffled, source_splits, tensorize)


class TestPopulationP10(unittest.TestCase):
    def test_host_legal_menu(self):
        w=World.generate(34145)
        X, mask, history=observe(w, [])
        self.assertEqual(X.shape,(6,20))
        self.assertEqual(mask.tolist(),[a in w.legal() for a in ACTIONS])
        self.assertTrue(np.all(X[~mask]==0))
        self.assertEqual(len(history),HISTORY)

    def test_synthetic_recovery_and_protected_cells(self):
        w=scenario(321,1,36)
        self.assertTrue(w.painted.isdisjoint(w.obstacles|w.protected))
        self.assertEqual(w.legal().count('STOP'),1)
        self.assertTrue(w.open((4,11)))
        self.assertFalse(w.open((5,11)))
        self.assertEqual(scenario(321,2,36).goal,(19,4))

    def test_train_validation_holdout_nonoverlap(self):
        s=source_splits(643,20,8,10)
        a,b,c=(set(s[k]) for k in ('train','validation','holdout'))
        self.assertFalse(a&b or a&c or b&c)
        self.assertNotEqual(s['holdout'],source_splits(644,20,8,10)['holdout'])

    @unittest.skipIf(torch is None, "P10 optional torch CPU dependency")
    def test_student_mask_never_generates_illegal(self):
        import torch
        torch.manual_seed(7)
        for mode in ('state_only','token_gru','history_sham'):
            p=build_model(mode,16)
            w=scenario(1102,2,20)
            X,M,H=observe(w,[])
            logits=p(torch.tensor(X[None]),torch.tensor(M[None]),torch.tensor(H[None]),torch.tensor([1]))
            self.assertEqual(logits.shape,(1,6))
            self.assertTrue(torch.all(logits[0,~torch.tensor(M)] < -1e8))
            v=play_student(p,NICHES[1],1102,2,20)
            self.assertLessEqual(v['decisions'],20)
            self.assertTrue(v['world'].painted.isdisjoint(v['world'].obstacles|v['world'].protected))

    @unittest.skipIf(torch is None, "P10 optional torch CPU dependency")
    def test_shuffled_preserves_multiset_and_padding(self):
        import torch
        h=torch.tensor([[0,0,0,1,2,3,4,5],[0,0,0,0,1,1,1,0]],dtype=torch.long)
        p=shuffled(h,torch.Generator().manual_seed(2))
        self.assertEqual(sorted(p[0].tolist()),sorted(h[0].tolist()))
        self.assertEqual(sorted(p[1].tolist()),sorted(h[1].tolist()))
        self.assertTrue(torch.equal(h[0,:3],p[0,:3]))
        self.assertTrue(torch.equal(h[1,:4],p[1,:4]))
        self.assertFalse(torch.equal(h[0],p[0]))

    @unittest.skipIf(torch is None, "P10 optional torch CPU dependency")
    def test_distillation_tensor_shape_and_target(self):
        X,M,H=observe(World.generate(42),[])
        target=M.astype(np.float32); target/=target.sum()
        x,m,h,n,y=tensorize([(X,M,H,0,target),(X,M,H,1,target)])
        self.assertEqual(tuple(x.shape),(2,6,20))
        self.assertEqual(tuple(y.shape),(2,6))
        self.assertAlmostEqual(float(y[0].sum()),1.)


if __name__=='__main__':unittest.main()
