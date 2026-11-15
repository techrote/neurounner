"""Independent cheap G1B correctness and leakage guards."""
import unittest
from experiments.memplant.postcards.prepare import new_cases
from experiments.memplant.g1.features import cases

class PostcardSources(unittest.TestCase):
    def test_fresh_scenarios_are_new_and_counterfactual(self):
        old=cases()
        novel=new_cases()
        self.assertEqual(len(novel),8)
        self.assertTrue(all(x.family not in {r.family for r in old} for x in novel))
        self.assertEqual(len({x.source_text for x in novel}),8)
        for x in novel:
            self.assertEqual(x.source_text.count("hardens"),1)
            self.assertEqual(x.source_text.count("softens"),1)
            y=next(z for z in novel if z.case_id==x.twin_id)
            self.assertNotEqual(x.answer,y.answer)
            self.assertEqual(x.family,y.family)

try:
    import torch
    from experiments.memplant.postcards.rescue import source_readout,choose_prefix
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"torch/numpy not installed")
class PostcardReadout(unittest.TestCase):
    def test_twin_swap_and_zero(self):
        from experiments.memplant.postcards.prepare import new_cases
        import numpy as np
        torch.manual_seed(12)
        # Tiny synthetic features for plumbing only; no numerical-transfer claim.
        xs=np.random.default_rng(12).normal(size=(48,2048)).astype("float32")
        rec=[c.record() for c in cases() if c.split in ("train","validation")]
        rec.extend(x.record() for x in new_cases())
        sc,alpha,xt,stats=source_readout(xs,rec)
        self.assertEqual(len(sc),48)
        self.assertEqual(stats["train"]["total"],32)
        self.assertEqual(stats["fresh_test"]["total"],8)
        prototypes=[torch.randn(1,8,24),torch.randn(1,8,24)]
        a=choose_prefix(prototypes,sc,40,rec,"genuine_gate",seed=7)
        b=choose_prefix(prototypes,sc,40,rec,"swapped_gate",seed=7)
        z=choose_prefix(prototypes,sc,40,rec,"zero",seed=7)
        self.assertEqual(a.shape,(1,8,24))
        self.assertEqual(b.shape,(1,8,24))
        self.assertTrue(torch.equal(z,torch.zeros_like(a)))
        self.assertTrue(torch.isfinite(choose_prefix(prototypes,sc,40,rec,"blend",seed=7)).all())

if __name__=="__main__":
    unittest.main()
