"""G1C audit regression tests: new family and immutable causal-swap fixtures."""
import unittest
from experiments.memplant.postcard_audit.prepare import confirmation_worlds
from experiments.memplant.postcards.prepare import new_cases as g1b_cases
from experiments.memplant.g1.features import cases as g1_cases

class NewWorlds(unittest.TestCase):
    def test_untouched_groups_and_twin_swaps(self):
        new=confirmation_worlds()
        self.assertEqual(len(new),16)
        self.assertEqual(len({x.family for x in new}),8)
        past={x.family for x in g1b_cases()}
        past.update(x.family for x in g1_cases())
        self.assertTrue(all(x.family not in past for x in new))
        idmap={x.case_id:x for x in new}
        self.assertEqual(len(idmap),16)
        for x in new:
            mate=idmap[x.twin_id]
            self.assertEqual(x.family,mate.family)
            self.assertNotEqual(x.answer,mate.answer)
            self.assertEqual(x.source_text.count("hardens"),1)
            self.assertEqual(x.source_text.count("softens"),1)

try:
    import torch
    from experiments.memplant.postcard_audit.audit import conditioned_prefix
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"torch unavailable")
class NumericalControl(unittest.TestCase):
    def test_same_source_matches_swapped_twin_prefix(self):
        m=[x.record() for x in confirmation_worlds()]
        state={"prefix_hardens":torch.randn(1,8,24),"prefix_softens":torch.randn(1,8,24)}
        scores=torch.linspace(-.4,.4,len(m))
        a,src_a=conditioned_prefix(state,scores,m,0,"genuine_blend")
        b,src_b=conditioned_prefix(state,scores,m,1,"swapped_blend")
        self.assertEqual(src_a,src_b)
        self.assertTrue(torch.equal(a,b))
        z,_=conditioned_prefix(state,scores,m,0,"zero")
        self.assertTrue(torch.equal(z,torch.zeros_like(z)))

if __name__=="__main__":
    unittest.main()
