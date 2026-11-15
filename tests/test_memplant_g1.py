"""Offline source/split and lightweight receiver-control checks for MEMPLANT G1."""
import unittest
from collections import Counter

from experiments.memplant.g1.features import (
    MATERIALS, QUESTION_A, QUESTION_B, TRAIT, cases, validate_cases)


class DatasetTests(unittest.TestCase):
    def test_group_disjoint_counterfactual_twins(self):
        xs=cases()
        self.assertEqual(len(xs),48)
        self.assertEqual(Counter(x.split for x in xs),
                         Counter({"train":32,"validation":8,"sealed":8}))
        self.assertEqual(len({x.source_sha256 for x in xs}),48)
        by_id={x.case_id:x for x in xs}
        for x in xs:
            y=by_id[x.twin_id]
            self.assertEqual(x.family,y.family)
            self.assertEqual(x.split,y.split)
            self.assertNotEqual(x.answer,y.answer)
            self.assertEqual(x.source_text.count("hardens"),1)
            self.assertEqual(x.source_text.count("softens"),1)

    def test_prompts_identical_across_donors(self):
        xs=cases()
        for q in (QUESTION_A,QUESTION_B):
            self.assertIn("hardens",q)
            self.assertIn("softens",q)
            self.assertNotIn("world-00",q)
            for x in xs:
                self.assertNotIn(x.source_text,q)
                self.assertNotIn(x.case_id,q)
        self.assertNotEqual(QUESTION_A,QUESTION_B)

    def test_template_family_separation(self):
        xs=cases()
        grouped={split:{x.family for x in xs if x.split==split}
                 for split in ("train","validation","sealed")}
        for a in grouped:
            for b in grouped:
                if a!=b:self.assertTrue(grouped[a].isdisjoint(grouped[b]))
        self.assertEqual({x.template for x in xs if x.split=="sealed"},{5})
        self.assertEqual({x.template for x in xs if x.split=="validation"},{4})
        self.assertEqual({x.template for x in xs if x.split=="train"},{0,1,2,3})

    def test_fail_closed_fixture_modification(self):
        xs=cases()
        with self.assertRaises(AssertionError):
            validate_cases(xs[:-1])


try:
    import torch
    from experiments.memplant.g1.features import fixed_projection,project_bins
    from experiments.memplant.g1.imprint import inferred_word,compute_condition
    from experiments.memplant.dreamgraft001.bridge import DreamPrefix
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"torch/numpy not installed")
class FeatureControlTests(unittest.TestCase):
    def test_projection_reproducible_and_geometry(self):
        p=fixed_projection(torch)
        self.assertEqual(p.shape,(896,128))
        self.assertTrue(torch.equal(p,fixed_projection(torch)))
        x=torch.arange(40*896,dtype=torch.float32).reshape(1,40,896)
        feature=project_bins(x,p,16)
        self.assertEqual(tuple(feature.shape),(2048,))
        self.assertTrue(torch.isfinite(feature).all())

    def test_swapped_and_zero_are_distinct(self):
        torch.manual_seed(61)
        model=DreamPrefix(2048,12,slots=8,bottleneck=16,scale=1)
        f=torch.stack([torch.randn(2048),torch.randn(2048)])  # avoid LayerNorm-equivalent constant vectors
        true=compute_condition(model,f,0,"genuine",twin_index=1,seed=13)
        switched=compute_condition(model,f,0,"swapped",twin_index=1,seed=13)
        zero=compute_condition(model,f,0,"zero",twin_index=1,seed=13)
        rnd=compute_condition(model,f,0,"random",twin_index=1,seed=13)
        self.assertFalse(torch.allclose(true,switched))
        self.assertTrue(torch.equal(zero,torch.zeros_like(true)))
        self.assertTrue(torch.allclose(rnd.norm(),true.norm(),atol=1e-5))
        self.assertEqual(inferred_word(0.4),"hardens")
        self.assertEqual(inferred_word(-0.4),"softens")


if __name__=="__main__":
    unittest.main()
