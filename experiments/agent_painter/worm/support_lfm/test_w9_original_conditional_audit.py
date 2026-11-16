#!/usr/bin/env python3
"""CPU-only adversarial W9 source/score attribution regressions."""
import unittest
from w9_original_conditional_audit import validate_pair, assess_probe

class W9AuditTest(unittest.TestCase):
    def pair(self):
        a={"origin":"W0-observed-TRAIN","family":"greebles","pair_id":"a",
            "prompt":'{"heading":0,"legal":["RIGHT","STOP"],"wall":1}',
            "completion":"RIGHT","legal":["RIGHT","STOP"]}
        b={"origin":"synthetic-TRAIN-not-W0","family":"greebles","pair_id":"a",
            "prompt":'{"heading":0,"legal":["STOP"],"wall":1}',
            "completion":"STOP","legal":["STOP"],"excluded_action":"RIGHT"}
        return a,b

    def probe(self):
        return {"scores":[0.,1.,3.,2.,4.,0.],"legal":["RIGHT","RETRACE","STOP"],
                "expected":"RIGHT","raw_score_top":"RETRACE",
                "legal_projected_top":"RETRACE","teacher_raw_match":False,
                "teacher_legal_projected_match":False,"synthetic_exclusion":{
                    "excluded":"RIGHT","raw_score_top":"RIGHT",
                    "new_legal_top":"RETRACE","unmasked_score_top_changed":True,
                    "scores":[0.,1.,6.,2.,4.,0.],
                    "synthetic_not_W0_observed":True}}

    def test_valid_original_synthetic_pair(self):
        validate_pair(*self.pair())

    def test_rejects_synthetic_world_mutation(self):
        a,b=self.pair()
        b["prompt"]='{"heading":2,"legal":["STOP"],"wall":1}'
        with self.assertRaises(ValueError):validate_pair(a,b)

    def test_rejects_fake_synthetic_target(self):
        a,b=self.pair()
        b["completion"]="RIGHT"
        with self.assertRaises(ValueError):validate_pair(a,b)

    def test_real_raw_and_masked_score_separated(self):
        r=assess_probe(self.probe())
        self.assertEqual(r["top"],"RETRACE")
        self.assertFalse(r["raw_teacher"])
        self.assertTrue(r["synthetic_changes_raw_top"])

    def test_rejects_falsified_top(self):
        x=self.probe();x["raw_score_top"]="RIGHT"
        with self.assertRaises(ValueError):assess_probe(x)

    def test_rejects_illegal_host_projection(self):
        x=self.probe();x["legal_projected_top"]="RIGHT"
        with self.assertRaises(ValueError):assess_probe(x)

    def test_rejects_fake_exclusion_change(self):
        x=self.probe();x["synthetic_exclusion"]["unmasked_score_top_changed"]=False
        with self.assertRaises(ValueError):assess_probe(x)

    def test_rejects_nonfinite_or_bad_scores(self):
        for v in ([1,2,3,4,5],[0,1,2,3,float("nan"),5]):
            x=self.probe();x["scores"]=v
            with self.subTest(v=v),self.assertRaises(ValueError):assess_probe(x)

if __name__=="__main__":
    unittest.main()
