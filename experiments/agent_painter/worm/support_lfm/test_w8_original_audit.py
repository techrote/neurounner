#!/usr/bin/env python3
"""Adversarial raw/EOS/host-legal accounting checks, no model required."""
import unittest
from w8_original_audit import score, sha, CODES

class W8Tests(unittest.TestCase):
    def base(self):
        return {"token_ids":[529,7],"raw_text":"4","raw_action":"RETRACE",
            "eos_terminated":True,"valid_first_pass":True,
            "raw_legal":True,"legal":["RETRACE","STOP"]}
    def test_true_eos_independent_of_host_legal(self):
        row=self.base()
        self.assertEqual(score(row),("RETRACE",True,True))
        row["legal"]=["STOP"]
        row["raw_legal"]=row["valid_first_pass"]=False
        row["raw_action"]="MODEL_REFUSED"
        self.assertEqual(score(row),("RETRACE",True,False))
    def test_no_silent_repair_of_bad_text(self):
        row=self.base()
        row.update(raw_text="4\n",raw_legal=False,valid_first_pass=False,
                   raw_action="MODEL_REFUSED")
        self.assertEqual(score(row),(None,False,False))
    def test_wrong_eos_or_host_claim_rejected(self):
        a=self.base()
        for mutation in ({"eos_terminated":False},
                         {"raw_legal":False},
                         {"valid_first_pass":False},
                         {"raw_action":"STRAIGHT"},
                         {"token_ids":[529,True]}):
            x={**a,**mutation}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                score(x)
    def test_action_vocabulary_pinned(self):
        self.assertEqual(CODES,dict(zip(("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP"),"012345")))
        self.assertEqual(len(sha(b"source")),64)

if __name__=="__main__":
    unittest.main()
