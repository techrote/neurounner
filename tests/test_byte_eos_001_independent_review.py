"""Source-only independent BYTE-EOS-001 oracle and fail-closed acceptance tests."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

import byte_eos_001_independent_review as review  # noqa:E402


class ByteEOSIndependentSourceTests(unittest.TestCase):
    def test_exact_review_source_constants_and_no_model_scope(self):
        self.assertEqual(review.REVIEWED_PR94_HEAD,
                         "b1c558f7d46f5d7c76b4e6336a37143886d16c08")
        self.assertEqual(review.CANDIDATE_BLOB,
                         "724a5040d2fc9eb06ef0b17143d44daa36a4aa89")
        self.assertEqual(review.TOKENIZER_SOURCE_SHA,
                         "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03")
        self.assertEqual(review.PREREG_COMMIT,
                         "a1b41c358d215974ea9ef2748baa13c3872b0fbb")
        self.assertEqual((review.PAD,review.BOS,review.EOS,review.VOCAB),
                         (256,257,258,259))
        self.assertEqual(review.MAX_UNITS,1024)

    def test_independent_oracle_masks_prompt_and_supervises_real_eos(self):
        p="Task: "
        c="café 🔧"
        x=review.oracle(p,c)
        pb=list(p.encode("utf-8"));cb=list(c.encode("utf-8"))
        self.assertEqual(x["input_ids"],pb+cb+[258])
        self.assertEqual(x["labels"],[-100]*len(pb)+cb+[258])
        self.assertEqual(x["attention_mask"],[1]*len(x["input_ids"]))
        self.assertEqual(x["prompt_tokenizer_units"],len(pb))
        self.assertEqual(x["supervised_tokenizer_units"],len(cb)+1)
        self.assertEqual(x["input_ids"][-1],258)
        self.assertTrue(x["terminal_eos_supervised"])

    def test_literal_eos_text_is_bytes_then_actual_terminal_eos(self):
        x=review.oracle("","<eos>")
        self.assertEqual(x["input_ids"],[60,101,111,115,62,258])
        self.assertEqual(x["labels"],[60,101,111,115,62,258])
        self.assertEqual(x["input_ids"].count(258),1)

    def test_empty_completion_still_supervises_one_real_eos(self):
        x=review.oracle("Prompt","")
        self.assertEqual(x["labels"],[-100]*6+[258])
        self.assertEqual(x["supervised_tokenizer_units"],1)

    def test_max_boundary_preserves_eos_and_one_more_refuses(self):
        x=review.oracle("","x"*1023,max_units=1024)
        self.assertEqual(len(x["input_ids"]),1024)
        self.assertEqual(x["input_ids"][-1],258)
        with self.assertRaises(RuntimeError):
            review.oracle("","x"*1024,max_units=1024)
        for value in (0,1025,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                review.oracle("a","b",max_units=value)

    def test_malformed_unicode_refused_without_replacement(self):
        with self.assertRaises(UnicodeEncodeError):
            review.oracle("ok","\ud800")

    def test_new_reviewer_cases_cover_ascii_unicode_literal_empty_and_boundary(self):
        cases=review.source_cases()
        names=[row[0] for row in cases]
        self.assertEqual(names,[
            "ascii","unicode","literal_eos_content","empty_completion",
            "empty_prompt","emoji_boundary","max_1024"])
        self.assertEqual(len(set(names)),len(names))

    def test_false_integration_impact_or_training_claim_refused(self):
        base={
            "training_authorized":False,
            "optimizer_steps":0,
            "model_weights_loaded":False,
            "historical_train002b_modified":False,
            "historical_impact_measured":False,
            "future_serializer_integrated_into_training":False,
        }
        review.guard_report(base)
        for key,value in (
            ("training_authorized",True),("optimizer_steps",1),
            ("model_weights_loaded",True),("historical_train002b_modified",True),
            ("historical_impact_measured",True),
            ("future_serializer_integrated_into_training",True),
        ):
            x=copy.deepcopy(base);x[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                review.guard_report(x)


if __name__=="__main__":
    unittest.main()
