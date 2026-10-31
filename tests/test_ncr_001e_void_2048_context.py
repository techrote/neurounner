"""Source-only sentinels for void.1 2048/HF.generate reviewer."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001e_void_2048_context as review  # noqa:E402


class Void2048SourceTests(unittest.TestCase):
    def test_fixed_source_identity_context_and_no_training(self):
        self.assertTrue(review.fixed_plan())
        self.assertEqual(review.LENGTHS,(512,1024,2048))
        self.assertEqual(review.TAIL,16)
        self.assertEqual(review.GENERATE_PREFIX,512)
        self.assertEqual(review.GENERATE_UNITS,4)
        self.assertEqual((review.PAD,review.BOS,review.EOS),(256,257,258))
        self.assertEqual(review.WEIGHT,
            "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3")
        self.assertEqual(review.HOOK,
            "5fa3dc8c0de153098d1800f7b0c809819620aef4")
        self.assertEqual(review.HOOK_BLOB,
            "3640624df6dbd994dc11b3ca5cd759bdef8757dd")
        self.assertEqual(review.TOKENIZER_SHA,
            "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03")

    def test_source_text_is_real_multibyte_utf8_beyond_full_context(self):
        raw=review.SOURCE_TEXT.encode("utf-8")
        self.assertGreater(len(raw),2048)
        self.assertTrue(any(ord(c)>127 for c in review.SOURCE_TEXT))
        self.assertIn("🔧",review.SOURCE_TEXT)
        self.assertIn("日本語",review.SOURCE_TEXT)

    def test_false_training_historical_impact_or_final_pass_refused(self):
        good={
            "train_authorized":False,
            "optimizer_steps":0,
            "actual_training":False,
            "heldout_evaluation":False,
            "historical_byte_eos_impact_measured":False,
            "final_four_model_72_conformance":False,
        }
        review.guard_claims(good)
        for key,value in (
            ("train_authorized",True),("optimizer_steps",1),
            ("actual_training",True),("heldout_evaluation",True),
            ("historical_byte_eos_impact_measured",True),
            ("final_four_model_72_conformance",True),
        ):
            changed=copy.deepcopy(good);changed[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                review.guard_claims(changed)

    def test_context_lengths_are_distinct_and_exactly_at_pinned_max(self):
        self.assertEqual(max(review.LENGTHS),2048)
        self.assertEqual(len(set(review.LENGTHS)),3)
        for n in review.LENGTHS:
            self.assertGreater(n,review.TAIL)
            self.assertEqual(n % 512,0)

    def test_generation_is_short_conformance_not_eos_propensity_study(self):
        self.assertEqual(review.GENERATE_UNITS,4)
        self.assertEqual(review.GENERATE_PREFIX,512)
        self.assertNotIn("TRAIN-002",review.SOURCE_TEXT)
        self.assertEqual(set(review.LANES),
                         {"smollm2","sprocket_chat","sprocket_instruct","void1"})


if __name__=="__main__":
    unittest.main()
