"""Stdlib-only right-gapped cache and source/false-admission guards; no model."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_pr102_rightpad_generate as wire  # noqa:E402


def synthetic():
    return [[3] + list(range(220, 248)) + [4],
            [3] + list(range(220, 234)) + [4]]


class RightPaddingSourceTests(unittest.TestCase):
    def test_source_locked_four_variant_model_and_generation_budget(self):
        self.assertTrue(wire.check_plan())
        self.assertEqual(wire.TARGET,
                         "3592841802d85c442ca2521479b5d8795bee7233")
        self.assertEqual(wire.SOURCE_IMPL,
                         "b95c010f9b2ba3cbd9f81317981316bb9c8acd19")
        self.assertEqual(wire.REVIEWER_SOURCE_BLOB,
                         "6ade653f0452e58f82c8c78e79d326018c30bcf1")
        self.assertEqual(wire.ROOT_SHA,
                         "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864")
        self.assertNotEqual(wire.ROOT_SHA, wire.CHAT_SHA)
        self.assertEqual(set(wire.MODELS),
                         {"smollm2","sprocket_chat","sprocket_instruct","void1"})
        self.assertEqual(wire.GENERATE_TOKENS, 4)
        self.assertEqual((wire.PAD, wire.END), (1, 5))

    def test_right_gapped_prefix_requires_masked_holes_and_appended_real_tokens(self):
        sequences = synthetic()
        observed = wire.gapped_right_prefix(sequences)
        self.assertEqual(observed["valid_prefix_lengths"], [26, 12])
        self.assertEqual(observed["width"], 26)
        self.assertEqual([len(row) for row in observed["prefix_ids"]], [26,26])
        self.assertEqual(observed["prefix_ids"][0], sequences[0][:-4])
        self.assertEqual(observed["prefix_ids"][1][:12], sequences[1][:-4])
        self.assertEqual(observed["prefix_ids"][1][12:], [wire.PAD]*14)
        self.assertEqual(observed["prefix_mask"][1],
                         [1]*12 + [0]*14)
        self.assertEqual(observed["prefix_position_ids"][1],
                         list(range(12)) + [0]*14)
        for i in range(4):
            self.assertEqual(observed["incremental_tokens"][i],
                             [sequences[0][-4+i],sequences[1][-4+i]])
            self.assertEqual(observed["incremental_position_ids"][i],
                             [26+i,12+i])
        self.assertEqual(observed["incremental_tokens"][-1], [4, 4])

    def test_reject_unknown_padding_wrong_shape_or_inadequate_source_gap(self):
        cases = [
            ([[3]*30,[4]*15],3,1),
            ([[3]*30,[4]*15],4,0),
            ([[3]*30,[4]*15],4,7),
            ([[3]*20],4,1),
            (["not a token vector",[1]*14],4,1),
            ([[3]*30,[]],4,1),
            ([[3]*14,[4]*13],4,1),
            ([[3]*30,[4]*12],4,1),
        ]
        for sequences, tail, pad in cases:
            with self.subTest(case=(len(sequences),tail,pad)), \
                    self.assertRaises(RuntimeError):
                wire.gapped_right_prefix(sequences,tail=tail,pad=pad)

    def test_masked_influence_sentinel_uses_real_control_id_and_token_positions(self):
        gap=wire.gapped_right_prefix(synthetic())
        self.assertEqual(wire.PAD,1)
        self.assertEqual(gap["prefix_ids"][1][12:], [1]*14)
        self.assertTrue(all(mask==0 for mask in gap["prefix_mask"][1][12:]))
        self.assertEqual(wire.SOURCE_IMPL[:7], "b95c010")
        self.assertEqual(wire.CACHED_ATOL,wire.CACHED_RTOL)
        self.assertEqual(wire.CACHED_ATOL,.0005)

    def test_reject_false_training_heldout_semantics_and_variant(self):
        good={
            "train_authorized":False,"optimizer_steps":0,
            "heldout_evaluation":False,"any_model_training":False,
            "semantic_tool_memory_qualified":False,
            "final_four_model_72_conformance":False,
            "variant":"sprocket_instruct",
        }
        wire.guard_claims(good)
        for key, bad in (
            ("train_authorized",True),("optimizer_steps",1),
            ("heldout_evaluation",True),("any_model_training",True),
            ("semantic_tool_memory_qualified",True),
            ("final_four_model_72_conformance",True),
            ("variant","sprocket_chat"),
        ):
            changed=copy.deepcopy(good)
            changed[key]=bad
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                wire.guard_claims(changed)

    def test_source_review_keeps_original_generation_and_archive_separate(self):
        self.assertNotEqual(wire.OUT.name, "pr102-independent-batch-review.json")
        self.assertIn("rightpad-generate", wire.OUT.name)
        self.assertEqual(wire.PRIMARY[-1][0],"user")
        self.assertEqual(wire.SECONDARY[-1][0],"user")


if __name__=="__main__":
    unittest.main()
