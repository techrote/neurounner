"""NCR-001E separate reviewer source/mask refusals, Python-stdlib only."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001e_pr102_masked_batch_review import (  # noqa: E402
    CONTROL_IDS, END, PRIMARY, SECONDARY, ROLE_IDS, TAIL, TARGET, SOURCE_IMPL,
    PUBLISHER, ROOT_SHA, CHAT_SHA, independent_wire, batch_layout,
    check_synthetic_plan, guard_report_claims, require,
)


class StubTokenEncoder:
    def __init__(self, broken=False):
        self.broken = broken
        self.pattern = re.compile("(" + "|".join(re.escape(t) for t in
                                  sorted(CONTROL_IDS, key=len, reverse=True)) + ")")
    def encode(self, text, add_special_tokens=False):
        tokens = []
        for part in self.pattern.split(text):
            if part in CONTROL_IDS:
                tokens.append(CONTROL_IDS[part])
            else:
                tokens.extend(200 + ord(c) for c in part)
        if self.broken and text.startswith("<|system|>"):
            return tokens + [999]
        return tokens


class ReviewerSourceTests(unittest.TestCase):
    def test_pinned_four_way_identity_and_new_source_inputs(self):
        self.assertEqual(TARGET, "3592841802d85c442ca2521479b5d8795bee7233")
        self.assertEqual(SOURCE_IMPL, "b95c010f9b2ba3cbd9f81317981316bb9c8acd19")
        self.assertEqual(PUBLISHER, "a003767e70d766bc14b40c89792543ee8641c8bc")
        self.assertEqual(ROOT_SHA, "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864")
        self.assertNotEqual(ROOT_SHA, CHAT_SHA)
        self.assertTrue(check_synthetic_plan())
        self.assertEqual(TAIL, 4)

    def test_raw_source_markers_and_assistant_label_mask(self):
        enc = StubTokenEncoder()
        ids, trained, wire = independent_wire(PRIMARY, enc)
        self.assertEqual((len(ids), len(trained)), (len(ids), len(ids)))
        self.assertEqual(ids[0], ROLE_IDS["system"])
        self.assertEqual(ids[-1], ROLE_IDS["assistant"])
        self.assertFalse(trained[-1])
        self.assertEqual([trained[i] for i, token in enumerate(ids) if token == 10],
                         [False, True, False])
        self.assertTrue(trained[ids.index(6)])   # assistant tool call
        self.assertFalse(trained[ids.index(7)])  # host tool result
        self.assertTrue(trained[ids.index(11)])  # assistant memory write
        self.assertTrue(wire.endswith("<|assistant|>"))
        self.assertEqual(ids.count(END), 12)
        other_ids, other_mask, _ = independent_wire(SECONDARY, enc)
        self.assertLess(len(other_ids), len(ids))
        self.assertEqual(other_ids[-1], ROLE_IDS["assistant"])
        self.assertFalse(other_mask[-1])

    def test_conversation_refuses_unknown_and_empty_roles(self):
        bad = [
            (("developer", "Untrusted"),),
            (("assistant", "Cannot finish on assistant"),),
            (("user", ""),),
            tuple([("user", "over maximum")]*13),
            ((1, "nonstring role"),),
        ]
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                independent_wire(rows, StubTokenEncoder())
        with self.assertRaises(RuntimeError):
            independent_wire(PRIMARY, StubTokenEncoder(broken=True))

    def test_padding_isolation_both_directions(self):
        sequences = [[3, 20, 21, 5, 4], [3, 30, 5, 4]]
        for side in ("left", "right"):
            padded, masks, spans = batch_layout(sequences, side)
            self.assertEqual(len(padded), 2)
            self.assertTrue(all(len(row)==5 for row in padded))
            for i, (a,b) in enumerate(spans):
                self.assertEqual(padded[i][a:b], sequences[i])
                self.assertTrue(all(masks[i][j]==1 for j in range(a,b)))
                self.assertEqual(sum(masks[i]),len(sequences[i]))
            if side=="left":
                self.assertEqual(padded[1][0],1)
                self.assertEqual(masks[1][0],0)
            else:
                self.assertEqual(padded[1][-1],1)
                self.assertEqual(masks[1][-1],0)

    def test_refuse_bad_layouts_and_false_training_pass(self):
        for args in (
            ([[3]],"left"),
            ([[3],[4]],"center"),
            ([[3],[]],"right"),
            ("text","left"),
        ):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                batch_layout(*args)
        false_report = {
            "train_authorized": False,
            "optimizer_steps": 0,
            "heldout_evaluation": False,
            "semantic_tool_or_memory_quality_qualified": False,
            "final_four_model_72_pass": False,
            "model_variant": "sprocket_instruct",
        }
        guard_report_claims(false_report)
        for field,value in (
            ("train_authorized",True),
            ("optimizer_steps",1),
            ("heldout_evaluation",True),
            ("semantic_tool_or_memory_quality_qualified",True),
            ("final_four_model_72_pass",True),
            ("model_variant","sprocket_chat"),
        ):
            broken=dict(false_report)
            broken[field]=value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                guard_report_claims(broken)

    def test_synthetic_no_train_negative_test_does_not_open_fixtures(self):
        for role, body in PRIMARY + SECONDARY:
            self.assertIn(role, ROLE_IDS)
            self.assertTrue(body)
        self.assertEqual(CONTROL_IDS["<|tool_result|>"], 7)
        self.assertEqual(CONTROL_IDS["<|memory_read|>"], 10)
        self.assertEqual(CONTROL_IDS["<|memory_write|>"], 11)


if __name__ == "__main__":
    unittest.main()
