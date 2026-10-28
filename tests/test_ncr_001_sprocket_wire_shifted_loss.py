"""Offline causal-shift and shared END/pad-ID refusals for NCR-001E.

These tests use a deterministic *fake* tokenizer for adversarial structures;
only the separate source-pinned CI job executes the real root tokenizer.
No model, optimizer, held-out fixture or training admission.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_wire_fullmask as original
import ncr_001_sprocket_wire_fullmask_custody as custody
import ncr_001_sprocket_wire_shifted_loss as alignment


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("unexpected automatic special tokens")
        markers = {"<|tool_call|>": 6, "<|memory_write|>": 11,
                   "<|memory_read|>": 10}
        for marker, token in markers.items():
            if text.startswith(marker):
                return [token] + [100 + ord(c) for c in text[len(marker):]]
        return [100 + ord(c) for c in text]


class ShiftedLossAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.turns = original.extract_synthetic_turns(original.HOST_SOURCE)
        cls.long_ids, cls.long_mask = original.independently_rebuild(
            cls.turns, FakeTokenizer())
        cls.short_ids, cls.short_mask = original.independently_rebuild(
            cls.turns[:3], FakeTokenizer())
        cls.good = alignment.construct(cls.long_ids, cls.long_mask,
                                       cls.short_ids, cls.short_mask)

    def build(self, *, long_ids=None, long_mask=None, short_ids=None,
              short_mask=None, pad_id=5):
        return alignment.construct(
            self.long_ids if long_ids is None else long_ids,
            self.long_mask if long_mask is None else long_mask,
            self.short_ids if short_ids is None else short_ids,
            self.short_mask if short_mask is None else short_mask,
            pad_id=pad_id)

    def test_positive_has_only_attended_next_token_targets(self):
        p = self.good
        self.assertEqual(len(p["supervised_positions"]), sum(self.long_mask))
        self.assertTrue(all(row == 1 and j > 0
                            for row, j in p["supervised_positions"]))
        self.assertEqual(p["shifted_targets"][0],
                         [-100] * (len(self.long_ids)-1))
        alignment.validate(p, copy.deepcopy(p))

    def test_real_end_id_remains_attended_even_when_used_for_padding(self):
        p = self.good
        self.assertEqual(p["input_ids"][0][:p["offset"]],
                         [5] * p["offset"])
        self.assertEqual(p["attention_mask"][0][:p["offset"]],
                         [0] * p["offset"])
        self.assertTrue(any(token == 5 and attn == 1
                            for row, mask in zip(p["input_ids"],
                                                 p["attention_mask"])
                            for token, attn in zip(row, mask)))

    def test_naive_token_equality_attention_is_refused(self):
        p = copy.deepcopy(self.good)
        p["attention_mask"] = [[int(v != 5) for v in row]
                               for row in p["input_ids"]]
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            alignment.validate(self.good, p)

    def test_pad_incorrectly_attended_is_refused(self):
        p = copy.deepcopy(self.good)
        p["attention_mask"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            alignment.validate(self.good, p)

    def test_causal_target_off_by_one_refused(self):
        p = copy.deepcopy(self.good)
        p["shifted_targets"][1][0] = 5
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            alignment.validate(self.good, p)

    def test_host_row_supervision_refused(self):
        mask = list(self.short_mask)
        mask[1] = True
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(short_mask=mask)

    def test_heldout_or_other_variant_not_substituted(self):
        ids = list(self.long_ids)
        ids[0] = 3
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_ids=ids)
        ids = list(self.long_ids)
        ids[-1] = 5
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_ids=ids)

    def test_closing_assistant_cue_must_not_be_a_loss(self):
        mask = list(self.long_mask)
        mask[-1] = True
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_mask=mask)

    def test_real_end_delimiter_cannot_be_deleted(self):
        ids = list(self.long_ids)
        i = ids.index(5)
        ids[i] = 123
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_ids=ids)

    def test_injected_end_marker_in_body_refused(self):
        ids = list(self.long_ids)
        ids[3] = 5
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_ids=ids)

    def test_bool_and_int_types_not_interchangeable(self):
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_mask=[int(v) for v in self.long_mask])
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_ids=[True] + self.long_ids[1:])

    def test_missing_target_or_length_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            self.build(long_mask=self.long_mask[:-1])
        p = copy.deepcopy(self.good)
        p["labels"][1].pop()
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            alignment.validate(self.good, p)

    def test_unmodified_original_custody_is_source_only(self):
        report = custody.audit()
        self.assertEqual(report["archive_members"], 4)
        self.assertFalse(report["new_model_execution"])
        self.assertFalse(report["train_authorized"])
        self.assertFalse(report["full_72_conformance_pass"])

    def test_four_model_protocol_is_still_closed(self):
        p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(p["train_authorized"], False)
        self.assertEqual(set(p["models"]), {
            "smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        self.assertNotEqual(p["models"]["sprocket_chat"]["weight_file"],
                            p["models"]["sprocket_instruct"]["weight_file"])


if __name__ == "__main__":
    unittest.main()
