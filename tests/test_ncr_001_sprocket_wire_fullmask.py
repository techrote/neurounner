"""Pure-Python mutation sentinels for NCR-001 Sprocket Instruct full-mask replay.

Do not execute a model, optimizer, evaluator or held-out fixtures.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_wire_fullmask as probe


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("unexpected BOS/EOS injection")
        return [100 + ord(char) for char in text]


class FullPositionMaskTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {"role": "system", "content": "Only test synthetic source."},
            {"role": "user", "content": "Read the meter."},
            {"role": "assistant", "content": "I will read it."},
            {"role": "tool", "content": '{"reading":42}'},
            {"role": "memory", "content": "Temperature is Celsius."},
            {"role": "assistant", "content": "Reading 42 C."},
            {"role": "user", "content": "Next."},
        ]
        self.ids, self.labels = probe.independently_rebuild(
            self.turns, FakeTokenizer())

    def verified(self, ids=None, labels=None):
        return probe.check_full_mask(
            self.ids if ids is None else ids,
            self.labels if labels is None else labels,
            self.ids, self.labels)

    def test_full_position_positive(self):
        result = self.verified()
        self.assertTrue(result["all_positions_checked"])
        self.assertEqual(result["assistant_units"], sum(self.labels))
        self.assertEqual(len(self.ids), result["units"])

    def test_compensating_host_and_assistant_body_mask_swaps_refused(self):
        """Preserve aggregate counts while moving loss to untrusted tool text."""
        labels = copy.copy(self.labels)
        assistant_body = next(i for i, value in enumerate(labels)
                              if value and self.ids[i] not in (5,))
        tool_header = self.ids.index(probe.ROLE_IDS["tool"])
        tool_body = tool_header + 1
        self.assertFalse(labels[tool_body])
        self.assertTrue(labels[assistant_body])
        labels[assistant_body] = False
        labels[tool_body] = True
        self.assertEqual(sum(labels), sum(self.labels))
        with self.assertRaisesRegex(RuntimeError, "body label drift"):
            self.verified(labels=labels)

    def test_host_memory_body_leak_refused(self):
        labels = copy.copy(self.labels)
        body = self.ids.index(probe.ROLE_IDS["memory"]) + 1
        self.assertFalse(labels[body])
        labels[body] = True
        with self.assertRaisesRegex(RuntimeError, "body label drift"):
            self.verified(labels=labels)

    def test_assistant_body_labels_cannot_disappear(self):
        labels = copy.copy(self.labels)
        labels[next(i for i, v in enumerate(labels) if v)] = False
        with self.assertRaisesRegex(RuntimeError, "body label drift"):
            self.verified(labels=labels)

    def test_assistant_role_header_not_supervised(self):
        labels = copy.copy(self.labels)
        at = self.ids.index(probe.ROLE_IDS["assistant"])
        self.assertFalse(labels[at])
        labels[at] = True
        with self.assertRaisesRegex(RuntimeError, "body label drift"):
            self.verified(labels=labels)

    def test_final_dangling_assistant_cue_not_supervised(self):
        labels = copy.copy(self.labels)
        self.assertEqual(self.ids[-1], probe.ROLE_IDS["assistant"])
        self.assertFalse(labels[-1])
        labels[-1] = True
        with self.assertRaisesRegex(RuntimeError, "body label drift"):
            self.verified(labels=labels)

    def test_end_marker_five_is_genuine_not_padding(self):
        self.assertIn(5, self.ids)
        assistant_end = next(i for i, v in enumerate(self.labels)
                             if v and self.ids[i] == 5)
        labels = copy.copy(self.labels)
        labels[assistant_end] = False
        with self.assertRaisesRegex(RuntimeError, "body label drift"):
            self.verified(labels=labels)

    def test_corrupted_token_refused_with_unchanged_labels(self):
        ids = copy.copy(self.ids)
        ids[len(ids)//2] += 7
        with self.assertRaisesRegex(RuntimeError, "reconstruction mismatch"):
            self.verified(ids=ids)

    def test_mask_ints_do_not_masquerade_as_bools(self):
        with self.assertRaisesRegex(RuntimeError, "type drift"):
            self.verified(labels=[int(v) for v in self.labels])

    def test_mask_length_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "length drift"):
            self.verified(labels=self.labels[:-1])

    def test_source_literal_extract_cannot_invent_a_fixture(self):
        turns = probe.extract_synthetic_turns(probe.HOST_SOURCE)
        self.assertEqual(len(turns), 12)
        self.assertEqual([x["role"] for x in turns],
            ["system", "memory", "user", "assistant", "tool", "assistant",
             "user", "assistant", "user", "assistant", "memory", "user"])

    def test_original_archive_readback_is_not_new_model_qualification(self):
        summary, receipt = probe.original_receipt()
        self.assertTrue(summary["retained_original_failures"])
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["full_72_admitted"])
        self.assertEqual(receipt["lane"], "sprocket_instruct")
        self.assertFalse(receipt["model_generated_tool_memory_semantics_qualified"])
        self.assertEqual(receipt["optimizer_steps"], 0)

    def test_training_protocol_stays_closed_with_four_separate_models(self):
        import json
        p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(p["train_authorized"], False)
        self.assertEqual(set(p["models"]),
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        self.assertEqual(p["models"]["sprocket_chat"]["weight_file"],
                         "chat/model.safetensors")
        self.assertEqual(p["models"]["sprocket_instruct"]["weight_file"],
                         "model.safetensors")


if __name__ == "__main__":
    unittest.main()
