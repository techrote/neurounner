"""Source-only negative tests for the independent Sprocket Chat generation review."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ncr_001e_sprocket_chat_generate_review as chat  # noqa:E402


class ChatGenerateSourceTests(unittest.TestCase):
    def test_fixed_source_identity_and_no_instruct_substitution(self):
        self.assertTrue(chat.source_only_plan())
        self.assertEqual(chat.HOOK_SHA,
                         "83a56835a7476f17a17980d28dd1a7c14bb6f2f6")
        self.assertEqual(chat.FORMAT_BLOB,
                         "39cf015735e74a1d1e08682b623404bc1e600b9b")
        self.assertEqual(chat.CHAT_SHA,
                         "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97")
        self.assertNotEqual(chat.CHAT_SHA, chat.INSTRUCT_SHA)
        self.assertEqual((chat.PAD, chat.END, chat.TAIL), (1, 5, 4))

    def test_fixed_prompts_use_only_ordinary_source_chat_roles(self):
        self.assertEqual([row["role"] for row in chat.PROMPT_A],
                         ["user", "assistant", "user"])
        self.assertEqual([row["role"] for row in chat.PROMPT_B], ["user"])
        for row in chat.PROMPT_A + chat.PROMPT_B:
            self.assertNotIn(row["role"], {"tool", "memory", "system"})
            for marker in ("<|tool_call|>", "<|tool_result|>",
                           "<|memory_read|>", "<|memory_write|>"):
                self.assertNotIn(marker, row["content"])

    def test_right_gap_masks_shorter_request_and_preserves_true_tail(self):
        seqs = [
            [3] + list(range(100, 130)) + [5, 4],
            [3] + list(range(200, 214)) + [5, 4],
        ]
        observed = chat.right_gap(seqs)
        self.assertEqual(observed["valid_prefix_lengths"], [29, 13])
        self.assertEqual(observed["width"], 29)
        self.assertEqual(observed["ids"][1][13:], [chat.PAD] * 16)
        self.assertEqual(observed["mask"][1][13:], [0] * 16)
        self.assertEqual(observed["positions"][1][13:], [0] * 16)
        self.assertEqual(observed["tail_tokens"][-1],
                         [seqs[0][-1], seqs[1][-1]])
        self.assertEqual(observed["tail_positions"][-1], [32, 16])

    def test_right_gap_rejects_bad_batch_and_inadequate_padding_difference(self):
        bad = (
            [[3] * 30],
            [[3] * 30, []],
            [[3] * 14, [4] * 13],
            ["not-token-list", [4] * 20],
        )
        for value in bad:
            with self.subTest(value=type(value[0]).__name__), \
                    self.assertRaises(RuntimeError):
                chat.right_gap(value)

    def test_false_training_tool_memory_or_final_pass_is_refused(self):
        good = {
            "variant": "chat",
            "train_authorized": False,
            "optimizer_steps": 0,
            "heldout_evaluation": False,
            "actual_training": False,
            "tool_memory_semantics_qualified": False,
            "final_four_model_72_conformance": False,
        }
        chat.guard_claims(good)
        for key, value in (
            ("variant", "instruct"),
            ("train_authorized", True),
            ("optimizer_steps", 1),
            ("heldout_evaluation", True),
            ("actual_training", True),
            ("tool_memory_semantics_qualified", True),
            ("final_four_model_72_conformance", True),
        ):
            altered = copy.deepcopy(good)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                chat.guard_claims(altered)

    def test_chat_review_does_not_reuse_instruct_evidence_file(self):
        self.assertEqual(chat.OUT.name, "sprocket-chat-generate-review.json")
        self.assertNotIn("instruct", chat.OUT.name)
        self.assertEqual(set(chat.MODEL_KEYS),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})


if __name__ == "__main__":
    unittest.main()
