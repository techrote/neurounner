"""NCR-001E Chat-only assistant-label/END/padding and independent CE sentinels.

Toy tokenization is deliberately NOT the original Sprocket tokenizer; an
independent job must execute SHA-verified source and the real frozen model.
No optimizer, heldouts, comparative NCA/MLP training, or tool/memory model.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_chat_native_ce as check


class FakeChatTokenizer:
    pad_token_id = 5
    controls = {"<|user|>": 3, "<|assistant|>": 4, "<|end|>": 5}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        result = []
        while text:
            marker = next((s for s in self.controls if text.startswith(s)), None)
            if marker:
                result.append(self.controls[marker])
                text = text[len(marker):]
            else:
                result.append(100 + ord(text[0]))
                text = text[1:]
        return result


def toy_source_formatter(messages, *, variant):
    if variant != "chat":
        raise ValueError("distinct Chat variant required")
    rows = list(messages)
    if not rows or len(rows) > 6:
        raise ValueError("typed source messages required")
    if rows[0]["role"] != "user" or rows[-1]["role"] != "user":
        raise ValueError("only alternating user/assistant source roles")
    s = ""
    for i, row in enumerate(rows):
        role = "user" if i % 2 == 0 else "assistant"
        if row["role"] != role or "<|" in row["content"]:
            raise ValueError("unqualified control or role")
        s += f"<|{role}|>{row['content']}<|end|>"
    return s + "<|assistant|>"


class ChatCausalMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = FakeChatTokenizer()
        cls.good = check.construct(cls.tok, toy_source_formatter)

    def test_positive_source_role_headers_and_distinct_example_sizes(self):
        obj = self.good
        self.assertEqual(len(obj["input_ids"]), 2)
        self.assertEqual(obj["input_ids"][0][-1], 5)
        self.assertEqual(obj["input_ids"][1][-1], 5)
        self.assertNotEqual(obj["items"][0]["prompt_units"],
                            obj["items"][1]["prompt_units"])
        self.assertGreater(obj["supervised_count"], 8)

    def test_host_context_and_synthetic_padding_never_supervised(self):
        obj = self.good
        for row in range(2):
            pad = obj["attention_mask"][row].index(1)
            self.assertTrue(all(x == -100 for x in obj["labels"][row][:pad]))
            self.assertEqual(obj["labels"][row][-1], 5)
            self.assertTrue(all(not (x != -100 and a == 0)
                                for x, a in zip(obj["labels"][row],
                                                obj["attention_mask"][row])))
        self.assertGreater(obj["attention_mask"][0].count(0), 0)

    def test_genuine_end_id5_cannot_be_treated_as_padding(self):
        obj = self.good
        self.assertTrue(any(t == 5 and a == 1
                            for row, mask in zip(obj["input_ids"], obj["attention_mask"])
                            for t, a in zip(row, mask)))
        self.assertNotEqual([[int(t != 5) for t in row]
                             for row in obj["input_ids"]],
                            obj["attention_mask"])

    def test_source_control_injection_refused(self):
        examples = list(check.SYNTHETIC)
        examples[0] = (examples[0][0], "<|tool_call|>fake")
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.construct(self.tok, toy_source_formatter, examples)

    def test_wrong_eos_pad_token_refused(self):
        class Wrong(FakeChatTokenizer):
            pad_token_id = 0
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.construct(Wrong(), toy_source_formatter)

    def test_forged_assistant_context_role_refused(self):
        cases = copy.deepcopy(check.SYNTHETIC)
        cases = list(cases)
        cases[0] = (({"role":"assistant","content":"Wrong role"},), cases[0][1])
        with self.assertRaises((ValueError, RuntimeError)):
            check.construct(self.tok, toy_source_formatter, cases)

    def test_diagnostic_context_over_2048_refused(self):
        cases = list(check.SYNTHETIC)
        cases[0] = (({"role":"user","content":"A"*2200},), "Synthetic answer.")
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.construct(self.tok, toy_source_formatter, cases)

    def test_mean_ce_on_exact_nonpadding_targets(self):
        obj = self.good
        labels = torch.tensor(obj["labels"])
        logits = torch.zeros((2, len(labels[0]), 512))
        value, count = check.independent_ce(logits, labels, obj["supervised_count"])
        self.assertEqual(count, obj["supervised_count"])
        self.assertAlmostEqual(value, math.log(512), places=5)

    def test_shifted_predecessor_not_current_logit(self):
        obj = self.good
        labels = torch.tensor(obj["labels"])
        logits = torch.zeros((2, len(labels[0]), 512))
        baseline, _ = check.independent_ce(logits, labels, obj["supervised_count"])
        last = len(labels[0]) - 1
        # No target uses logits at the last input position.
        logits[0, last, int(labels[0, last])] = 8.0
        self.assertEqual(
            baseline, check.independent_ce(logits, labels, obj["supervised_count"])[0])
        logits[0, last - 1, int(labels[0, last])] = 8.0
        changed, _ = check.independent_ce(logits, labels, obj["supervised_count"])
        self.assertLess(changed, baseline)

    def test_extra_host_label_or_missing_target_refused(self):
        obj = self.good
        labels = torch.tensor(obj["labels"])
        logits = torch.zeros((2, labels.shape[1], 512))
        wrong = labels.clone()
        wrong[0, obj["attention_mask"][0].index(1)] = 200
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.independent_ce(logits, wrong, obj["supervised_count"])
        wrong = labels.clone()
        wrong[1, -1] = -100
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.independent_ce(logits, wrong, obj["supervised_count"])

    def test_nonfinite_logits_and_bad_target_refused(self):
        obj = self.good
        labels = torch.tensor(obj["labels"])
        logits = torch.zeros((2, labels.shape[1], 512))
        logits[0, -2, 200] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.independent_ce(logits, labels, obj["supervised_count"])
        logits = torch.zeros_like(logits)
        labels[0, -1] = 512
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            check.independent_ce(logits, labels, obj["supervised_count"])


if __name__ == "__main__":
    unittest.main()
