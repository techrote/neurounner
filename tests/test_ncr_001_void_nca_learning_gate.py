"""Synthetic-only fail-closed sentinels for NCR-001E void.1 diagnostic gradients.

No HF model/tokenizer, network, heldout fixture, optimizer step, or inference.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001_void_nca_learning_gate import (  # noqa: E402
    PROMPT, COMPLETION, EXPECTED_GRADIENTS, check_gradients,
    finite_loss, make_sample,
)


class FrozenBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3), requires_grad=False)


class FakeNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(4, 4, bias=False)
        self.observe = nn.Linear(4, 4, bias=False)
        self.perception = nn.Conv2d(1, 1, 3)
        self.rule = nn.Conv2d(1, 1, 1)
        self.input_norm = nn.LayerNorm(4)

    def add_grads(self, *, warmed):
        for name, param in self.named_parameters():
            param.grad = torch.ones_like(param) if (
                warmed or name == "readout.weight") else torch.zeros_like(param)
        return self


def good_serializer(_tok, prompt, completion):
    assert prompt == PROMPT and completion == COMPLETION
    prefix, tail = list(prompt.encode("utf-8")), list(completion.encode("utf-8"))
    return {
        "input_ids": prefix + tail + [258],
        "labels": [-100] * len(prefix) + tail + [258],
        "attention_mask": [1] * (len(prefix) + len(tail) + 1),
        "causal_shift_supervised_units": len(tail) + 1,
    }


class VoidLearningSentinels(unittest.TestCase):
    def test_actual_byte_eos_and_prompt_mask(self):
        ids, labels, mask, w = make_sample(good_serializer, None)
        self.assertEqual(w["supervised_eos_id"], 258)
        self.assertEqual(int(ids[0, -1]), 258)
        self.assertEqual(int(labels[0, -1]), 258)
        self.assertEqual(int((labels != -100).sum()), w["supervised_after_shift"])
        self.assertTrue(bool(torch.all(mask == 1)))
        self.assertTrue(bool(torch.all(labels[0, :w["prompt_bytes"]] == -100)))
        self.assertEqual(w["source_text_utf8_bytes"],
                         len((PROMPT + COMPLETION).encode("utf-8")))
        self.assertFalse(w["heldout"])

    def test_text_eos_bytes_rejected(self):
        def broken(tok, prompt, completion):
            row = good_serializer(tok, prompt, completion)
            row["input_ids"][-1:] = [60, 101, 111, 115, 62]
            row["labels"][-1:] = [60, 101, 111, 115, 62]
            row["attention_mask"] = [1] * len(row["input_ids"])
            return row
        with self.assertRaisesRegex(RuntimeError, "literal EOS"):
            make_sample(broken, None)

    def test_supervision_prompt_leak_rejected(self):
        def broken(tok, prompt, completion):
            row = good_serializer(tok, prompt, completion)
            row["labels"][0] = row["input_ids"][0]
            return row
        with self.assertRaisesRegex(RuntimeError, "prompt leak"):
            make_sample(broken, None)

    def test_fake_supervised_length_rejected(self):
        def broken(tok, prompt, completion):
            row = good_serializer(tok, prompt, completion)
            row["causal_shift_supervised_units"] += 1
            return row
        with self.assertRaisesRegex(RuntimeError, "falsified"):
            make_sample(broken, None)

    def test_required_gradients_warmup(self):
        base, nca = FrozenBase(), FakeNCA().add_grads(warmed=False)
        first = check_gradients(base, nca, warmed=False)
        self.assertGreater(first["readout.weight"], 0)
        self.assertEqual(first["observe.weight"], 0)
        warm = check_gradients(base, nca.add_grads(warmed=True), warmed=True)
        self.assertTrue(all(warm[n] > 0 for n in EXPECTED_GRADIENTS))

    def test_early_upstream_gradient_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "bypassed zero readout"):
            check_gradients(FrozenBase(), FakeNCA().add_grads(warmed=True),
                            warmed=False)

    def test_missing_post_warmup_gradient_rejected(self):
        nca = FakeNCA().add_grads(warmed=True)
        nca.perception.weight.grad = None
        with self.assertRaisesRegex(RuntimeError, "absent post-warmup"):
            check_gradients(FrozenBase(), nca, warmed=True)

    def test_nonfinite_gradient_rejected(self):
        nca = FakeNCA().add_grads(warmed=True)
        nca.rule.weight.grad.flatten()[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            check_gradients(FrozenBase(), nca, warmed=True)

    def test_unfrozen_base_rejected(self):
        base = FrozenBase()
        base.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "source backbone"):
            check_gradients(base, FakeNCA().add_grads(warmed=True), warmed=True)

    def test_bad_supervised_losses_rejected(self):
        self.assertEqual(finite_loss(torch.tensor(2.0, requires_grad=True)), 2.0)
        for loss in (
            torch.tensor(float("nan"), requires_grad=True),
            torch.tensor(float("inf"), requires_grad=True),
            torch.tensor(0.0, requires_grad=True),
            torch.tensor(-1.0, requires_grad=True),
            torch.tensor(2.0),
            torch.ones(3, requires_grad=True),
        ):
            with self.subTest(loss=repr(loss)):
                with self.assertRaisesRegex(RuntimeError, "native supervised"):
                    finite_loss(loss)

    def test_four_frozen_lanes_and_no_evaluation(self):
        protocol = json.loads(
            (ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(protocol["train_authorized"], False)
        self.assertEqual(set(protocol["models"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        src = (ROOT / "scripts/ncr_001_void_nca_learning_gate.py").read_text()
        self.assertIn('"comparative_training_performed": False', src)
        self.assertIn('"full_72_conformance_pass": False', src)
        self.assertIn('"diagnostic_optimizer_steps": 0', src)
        self.assertIn('REPORT["diagnostic_optimizer_steps"] = 1', src)
        self.assertNotIn("datasets/train-003/", src)
        self.assertNotIn("eval-final", src)


if __name__ == "__main__":
    unittest.main()
