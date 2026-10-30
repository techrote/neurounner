"""NCR-001 source-only adversarial guards for one-step NCA learning conformance.

These tests never load any frozen weight, optimize parameters or open heldouts.
The physical Chat/Instruct tests are separate pinned GitHub Actions matrix jobs.
"""
from pathlib import Path
import json
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001_sprocket_learning_gate import (  # noqa: E402
    finite_supervised_loss, supervised_sample, verify_gradient_gate,
)


class FakeBackbone(nn.Module):
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

    def with_grads(self, *, warmed):
        for name, p in self.named_parameters():
            p.grad = torch.ones_like(p) if (warmed or name == "readout.weight") \
                else torch.zeros_like(p)
        return self


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        suffix = "<|end|>"
        if text.endswith(suffix):
            return {"input_ids": [100 + ord(c) for c in text[:-len(suffix)]] + [5]}
        return {"input_ids": [100 + ord(c) for c in text]}


def fake_serialize(rows, *, variant):
    assert variant in ("chat", "instruct")
    return "<|user|>" + rows[0]["content"] + "<|end|><|assistant|>"


class LearningGateSentinels(unittest.TestCase):
    def test_supervision_only_assistant_completion_and_real_eos(self):
        for variant in ("chat", "instruct"):
            with self.subTest(variant=variant):
                ids, labels, attention, desc = supervised_sample(
                    FakeTokenizer(), fake_serialize, variant)
                self.assertEqual(desc["assistant_eos_id"], 5)
                self.assertEqual(desc["sequence_units"], ids.shape[1])
                self.assertTrue(torch.all(attention == 1))
                self.assertTrue(torch.all(labels[0, :desc["prefix_units"]] == -100))
                self.assertEqual(labels[0, -1].item(), 5)
                self.assertEqual(
                    int((labels != -100).sum()), desc["supervised_units"])

    def test_literal_eos_replacement_is_not_accepted(self):
        class BrokenTokenizer(FakeTokenizer):
            def __call__(self, text, *, add_special_tokens=False):
                return {"input_ids": [101, 102, 103]}
        with self.assertRaisesRegex(RuntimeError, "genuine end token"):
            supervised_sample(BrokenTokenizer(), fake_serialize, "chat")

    def test_first_step_zero_readout_gradient_handoff(self):
        b = FakeBackbone()
        n = FakeNCA().with_grads(warmed=False)
        observed = verify_gradient_gate(b, n, warmed=False)
        self.assertGreater(observed["readout.weight"], 0)
        self.assertEqual(observed["observe.weight"], 0)

    def test_first_step_illegal_upstream_gradient_refused(self):
        b = FakeBackbone()
        n = FakeNCA().with_grads(warmed=True)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly touched upstream"):
            verify_gradient_gate(b, n, warmed=False)

    def test_post_warmup_all_required_gradients_positive(self):
        b = FakeBackbone()
        n = FakeNCA().with_grads(warmed=True)
        observed = verify_gradient_gate(b, n, warmed=True)
        self.assertTrue(all(v > 0 for v in observed.values()))

    def test_post_warmup_absent_upstream_gradient_refused(self):
        b = FakeBackbone()
        n = FakeNCA().with_grads(warmed=True)
        n.perception.weight.grad.zero_()
        with self.assertRaisesRegex(RuntimeError, "Post-warmup"):
            verify_gradient_gate(b, n, warmed=True)

    def test_nonfinite_gradient_refused(self):
        b = FakeBackbone()
        n = FakeNCA().with_grads(warmed=True)
        n.observe.weight.grad.flatten()[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            verify_gradient_gate(b, n, warmed=True)

    def test_frozen_model_gradient_refused(self):
        b = FakeBackbone()
        n = FakeNCA().with_grads(warmed=True)
        b.weight.grad = torch.ones_like(b.weight)
        with self.assertRaisesRegex(RuntimeError, "Frozen"):
            verify_gradient_gate(b, n, warmed=True)

    def test_supervised_loss_requires_nonzero_finite_differentiable_scalar(self):
        p = torch.tensor(2.0, requires_grad=True)
        self.assertEqual(finite_supervised_loss(p), 2.0)
        for bad in (torch.tensor(float("nan"), requires_grad=True),
                    torch.tensor(0.0, requires_grad=True),
                    torch.tensor(-1.0, requires_grad=True),
                    torch.tensor(2.0),
                    torch.ones(2, requires_grad=True)):
            with self.assertRaisesRegex(RuntimeError, "(?i)supervised loss"):
                finite_supervised_loss(bad)

    def test_four_lane_protocol_and_no_comparative_runner(self):
        protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertIs(protocol["train_authorized"], False)
        self.assertEqual(set(protocol["models"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        src = (ROOT / "scripts/ncr_001_sprocket_learning_gate.py").read_text()
        self.assertIn('"comparative_training_performed": False', src)
        self.assertIn('"full_72_conformance_pass": False', src)
        self.assertIn('"tool_memory_semantics_qualified": False', src)
        self.assertIn('REPORT["diagnostic_optimizer_steps"] = 1', src)
        self.assertNotIn("heldout.json", src)
        self.assertNotIn("eval-final", src)


if __name__ == "__main__":
    unittest.main()
