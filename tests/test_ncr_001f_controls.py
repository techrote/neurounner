"""NCR-001F offline fairness and leakage tests. No model downloads or training."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001_core import PersistentDepthNCA  # noqa: E402
from ncr_001f_controls import GatedMLPControl, matched_control_contract  # noqa: E402
from ncr_001f_plan import build_plan, fixed_schedule, validate_split, load_json  # noqa: E402


class FrozenToy(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.backbone = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(6))
        self.out = nn.Linear(hidden_size, 11)
        self.adapter = GatedMLPControl(hidden_size)
        for parameter in [*self.backbone.parameters(), *self.out.parameters()]:
            parameter.requires_grad_(False)

    def forward(self, hidden, enabled=True):
        tap = 0
        for i, layer in enumerate(self.backbone):
            hidden = torch.tanh(layer(hidden))
            if enabled and i in (1, 3, 5):
                hidden = self.adapter.step(hidden, tap_index=tap)
                tap += 1
        return self.out(hidden)


class MatchAndLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_parameter_matching_for_four_models(self):
        model_widths = {"smollm2": 576, "sprocket_chat": 1280,
                        "sprocket_instruct": 1280, "void1": 640}
        expected = {
            576: (297048, 296576),
            1280: (658904, 658432),
            640: (329944, 329472),
        }
        for key, width in model_widths.items():
            nca = PersistentDepthNCA(width)
            mlp = GatedMLPControl(width)
            self.assertEqual((nca.trainable_parameters(), mlp.trainable_parameters()),
                             expected[width], key)
            match = matched_control_contract(
                width, nca_parameters=nca.trainable_parameters())
            self.assertTrue(match["within_limit"])
            self.assertLess(match["relative_absolute_difference"], 0.002)
            self.assertFalse(match["compares_FLOPs"])

    def test_finite_exact_initial_noop_at_three_taps(self):
        torch.manual_seed(21)
        core = GatedMLPControl(24)
        hidden = torch.randn(2, 4, 24)
        for tap in (0, 1, 2):
            out = core.step(hidden, tap_index=tap)
            self.assertTrue(torch.equal(out, hidden))
            self.assertTrue(torch.isfinite(out).all())

    def test_per_token_batch_and_order_independence(self):
        torch.manual_seed(22)
        core = GatedMLPControl(16)
        core.down.weight.data.fill_(0.02)
        x = torch.randn(2, 6, 16)
        y = x.clone()
        y[0, 5] += 50
        y[1, 0] -= 50
        left = core.step(x, tap_index=1)
        right = core.step(y, tap_index=1)
        self.assertTrue(torch.equal(left[0, :5], right[0, :5]))
        self.assertTrue(torch.equal(left[1, 1:], right[1, 1:]))
        order = torch.tensor([5, 2, 0, 4, 3, 1])
        permuted = core.step(x[:, order], tap_index=1)
        self.assertTrue(torch.allclose(left[:, order], permuted, rtol=1e-6, atol=1e-6))
        self.assertFalse(hasattr(core, "field"))
        self.assertFalse(hasattr(core, "state"))

    def test_frozen_toy_readout_then_upstream_grad(self):
        torch.manual_seed(23)
        m = FrozenToy(24)
        x = torch.randn(1, 5, 24)
        original = m(x, enabled=False)
        self.assertTrue(torch.equal(original, m(x)))
        frozen = {name: p.detach().clone() for name, p in m.backbone.named_parameters()}
        opt = torch.optim.AdamW(m.adapter.parameters(), lr=0.01)
        loss = m(x).square().sum()
        loss.backward()
        self.assertGreater(m.adapter.down.weight.grad.abs().sum().item(), 0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(original, m(x)))
        m(x).square().sum().backward()
        self.assertGreater(m.adapter.up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(m.adapter.feature_gate.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertTrue(all(torch.equal(p, frozen[name]) for name, p in
                            m.backbone.named_parameters()))

    def test_state_dict_architecture_identity(self):
        torch.manual_seed(24)
        m = GatedMLPControl(32)
        raw = io.BytesIO()
        torch.save(m.state_dict(), raw)
        raw.seek(0)
        restored = GatedMLPControl(32)
        restored.load_state_dict(torch.load(raw, weights_only=True))
        for key, value in m.state_dict().items():
            copy = restored.state_dict()[key]
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, copy), key)
            else:
                self.assertEqual(value, copy)
        with self.assertRaises(RuntimeError):
            GatedMLPControl(32, bottleneck=128).load_state_dict(m.state_dict())
        with self.assertRaises(ValueError):
            m.step(torch.randn(2, 1, 32), tap_index=3)
        with self.assertRaises(ValueError):
            m.step(torch.randn(2, 1, 16), tap_index=2)

    def test_match_contract_rejects_wrong_size(self):
        with self.assertRaises(RuntimeError):
            matched_control_contract(576, nca_parameters=1)
        with self.assertRaises(ValueError):
            matched_control_contract(576, nca_parameters=297048, limit=0.1)

    def test_locked_pretraining_gates_fail_closed_under_optimization(self):
        """Must reject changed preregistration even when Python strips asserts."""
        original = load_json(ROOT / "experiments/ncr-001/protocol.json")
        cases = (
            ("field scalar count", ("architecture", "state_geometry", "scalars_per_token"), 128),
            ("tap count", ("architecture", "tap_count"), 2),
            ("steps per tap", ("architecture", "nca_steps_per_tap"), 1),
            ("optimizer", ("training", "optimizer"), "SGD"),
            ("optimizer learning rate", ("training", "optimizer_lr"), 0.08),
            ("optimizer betas", ("training", "optimizer_betas"), [0.9, 0.9]),
            ("weight decay", ("training", "weight_decay"), 0.1),
            ("gradient clip", ("training", "grad_clip_norm"), 2),
            ("gradient clip", ("training", "grad_clip_norm"), True),
            ("MLP bottleneck", ("match_constraints", "conventional_adapter_bottleneck"), 128),
        )
        for label, keys, replacement in cases:
            with self.subTest(gate=label, replacement=replacement):
                changed = json.loads(json.dumps(original))
                current = changed
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = replacement
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "altered-protocol.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, f"NCR-001F preregistered {label} drift"
                    ):
                        build_plan(protocol_path=path)

    def test_train_heldout_family_collision_is_blocking(self):
        m = {"train_examples": 1, "heldout_examples": 1,
             "train_family_count": 1, "heldout_family_count": 1,
             "labels": ["VALID"],
             "label_counts": {"train": {"VALID": 1}, "heldout": {"VALID": 1}}}
        a = {"id": "a", "family": "same", "split": "train",
             "output": {"decision": "VALID"}}
        b = {"id": "b", "family": "same", "split": "eval",
             "output": {"decision": "VALID"}}
        with self.assertRaisesRegex(RuntimeError, "families overlap"):
            validate_split([a], [b], m)
        b["family"] = "other"
        self.assertEqual(validate_split([a], [b], m)["family_overlap"], 0)
        b["id"] = "a"
        with self.assertRaisesRegex(RuntimeError, "leaked into train"):
            validate_split([a], [b], m)

    def test_schedule_determinism_and_seed_difference(self):
        a = fixed_schedule(768, 192, 20261026)
        b = fixed_schedule(768, 192, 20261026)
        c = fixed_schedule(768, 192, 20261027)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 192)
        self.assertTrue(all(0 <= row < 768 for row in a))

    def test_four_model_plan_is_read_only_training_blocked(self):
        plan = build_plan()
        self.assertFalse(plan["training_authorized"])
        self.assertFalse(plan["all_model_conformance_gate_pass"])
        self.assertEqual(set(plan["lanes"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        self.assertEqual(plan["dataset_validation"]["train_rows"], 768)
        self.assertEqual(plan["dataset_validation"]["heldout_rows"], 128)
        self.assertEqual(len(plan["sealed_fixtures"]), 5)
        self.assertEqual(len(plan["schedules"]), 3)
        self.assertTrue(all(not row["training_allowed"] for row in plan["lanes"].values()))
        self.assertEqual(plan["lanes"]["void1"]["token_limit_unit"], "byte_tokenizer_unit")

    def test_tampered_fixture_hash_and_training_toggle_fail_closed(self):
        original = load_json(ROOT / "experiments/ncr-001/protocol.json")
        for change in ("hash", "training"):
            p = json.loads(json.dumps(original))
            if change == "hash":
                p["datasets"]["sprocket_chat_probes"]["sha256"] = "0" * 64
            else:
                p["train_authorized"] = True
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "protocol.json"
                path.write_text(json.dumps(p), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    build_plan(protocol_path=path)


if __name__ == "__main__":
    unittest.main()
