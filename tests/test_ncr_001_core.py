"""Fast offline tests; never import models, downloads, GPU, or run training jobs."""
from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ncr_001_core", ROOT / "scripts/ncr_001_core.py")
assert spec and spec.loader
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
PersistentDepthNCA = module.PersistentDepthNCA
DepthFieldState = module.DepthFieldState


class ToyDepthModel(nn.Module):
    """Frozen 6-block toy: effect of NCA tested without a real LM."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.backbone = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(6))
        self.head = nn.Linear(hidden_size, 5)
        self.nca = PersistentDepthNCA(hidden_size)
        for p in list(self.backbone.parameters()) + list(self.head.parameters()):
            p.requires_grad_(False)

    def forward(self, hidden, *, with_nca=True, steps=None, reset=False):
        state = None
        for block_id, block in enumerate(self.backbone):
            hidden = torch.tanh(block(hidden))
            if with_nca and block_id in (1, 3, 5):
                tap = (1, 3, 5).index(block_id)
                hidden, state = self.nca.step(
                    hidden, state, tap_index=tap, iterations=steps,
                    reset_each_tap=reset,
                )
        return self.head(hidden)


class NCRCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_geometry_locality_and_boundary(self):
        torch.manual_seed(71)
        core = PersistentDepthNCA(16)
        quiet = torch.zeros(1, 1, 4, 8, 8)
        impulse = quiet.clone()
        impulse[0, 0, 0, 3, 3] = 1.0
        diff = core.evolve(impulse, steps=1) - core.evolve(quiet, steps=1)
        for row in range(8):
            for col in range(8):
                if max(abs(row - 3), abs(col - 3)) > 1:
                    self.assertTrue(torch.equal(diff[0, 0, :, row, col],
                                                torch.zeros(4)))
        impulse_corner = quiet.clone()
        impulse_corner[0, 0, 1, 0, 0] = 0.7
        out_corner = core.evolve(impulse_corner, steps=1) - core.evolve(quiet, steps=1)
        self.assertTrue(torch.equal(out_corner[0, 0, :, -1, -1], torch.zeros(4)))
        self.assertTrue(torch.isfinite(core.evolve(impulse, steps=100)).all())

    def test_true_zero_readout_and_depth_evolution(self):
        torch.manual_seed(70)
        core = PersistentDepthNCA(16)
        h = torch.randn(2, 3, 16)
        s = None
        seen = []
        for t in range(3):
            mod, s = core.step(h, s, tap_index=t)
            self.assertTrue(torch.equal(mod, h), "initial adapter must be no-op")
            self.assertEqual(s.field.shape, (2, 3, 4, 8, 8))
            self.assertTrue(torch.isfinite(s.field).all())
            self.assertLessEqual(s.field.abs().max().item(), 1.0)
            seen.append(s.field.detach().clone())
        self.assertFalse(torch.allclose(seen[0], seen[1]))
        self.assertFalse(torch.allclose(seen[1], seen[2]))
        self.assertGreater(core.trainable_parameters(), 0)

    def test_per_token_and_per_batch_independence(self):
        torch.manual_seed(72)
        core = PersistentDepthNCA(12)
        a = torch.randn(2, 4, 12)
        b = a.clone()
        b[0, 3] += 100
        b[1, 1] -= 90
        # Three depth taps: all fields for same other token position remain same.
        sa = sb = None
        for tap in range(3):
            _, sa = core.step(a, sa, tap_index=tap)
            _, sb = core.step(b, sb, tap_index=tap)
            self.assertTrue(torch.equal(sa.field[0, :3], sb.field[0, :3]))
            self.assertTrue(torch.equal(sa.field[1, 0], sb.field[1, 0]))
            self.assertTrue(torch.equal(sa.field[1, 2:], sb.field[1, 2:]))
        # A new request always recomputes its field from only its own input.
        _, again = core.step(a, None, tap_index=0)
        _, first = core.step(a, None, tap_index=0)
        self.assertTrue(torch.equal(again.field, first.field))

    def test_token_permutation_equivariance(self):
        torch.manual_seed(73)
        core = PersistentDepthNCA(9)
        h = torch.randn(2, 5, 9)
        order = torch.tensor([4, 1, 3, 0, 2])
        s = p = None
        for tap in range(3):
            a, s = core.step(h, s, tap_index=tap)
            b, p = core.step(h[:, order], p, tap_index=tap)
            self.assertTrue(torch.allclose(a[:, order], b, rtol=1e-6, atol=1e-6))
            self.assertTrue(torch.allclose(s.field[:, order], p.field,
                                            rtol=1e-6, atol=1e-6))

    def test_rejects_misordered_or_reused_depth_state(self):
        core = PersistentDepthNCA(8)
        x = torch.randn(1, 2, 8)
        _, state = core.step(x, None, tap_index=0)
        with self.assertRaises(ValueError):
            core.step(x, None, tap_index=1)
        with self.assertRaises(ValueError):
            core.step(x, state, tap_index=2)
        with self.assertRaises(ValueError):
            core.step(torch.randn(2, 2, 8), state, tap_index=1)
        with self.assertRaises(ValueError):
            core.step(x, state, tap_index=3)
        with self.assertRaises(ValueError):
            core.evolve(torch.ones(1, 2, 8, 8), steps=1)
        with self.assertRaises(ValueError):
            core.evolve(torch.ones(1, 1, 4, 8, 8), steps=0)

    def test_synthetic_frozen_backbone_first_step_then_upstream_grad(self):
        torch.manual_seed(74)
        toy = ToyDepthModel()
        x = torch.randn(1, 3, 16)
        base_copy = {k: p.detach().clone() for k, p in toy.backbone.named_parameters()}
        out_base = toy(x, with_nca=False)
        self.assertTrue(torch.equal(out_base, toy(x)))
        opt = torch.optim.AdamW(toy.nca.parameters(), lr=1e-2)
        loss = toy(x).square().sum()
        loss.backward()
        self.assertGreater(toy.nca.readout.weight.grad.abs().sum().item(), 0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(out_base, toy(x)))
        loss2 = toy(x).square().sum()
        loss2.backward()
        self.assertGreater(toy.nca.perception.weight.grad.abs().sum().item(), 0)
        self.assertGreater(toy.nca.observe.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in toy.backbone.parameters()))
        self.assertTrue(all(torch.equal(p, base_copy[k]) for k, p in
                            toy.backbone.named_parameters()))

    def test_model_specific_widths_checkpoint_and_no_tap_memory(self):
        for width in (576, 1280, 640):
            torch.manual_seed(75)
            core = PersistentDepthNCA(width)
            h = torch.randn(1, 1, width)
            out, state = core.step(h, None, tap_index=0)
            self.assertTrue(torch.equal(out, h))
            self.assertEqual(state.next_tap, 1)
            self.assertNotIn("field", vars(core))
            buffer = io.BytesIO()
            torch.save(core.state_dict(), buffer)
            buffer.seek(0)
            restored = PersistentDepthNCA(width)
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            for key, param in core.state_dict().items():
                other = restored.state_dict()[key]
                if isinstance(param, torch.Tensor):
                    self.assertTrue(torch.equal(param, other), key)
                else:
                    self.assertEqual(param, other)
            wrong_width = PersistentDepthNCA(width + 1)
            with self.assertRaises(RuntimeError):
                wrong_width.load_state_dict(core.state_dict())


if __name__ == "__main__":
    unittest.main()
