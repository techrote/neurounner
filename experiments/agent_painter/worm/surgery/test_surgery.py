"""Fast actual FP32 tensor tests for surgical effects and restoration."""
import importlib.util
import pathlib
import unittest
import sys

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("surgery_core", ROOT / "core.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


class Tests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.parameters = [
            ("layer.0.lora_A.default.weight",
             torch.nn.Parameter(torch.linspace(-2, 2, 40).reshape(5, 8))),
            ("layer.0.lora_B.default.weight",
             torch.nn.Parameter(torch.linspace(-1, 1, 40).reshape(8, 5)))]
        self.original = core.flatten(self.parameters)
        self.scores = torch.arange(80, dtype=torch.float32) / 80

    def test_identity_restores(self):
        values, receipt = core.mutate(self.original, self.parameters, core.TARGETED[0], self.scores)
        self.assertTrue(self.t.equal(values, self.original))
        self.assertEqual(receipt["changed_count"], 0)
        core.apply_flat(self.parameters, values)
        self.assertTrue(self.t.equal(core.flatten(self.parameters), self.original))

    def test_top_saliency_zero(self):
        values, receipt = core.mutate(self.original, self.parameters, core.TARGETED[1], self.scores)
        indices = core.selected_indices(self.scores, .10)
        self.assertEqual(len(indices), 8)
        self.assertTrue(self.t.all(values[indices] == 0).item())
        self.assertGreater(receipt["changed_count"], 0)

    def test_hybrid_disjoint(self):
        values, receipt = core.mutate(self.original, self.parameters, core.TARGETED[2], self.scores)
        self.assertEqual(receipt["selected_count"], 8)
        self.assertEqual(receipt["extreme_count"], 40)
        self.assertEqual(int((values == self.t.finfo(self.t.float32).min).sum().item()), 40)

    def test_global_half_extreme_and_random(self):
        values, receipt = core.mutate(self.original, self.parameters, core.CHAOS[1])
        self.assertEqual(receipt["extreme_count"], 40)
        self.assertTrue(self.t.isfinite(values).all().item())
        a, _ = core.mutate(self.original, self.parameters, core.CHAOS[3])
        b, _ = core.mutate(self.original, self.parameters, core.CHAOS[3])
        self.assertTrue(self.t.equal(a, b))
        self.assertFalse(self.t.equal(a, self.original))

    def test_mid_and_restore(self):
        values, _ = core.mutate(self.original, self.parameters, core.CHAOS[2])
        self.assertEqual(len(values), 80)
        core.apply_flat(self.parameters, values)
        self.assertFalse(self.t.equal(core.flatten(self.parameters), self.original))
        core.apply_flat(self.parameters, self.original)
        self.assertTrue(self.t.equal(core.flatten(self.parameters), self.original))

    def test_no_positive_right_attr_refusal(self):
        with self.assertRaises(ValueError):
            core.selected_indices(self.t.zeros(80), .5)


if __name__ == "__main__":
    unittest.main()
