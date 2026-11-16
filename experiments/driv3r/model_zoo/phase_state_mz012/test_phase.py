"""MZ-012 host, recurrence and cached-state contracts."""
import unittest
from experiments.driv3r.model_zoo.phase_state_mz012.run import (
    MAX_T, VariableCorridorWorld, build_models, counts, dataset,
    demonstration, incremental, phase_model, tensors,
)
from experiments.driv3r.model_zoo.material_forks_mz009 import run as host

class PhaseContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(61212)

    def test_hidden_cues_and_longer_out_of_range_corridors(self):
        a = VariableCorridorWorld(7009, 0, 0, True)
        b = VariableCorridorWorld(7009, 1, 0, True)
        self.assertEqual(a.lengths, b.lengths)
        self.assertGreaterEqual(a.lengths[0], 14)
        self.assertNotEqual(a.observe(), b.observe())
        for _ in range(a.lengths[0]):
            a.step(0)
            b.step(0)
        self.assertEqual(a.stage, "fork")
        self.assertEqual(a.observe(), b.observe())
        self.assertNotEqual(a.preferred_action(), b.preferred_action())

    def test_real_recovery_and_protected_material_injection(self):
        for cue in (0, 1):
            for inv in (0, 1):
                seq, labels, _ = demonstration(7011, cue, inv, 1, ood=True)
                self.assertGreater(len(seq), 30)
                self.assertLessEqual(len(seq), MAX_T)
                self.assertIn(3, labels)  # RETRACE
                self.assertEqual(labels[-1], 4)  # STOP after painted three
                self.assertTrue(any(x[2] for x in seq))  # hazard stage

    def test_source_split_disjoint(self):
        train = dataset(range(1000, 1003))
        shifted = dataset(range(6000, 6003), ood=True)
        self.assertEqual(len(train), 36)
        self.assertEqual(len(shifted), 36)
        self.assertTrue({r[2][0] for r in train}.isdisjoint(
            {r[2][0] for r in shifted}))

    def test_models_and_recurrent_phase_gradients(self):
        torch = self.torch
        models = build_models(torch)
        n = counts(models)
        self.assertGreater(n["phase"], 100_000)
        self.assertLess(n["phase"], 300_000)
        self.assertLess(n["phase"], n["small_gru"])
        x, y = tensors(torch, dataset([1001, 1002]))
        logits = models["phase"](x)
        self.assertEqual(tuple(logits.shape), (24, MAX_T, len(host.ACTIONS)))
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1,2), y, ignore_index=-100)
        loss.backward()
        self.assertGreater(
            float(models["phase"].cells[0].proj.weight.grad.abs().sum()), 0.)
        self.assertIsNotNone(models["phase"].cells[0].angle.grad)

    def test_prefix_and_caching_exact_action_parity(self):
        torch = self.torch
        for name, model in build_models(torch).items():
            model.eval()
            x = torch.randn(2, 15, host.D)
            state = None
            with torch.inference_mode():
                full = model(x)
                for step in range(x.shape[1]):
                    latest, state = incremental(torch, model, x[:,step], state)
                    self.assertTrue(torch.allclose(
                        latest, full[:,step], atol=3e-5, rtol=3e-5),
                        msg=f"{name} prefix mismatch at step {step}")

if __name__ == "__main__":
    unittest.main()
