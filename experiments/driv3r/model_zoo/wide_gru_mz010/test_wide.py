"""MZ-010 source-only scope, memory-law and actual large-GRU gradient tests."""
import unittest

from experiments.driv3r.model_zoo.material_forks_mz009 import run as env
from experiments.driv3r.model_zoo.wide_gru_mz010 import run as mz010


class MZ010Tests(unittest.TestCase):
    def test_world_contract_and_nonleaking_cues(self):
        for cue in (0, 1):
            for invert in (0, 1):
                w = env.World(7111, cue, invert)
                for _ in range(8):
                    w.step(w.preferred_action())
                self.assertEqual(w.stage, "fork")
                self.assertEqual(w.observe()[17:21], [0.0] * 4)
                self.assertTrue(w.legal()[w.preferred_action()])

    def test_size_and_independent_parameters(self):
        import torch
        torch.set_num_threads(2)
        models = mz010.build_models(torch)
        n = mz010.model_counts(models)
        self.assertGreaterEqual(n["wide"], 20_000_000)
        self.assertLessEqual(n["wide"], 24_000_000)
        self.assertLess(n["small"], 250_000)
        self.assertLess(n["current"], 10_000)
        self.assertEqual(models["wide"].core.num_layers, 2)
        self.assertEqual(models["wide"].core.hidden_size, 1504)

    def test_actual_large_recurrent_gradient_and_recovery_labels(self):
        import torch
        torch.set_num_threads(2)
        examples = env.dataset([9000])
        x, y = env.to_tensors(torch, examples)
        model = mz010.build_models(torch)["wide"]
        logits = model(x[:2])
        self.assertEqual(tuple(logits.shape), (2, env.MAX_STEPS, len(env.ACTIONS)))
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y[:2], ignore_index=-100)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(float(model.core.weight_hh_l0.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.core.weight_hh_l1.grad.abs().sum()), 0.0)
        for obs, labels, _ in examples:
            for step, target in zip(obs, labels):
                self.assertTrue(bool(step[8 + target]))

    def test_seed_families_are_disjoint(self):
        a = env.dataset(range(1000, 1004))
        b = env.dataset(range(3000, 3004))
        c = env.dataset(range(5000, 5004))
        keys = [{r[2][0] for r in rows} for rows in (a, b, c)]
        self.assertFalse(keys[0] & keys[1])
        self.assertFalse(keys[0] & keys[2])
        self.assertFalse(keys[1] & keys[2])


if __name__ == "__main__":
    unittest.main()
