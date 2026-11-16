"""MZ-011 tests: causal host boundary, stable diagonal state, legal masked play."""
import unittest

from experiments.driv3r.model_zoo.material_forks_mz009 import run as env
from experiments.driv3r.model_zoo.diag_state_mz011.run import (
    diagonal_model, build_models, stateful_play, parameter_counts,
)


class DiagonalStateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(61111)

    def test_size_and_distinct_architecture(self):
        torch = self.torch
        models = build_models(torch)
        p = parameter_counts(models)
        self.assertGreater(p["diagonal"], 100_000)
        self.assertLess(p["diagonal"], 200_000)
        self.assertGreater(p["small_gru"], p["diagonal"])
        self.assertEqual(len(models["diagonal"].cells), 2)
        self.assertFalse(any(isinstance(m, torch.nn.GRU)
                             for m in models["diagonal"].modules()))

    def test_persistent_state_equivalent_to_full_prefix(self):
        torch = self.torch
        model = diagonal_model(torch).eval()
        x = torch.randn(2, 16, env.D)
        state = None
        with torch.inference_mode():
            full = model(x)
            for t in range(x.shape[1]):
                logits, state = model.step(x[:, t], state)
                self.assertTrue(torch.allclose(
                    logits, full[:, t], rtol=2e-5, atol=2e-5),
                    msg=f"cached state != full-history forward at {t}")
        self.assertEqual(len(state), 2)
        self.assertEqual(tuple(state[0].shape), (2, 224))

    def test_state_has_real_trainable_temporal_gradient(self):
        torch = self.torch
        model = diagonal_model(torch)
        x, y = env.to_tensors(torch, env.dataset([9001, 9002]))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (24, env.MAX_STEPS,
                                                len(env.ACTIONS)))
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y, ignore_index=-100)
        loss.backward()
        for cell in model.cells:
            self.assertGreater(float(cell.proj.weight.grad.abs().sum()), 0.0)
        with torch.no_grad():
            initial_retain = torch.sigmoid(
                model.cells[0].proj.bias[:224])
        self.assertGreater(float(initial_retain.mean()), .90)

    def test_hidden_history_and_host_legal_actions(self):
        worlds = [env.World(8123, cue, inversion)
                  for cue in range(2) for inversion in range(2)]
        for world in worlds:
            for _ in range(8):
                world.step(0)
        obs = [tuple(world.observe()) for world in worlds]
        self.assertEqual(len(set(obs)), 1)
        self.assertEqual(worlds[0].legal(), [0, 1, 1, 0, 1])
        self.assertNotEqual(worlds[0].preferred_action(),
                            worlds[1].preferred_action())
        model = diagonal_model(self.torch).eval()
        result = stateful_play(self.torch, model, 8123, 0, 1)
        self.assertLessEqual(result["decisions"], env.MAX_STEPS)
        self.assertGreaterEqual(result["painted"], 0)
        self.assertGreaterEqual(result["collisions"], 0)

    def test_training_episode_groups_are_disjoint(self):
        train = env.dataset(range(1000, 1003))
        test = env.dataset(range(5000, 5003))
        self.assertEqual(len(train), 36)
        self.assertTrue({r[2][0] for r in train}.isdisjoint(
            {r[2][0] for r in test}))


if __name__ == "__main__":
    unittest.main()
