"""MZ-013 synthetic host and model qualification contracts."""
import unittest

from experiments.driv3r.model_zoo.spatial_paint_mz013 import run as mz


class WorldTests(unittest.TestCase):
    def test_variable_seed_worlds_teacher_can_finish(self):
        for ood in (False, True):
            for seed in (0, 1, 11, 99):
                for cue in (0, 1):
                    for inversion in (0, 1):
                        for forced in (-1, 1):
                            seq, targets, _ = mz.episode(
                                seed, cue, inversion, bad=forced, ood=ood)
                            self.assertLessEqual(len(seq), mz.MAX_T)
                            self.assertEqual(targets[-1], mz.STOP)
                            self.assertEqual(len(seq), len(targets))

    def test_host_rejects_illegal_and_early_stop_fails(self):
        w = mz.World(17, 0, 0)
        with self.assertRaises(ValueError):
            w.step(mz.PAINT)
        self.assertTrue(w.legal()[mz.STOP])
        w.step(mz.STOP)
        self.assertTrue(w.failed)
        with self.assertRaises(RuntimeError):
            w.step(mz.STOP)

    def test_protected_collision_requires_retrace(self):
        w = mz.World(23, 0, 0)
        for _ in range(4):
            w.step(mz.WAIT)
        while w.x != w.stations[0]:
            w.step(mz.RIGHT)
        wrong = mz.DOWN if w.teacher() == mz.UP else mz.UP
        w.step(wrong)
        self.assertTrue(w.hazard)
        self.assertEqual(w.collisions, 1)
        self.assertFalse(w.legal()[mz.PAINT])
        self.assertTrue(w.legal()[mz.RETRACE])
        w.step(mz.RETRACE)
        self.assertFalse(w.hazard)
        self.assertEqual(w.y, 3)
        w.step(w.teacher())
        self.assertTrue(w.legal()[mz.PAINT])
        w.step(mz.PAINT)
        self.assertIn(0, w.painted)
        self.assertEqual(w.tile(w.x, w.y), 4)
        w.step(mz.RETRACE)
        self.assertEqual(w.y, 3)

    def test_disappearing_information_is_not_leaked(self):
        a, b = mz.World(53, 0, 0), mz.World(53, 1, 0)
        first_a, first_b = a.observe(), b.observe()
        self.assertNotEqual(first_a, first_b)
        for _ in range(4):
            a.step(mz.WAIT)
            b.step(mz.WAIT)
        self.assertEqual(a.observe(), b.observe())
        self.assertNotEqual(a.teacher(), b.teacher()) if a.station_index() >= 0 else None

    def test_legal_and_observation_dimension(self):
        for seed in range(8):
            w = mz.World(seed, seed%2, (seed//2)%2)
            for _ in range(4):
                x = w.observe()
                self.assertEqual(len(x), mz.D)
                self.assertEqual(x[131:139], w.legal())
                w.step(mz.WAIT)
            while not w.done:
                x = w.observe()
                self.assertEqual(len(x), mz.D)
                action = w.teacher()
                self.assertTrue(w.legal()[action])
                w.step(action)
            self.assertFalse(w.failed)


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)

    def test_models_are_real_trainable_and_host_legal(self):
        torch = self.torch
        nets = mz.models(torch)
        params = {name: sum(p.numel() for p in net.parameters())
                  for name, net in nets.items()}
        self.assertTrue(20000000 <= params["transformer20m"] <= 30000000)
        xs, ys = mz.tensors(torch, [
            mz.episode(12, 0, 1), mz.episode(13, 1, 1)])
        for net in nets.values():
            with torch.no_grad():
                output = net(xs[:, :8])
            self.assertEqual(tuple(output.shape), (2, 8, len(mz.ACTIONS)))
            self.assertTrue(bool(torch.isfinite(output).all()))
        for name in ("diagonal", "gru"):
            net = nets[name].eval()
            with torch.inference_mode():
                prefix = net(xs[:, :8])[:, -1]
                state = None
                for t in range(8):
                    incremental, state = net.step(xs[:, t], state)
            self.assertTrue(torch.allclose(prefix, incremental, atol=1e-5),
                            name)

    def test_teacher_action_is_masked_correctly(self):
        w = mz.World(12, 0, 1)
        for _ in range(4):
            w.step(mz.WAIT)
        while w.x != w.stations[0]:
            w.step(mz.RIGHT)
        self.assertTrue(w.legal()[mz.UP])
        self.assertTrue(w.legal()[mz.DOWN])
        self.assertFalse(w.legal()[mz.PAINT])


if __name__ == "__main__":
    unittest.main()
