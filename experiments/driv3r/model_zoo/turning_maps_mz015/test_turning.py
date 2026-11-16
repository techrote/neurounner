"""Deterministic, host-legal, 2-D turning-map MZ-015 qualification."""
import unittest

from experiments.driv3r.model_zoo.turning_maps_mz015.run import (
    World, episode, D, ACTIONS, PAINT, RETRACE, STOP
)


class TurningMapTests(unittest.TestCase):
    def test_all_rotations_mirrors_and_material_combinations(self):
        for seed in range(16):
            for cue in (0, 1):
                for invert in (0, 1):
                    for bad in (-1, 1):
                        x, y = episode(seed, cue, invert, bad=bad)
                        self.assertEqual(len(x), len(y))
                        self.assertTrue(all(len(a) == D for a in x))
                        self.assertLessEqual(len(x), 96)

    def test_unseen_length_family(self):
        for seed in range(48, 64):
            episode(seed, seed % 2, (seed//2)%2, ood=True)

    def test_hidden_cue_and_host_legal_mask(self):
        a = World(4, 0, 1)
        b = World(4, 1, 0)
        self.assertEqual(a.safe, b.safe)
        for tick in range(4):
            self.assertNotEqual(a.observe()[-8:-4], b.observe()[-8:-4])
            self.assertEqual(a.teacher(), b.teacher())
            action = a.teacher()
            self.assertTrue(a.legal()[action])
            a.step(action)
            b.step(action)
        self.assertEqual(a.observe()[-8:-4], [0.,0.,0.,0.])
        self.assertEqual(b.observe()[-8:-4], [0.,0.,0.,0.])

    def test_collision_retrace_and_safe_paint(self):
        w = World(13, 0, 0)
        saw_paint = False
        for _ in range(96):
            action = w.teacher()
            self.assertTrue(w.legal()[action])
            saw_paint |= action == PAINT
            w.step(action)
            if w.done:
                break
        self.assertTrue(w.done)
        self.assertFalse(w.failed)
        self.assertEqual(w.collisions, 0)
        self.assertTrue(saw_paint)
        self.assertEqual(len(w.painted), 3)

    def test_stop_is_legal_but_premature_stop_fails(self):
        w = World(1, 1, 1)
        self.assertTrue(w.legal()[STOP])
        w.step(STOP)
        self.assertTrue(w.failed)

if __name__ == "__main__":
    unittest.main()
