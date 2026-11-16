"""Synthetic population guardrails; never touches the actual game."""
import json
import pathlib
import tempfile
import unittest

import numpy as np

try:
    from .population_lab import World, Policy, experiment, softmax, NICHES, stable_seed
except ImportError:
    from population_lab import World, Policy, experiment, softmax, NICHES, stable_seed


class PopulationTest(unittest.TestCase):
    def test_seed_replay_and_split(self):
        a = World.generate(619)
        b = World.generate(619)
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.protected, b.protected)
        self.assertEqual(a.legal(), b.legal())
        self.assertNotEqual(stable_seed(619, "train", 0), stable_seed(619, "test", 0))

    def test_obstacles_protected_never_painted(self):
        for seed in range(20):
            w = World.generate(seed)
            for _ in range(30):
                actions = w.legal()
                a = next((x for x in actions if x in ("N", "E", "S", "W")), None)
                a = a or next((x for x in actions if x != "STOP"), "STOP")
                w.step(a)
                self.assertFalse(w.painted & w.protected)
                self.assertFalse(w.painted & w.obstacles)
                if w.stopped:
                    break

    def test_illegal_action_fails_without_mutating(self):
        w = World.generate(4)
        before = (set(w.painted), w.cursor, w.steps)
        with self.assertRaises(ValueError):
            w.step("FREE_COORDINATES")
        self.assertEqual((w.painted, w.cursor, w.steps), before)

    def test_backtrack_cannot_paint(self):
        w = World.generate(11)
        w.step(next(a for a in w.legal() if a in ("N", "E", "S", "W")))
        count = len(w.painted)
        self.assertIn("RETRACE", w.legal())
        w.step("RETRACE")
        self.assertEqual(len(w.painted), count)
        w.step("STOP")
        self.assertEqual(w.legal(), ["STOP"])

    def test_gradient_updates_are_real(self):
        p = Policy(15)
        w = World.generate(8)
        x = np.stack([w.feature(a) for a in w.legal()])
        old = p.p["W1"].copy()
        loss = p.teach(x, softmax(np.arange(len(x))))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(old, p.p["W1"]))
        self.assertEqual(p.updates, 1)

    def test_softmax_numerical_stability(self):
        self.assertTrue(np.allclose(softmax(np.array([10000., 10001.])),
                                    [.26894142, .73105858]))

    def test_campaign_outputs_are_source_traced(self):
        with tempfile.TemporaryDirectory() as t:
            output = pathlib.Path(t) / "campaign"
            m = experiment(output, seed=619, population=1, train_episodes=3,
                           selection_episodes=2, test_episodes=2)
            self.assertEqual(set(m["champion_ids"]), set(NICHES))
            self.assertEqual(m["candidate_count"], 12)
            self.assertEqual(len(list((output / "gallery").glob("*.svg"))), 6)
            for niche in NICHES:
                result = json.loads((output / "gallery" / (niche + "-trajectory.json")).read_text())
                self.assertTrue(result["synthetic"])
                self.assertFalse(result["native"])
                self.assertFalse({tuple(p) for p in result["painted"]}
                                 & {tuple(p) for p in result["protected"]})
                self.assertTrue((output / "models" / (niche + ".json")).exists())
            self.assertIn("nonstop_random", m["heldout_controls"]["navigator"])


if __name__ == "__main__":
    unittest.main()
