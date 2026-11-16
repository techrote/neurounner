"""Guardrails for paired P2 optimizer-time novelty training."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from .population_lab import NICHES, Policy, World, experiment, stable_seed
    from .population_p2 import (
        ARMS, STYLES, split_seeds, train_arm, novelty_key,
        morphology_bonus, trajectory_distance, heldout_dispersion, run_p2,
    )
except ImportError:
    from population_lab import NICHES, Policy, World, experiment, stable_seed
    from population_p2 import (
        ARMS, STYLES, split_seeds, train_arm, novelty_key,
        morphology_bonus, trajectory_distance, heldout_dispersion, run_p2,
    )


class PopulationP2Tests(unittest.TestCase):
    def test_same_start_different_actual_optimizer_updates(self):
        a, b = Policy(19), Policy(19)
        same = np.array_equal(a.p["W1"], b.p["W1"])
        self.assertTrue(same)
        seeds = split_seeds(625, 4, 2, 2)["train"]
        r0 = train_arm(a, "branch_artist", seeds, "zigzag", ARMS[0])
        r1 = train_arm(b, "branch_artist", seeds, "zigzag", ARMS[1])
        self.assertGreater(r0["adam_steps"], 0)
        self.assertGreater(r1["adam_steps"], 0)
        self.assertFalse(np.array_equal(a.p["W1"], b.p["W1"]))
        self.assertTrue(np.isfinite(a.p["W1"]).all())
        self.assertTrue(np.isfinite(b.p["W1"]).all())

    def test_host_legality_and_noninvented_actions(self):
        w = World.generate(619)
        legal = w.legal()
        self.assertIn("STOP", legal)
        for action in legal:
            vec = w.feature(action)
            self.assertTrue(np.isfinite(morphology_bonus(w, action, vec, "zigzag")))
            self.assertTrue(novelty_key(w, action))
        self.assertFalse(w.painted & w.protected)
        self.assertFalse(w.painted & w.obstacles)

    def test_same_world_distance_properties(self):
        w = World.generate(619)
        v = World.generate(619)
        self.assertEqual(trajectory_distance(w, v), 0)
        w.step("STOP")
        self.assertGreater(trajectory_distance(w, v), 0)
        self.assertEqual(trajectory_distance(w, v), trajectory_distance(v, w))

    def test_seed_splits_disjoint_and_stable(self):
        a = split_seeds(625, 9, 4, 6)
        b = split_seeds(625, 9, 4, 6)
        self.assertEqual(a, b)
        flat = sum(a.values(), [])
        self.assertEqual(len(flat), len(set(flat)))
        self.assertNotEqual(a["train"], a["holdout"])

    def test_bad_configuration_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p0 = Path(d) / "p0"
            experiment(p0, seed=619, population=1,
                       train_episodes=3, selection_episodes=2, test_episodes=2)
            for value in (0, -1, 3):
                with self.assertRaises(ValueError):
                    run_p2(p0, Path(d) / f"bad-{value}",
                           members=2, train_episodes=4,
                           selection_episodes=2, holdout_episodes=2,
                           strength=value)
            with self.assertRaises(ValueError):
                run_p2(p0, Path(d) / "bad-count", members=1)

    def test_full_small_campaign_artifacts_provenance_replay(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            p0 = base / "p0"
            experiment(p0, seed=619, population=1,
                       train_episodes=3, selection_episodes=2, test_episodes=2)
            result = run_p2(p0, base / "p2", seed=625, members=2,
                            train_episodes=4, selection_episodes=2,
                            holdout_episodes=2)
            self.assertEqual(result["candidate_count"], 24)
            self.assertEqual(result["optimizer_trained_count"], 24)
            self.assertFalse(result["native"])
            self.assertFalse(result["language_model"])
            self.assertEqual(len(list((base / "p2/models").glob("*.json"))), 24)
            self.assertEqual(len(list((base / "p2/gallery").glob("*.svg"))), 12)
            for niche in NICHES:
                for arm in ARMS:
                    group = result["comparisons"][niche][arm]
                    self.assertTrue((base / "p2/models" /
                                     (group["champion_id"] + ".json")).exists())
                    self.assertGreater(group["heldout_dispersion"]["phenotype_signatures"], 0)
                    trace = json.loads((base / "p2/gallery" /
                                        (f"{niche}-{arm}-trace.json")).read_text())
                    painted = set(map(tuple, trace["painted"]))
                    self.assertFalse(painted & set(map(tuple, trace["protected"])))
                    self.assertFalse(painted & set(map(tuple, trace["obstacles"])))
                    self.assertTrue(set(h["action"] for h in trace["trajectory"])
                                    <= {"N", "E", "S", "W", "RETRACE", "STOP"})
            with self.assertRaises(FileExistsError):
                run_p2(p0, base / "p2", seed=625, members=2,
                       train_episodes=4, selection_episodes=2, holdout_episodes=2)


if __name__ == "__main__":
    unittest.main()
