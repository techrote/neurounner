"""P3 four-arm causal isolation, lineage, train/heldout and legal-action tests."""
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

import numpy as np

try:
    from .population_lab import NICHES, Policy, World, experiment
    from .population_p2 import STYLES, novelty_key
    from .population_p3 import ARMS, split_seeds, teacher_scores, train_arm, long_horizon, run_p3
except ImportError:
    from population_lab import NICHES, Policy, World, experiment
    from population_p2 import STYLES, novelty_key
    from population_p3 import ARMS, split_seeds, teacher_scores, train_arm, long_horizon, run_p3


class PopulationP3Tests(unittest.TestCase):
    def test_factorial_components_are_separable(self):
        w = World.generate(619)
        menu = w.legal()
        X = np.stack([w.feature(a) for a in menu])
        archive, local = Counter(), Counter()
        a = {arm: teacher_scores(w, "branch_artist", menu, X, arm, "zigzag",
                                 archive, local, .8) for arm in ARMS}
        self.assertTrue(np.array_equal(a["combined"] - a["rarity_only"],
                                       a["morphology_only"] - a["reward_only"])
                        or np.allclose(a["combined"] - a["rarity_only"],
                                       a["morphology_only"] - a["reward_only"]))
        self.assertTrue(np.allclose(a["combined"] - a["morphology_only"],
                                    a["rarity_only"] - a["reward_only"]))
        self.assertFalse(np.array_equal(a["rarity_only"], a["reward_only"]))
        self.assertFalse(np.array_equal(a["morphology_only"], a["reward_only"]))
        for bad in ("illegal", "morph+random"):
            with self.assertRaises(ValueError):
                teacher_scores(w, "branch_artist", menu, X, bad, "zigzag",
                               archive, local, .8)

    def test_matched_initial_parameters_train_differently(self):
        seeds = split_seeds(627, 4, 2, 2)["train"]
        variants = [Policy(17) for arm in ARMS]
        self.assertTrue(all(np.array_equal(variants[0].p["W1"], v.p["W1"])
                            for v in variants))
        result = [train_arm(policy, "navigator", seeds, STYLES[0], arm,
                            Counter(), .8)
                  for policy, arm in zip(variants, ARMS)]
        self.assertTrue(all(x["adam_steps"] > 0 for x in result))
        self.assertGreater(sum(not np.array_equal(variants[0].p["W1"], p.p["W1"])
                               for p in variants[1:]), 0)
        self.assertTrue(all(np.isfinite(v.p["W1"]).all() for v in variants))

    def test_split_and_world_mask(self):
        a = split_seeds(627, 9, 3, 4)
        self.assertEqual(a, split_seeds(627, 9, 3, 4))
        self.assertEqual(16, len(set(sum(a.values(), []))))
        w = World.generate(a["holdout"][0], max_steps=60)
        while w.steps < 60 and not w.stopped:
            action = next((a for a in w.legal() if a != "STOP"), "STOP")
            w.step(action)
        self.assertFalse(w.painted & w.obstacles)
        self.assertFalse(w.painted & w.protected)
        self.assertTrue(all(h["action"] in ("N", "S", "E", "W", "RETRACE", "STOP")
                            for h in w.history))
        self.assertTrue(novelty_key(w, "STOP"))

    def test_long_horizon_is_separate_from_short_evaluation(self):
        x = long_horizon([Policy(1), Policy(2)], "navigator", [623, 624], 60)
        self.assertEqual(x["horizon"], 60)
        self.assertGreaterEqual(x["mean_distance"], 0)
        self.assertLessEqual(x["mean_distance"], 1)
        self.assertGreaterEqual(x["phenotype_signatures"], 1)

    def test_bad_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p0 = Path(tmp) / "p0"
            experiment(p0, seed=619, population=1,
                       train_episodes=3, selection_episodes=2, test_episodes=2)
            for kwargs in (dict(members=1), dict(strength=0), dict(horizon=30),
                           dict(horizon=129), dict(train_episodes=101)):
                with self.assertRaises(ValueError):
                    run_p3(p0, Path(tmp) / "invalid", **kwargs)

    def test_replay_complete_48_candidate_miniature_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p0 = root / "p0"
            experiment(p0, seed=619, population=1,
                       train_episodes=3, selection_episodes=2, test_episodes=2)
            result = run_p3(p0, root / "p3", seed=627, members=2,
                            train_episodes=4, selection_episodes=2,
                            holdout_episodes=2, horizon=36)
            self.assertEqual(result["candidate_count"], 6 * 4 * 2)
            self.assertEqual(result["optimizer_trained_count"], 48)
            self.assertGreater(result["total_new_adam_steps"], 0)
            self.assertFalse(result["language_model"])
            self.assertFalse(result["native"])
            self.assertEqual(len(list((root / "p3/models").glob("*.json"))), 48)
            self.assertEqual(len(list((root / "p3/gallery").glob("*.svg"))), 24)
            for niche in NICHES:
                for arm in ARMS:
                    group = result["comparison"][niche][arm]
                    self.assertTrue(group["champion_id"] in group["members"])
                    self.assertIn("long_horizon", group)
                    trace = json.loads((root / "p3/gallery" /
                                       (f"{niche}-{arm}-trace.json")).read_text())
                    assert not (set(map(tuple, trace["painted"]))
                                & set(map(tuple, trace["protected"])))
                    assert not (set(map(tuple, trace["painted"]))
                                & set(map(tuple, trace["obstacles"])))
            with self.assertRaises(FileExistsError):
                run_p3(p0, root / "p3", seed=627, members=2,
                       train_episodes=4, selection_episodes=2,
                       holdout_episodes=2, horizon=36)


if __name__ == "__main__":
    unittest.main()
