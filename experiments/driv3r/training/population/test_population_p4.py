"""P4 feedback, causality, provenance, reproducibility and legality tests."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
try:
    from .population_lab import DIRECTIONS, Policy, World, experiment
    from .population_p4 import ARMS, splits_for, phenotype, trajectory_novelty, feedback_target, train_member, run_p4
except ImportError:
    from population_lab import DIRECTIONS, Policy, World, experiment
    from population_p4 import ARMS, splits_for, phenotype, trajectory_novelty, feedback_target, train_member, run_p4


class PopulationP4Tests(unittest.TestCase):
    def test_phenotype_is_observed_only(self):
        w = World.generate(629)
        self.assertEqual(len(phenotype(w)), 14)
        self.assertEqual(trajectory_novelty(w, []), .5)
        self.assertEqual(trajectory_novelty(w, [phenotype(w)]), 0.)
        w.step(next(a for a in w.legal() if a != "STOP"))
        self.assertGreater(trajectory_novelty(w, [phenotype(World.generate(629))]), 0)

    def test_positive_negative_feedback_are_different(self):
        target = np.array([.2, .5, .3])
        pos = feedback_target(target, 1, .3, .8)
        neg = feedback_target(target, 1, -.3, .8)
        self.assertAlmostEqual(pos.sum(), 1)
        self.assertAlmostEqual(neg.sum(), 1)
        self.assertGreater(pos[1], target[1])
        self.assertLess(neg[1], target[1])
        np.testing.assert_array_equal(feedback_target(np.array([1.]), 0, .5, .8), [1.])
        with self.assertRaises(ValueError):
            feedback_target(target, 0, .5, 1.1)

    def test_real_optimizer_updates_and_matched_initialisation(self):
        models = [Policy(17) for _ in ARMS]
        seeds = splits_for(629, 4, 2, 2)["train"]
        archives = [[] for _ in ARMS]
        records = [train_member(p, "branch_artist", seeds, arm, "zigzag", archive, .8, 777)
                   for p, arm, archive in zip(models, ARMS, archives)]
        self.assertTrue(all(r["adam_steps"] > 0 for r in records))
        self.assertTrue(all(len(a) == len(seeds) for a in archives))
        self.assertGreater(sum(not np.array_equal(models[0].p["W1"], p.p["W1"])
                               for p in models[1:]), 0)

    def test_nonoverlapping_split_and_replay(self):
        seeds = splits_for(629, 20, 8, 12)
        self.assertEqual(len(set(sum(seeds.values(), []))), 40)
        self.assertEqual(seeds, splits_for(629, 20, 8, 12))

    def test_replay_all_48_policies_and_host_legality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment(root / "p0", seed=619, population=1,
                       train_episodes=3, selection_episodes=2, test_episodes=2)
            args = dict(seed=629, members=2, train_episodes=4,
                        selection_episodes=2, holdout_episodes=2, horizon=36)
            a = run_p4(root / "p0", root / "first", **args)
            run_p4(root / "p0", root / "second", **args)
            self.assertEqual((root / "first/results.json").read_bytes(),
                             (root / "second/results.json").read_bytes())
            self.assertEqual(a["candidate_count"], 48)
            self.assertGreater(a["adam_updates"], 0)
            self.assertEqual(len(list((root / "first/gallery").glob("*.svg"))), 24)
            for path in (root / "first/gallery").glob("*-trace.json"):
                trace = json.loads(path.read_text())
                self.assertTrue(trace["synthetic"] and not trace["native"])
                self.assertFalse(set(map(tuple, trace["painted"])) & set(map(tuple, trace["protected"])))
                self.assertFalse(set(map(tuple, trace["painted"])) & set(map(tuple, trace["obstacles"])))
                self.assertTrue(all(t["action"] in (*DIRECTIONS, "RETRACE", "STOP")
                                    for t in trace["trajectory"]))
            with self.assertRaises(FileExistsError):
                run_p4(root / "p0", root / "first", **args)

    def test_invalid_bounds_and_source_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p0 = Path(directory) / "p0"
            p0.mkdir()
            (p0 / "results.json").write_text(json.dumps({"format": "wrong"}))
            with self.assertRaises(ValueError):
                run_p4(p0, Path(directory) / "bad", members=2)
            for bad in (dict(members=1), dict(strength=0), dict(horizon=129),
                        dict(train_episodes=2)):
                with self.assertRaises(ValueError):
                    run_p4(p0, Path(directory) / "bad", **bad)


if __name__ == "__main__":
    unittest.main()
