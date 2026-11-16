"""P5 matched-sham feedback, frozen trajectories, selection and host legality."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    from .population_lab import DIRECTIONS, Policy, World, experiment, stable_seed
    from .population_p5 import (ARMS, behaviour_bin, challenge_world, episode_advantages,
                                elite_ids, feedback_target, frozen_rollouts, matched_sham,
                                run_p5, splits_for, train_arms)
except ImportError:
    from population_lab import DIRECTIONS, Policy, World, experiment, stable_seed
    from population_p5 import (ARMS, behaviour_bin, challenge_world, episode_advantages,
                                elite_ids, feedback_target, frozen_rollouts, matched_sham,
                                run_p5, splits_for, train_arms)


class P5Tests(unittest.TestCase):
    def test_matched_signal_exact_multiset_and_different_alignment(self):
        signals = episode_advantages([.1, .3, .2, .6, .1, .8, .9, .4])
        fake = matched_sham(signals, 7)
        self.assertEqual(sorted(signals.tolist()), sorted(fake.tolist()))
        self.assertFalse(np.array_equal(signals, fake))
        np.testing.assert_array_equal(episode_advantages([4., 4., 4.]), [0., 0., 0.])
        with self.assertRaises(ValueError):
            episode_advantages([float("nan")])

    def test_protected_material_dead_end_forces_legal_retrace(self):
        w = challenge_world(1234, 60)
        self.assertIn("E", w.legal())
        w.step("E")
        self.assertIn("RETRACE", w.legal())
        self.assertFalse(any(a in DIRECTIONS for a in w.legal()))
        w.step("RETRACE")
        self.assertEqual(w.cursor, (3, 11))
        self.assertFalse(w.painted & (w.protected | w.obstacles))

    def test_training_same_frozen_data_and_adam_updates(self):
        source = Policy(313)
        seeds = splits_for(631, 6, 2, 2)["train"]
        policies, training, signals = train_arms(source, "recoverer", seeds, "zigzag", 2027, .8)
        self.assertEqual(set(policies), set(ARMS))
        self.assertEqual(len(set(t["source_sha256"] for t in training.values())), 1)
        self.assertTrue(all(t["adam_steps"] == training["reward_only"]["adam_steps"] for t in training.values()))
        self.assertTrue(signals["exact_signal_multiset_match"])
        self.assertTrue(signals["signal_alignment_changed"])
        self.assertGreater(training["reward_only"]["adam_steps"], 0)
        self.assertGreater(signals["decisions"], 0)
        self.assertFalse(np.array_equal(policies["reward_only"].p["W1"], policies["measured_novelty"].p["W1"]))

    def test_selection_elites_without_heldout(self):
        self.assertEqual(behaviour_bin({"mean_descriptor": [.6, .1, .5, .1, .8]}), (1, 0, 1, 1))
        records = {str(i): {"selection_bin": list(bin_), "selection": {"mean_reward": score}}
                   for i, bin_, score in [(0, (1, 1, 1, 1), 4.), (1, (1, 1, 1, 1), 5.),
                                         (2, (0, 0, 2, 0), 3.)]}
        self.assertEqual(elite_ids(list(records), records, 3), ["1", "2"])

    def test_full_replay_and_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment(root / "p0", seed=619, population=1,
                       train_episodes=3, selection_episodes=2, test_episodes=2)
            args = dict(seed=631, members=2, train_episodes=4,
                        selection_episodes=2, holdout_episodes=2, horizon=36)
            first = run_p5(root / "p0", root / "first", **args)
            run_p5(root / "p0", root / "second", **args)
            self.assertEqual((root / "first/results.json").read_bytes(),
                             (root / "second/results.json").read_bytes())
            self.assertEqual(first["candidate_count"], 48)
            self.assertTrue(first["adam_updates"] > 0)
            self.assertEqual(len(list((root / "first/models").glob("*.json"))), 48)
            for path in (root / "first/gallery").glob("*-trace.json"):
                trace = json.loads(path.read_text())
                self.assertTrue(trace["synthetic"] and not trace["native"])
                self.assertFalse(set(map(tuple, trace["painted"])) & set(map(tuple, trace["protected"])))
                self.assertFalse(set(map(tuple, trace["painted"])) & set(map(tuple, trace["obstacles"])))
                self.assertTrue(all(t["action"] in (*DIRECTIONS, "RETRACE", "STOP")
                                    for t in trace["trajectory"]))
            with self.assertRaises(FileExistsError):
                run_p5(root / "p0", root / "first", **args)

    def test_invalid_source_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "p0").mkdir()
            (root / "p0/results.json").write_text(json.dumps({"format": "wrong"}))
            with self.assertRaises(ValueError):
                run_p5(root / "p0", root / "bad")
            for kwargs in (dict(members=1), dict(train_episodes=2), dict(horizon=500), dict(strength=0)):
                with self.assertRaises(ValueError):
                    run_p5(root / "p0", root / "bad", **kwargs)
            with self.assertRaises(ValueError):
                feedback_target(np.array([1.]), 0, .5, 0)


if __name__ == "__main__":
    unittest.main()
