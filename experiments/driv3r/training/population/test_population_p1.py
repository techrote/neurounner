"""P1 population novelty and live-vs-frozen observations, synthetic only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from .population_lab import Policy, World, experiment, NICHES
    from .population_p1 import (
        candidate_behaviour, choose_survivors, experiment_p1,
        load_policy, novelty_scores, observation_probe, select_seed_lists,
    )
except ImportError:
    from population_lab import Policy, World, experiment, NICHES
    from population_p1 import (
        candidate_behaviour, choose_survivors, experiment_p1,
        load_policy, novelty_scores, observation_probe, select_seed_lists,
    )


class PopulationP1Tests(unittest.TestCase):
    def test_policy_roundtrip_exact_scores_and_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.json"
            p = Policy(33)
            p.updates = 129
            path.write_text(json.dumps(p.serialized()))
            q = load_policy(path)
            w = World.generate(33)
            x = np.stack([w.feature(a) for a in w.legal()])
            self.assertTrue(np.array_equal(p.logits(x), q.logits(x)))
            self.assertEqual(q.updates, 0)
            self.assertTrue(all(np.count_nonzero(m) == 0 for m in q.m.values()))

    def test_corrupted_model_schema_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.json"
            source = Policy(7).serialized()
            source["features"] = ["changed"]
            path.write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                load_policy(path)
            source = Policy(7).serialized()
            source["parameters"]["W2"][0] = float("nan")
            path.write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                load_policy(path)

    def test_novelty_distances_and_archive_use_selection_only(self):
        items = []
        for n, v in enumerate((0., .01, .91)):
            items.append(dict(id=f"navigator-{n}", niche="navigator",
                              selection=dict(mean_reward=float(10-n),
                                             mean_vector=[v] * 14)))
        n = novelty_scores(items, k=1)
        self.assertGreater(n[2], n[0])
        chosen, archive = choose_survivors(items)
        self.assertEqual(chosen["navigator"]["reward"], "navigator-0")
        self.assertTrue(archive)

    def test_split_separation(self):
        seeds = select_seed_lists(620, 40, 15, 12)
        all_seeds = sum(seeds.values(), [])
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

    def test_observation_probe_always_host_legal(self):
        p = Policy(13)
        r = observation_probe(p, "navigator", [777, 778])
        self.assertGreater(r["decision_comparisons"], 0)
        self.assertGreaterEqual(r["same_state_action_divergence"], 0)
        self.assertLessEqual(r["same_state_action_divergence"], 1)
        self.assertLessEqual(r["same_state_total_variation"], 1)
        self.assertTrue(np.isfinite(r["frozen_prior_mean_reward"]))

    def test_selection_behavior_is_repeatable(self):
        p = Policy(4)
        a = candidate_behaviour(p, "branch_artist", [202, 203])
        b = candidate_behaviour(p, "branch_artist", [202, 203])
        self.assertEqual(a, b)
        self.assertEqual(len(a["mean_vector"]), 14)

    def test_training_campaign_provenance_and_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p0 = root / "p0"
            p1 = root / "p1"
            experiment(p0, seed=619, population=1, train_episodes=3,
                       selection_episodes=2, test_episodes=2)
            result = experiment_p1(p0, p1, seed=620, offspring=2, train_episodes=4,
                                   selection_episodes=2, holdout_episodes=2)
            self.assertEqual(result["format"], "population-research-p1")
            self.assertEqual(result["generated_candidates"], 18)
            self.assertEqual(result["trained_offspring"], 12)
            self.assertFalse(result["native"])
            self.assertFalse(result["language_model"])
            self.assertEqual(len(list((p1 / "models").glob("*.json"))), 18)
            self.assertEqual(len(list((p1 / "gallery").glob("*.svg"))), 12)
            self.assertEqual(len(result["archive_ids"]), result["archive_bins"])
            self.assertEqual(set(result["survivor_ids"]), set(NICHES))
            for niche in NICHES:
                self.assertIn("p0_champion", result["heldout_comparison"][niche]["controls"])
                for mode in ("reward", "diversity"):
                    chosen = result["survivor_ids"][niche][mode]
                    self.assertTrue((p1 / "models" / (chosen + ".json")).exists())
                    trace = json.loads((p1 / "gallery" / f"{niche}-{mode}-trace.json").read_text())
                    self.assertTrue(trace["synthetic"])
                    self.assertFalse(trace["native"])
                    self.assertFalse(set(map(tuple, trace["painted"])) & set(map(tuple, trace["protected"])))
                    self.assertFalse(set(map(tuple, trace["painted"])) & set(map(tuple, trace["obstacles"])))
                    self.assertTrue(set(x["action"] for x in trace["trajectory"])
                                    <= {"N", "E", "S", "W", "RETRACE", "STOP"})
            self.assertTrue((p1 / "results.json").exists())
            with self.assertRaises(FileExistsError):
                experiment_p1(p0, p1, seed=620, offspring=2, train_episodes=4,
                              selection_episodes=2, holdout_episodes=2)


if __name__ == "__main__":
    unittest.main()
