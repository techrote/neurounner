"""P6 archive-evolution tests; no CyberSand game imports/writes."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    from .population_lab import NICHES, Policy, World, experiment
    from .population_p6 import (ARMS, parents_for, portfolio_ids, run_p6, split_seeds,
                                topology_bin, train_child, update_archive)
except ImportError:
    from population_lab import NICHES, Policy, World, experiment
    from population_p6 import (ARMS, parents_for, portfolio_ids, run_p6, split_seeds,
                               topology_bin, train_child, update_archive)


class PopulationP6Tests(unittest.TestCase):
    def test_split_independence(self):
        splits = split_seeds(634, 5, 4, 3, 3)
        flat = splits["selection"] + splits["holdout"] + sum(splits["train"], [])
        self.assertEqual(len(flat), len(set(flat)))
        self.assertNotEqual(split_seeds(635, 5, 4, 3, 3), splits)

    def test_archive_replaces_only_with_better_reward(self):
        def entry(i, bin_, reward):
            return dict(id=i, selection_bin=list(bin_), selection={"mean_reward":reward})
        a = entry("a", (0, 0, 0, 0, 0), 4)
        b = entry("b", (0, 0, 0, 0, 0), 2)
        c = entry("c", (0, 1, 0, 0, 0), 1)
        d = entry("d", (0, 0, 0, 0, 0), 5)
        archive = update_archive({}, [a,b,c])
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive[(0, 0, 0, 0, 0)]["id"], "a")
        self.assertEqual(update_archive(archive,[d])[(0, 0, 0, 0, 0)]["id"], "d")
        self.assertEqual(len(archive), 2)

    def test_parent_sampling_uses_selection_only(self):
        a = dict(id="a", selection={"mean_reward":4}, selection_bin=[0])
        b = dict(id="b", selection={"mean_reward":3}, selection_bin=[1])
        archive = {(0,):a, (1,):b}
        self.assertIsNone(parents_for("map_elites", archive, a, [a,b], 0, 0, 633, "navigator"))
        self.assertEqual(parents_for("reward_pbt", archive,a,[a,b],1,0,633,"navigator"), a)
        self.assertIn(parents_for("map_elites", archive,a,[a,b],1,0,633,"navigator"),[a,b])
        self.assertEqual(len(portfolio_ids([a,b],archive,"map_elites",2,7)),2)

    def test_real_adam_and_host_legal_curriculum(self):
        policy = Policy(2026)
        result = train_child(policy, "recoverer", split_seeds(634,5,3,3,2)["train"][0])
        self.assertEqual(result["adam_steps"], policy.updates)
        self.assertGreater(result["adam_steps"], 0)
        self.assertGreater(result["curriculum_actions"], 0)
        self.assertEqual(len(topology_bin({"mean_vector":[.5,.1,.1,.1,.1,.5,.1,.1,.3]+[0]*5})),5)
        with self.assertRaises(ValueError):
            topology_bin({"mean_vector":[float("nan")]*14})

    def test_end_to_end_replay_and_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            experiment(root/"p0", seed=619, population=1, train_episodes=3,
                       selection_episodes=2, test_episodes=2)
            kwargs = dict(seed=634, generations=2, offspring=3, train_episodes=4,
                          selection_episodes=3, holdout_episodes=3, horizon=32, portfolio=2)
            a = run_p6(root/"p0", root/"one", **kwargs)
            b = run_p6(root/"p0", root/"two", **kwargs)
            self.assertEqual((root/"one/results.json").read_bytes(),(root/"two/results.json").read_bytes())
            self.assertEqual(a["candidate_count"], len(NICHES)*len(ARMS)*2*3)
            self.assertGreater(a["adam_updates"], 0)
            self.assertFalse(a["language_model"] or a["native"])
            self.assertEqual(len(list((root/"one/models").glob("*.json"))),a["candidate_count"])
            for t in (root/"one/gallery").glob("*-trace.json"):
                x = json.loads(t.read_text())
                self.assertFalse(set(map(tuple,x["painted"])) & set(map(tuple,x["protected"])))
                self.assertFalse(set(map(tuple,x["painted"])) & set(map(tuple,x["obstacles"])))
            with self.assertRaises(FileExistsError):
                run_p6(root/"p0",root/"one", **kwargs)
            for altered in (dict(generations=1),dict(offspring=1),dict(train_episodes=2),
                            dict(selection_episodes=1),dict(holdout_episodes=1),dict(horizon=300)):
                with self.assertRaises(ValueError):
                    run_p6(root/"p0", root/"invalid", **{**kwargs,**altered})
            self.assertEqual(a["summary"], b["summary"])


if __name__ == "__main__":
    unittest.main()
