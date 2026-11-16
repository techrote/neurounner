"""FT-006 stop-target ablation: legal actions, custody and equal exposure."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

import numpy as np
import ft006_stop_ablation as f


class TestFT006(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((Path(__file__).parent / "request_ft006.json").read_text())

    def test_source_request_and_fail_closed_budgets(self):
        self.assertEqual(f.checked_request(self.request), self.request)
        for key, value in (
            ("ft003_artifact_id", 0), ("ft003_zip_sha256", "0"*64),
            ("train_seeds", [9401, 9401]), ("training_worlds_per_scenario", 999),
            ("early_stop_horizon", 60), ("early_stop_logit_penalty", 0.),
            ("max_seconds_per_seed", 99999), ("max_parallel_runners", 16),
            ("native_material_edits", 1), ("pretrained_slm_optimizer_steps", 1)
        ):
            r = copy.deepcopy(self.request)
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.checked_request(r)

    def test_lease_and_seed_domains(self):
        def stable(*parts):
            return int.from_bytes(hashlib.sha256(
                "|".join(map(str,parts)).encode()).digest()[:8],"big") % (2**32)
        pop = types.SimpleNamespace(stable_seed=stable)
        leases = [f.lease(seed,niche,arm)
                  for seed in f.SEEDS for niche in f.base.NICHES for arm in f.ARMS]
        self.assertEqual(len(leases), len(set(leases)))
        self.assertEqual(len(f.custody_digest(pop)), 64)
        self.assertNotEqual(f.world_seed(pop,"TRAIN",9401,"navigator","long60",0),
                            f.world_seed(pop,"TEST",9401,"navigator","long60",0))
        with self.assertRaises(ValueError):
            f.world_seed(pop,"TEST",9401,"navigator","long60",999)
        with self.assertRaises(ValueError):
            f.lease(9401,"unapproved",f.ARMS[0])

    def test_stop_penalty_only_when_early_and_escape_possible(self):
        pop = types.SimpleNamespace(
            utility=lambda niche,x,w:float(x[0]),
            softmax=lambda a:np.exp(a-np.max(a))/np.exp(a-np.max(a)).sum())
        menu=["STOP","E"]
        X=np.asarray([[0.],[1.]])
        early=types.SimpleNamespace(steps=2)
        plain=f.reward_targets(pop,"navigator",early,menu,X,False)
        penalized=f.reward_targets(pop,"navigator",early,menu,X,True)
        self.assertLess(penalized[0],plain[0])
        late=types.SimpleNamespace(steps=24)
        np.testing.assert_array_equal(f.reward_targets(pop,"navigator",late,menu,X,False),
                                      f.reward_targets(pop,"navigator",late,menu,X,True))
        np.testing.assert_array_equal(f.reward_targets(pop,"navigator",early,["STOP"],X[:1],False),
                                      f.reward_targets(pop,"navigator",early,["STOP"],X[:1],True))

    def test_teacher_never_paints_illegal_or_censors_terminal_stop(self):
        pop=types.SimpleNamespace(DIRECTIONS={"N","E","S","W"})
        world=types.SimpleNamespace()
        rng=np.random.default_rng(44)
        for menu in (["STOP"],["STOP","RETRACE"],["STOP","E","W"]):
            for _ in range(20):
                index=f.teacher_choice(pop,world,menu,rng)
                self.assertIn(menu[index],menu)
                self.assertNotEqual(menu[index],"STOP") if len(menu)>1 else self.assertEqual(menu[index],"STOP")


if __name__ == "__main__":
    unittest.main()
