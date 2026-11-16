"""FT-014: deterministic CPU-only contracts for trained STOP gates and custody."""
import copy
import json
from pathlib import Path
import types
import unittest

import numpy as np
import ft014_stop_sidecar as ft


class Toy:
    def __init__(self, dist):
        self.cursor=(0,0)
        self.goal=(int(dist),0)
    def legal(self):
        return ["E","STOP"]
    def feature(self, action):
        assert action=="STOP"
        x=np.zeros(20,dtype=np.float64)
        x[0]=1.
        x[3]=1.
        x[12]=0.75
        return x


class FT014Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=json.loads((Path(__file__).parent/"request_ft014.json").read_text())

    def test_immutable_request_and_budget(self):
        self.assertEqual(ft.checked_request(copy.deepcopy(self.request)),self.request)
        for key,value in (("source_blob","drift"),("ft003_zip_sha256","fake"),
                          ("max_parallel_runner_jobs",8),("optimizer_updates_per_sidecar",0),
                          ("native_material_edits",1),("validation_false_positive_cap",.5),
                          ("pretrained_slm_optimizer_steps",1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                doc=copy.deepcopy(self.request)
                doc[key]=value
                ft.checked_request(doc)

    def test_exact_24_independent_optimizer_leases(self):
        keys=[ft.lease(s,n,a) for s in ft.SEEDS for n in ft.base.NICHES
              for a in ft.GATE_ARMS]
        self.assertEqual(len(keys),24)
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(sum(len(n) for n in ft.COHORTS.values()),6)
        self.assertEqual(set(sum((list(n) for n in ft.COHORTS.values()),[])),
                         set(ft.base.NICHES))
        with self.assertRaises(ValueError):
            ft.lease(9401,"navigator","frozen_baseline")

    def test_disjoint_custody_and_unknown_splits(self):
        pop=types.SimpleNamespace(stable_seed=lambda *args: repr(args))
        keys=[ft.world_seed(pop,split,9401,"navigator","dense60",0)
              for split in ft.SPLITS]
        self.assertEqual(len(set(keys)),5)
        with self.assertRaises(ValueError):
            ft.world_seed(pop,"FT013_FRESH",9401,"navigator","dense60",0)

    def test_original_host_features_are_blind_to_absolute_goal_distance(self):
        near=Toy(1)
        goal=Toy(0)
        self.assertTrue(np.array_equal(ft.features(goal,False),ft.features(near,False)))
        self.assertNotEqual(float(ft.features(goal,True)[-1]),
                            float(ft.features(near,True)[-1]))
        self.assertEqual(len(ft.features(goal,True)),21)
        self.assertEqual(len(ft.features(goal,False)),20)

    def test_real_optimizer_and_pinned_training_exposure(self):
        cases=[(Toy(0),1) for _ in range(20)]
        cases += [(Toy(i),0) for i in [1,2,3,4,6]*8]
        ordinary=ft.StopGate(False)
        geometric=ft.StopGate(True)
        o=ordinary.fit(cases)
        g=geometric.fit(cases)
        self.assertEqual(o["adam_updates"],600)
        self.assertEqual(g["adam_updates"],600)
        self.assertGreater(geometric.probability(ft.features(Toy(0),True)),
                           geometric.probability(ft.features(Toy(1),True)))
        self.assertAlmostEqual(
            ordinary.probability(ft.features(Toy(0),False)),
            ordinary.probability(ft.features(Toy(1),False)),places=10)
        self.assertTrue(np.isfinite(geometric.w).all())
        self.assertEqual(len(geometric.serialize()["w"]),21)

    def test_no_heldout_tuning_in_threshold_calibration(self):
        gate=ft.StopGate(True)
        gate.w[0]=2.
        gate.w[-1]=-10.
        validation=[(Toy(0),1)]*10+[(Toy(1),0)]*10
        tune=ft.choose_threshold(gate,validation)
        self.assertEqual(tune["validation_positive"],10)
        self.assertEqual(tune["false_positive_rate"],0)
        heldout=ft.score_gate(gate,[(Toy(0),1),(Toy(1),0)])
        self.assertEqual(heldout["tp"],1)
        self.assertEqual(heldout["fp"],0)

    def test_strict_limits(self):
        with self.assertRaises(TimeoutError):
            ft.budget(-1e20)


if __name__=="__main__":
    unittest.main()
