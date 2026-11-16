"""FT-013 guardrails: exact custody, legal on-policy domains, paired STOP targets."""
import copy
import json
from pathlib import Path
import types
import unittest

import numpy as np
import ft013_rehearsal_probe as ft


class FT013Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = json.loads((Path(__file__).parent/"request_ft013.json").read_text())

    def test_immutable_request_and_budget(self):
        self.assertEqual(ft.checked_request(copy.deepcopy(self.request)),self.request)
        for key,value in (("ft003_zip_sha256","wrong"),
                          ("source_blob","changed"),("max_parallel_runner_jobs",4),
                          ("native_material_edits",1),("pretrained_slm_optimizer_steps",1),
                          ("negative_stop_logit_penalty",500),
                          ("sparse_negative_stride",99),
                          ("goal_rehearsal_examples_per_round",0),
                          ("max_seconds_per_seed",0),("fresh_heldout_goals_per_group",0)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                mutated=copy.deepcopy(self.request)
                mutated[key]=value
                ft.checked_request(mutated)

    def test_unique_optimizer_leases(self):
        ids=[ft.lease(s,n,a) for s in ft.SEEDS
             for n in ft.base.NICHES for a in ft.ARMS]
        self.assertEqual(len(ids),36)
        self.assertEqual(len(set(ids)),36)
        for args in ((0,ft.base.NICHES[0],ft.ARMS[0]),
                     (9401,"wrong",ft.ARMS[1]),
                     (9403,ft.base.NICHES[0],"worm001")):
            with self.assertRaises(ValueError):
                ft.lease(*args)

    def test_world_custody(self):
        pop=types.SimpleNamespace(stable_seed=lambda *a:repr(a))
        s=[ft.seed_for(pop,split,9401,ft.base.NICHES[0],"dense60",0)
           for split in ("TRAIN","MINE","FRESH_GOAL","AUTONOMOUS")]
        self.assertEqual(len(set(s)),4)
        self.assertIn("TRAIN_CANDIDATES",s[0])
        self.assertIn("FT-013",s[1])
        with self.assertRaises(ValueError):
            ft.seed_for(pop,"FT011_HELDOUT",9401,ft.base.NICHES[0],"dense60",0)
        with self.assertRaises(ValueError):
            ft.seed_for(pop,"MINE",9401,ft.base.NICHES[0],"dense60",48)

    def test_hard_negative_contrast_only_suppresses_stop(self):
        vals=np.array([1.2,-.2,0.7],dtype=np.float64)
        si=2
        regular=np.exp(vals/ft.TEMPERATURE-np.max(vals/ft.TEMPERATURE))
        regular=regular/regular.sum()
        altered=vals.copy()
        altered[si]-=ft.STOP_NEGATIVE_LOGIT
        hard=np.exp(altered/ft.TEMPERATURE-np.max(altered/ft.TEMPERATURE))
        hard=hard/hard.sum()
        self.assertLess(hard[si],regular[si])
        self.assertGreater(hard[0],regular[0])
        self.assertEqual(ft.ARMS,("goal_only_control","sparse_hardnegative","sparse_negative_plus_goal_rehearsal"))
        self.assertEqual(ft.GOAL_REPEATS,ft.previous.GOAL_REPEATS)

    def test_three_arm_matched_rehearsal_contract(self):
        self.assertEqual(ft.SPARSE_STRIDE,2)
        self.assertEqual(ft.REHEARSAL_PER_ROUND,ft.TRAIN_GOALS)
        self.assertEqual(ft.previous.STOP_BONUS,5.0)
        self.assertEqual(ft.FRESH_GOALS,6)
        self.assertEqual(len(ft.ARMS),3)

    def test_wall_budget_fails_closed(self):
        with self.assertRaises(TimeoutError):
            ft.budget(-1e10)


if __name__=="__main__":
    unittest.main()
