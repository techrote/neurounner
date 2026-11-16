"""Adversarial FT-011 source/checkpoint/STOP and heldout authority tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import types
import unittest
import numpy as np
import ft011_goal_stop as ft


class GoalStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = json.loads((Path(__file__).parent / "request_ft011.json").read_text())

    def test_exact_source_checkpoint_budget_and_scope(self):
        self.assertEqual(ft.checked_request(copy.deepcopy(self.valid)), self.valid)
        for k, value in (
            ("ft003_zip_sha256", "unverified"), ("source_blob", "different"),
            ("native_material_edits", 1), ("pretrained_slm_optimizer_steps", 1),
            ("max_parallel_runner_jobs", 4), ("fresh_heldout_goals_per_group", 0),
            ("goal_stop_logit_teacher_bonus", 9999)
        ):
            with self.subTest(key=k), self.assertRaises(ValueError):
                a = copy.deepcopy(self.valid)
                a[k] = value
                ft.checked_request(a)

    def test_leases_cannot_reuse_or_invent_identities(self):
        ids = [ft.lease(s, n, a) for s in ft.SEEDS
               for n in ft.base.NICHES for a in ft.ARMS]
        self.assertEqual(len(ids), len(set(ids)))
        for args in ((-1,"navigator",ft.ARMS[0]),
                     (9401,"unknown",ft.ARMS[0]),
                     (9401,"navigator","optimizer_worm001")):
            with self.assertRaises(ValueError):
                ft.lease(*args)

    def test_train_heldout_and_autonomy_separate(self):
        pop = types.SimpleNamespace(stable_seed=lambda *args: repr(args))
        s = [ft.world_seed(pop, split, 9401, "navigator", "dense60", 0)
             for split in ("TRAIN","FRESH_HELDOUT_GOAL","AUTONOMOUS")]
        self.assertEqual(len(set(s)),3)
        self.assertIn("TRAIN_CANDIDATES",s[0])
        self.assertNotIn("HELDOUT_CANDIDATES",str(s))
        with self.assertRaises(ValueError):
            ft.world_seed(pop,"FT010_HELDOUT",9401,"navigator","dense60",0)
        with self.assertRaises(ValueError):
            ft.world_seed(pop,"FRESH_HELDOUT_GOAL",9401,"navigator","dense60",48)

    def test_teacher_bonus_only_at_reached_goal_with_legal_menu(self):
        class World:
            def legal(self):
                return ["E","STOP"]
            def feature(self,a):
                return np.ones(20)*float(a=="STOP")
        world = World()
        pop = types.SimpleNamespace(
            utility=lambda niche, x, w: 0.0,
            softmax=lambda x: np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum())
        X, control = ft.teacher(pop,"navigator",world,world.legal(),ft.ARMS[0],True)
        _, goal = ft.teacher(pop,"navigator",world,world.legal(),ft.ARMS[1],True)
        _, early = ft.teacher(pop,"navigator",world,world.legal(),ft.ARMS[1],False)
        self.assertEqual(X.shape,(2,20))
        self.assertAlmostEqual(control[1],.5)
        self.assertGreater(goal[1],.999)
        self.assertAlmostEqual(early[1],control[1])
        with self.assertRaises(ValueError):
            ft.teacher(pop,"navigator",world,["STOP"],ft.ARMS[1],True)

    def test_invalid_or_nonfinite_policy_scores_rejected(self):
        class W:
            def legal(self): return ["E","STOP"]
            def feature(self,a): return np.ones(20)
        class P:
            def logits(self,X): return np.array([0.,np.nan])
        with self.assertRaises(AssertionError):
            ft.stop_scores(None,P(),W())
        class Good:
            def logits(self,X): return np.array([1.,0.])
        result=ft.stop_scores(None,Good(),W())
        self.assertFalse(result["stop_top1"])
        self.assertAlmostEqual(result["stop_margin"],-1)

    def test_goal_scaffold_never_teleports_and_budget_is_finite(self):
        pop = types.SimpleNamespace(stable_seed=lambda *args:repr(args))
        with self.assertRaises(ValueError):
            ft.goal_cases(pop,"EXPOSED_HELDOUT",9401,"navigator","long60",1,set())
        with self.assertRaises(TimeoutError):
            ft.check_budget(-1e10)


if __name__=="__main__":
    unittest.main()
