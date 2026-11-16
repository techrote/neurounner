"""FT-016 ending taxonomy, fixed-horizon denominator and unique dispatch tests."""
import copy
import json
from pathlib import Path
import unittest

import ft016_endings as ft


class TestFT016(unittest.TestCase):
    def setUp(self):
        self.req=json.loads((Path(__file__).parent/"request_ft016.json").read_text())

    def test_request_is_strict_and_pinned(self):
        self.assertEqual(ft.check_request(self.req),self.req)
        for k,v in (
            ("max_parallel_runner_jobs",8),("optimizer_updates",1),
            ("native_material_edits",1),("ft015_git_blob","drift"),
            ("ft014_actions_run",0),("worlds_per_scenario",100),
        ):
            bad=copy.deepcopy(self.req)
            bad[k]=v
            with self.assertRaises(ValueError):
                ft.check_request(bad)

    def test_four_unique_job_leases(self):
        ids=[ft.lease(s,c) for s in ft.SEEDS for c in ft.COHORTS]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),4)
        with self.assertRaises(ValueError):
            ft.lease(0,"art")

    def test_immutable_world_domains(self):
        class Pop:
            @staticmethod
            def stable_seed(*args):
                return repr(args)
        autonomous={ft.world_seed(Pop,s,"navigator",x,i,"AUTONOMOUS")
                    for s in ft.SEEDS for x in ft.SCENARIOS for i in range(ft.WORLDS)}
        scaffold={ft.world_seed(Pop,s,"navigator",x,i,"SCAFFOLD")
                  for s in ft.SEEDS for x in ft.SCENARIOS for i in range(ft.PROBES)}
        self.assertEqual(len(autonomous),2*4*ft.WORLDS)
        self.assertEqual(len(scaffold),2*4*ft.PROBES)
        self.assertFalse(autonomous & scaffold)
        with self.assertRaises(ValueError):
            ft.world_seed(Pop,9401,"navigator","bad",0,"AUTONOMOUS")

    def test_stop_taxonomy_explicitly_disambiguates_host_compulsion(self):
        self.assertEqual(ft.ending_kind("STOP",["STOP"],False),"forced_only_stop_nongoal")
        self.assertEqual(ft.ending_kind("STOP",["STOP"],True),"forced_only_stop_at_goal")
        self.assertEqual(ft.ending_kind("STOP",["N","STOP"],False),"voluntary_nongoal_stop")
        self.assertEqual(ft.ending_kind("STOP",["N","STOP"],True),"voluntary_goal_stop")
        self.assertEqual(ft.ending_kind("E",["E","STOP"],False,True),"horizon_exhausted")
        self.assertEqual(ft.ending_kind("E",["E","STOP"],False,False),"continuing")
        with self.assertRaises(AssertionError):
            ft.ending_kind("STOP",["N"],False)

    def test_episode_summary_denominator_is_full_budget(self):
        rows=[]
        for kind,utility in [("horizon_exhausted",20),("voluntary_goal_stop",3),
                             ("forced_only_stop_nongoal",-2)]:
            rows.append(dict(ending=kind,horizon=120,decisions=10,
                             total_shaped_utility=utility,
                             utility_per_budget_step=utility/120,
                             utility_per_decision=utility/10,new_ink=7,
                             retraces=2,goal_visits=0,switch_occurred=False,
                             post_switch_decisions=0,
                             post_switch_voluntary_nongoal_stop=0))
        got=ft.summarize(rows)
        self.assertEqual(got["fixed_horizon_steps"],360)
        self.assertEqual(got["endings"]["voluntary_goal_stop"],1)
        self.assertEqual(got["endings"]["forced_only_stop_nongoal"],1)
        self.assertAlmostEqual(got["mean_utility_per_budget_step"],21/360)

    def test_matched_reward_is_total_not_reward_per_decision(self):
        base=dict(world_seed=1,new_ink=5,total_shaped_utility=10)
        worse=dict(world_seed=1,new_ink=6,total_shaped_utility=9)
        cases={a:[base if a=="frozen_baseline" else worse] for a in ft.ARMS}
        got=ft.paired(cases)
        for arm in ft.ARMS[1:]:
            self.assertEqual(got[arm]["total_utility_delta_mean"],-1)
            self.assertEqual(got[arm]["losses_total_utility"],1)

    def test_budget_timeout(self):
        with self.assertRaises(TimeoutError):
            ft.budget(-1e9)


if __name__=="__main__":
    unittest.main()
