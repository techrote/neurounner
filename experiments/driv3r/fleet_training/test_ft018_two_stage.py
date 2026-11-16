"""FT-018 two-goal host authorization, phase accuracy and dispatch tests."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import ft018_two_stage as ft


class TestFT018(unittest.TestCase):
    def test_request_is_strict_and_research_only(self):
        request=json.loads((Path(__file__).parent/"request_ft018.json").read_text())
        self.assertEqual(ft.check_request(request),ft.expected_request())
        for k,v in (("ft017_git_blob","wrong"),("max_parallel_runner_jobs",8),
                    ("optimizer_updates",1),("native_material_edits",1),
                    ("evaluation_horizon",120),("worlds_per_scenario",99)):
            broken=copy.deepcopy(request)
            broken[k]=v
            with self.assertRaises(ValueError):
                ft.check_request(broken)

    def test_unique_wrangle_and_world_domains(self):
        jobs=[ft.lease(s,c) for s in ft.SEEDS for c in ft.COHORTS]
        self.assertEqual(len(jobs),4)
        self.assertEqual(len(set(jobs)),4)
        class Pop:
            @staticmethod
            def stable_seed(*values):return repr(values)
        source={ft.world_seed(Pop,"TEACHER",9401,"navigator",s,i)
                for s in ft.SCENARIOS for i in range(ft.CANDIDATES)}
        target={ft.world_seed(Pop,"AUTONOMOUS",9401,"navigator",s,i)
                for s in ft.SCENARIOS for i in range(ft.WORLDS)}
        self.assertFalse(source & target)
        with self.assertRaises(ValueError):
            ft.lease(0,"navigation")
        with self.assertRaises(ValueError):
            ft.world_seed(Pop,"AUTONOMOUS",9401,"navigator","bogus",0)

    def test_phase_controller_must_not_abort_first_objective(self):
        class World:
            def __init__(self,phase):
                self.switched=phase;self.cursor=(2,2);self.goal=(2,2)
            def legal(self):return ["E","STOP"]
            def feature(self,a):
                x=np.zeros(20)
                x[3]=float(a=="STOP")
                return x
        class Frozen:
            def logits(self,x):return 1-x[:,3]
        class Gate:
            threshold=.5
            def probability(self,x):return np.float64(.95)
        for phase in (False,True):
            w=World(phase)
            self.assertEqual(ft.choose(None,Frozen(),Gate(),"exact_current_goal",w)[0],"STOP")
            self.assertEqual(ft.choose(None,Frozen(),Gate(),"phase_exact",w)[0],
                             "STOP" if phase else "E")
            self.assertEqual(ft.choose(None,Frozen(),Gate(),"phase_gate",w)[0],
                             "STOP" if phase else "E")

    def test_second_stage_search_respects_only_host_open_cells(self):
        class Pop:
            DIRECTIONS={"E":(1,0),"W":(-1,0)}
        class World:
            switched=True;stopped=False;goal=(3,1);second_goal=(3,1)
            cursor=(1,1);max_steps=80;steps=40
            parent={};painted={(1,1)}
            def open(self,p):return p in {(2,1),(3,1)}
        w=World()
        self.assertEqual(ft.second_route(Pop,w),("E","E"))
        w.switched=False
        with self.assertRaises(ValueError):
            ft.second_route(Pop,w)

    def test_stop_results_use_final_goal_completion_not_shaped_reward(self):
        episodes=[dict(completed_final_goal=1,premature_voluntary_stop=0,
            host_forced_stop=0,first_goal_visits=1,final_goal_visits=1,
            post_switch_decisions=15,new_ink=24,decisions=50,retraces=4,total_utility=7),
            dict(completed_final_goal=0,premature_voluntary_stop=1,
            host_forced_stop=0,first_goal_visits=1,final_goal_visits=0,
            post_switch_decisions=0,new_ink=8,decisions=20,retraces=1,total_utility=30)]
        result=ft.summary(episodes)
        self.assertEqual(result["final_goal_completions"],1)
        self.assertEqual(result["premature_voluntary_stops"],1)
        self.assertEqual(result["switch_episodes"],1)
        self.assertEqual(result["mean_ink"],16)
        self.assertEqual(result["mean_decisions"],35)


if __name__=="__main__":
    unittest.main()
