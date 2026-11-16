"""FT019 strict custody, protected host menus, trained optimizer and phase tests."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import ft019_two_goal_training as ft

class TestFT019(unittest.TestCase):
    def test_locked_request_and_game_boundary(self):
        r=json.loads((Path(__file__).parent/"request_ft019.json").read_text())
        self.assertEqual(ft.check_request(r),ft.expected_request())
        for key,v in (("source_blob","foreign"),("ft018_git_blob","foreign"),
                      ("max_parallel_runner_jobs",16),("adam_updates_per_continuation",0),
                      ("native_material_edits",1),("heldout_worlds_per_group",999)):
            wrong=copy.deepcopy(r);wrong[key]=v
            with self.assertRaises(ValueError):ft.check_request(wrong)

    def test_unique_model_and_split_leases(self):
        ids=[ft.lease(s,c,n,a) for s in ft.SEEDS for c in ft.COHORTS
             for n in ft.COHORTS[c] for a in ft.ARMS]
        self.assertEqual(len(ids),36)
        self.assertEqual(len(ids),len(set(ids)))
        class Pop:
            @staticmethod
            def stable_seed(*args):return repr(args)
        worlds=[ft.world_seed(Pop,d,9401,"navigator","dense120",i)
                for d,num in (("TEACHER",ft.CANDIDATES),
                              ("RECOVERY",ft.RECOVERY_WORLDS),
                              ("HELDOUT",ft.TEST_WORLDS))
                for i in range(num)]
        self.assertEqual(len(worlds),len(set(worlds)))
        with self.assertRaises(ValueError):ft.world_seed(Pop,"HELDOUT",9401,"navigator","dense120",1000)
        with self.assertRaises(ValueError):ft.lease(9401,"art","navigator","frozen")

    def test_stop_exact_final_phase_and_host_only(self):
        class World:
            switched=False;cursor=(3,4);goal=(3,4)
            def legal(self):return ["E","STOP"]
            def feature(self,a):
                x=np.zeros(20);x[0]=1;x[1]=int(a=="E")
                return x
        class Model:
            def logits(self,X):return X[:,1]
        w=World()
        self.assertEqual(ft.choose(w,Model()),("E",False))
        w.switched=True
        self.assertEqual(ft.choose(w,Model()),("STOP",True))
        w.cursor=(2,4)
        self.assertEqual(ft.choose(w,Model()),("E",False))
        w.legal=lambda:["STOP"]
        self.assertEqual(ft.choose(w,Model()),("STOP",False))

    def test_supervision_excludes_stop_and_one_hot_is_legal(self):
        class World:
            def legal(self):return ["N","E","STOP"]
            def feature(self,a):
                x=np.zeros(20);x[0]=1;x[2]=int(a=="E");return x
        w=World()
        X,y=ft.example(w,"E")
        self.assertEqual(X.shape,(2,20))
        self.assertEqual(y.tolist(),[0.0,1.0])
        with self.assertRaises(AssertionError):ft.example(w,"STOP")

    def test_real_updates_equal_exposure_and_no_data_fallback(self):
        class Policy:
            def __init__(self):self.updates=0
            def teach(self,X,y):
                self.updates+=1
                return 0.1
        class Pop:
            @staticmethod
            def stable_seed(*args):return 19877
        source=[(np.eye(2,20),np.array([1.,0.]))]
        with patch.object(ft.shift,"load_policy",side_effect=lambda *_: Policy()):
            p,m=ft.fit(Pop(),Path("unused"),"legal_route",source,[],9401,"navigator")
            self.assertEqual(m["adam_updates"],ft.UPDATES)
            self.assertEqual(p.updates,ft.UPDATES)
            p,m=ft.fit(Pop(),Path("unused"),"recovery_rehearsal",source,source,9401,"navigator")
            self.assertEqual(m["adam_updates"],ft.UPDATES)
            with self.assertRaises(ValueError):
                ft.fit(Pop(),Path("unused"),"recovery_rehearsal",source,[],9401,"navigator")

    def test_cumulative_goal_not_reward_or_early_stop(self):
        rows=[dict(completed_final_goal=1,host_forced_stop=0,premature_stop=0,
                   goal_visits=1,new_ink=7,retraces=2,decisions=50,neural_decisions=49),
              dict(completed_final_goal=0,host_forced_stop=1,premature_stop=0,
                   goal_visits=0,new_ink=5,retraces=0,decisions=60,neural_decisions=59)]
        result=ft.summary(rows)
        self.assertEqual(result["final_goal_completions"],1)
        self.assertEqual(result["host_forced"],1)
        self.assertEqual(result["mean_ink"],6)

if __name__=="__main__":unittest.main()
