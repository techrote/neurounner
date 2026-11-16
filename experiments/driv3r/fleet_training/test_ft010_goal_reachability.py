"""FT-010 adversarial corpus contract: no teleport or held-out leakage."""
import copy
import json
from pathlib import Path
import types
import unittest
import ft010_goal_reachability as ft

class GoalCorpusTests(unittest.TestCase):
    def setUp(self):
        self.r=json.loads((Path(__file__).parent/"request_ft010.json").read_text())
    def test_immutable_source_and_budget(self):
        self.assertEqual(ft.checked_request(self.r),self.r)
        for k,v in (("optimizer_updates",1),("native_material_edits",1),
                    ("max_parallel_runners",4),("pool_per_group",2000),
                    ("source_blob","not-the-pinned-population-blob")):
            with self.subTest(k=k),self.assertRaises(ValueError):
                altered=copy.deepcopy(self.r);altered[k]=v
                ft.checked_request(altered)
    def test_domains_cannot_share_seeds(self):
        pop=types.SimpleNamespace(stable_seed=lambda *v:repr(v))
        a=ft.seed_for(pop,"TRAIN_CANDIDATES",9401,"navigator","long60",0)
        b=ft.seed_for(pop,"HELDOUT_CANDIDATES",9401,"navigator","long60",0)
        self.assertNotEqual(a,b)
        with self.assertRaises(ValueError):
            ft.seed_for(pop,"WRONG",9401,"navigator","long60",0)
        with self.assertRaises(ValueError):
            ft.seed_for(pop,"TRAIN_CANDIDATES",9401,"navigator","long60",48)
    def test_material_detour_and_impossible_goal(self):
        pop=types.SimpleNamespace(DIRECTIONS={
            "N":(0,-1),"E":(1,0),"S":(0,1),"W":(-1,0)})
        class World:
            cursor=(0,0);goal=(2,0);steps=0;stopped=False;switched=False
            max_steps=60
            def open(self,p):
                return p in {(0,1),(1,1),(2,1),(2,0)}
        self.assertEqual(ft.legal_route(pop,World()),("S","E","E","N"))
        w=World();w.open=lambda p:False
        self.assertIsNone(ft.legal_route(pop,w))
        w=World();w.steps=1
        with self.assertRaises(ValueError):
            ft.legal_route(pop,w)
    def test_do_not_use_goal_switch_step_as_terminal_goal(self):
        pop=types.SimpleNamespace(DIRECTIONS={"E":(1,0)})
        class World:
            cursor=(0,0);goal=(30,0);steps=0;stopped=False;switched=False
            max_steps=60
            def open(self,p):return p[1]==0 and 1<=p[0]<=30
        # Arriving on step 30 would change the objective; must reject.
        self.assertIsNone(ft.legal_route(pop,World()))

if __name__=="__main__":
    unittest.main()
