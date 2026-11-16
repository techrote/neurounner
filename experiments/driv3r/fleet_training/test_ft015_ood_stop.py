"""FT-015: OOD lease, sealed artifact and host-legal replay contracts."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import numpy as np
import ft015_ood_stop as ft


class Toy:
    def __init__(self, goal):
        self.cursor=(3,11); self.goal=goal
    def legal(self):
        return ["E","STOP"]
    def feature(self, action):
        self_action=action
        assert self_action=="STOP"
        x=np.zeros(20); x[0]=x[3]=1.
        return x


class TestFT015(unittest.TestCase):
    def setUp(self):
        self.req=json.loads((Path(__file__).parent/"request_ft015.json").read_text())

    def test_strict_request(self):
        self.assertEqual(ft.checked_request(self.req),self.req)
        for k,v in [("optimizer_updates",1),("native_material_edits",1),
                    ("max_parallel_runner_jobs",8),
                    ("ft014_source_head","drift"),("max_seconds_per_job",300)]:
            copy_req=copy.deepcopy(self.req);copy_req[k]=v
            with self.assertRaises(ValueError):
                ft.checked_request(copy_req)

    def test_four_distinct_leases_and_artifacts(self):
        keys=[ft.job_lease(s,c) for s in ft.SEEDS for c in ft.COHORTS]
        self.assertEqual(len(keys),4)
        self.assertEqual(len(set(keys)),4)
        self.assertEqual(len(set(ft.ARTIFACTS.values())),4)
        for s in ft.SEEDS:
            for c in ft.COHORTS:
                self.assertIn(f"{s}/{c}",ft.ARTIFACTS)
        with self.assertRaises(ValueError):
            ft.job_lease(1,"art")

    def test_ood_custody_independent_of_ft014(self):
        pop=type("Pop",(),{"stable_seed":staticmethod(lambda *x:repr(x))})
        seeds=[ft.world_seed(pop,s,"navigator",sc,0) for s in ft.SEEDS
               for sc in ft.SCENARIOS]
        self.assertEqual(len(set(seeds)),len(seeds))
        with self.assertRaises(ValueError):
            ft.world_seed(pop,9401,"navigator","fresh30",0)

    def test_immutable_saved_gate_loader(self):
        model=ft.ft.StopGate(True)
        model.w[-1]=-9
        model.steps=ft.ft.STEPS
        model.threshold=.5
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"sidecar.json"
            p.write_text(json.dumps(model.serialize())+"\n")
            digest=ft.base.sha_file(p)
            loaded=ft.read_gate(p,True,digest)
            self.assertEqual(loaded.steps,600)
            self.assertTrue(np.array_equal(loaded.w,model.w))
            self.assertGreater(
                loaded.probability(ft.ft.features(Toy((3,11)),True)),
                loaded.probability(ft.ft.features(Toy((4,11)),True)))
            with self.assertRaises(ValueError):
                ft.read_gate(p,True,"0"*64)
            with self.assertRaises(ValueError):
                ft.read_gate(p,False,digest)

    def test_exact_goal_rule_not_clairvoyant_on_distance(self):
        self.assertEqual(ft.ft.distance(Toy((3,11))),0)
        self.assertEqual(ft.ft.distance(Toy((4,11))),1)

    def test_limit(self):
        with self.assertRaises(TimeoutError):
            ft.budget(-1e9)


if __name__=="__main__":
    unittest.main()
