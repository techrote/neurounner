"""FT-004 request, checkpoint, world-custody and receipt adversarial regression."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import ft004_shift as t


class TinyPolicy:
    def __init__(self, *args, **kwargs):
        self.p = {"W1":np.zeros((20,24)), "b1":np.zeros(24),
                  "W2":np.zeros(24), "b2":np.zeros(1)}


class TestFT004(unittest.TestCase):
    def setUp(self):
        self.req = json.loads((Path(__file__).parent / "request_ft004.json").read_text())

    def test_immutable_bounded_zero_optimizer_request(self):
        self.assertIs(t.request_check(self.req), self.req)
        for field, value in (("optimizer_steps",1), ("worlds_per_scenario",5000),
                             ("rss_cap_kib",99999999), ("scenarios",["fresh30"]),
                             ("ft003_artifact",0), ("ft003_archive_sha256","0"*64),
                             ("seeds",[9401,9401]), ("max_seconds",999999)):
            a = copy.deepcopy(self.req)
            a[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                t.request_check(a)

    def test_checkpoint_needs_real_optimizer_and_exact_features(self):
        pop = types.SimpleNamespace(Policy=TinyPolicy, FEATURES=tuple(f"x{i}" for i in range(20)))
        valid = {"format":"population-mlp-v1","features":list(pop.FEATURES),
                 "hidden":24, "optimizer_steps":1,"lr":0.004,"temperature":1.,
                 "parameters":{k:v.tolist() for k,v in TinyPolicy().p.items()}}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "policy.json"
            p.write_text(json.dumps(valid))
            loaded = t.load_policy(pop,p)
            self.assertEqual(loaded.p["W1"].shape,(20,24))
            for key, value in (("optimizer_steps",0),("hidden",23),("features",["fake"])):
                bad = copy.deepcopy(valid)
                bad[key] = value
                p.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    t.load_policy(pop,p)
            bad = copy.deepcopy(valid)
            bad["parameters"]["W2"][0] = float("nan")
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                t.load_policy(pop,p)

    def test_replay_rejects_menus_turns_and_material_hashes(self):
        class World:
            seed = 1
            cursor = (2,2)
            goal = (4,2)
            heading = "E"
            steps = 0
            stopped = False
            switched = False
            painted = {(2,2)}
            protected = {(1,1)}
            obstacles = {(3,3)}
            parent = {}
            def legal(self): return ["E","STOP"] if self.steps == 0 else ["STOP"]
            def step(self,a):
                if a not in self.legal(): raise ValueError(a)
                self.steps += 1
                if a == "E": self.cursor = (3,2); self.painted = {(2,2),(3,2)}
                if a == "STOP": self.stopped = True
        def make(): return World()
        old = t.world_for
        t.world_for = lambda pop,seed,scenario:make()
        try:
            w = make()
            before = t.state_digest(w)
            w.step("E")
            after = t.state_digest(w)
            rec = {"turn":1,"menu":["E","STOP"],"choice":"E",
                   "before":before,"after":after}
            self.assertTrue(t.audit_receipts(None,[rec],"test",1,"fresh30"))
            for field, value in (("menu",["E"]),("choice","W"),
                                 ("turn",2),("before","0"*64),("after","0"*64)):
                x = copy.deepcopy(rec)
                x[field] = value
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    t.audit_receipts(None,[x],"test",1,"fresh30")
        finally:
            t.world_for = old

    def test_new_domain_world_seeds_are_distinct(self):
        def stable(*args):
            import hashlib
            return int.from_bytes(hashlib.sha256("|".join(map(str,args)).encode()).digest()[:8],"big")%2**32
        pop = types.SimpleNamespace(stable_seed=stable)
        a=[t.heldout_world_seed(pop,9401,i) for i in range(t.WORLDS)]
        b=[t.heldout_world_seed(pop,9403,i) for i in range(t.WORLDS)]
        self.assertEqual(len(set(a+b)),12)
        self.assertTrue(set(a).isdisjoint(b))


if __name__ == "__main__":
    unittest.main()
