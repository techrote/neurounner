"""FT-005 adversarial custody and bounded optimizer tests."""
import copy
import hashlib
import json
from pathlib import Path
import time
import types
import unittest

import numpy as np
import ft005_curriculum as f

class TestFT005(unittest.TestCase):
    def setUp(self):
        self.request=json.loads((Path(__file__).parent/"request_ft005.json").read_text())

    def test_immutable_request_and_resource_ceiling(self):
        self.assertIs(f.checked_request(self.request),self.request)
        changes=(
            ("ft003_artifact_id",0),("ft003_zip_sha256","0"*64),
            ("train_seeds",[9401,9401]),("frozen_dose",12),
            ("training_worlds_per_scenario",2000),("test_worlds_per_scenario",2000),
            ("training_scenarios",["fresh30"]),("test_scenarios",["long60"]),
            ("max_seconds_per_seed",99999),("max_parallel_runners",16),
            ("native_material_edits",1),("pretrained_slm_optimizer_steps",1)
        )
        for key,value in changes:
            t=copy.deepcopy(self.request)
            t[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                f.checked_request(t)

    def test_lease_uniqueness_and_disallowed_family(self):
        leases=[f.lease_id(seed,niche) for seed in f.SEEDS for niche in f.base.NICHES]
        self.assertEqual(len(leases),12)
        self.assertEqual(len(set(leases)),12)
        with self.assertRaises(ValueError):
            f.lease_id(9401,"unauthorised_backbone")
        with self.assertRaises(ValueError):
            f.lease_id(0,f.base.NICHES[0])

    def test_training_and_test_domains_do_not_overlap_or_reuse_upstream(self):
        def stable(*parts):
            return int.from_bytes(hashlib.sha256(
                "|".join(map(str,parts)).encode()).digest()[:8],"big")%(2**32)
        pop=types.SimpleNamespace(stable_seed=stable)
        witness=f.check_seed_custody(pop)
        self.assertEqual(len(witness),64)
        self.assertEqual(len(set([f.domain_seed(pop,"TRAIN",9401,"navigator","long60",i)
                                  for i in range(8)])),8)
        with self.assertRaises(ValueError):
            f.domain_seed(pop,"TEST",9401,"navigator","long60",999)
        with self.assertRaises(ValueError):
            f.domain_seed(pop,"TRAIN",9401,"navigator","fresh30",0)
        with self.assertRaises(ValueError):
            f.domain_seed(pop,"CONTROL",9401,"navigator","long60",0)

    def test_real_optimizer_path_with_host_legal_stop_and_brush(self):
        class FakeWorld:
            def __init__(self):
                self.stopped=False
                self.steps=0
                self.max_steps=60
                self.cursor=(1,1)
                self.painted={(1,1)}
                self.protected={(9,9)}
                self.obstacles={(8,8)}
            def legal(self):
                if self.steps>=2: return ["STOP"]
                return ["E","STOP"]
            def feature(self,action):
                return np.array([1.,1. if action=="E" else 0.])
            def step(self,action):
                if action not in self.legal(): raise ValueError("illegal")
                if action=="E":
                    self.cursor=(self.cursor[0]+1,1)
                    self.painted.add(self.cursor)
                else: self.stopped=True
                self.steps+=1
        class FakePolicy:
            def __init__(self): self.updates=0
            def teach(self,X,target): self.updates+=1
            def choice(self,X,rng): return 0
        def stable(*parts):
            return int.from_bytes(hashlib.sha256(
                "|".join(map(str,parts)).encode()).digest()[:8],"big")%(2**32)
        pop=types.SimpleNamespace(stable_seed=stable,
                softmax=lambda a:np.exp(a-np.max(a))/np.exp(a-np.max(a)).sum(),
                utility=lambda niche,x,w:float(x[1]))
        original=f.transfer.world_for
        f.transfer.world_for=lambda pop,seed,scenario:FakeWorld()
        try:
            policy=FakePolicy()
            receipts=f.train_continuation(pop,policy,"navigator",9401,time.monotonic())
            self.assertEqual(len(receipts),24)
            self.assertEqual(policy.updates,sum(r["optimizer_updates"] for r in receipts))
            self.assertGreater(policy.updates,0)
            self.assertTrue(all(r["decisions"]>=1 and r["decisions"]<=3 for r in receipts))
        finally:
            f.transfer.world_for=original

if __name__=="__main__":
    unittest.main()
