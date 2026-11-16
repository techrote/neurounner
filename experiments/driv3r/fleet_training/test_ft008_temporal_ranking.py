"""FT-008 source/custody/paired optimizer contract tests."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

import numpy as np
import ft008_temporal_ranking as f


class FT008Tests(unittest.TestCase):
    def setUp(self):
        self.request=json.loads((Path(__file__).parent/"request_ft008.json").read_text())

    def test_frozen_request_and_limits(self):
        self.assertEqual(f.validate_request(self.request),self.request)
        for key, replacement in (
            ("source_blob","x"*40),("ft003_artifact_id",123),
            ("max_parallel_runner_jobs",4),("max_seconds_per_seed",1800),
            ("native_edits",1),("pretrained_slm_optimizer_steps",1),
            ("train_worlds_per_scenario",1000),("arms",["one","two"]),
            ("late_stop_bonus",0),("frozen_dose",12)):
            changed=copy.deepcopy(self.request)
            changed[key]=replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                f.validate_request(changed)

    def test_unique_lease_and_train_test_diagnostic_custody(self):
        stable=lambda *v: int.from_bytes(hashlib.sha256(
            "|".join(map(str,v)).encode()).digest()[:8],"big")%2**32
        pop=types.SimpleNamespace(stable_seed=stable)
        self.assertEqual(len(f.custody_hash(pop)),64)
        worlds=[f.domain_seed(pop,d,9401,"navigator","long60",0)
                for d in ("TRAIN","EVAL","DIAG")]
        self.assertEqual(len(set(worlds)),3)
        with self.assertRaises(ValueError):
            f.domain_seed(pop,"PRODUCTION",9401,"navigator","long60",0)
        with self.assertRaises(ValueError):
            f.lease(9401,"navigator","unknown")
        with self.assertRaises(ValueError):
            f.domain_seed(pop,"TRAIN",9401,"navigator","long60",100)

    def test_early_and_late_targets_change_only_contrast_arm(self):
        def softmax(x):
            z=np.exp(x-np.max(x))
            return z/z.sum()
        pop=types.SimpleNamespace(FEATURES=("early","stop"),
              utility=lambda niche,row,w: -2.5 if row[1] else 1.0,
              softmax=softmax)
        w=types.SimpleNamespace(steps=50,max_steps=60,cursor=(2,2),goal=(10,10))
        menu=["E","STOP"]
        Xhigh=np.array([[1.,0.],[1.,1.]])
        Xlate=np.array([[0.,0.],[0.,1.]])
        plain_hi=f.supervised_targets(pop,"navigator",w,Xhigh,menu,f.ARMS[0],1.0)
        plain_lo=f.supervised_targets(pop,"navigator",w,Xlate,menu,f.ARMS[0],0.0)
        rank_hi=f.supervised_targets(pop,"navigator",w,Xhigh,menu,f.ARMS[1],1.0)
        rank_lo=f.supervised_targets(pop,"navigator",w,Xlate,menu,f.ARMS[1],0.0)
        self.assertAlmostEqual(plain_hi[1],plain_lo[1])
        self.assertLess(rank_hi[1],plain_hi[1])
        self.assertGreater(rank_lo[1],plain_lo[1])
        np.testing.assert_array_equal(Xhigh,np.array([[1.,0.],[1.,1.]]))
        with self.assertRaises(AssertionError):
            f.supervised_targets(pop,"navigator",w,Xhigh,menu,f.ARMS[1],0.0)

    def test_no_unsafe_forced_stop_when_host_has_no_alternative(self):
        softmax=lambda x: np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum()
        pop=types.SimpleNamespace(FEATURES=("early","stop"),softmax=softmax,
                                  utility=lambda n,x,w: 0.)
        world=types.SimpleNamespace(steps=59,max_steps=60,cursor=(0,0),goal=(5,5))
        X=np.array([[0.,1.]])
        for arm in f.ARMS:
            np.testing.assert_allclose(
                f.supervised_targets(pop,"navigator",world,X,["STOP"],arm,0.0),
                [1.])

    def test_counterfactual_inputs_must_match_whole_host_menu(self):
        pop=types.SimpleNamespace(FEATURES=("early","stop"),
            utility=lambda *args:0.,softmax=lambda x:np.ones(len(x))/len(x))
        w=types.SimpleNamespace(steps=1,max_steps=60,cursor=(0,0),goal=(1,1))
        X=np.array([[1.,0.],[0.,1.]])
        with self.assertRaises(AssertionError):
            f.supervised_targets(pop,"navigator",w,X,["E","STOP"],f.ARMS[1],1.0)


if __name__=="__main__":
    unittest.main()
