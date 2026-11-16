"""FT-009 immutable custody, real-time-feature and situational STOP gates."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest

import numpy as np
import ft009_situational_stop as f

class FT009Tests(unittest.TestCase):
    def setUp(self):
        self.r=json.loads((Path(__file__).parent/"request_ft009.json").read_text())

    def test_frozen_request_and_authority(self):
        self.assertEqual(f.validate_request(self.r),self.r)
        for k,v in (("max_parallel_runner_jobs",4),("native_edits",1),
                    ("pretrained_slm_optimizer_steps",1),("terminal_stop_bonus",100),
                    ("ft003_artifact_id",0),("train_worlds_per_scenario",1000)):
            mutated=copy.deepcopy(self.r);mutated[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):
                f.validate_request(mutated)

    def test_world_custody_is_unique(self):
        pop=types.SimpleNamespace(stable_seed=lambda *v:int.from_bytes(
            hashlib.sha256("|".join(map(str,v)).encode()).digest()[:8],"big")%2**32)
        self.assertEqual(len(f.custody_hash(pop)),64)
        x=f.domain_seed(pop,"TRAIN",9401,"navigator","long60",0)
        y=f.domain_seed(pop,"EVAL",9401,"navigator","long60",0)
        self.assertNotEqual(x,y)
        with self.assertRaises(ValueError):
            f.domain_seed(pop,"TRAIN",9401,"navigator","fresh30",0)
        with self.assertRaises(ValueError):
            f.lease(9401,"navigator","unregistered")

    def test_retrace_stays_a_recovery_not_a_terminal(self):
        class W:
            steps=35;max_steps=60;cursor=(3,4);goal=(18,18);painted={(3,4)}
            def legal(self):return ["RETRACE","STOP"]
        w=W()
        self.assertFalse(f.terminal_context(w,w.legal()))
        w.cursor=w.goal
        self.assertTrue(f.terminal_context(w,w.legal()))
        w.cursor=(3,4);w.steps=58
        self.assertFalse(f.terminal_context(w,w.legal()))
        w.painted={(i,4) for i in range(10)}
        self.assertTrue(f.terminal_context(w,w.legal()))

    def test_genuine_time_and_terminal_supervision(self):
        class W:
            steps=7;max_steps=60;cursor=(3,4);goal=(18,18);painted={(3,4)}
            def legal(self):return ["E","STOP"]
        w=W()
        pop=types.SimpleNamespace(FEATURES=("early","stop"),
            utility=lambda n,x,w:0. if x[1]==0 else -2.5,
            softmax=lambda x:np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum())
        time=1.-w.steps/w.max_steps
        X=np.array([[time,0.],[time,1.]])
        a=f.supervised_targets(pop,"navigator",w,X,w.legal(),f.ARMS[0],time)
        b=f.supervised_targets(pop,"navigator",w,X,w.legal(),f.ARMS[1],time)
        self.assertLess(b[1],a[1])
        w.cursor=w.goal
        c=f.supervised_targets(pop,"navigator",w,X,w.legal(),f.ARMS[1],time)
        self.assertGreater(c[1],a[1])
        with self.assertRaises(AssertionError):
            f.supervised_targets(pop,"navigator",w,X,w.legal(),f.ARMS[1],0.0)
        with self.assertRaises(ValueError):
            f.supervised_targets(pop,"navigator",w,X,["STOP"],f.ARMS[1],time)

if __name__=="__main__":
    unittest.main()
