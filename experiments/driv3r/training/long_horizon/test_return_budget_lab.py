"""RETURN-BUDGET-001: source legality, branch-conflict and verifier guardrails."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import return_budget_lab as rb
from .world import World, specs, replay


class ReturnBudgetTests(unittest.TestCase):
    def fixture(self,split="development"):
        s=replace(specs(split,1)[0],challenge="trap_corridor")
        w=World.create(s)
        rb.initial_corridor(w,2)
        return s,w

    def test_same_state_different_budget_teacher(self):
        sp,lo=self.fixture("train")
        hi=World.create(sp);rb.initial_corridor(hi,8)
        self.assertEqual(lo.state_sha(),hi.state_sha())
        self.assertEqual(lo.menu(2),hi.menu(8))
        import random
        self.assertEqual(rb.correction_teacher(lo,128,2,random.Random(8)),"RELOCATE_START")
        self.assertEqual(rb.correction_teacher(hi,128,8,random.Random(8)),"BACKTRACK")

    def test_capacity_is_model_only_and_host_menu_stays_authoritative(self):
        _,w=self.fixture()
        for cap in rb.CAPS:
            _,base=rb.matrix(w,128,cap,"blind")
            offered,visible=rb.matrix(w,128,cap,"capacity_visible")
            self.assertEqual(offered,w.menu(cap))
            self.assertTrue(np.array_equal(base[:,:-5],visible[:,:-5]))
            self.assertTrue(np.array_equal(base[:,-5:],np.zeros_like(base[:,-5:])))
            self.assertFalse(np.array_equal(base,visible))
            self.assertEqual(w.recoveries,0)
        self.assertTrue(np.array_equal(rb.matrix(w,128,2,"blind")[1],rb.matrix(w,128,8,"blind")[1]))

    def test_backtrack_with_small_budget_can_exhaust_return(self):
        sp,w=self.fixture()
        w.apply("BACKTRACK",2);w.apply("BACKTRACK",2)
        self.assertEqual(w.menu(2),["STOP"])
        self.assertEqual(replay(sp,w.decisions,2).state_sha(),w.state_sha())

    def test_relocate_is_host_legal_and_preserves_material(self):
        sp,w=self.fixture()
        original_base=w.base_sha()
        w.apply("RELOCATE_START",2)
        self.assertEqual(w.cursor,(48,48))
        self.assertEqual(w.base_sha(),original_base)
        self.assertEqual(w.recoveries,1)
        self.assertEqual(replay(sp,w.decisions,2).state_sha(),w.state_sha())

    def test_corrupt_model_probability_is_rejected(self):
        sp,_=self.fixture()
        m=rb.Ranker(594,"blind")
        ep=rb.rollout(sp,m,2,128)
        rb.verify_episode(ep,m,sp)
        ep["rows"][6]["probabilities"]["STOP"]+=.04
        with self.assertRaises(ValueError):rb.verify_episode(ep,m,sp)

    def test_corrupt_host_menu_is_rejected(self):
        sp,_=self.fixture()
        m=rb.Ranker(594,"capacity_visible")
        ep=rb.rollout(sp,m,8,128)
        ep["rows"][6]["event"]["legal"]=["STOP"]
        with self.assertRaises(ValueError):rb.verify_episode(ep,m,sp)

    def test_sealed_heldout_rejected_without_creation(self):
        sp,_=self.fixture()
        m=rb.Ranker(594,"blind")
        with self.assertRaises(ValueError):rb.rollout(replace(sp,split="train",seed=1000),m,2,128)
        with self.assertRaises(ValueError):rb.rollout(sp,m,12,128)
        with self.assertRaises(ValueError):rb.rollout(sp,m,2,200)

    def test_genuine_optimizer_restore_and_original_replay(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"new_run"
            r=rb.run(out,seed=594,updates=1)
            self.assertEqual(r["episodes"],64)
            self.assertTrue(all(v["updates"]==3 for v in r["checkpoints"].values()))
            checked=rb.verify(out)
            self.assertEqual(checked["episodes_replayed"],64)
            self.assertEqual(checked["checkpoints_restored"],2)
            sample=next((out/"trajectories").glob("*.json"))
            data=json.loads(sample.read_text());data["final_sha"]="forged"
            sample.write_text(json.dumps(data))
            with self.assertRaises(ValueError):rb.verify(out)

if __name__=="__main__":unittest.main()
