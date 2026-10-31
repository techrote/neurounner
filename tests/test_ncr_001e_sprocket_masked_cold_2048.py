"""Offline adversarial guards for NCR-001E dual-variant 2048 cold cache probe.

Only synthetic integer tokens; no model checkpoint, optimizer or heldouts.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import ncr_001e_sprocket_masked_cold_2048 as probe


class MaskedLongContextContract(unittest.TestCase):
    def setUp(self):
        self.long=torch.arange(64,dtype=torch.long)+12
        self.short=torch.arange(40,dtype=torch.long)+110

    def layout(self, side):
        return probe.geometry(self.long,self.short,side)

    def test_left_prefix_mask_and_true_end_are_distinct(self):
        long=self.long.clone();long[-1]=5
        x,m,pos=probe.geometry(long,self.short,"left")
        self.assertEqual(tuple(x.shape),(2,64))
        self.assertTrue((m[1,:24]==0).all().item())
        self.assertTrue((x[1,:24]==5).all().item())
        self.assertTrue((m[1,-4:]==1).all().item())
        self.assertEqual(int(x[0,-1]),5)
        self.assertEqual(int(m[0,-1]),1)
        self.assertEqual(int(pos[1,-1]),39)

    def test_right_gap_is_masked_before_four_real_cache_steps(self):
        x,m,pos=self.layout("right")
        self.assertTrue((m[1,36:60]==0).all().item())
        self.assertTrue((x[1,36:60]==5).all().item())
        self.assertTrue((m[1,-4:]==1).all().item())
        self.assertTrue((x[1,-4:]==self.short[-4:]).all().item())
        self.assertEqual(int(pos[1,-4]),36)
        self.assertEqual(int(pos[1,-1]),39)

    def test_both_rows_real_continuations_and_source_counts(self):
        for side in ("left","right"):
            with self.subTest(side=side):
                x,m,pos=self.layout(side)
                self.assertEqual(m.sum(dim=1).tolist(),[64,40])
                self.assertEqual(pos[0,-4:].tolist(),[60,61,62,63])
                self.assertEqual(pos[1,-4:].tolist(),[36,37,38,39])
                self.assertEqual(x[0,-4:].tolist(),self.long[-4:].tolist())

    def test_reject_changed_pad_id_and_unregistered_orientation(self):
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.geometry(self.long,self.short,"right",pad=7)
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.geometry(self.long,self.short,"middle")

    def test_reject_truncated_or_equal_sized_or_noninteger_rows(self):
        for short in (self.long,self.long[:60],self.long[:8]):
            with self.subTest(n=short.numel()),self.assertRaisesRegex(RuntimeError,"REFUSED"):
                probe.geometry(self.long,short,"left")
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.geometry(self.long.to(torch.float32),self.short,"left")
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.geometry(self.long[:,None],self.short,"left")

    def test_reject_wrong_token_id_or_oversized_context(self):
        bad=self.short.clone();bad[-1]=32000
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.geometry(self.long,bad,"right")
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.geometry(torch.arange(128),torch.arange(64),"right")

    def test_comparison_positive_and_adversarial_negative(self):
        original=torch.ones((2,4,8),dtype=torch.float32)
        probe.REPORT.clear()
        probe.score("unaltered",original,original.clone(),exact=True,atol=0,rtol=0)
        self.assertTrue(probe.REPORT["checks"]["unaltered"]["pass"])
        forged=original.clone()
        forged[1,-1,0]+=0.2
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.score("tampered_cache",original,forged)
        with self.assertRaisesRegex(RuntimeError,"REFUSED"):
            probe.score("nonfinite",original,original*float("nan"))

    def test_contract_keeps_four_models_and_training_closed(self):
        p=json.loads((ROOT/"experiments/ncr-001/protocol.json").read_text())
        s=json.loads((ROOT/"experiments/ncr-001/source-pins.json").read_text())
        self.assertIs(p["train_authorized"],False)
        self.assertEqual(set(p["models"]),{"smollm2","sprocket_chat","sprocket_instruct","void1"})
        for lane in ("chat","instruct"):
            source=probe.source.verify_contract(lane,p,s)
            self.assertEqual(source["weight_path"],probe.source.LANE_WEIGHTS[lane][0])
        self.assertNotEqual(probe.source.LANE_WEIGHTS["chat"][1],
                            probe.source.LANE_WEIGHTS["instruct"][1])
        changed=copy.deepcopy(p);changed["train_authorized"]=True
        with self.assertRaisesRegex(RuntimeError,"Training gate"):
            probe.source.verify_contract("chat",changed,s)

    def test_no_training_or_heldout_path(self):
        script=(ROOT/"scripts/ncr_001e_sprocket_masked_cold_2048.py").read_text()
        self.assertNotIn("optimizer.step(",script)
        self.assertNotIn(".backward(",script)
        self.assertIn('"train_authorized": False',script)
        self.assertIn('"full_72_conformance_pass": False',script)
        self.assertIn('short_ids[:SHORT]',script)


if __name__=="__main__":
    unittest.main()
