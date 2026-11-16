#!/usr/bin/env python3
"""Adversarial source-only checks for VOID A02. No training, native Art or FINAL."""
import copy
import json
from pathlib import Path
import unittest
from v4_shaft_probe import shaft_witness
from compiler import compile_program

DOC=Path(__file__).resolve().with_name("v4_shaft_proposals.json")
def proposed():
    return json.loads(DOC.read_text(encoding="utf-8"))["candidates"][0]["program"]

class V4ShaftTests(unittest.TestCase):
    def test_vertical_two_portal_three_offset_decks(self):
        p=proposed()
        self.assertTrue(shaft_witness(p))
        self.assertEqual([c["kind"] for c in p["components"]],
                         ["room","platform","platform","platform","support","support"])
        self.assertEqual([o["side"] for o in p["components"][0]["openings"]],["top","bottom"])
    def test_both_compiler_seed_views_5_clearance(self):
        for seed in (104729,130363):
            with self.subTest(seed=seed):
                q=compile_program(proposed(),seed=seed,focus=(0,0,96,96),
                                  material=58,allowed_materials=(58,))
                m=q["structural_preflight"]["metrics"]
                self.assertTrue(q["structural_preflight"]["passed"])
                self.assertTrue(m["clearance_route_passed"])
                self.assertEqual((m["room_count"],m["opening_count"],m["clearance_cells"]),(1,2,5))
    def test_horizontal_facing_room_variant_refused(self):
        p=proposed()
        p["components"][0]["box"]=[15,28,74,40]
        self.assertFalse(shaft_witness(p))
    def test_side_entry_or_missing_top_portal_refused(self):
        p=proposed()
        p["components"][0]["openings"][0]["side"]="left"
        self.assertFalse(shaft_witness(p))
        p=proposed()
        p["components"][0]["openings"].pop()
        self.assertFalse(shaft_witness(p))
    def test_hall_stair_or_extra_room_refused(self):
        p=proposed()
        p["components"].append({"kind":"room","box":[2,2,18,20],"thickness":2,"openings":[]})
        self.assertFalse(shaft_witness(p))
    def test_center_blocking_platform_is_refused_by_shaft_witness(self):
        # A compiler free-space path could detour outside through portals;
        # that would NOT preserve the intended unobstructed central shaft.
        p=proposed()
        p["components"][1]["box"]=[44,28,10,2]
        self.assertFalse(shaft_witness(p))
    def test_no_claim_of_native_or_training_admission(self):
        d=json.loads(DOC.read_text(encoding="utf-8"))
        self.assertIs(d["final_targets_opened"],False)
        self.assertIs(d["development_targets_used"],False)
        self.assertEqual((d["new_shared_train_groups"],d["model_optimizer_updates"]),(0,0))
        c=d["candidates"][0]
        self.assertIs(c["training_admitted"],False)
        self.assertIs(c["native_qualified"],False)
        self.assertEqual(c["review_status"],"PENDING_GLOBAL_SPATIAL_AND_HELDOUT_REVIEW")

if __name__=="__main__": unittest.main(verbosity=2)
