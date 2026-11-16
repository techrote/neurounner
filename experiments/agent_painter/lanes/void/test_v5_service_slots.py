#!/usr/bin/env python3
"""Twelve actual model-free v5 geometry/source and adversarial failure checks."""
import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]))
from compiler import compile_program,rasterize_operations
from v5_service_slots import program_witness,material_witness,PROGRAM_SHA,SEEDS

class TestA02Slots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc=json.loads((HERE/"v5_support_proposals.json").read_text())
        cls.p=cls.doc["candidates"][0]["program"]
        cls.masks=[rasterize_operations(compile_program(cls.p,seed=s,focus=(0,0,96,96),
                    material=58,allowed_materials=(58,))["operations"]) for s in SEEDS]

    def test_01_program_pin(self):
        self.assertEqual(program_witness(self.doc)["source_program_sha256"],PROGRAM_SHA)
    def test_02_three_cell_service_slots(self):
        self.assertEqual(program_witness(self.doc)["side_service_slot_widths"],[3,3])
    def test_03_two_compiler_seeds(self):
        for m in self.masks:
            self.assertEqual(material_witness(m)["original_hard_wall_cells"],662)
    def test_04_left_void_obstruction(self):
        m=self.masks[0].copy();m[70*96+31]=58
        with self.assertRaisesRegex(ValueError,"left service"):material_witness(m)
    def test_05_right_void_obstruction(self):
        m=self.masks[0].copy();m[70*96+66]=58
        with self.assertRaisesRegex(ValueError,"right service"):material_witness(m)
    def test_06_left_post_missing(self):
        m=self.masks[0].copy();m[70*96+33]=0
        with self.assertRaisesRegex(ValueError,"post missing"):material_witness(m)
    def test_07_right_post_missing(self):
        m=self.masks[0].copy();m[70*96+64]=0
        with self.assertRaisesRegex(ValueError,"post missing"):material_witness(m)
    def test_08_floor_contact_missing(self):
        m=self.masks[0].copy();m[82*96+33]=0
        with self.assertRaisesRegex(ValueError,"floor contact"):material_witness(m)
    def test_09_central_route_obstructed(self):
        m=self.masks[0].copy();m[70*96+48]=58
        with self.assertRaisesRegex(ValueError,"central route"):material_witness(m)
    def test_10_modified_native_program_refused(self):
        d=copy.deepcopy(self.doc);d["candidates"][0]["program"]["components"][4]["box"][0]+=1
        with self.assertRaisesRegex(ValueError,"program changed"):program_witness(d)
    def test_11_fake_train_admission_refused(self):
        d=copy.deepcopy(self.doc);d["new_shared_train_groups"]=1
        with self.assertRaisesRegex(ValueError,"unapproved"):program_witness(d)
    def test_12_fake_native_admission_refused(self):
        d=copy.deepcopy(self.doc);d["candidates"][0]["native_qualified"]=True
        with self.assertRaisesRegex(ValueError,"research classification"):program_witness(d)

if __name__=="__main__":unittest.main(verbosity=2)
