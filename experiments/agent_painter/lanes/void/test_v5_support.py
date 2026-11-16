#!/usr/bin/env python3
"""Eight adversarial source-only support tests; no model/trainer or FINAL."""
import copy,json
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from compiler import compile_program
from v5_support_probe import witness,OLD_SHA
ROOT=Path(__file__).resolve().parent
old=json.loads((ROOT/"v4_shaft_proposals.json").read_text())["candidates"][0]["program"]
doc=json.loads((ROOT/"v5_support_proposals.json").read_text())
new=doc["candidates"][0]["program"]
class Tests(unittest.TestCase):
 def test_positive_strict_parent_revision(self):
  self.assertTrue(witness(old,new))
 def test_original_native_program_identity(self):
  c=compile_program(old,seed=104729,focus=(0,0,96,96),material=58,allowed_materials=(58,))
  self.assertEqual(c["program_sha256"],OLD_SHA)
  self.assertEqual(c["unique_cells"],524)
 def test_both_seed_compiler_routes(self):
  for seed in (104729,130363):
   c=compile_program(new,seed=seed,focus=(0,0,96,96),material=58,allowed_materials=(58,))
   self.assertTrue(c["structural_preflight"]["metrics"]["clearance_route_passed"])
   self.assertGreater(c["unique_cells"],524)
 def test_floating_post_refused(self):
  p=copy.deepcopy(new);p["components"][4]["box"][3]-=1
  self.assertFalse(witness(old,p))
 def test_detached_post_refused(self):
  p=copy.deepcopy(new);p["components"][5]["box"][1]+=1
  self.assertFalse(witness(old,p))
 def test_off_platform_refused(self):
  p=copy.deepcopy(new);p["components"][5]["box"][0]=45
  self.assertFalse(witness(old,p))
 def test_changed_room_refused(self):
  p=copy.deepcopy(new);p["components"][0]["box"][0]+=1
  self.assertFalse(witness(old,p))
 def test_no_double_count_or_native_claim(self):
  self.assertFalse(doc["candidates"][0]["native_qualified"])
  self.assertFalse(doc["candidates"][0]["training_admitted"])
  self.assertEqual(doc["new_shared_train_groups"],0)
  self.assertEqual(doc["model_optimizer_updates"],0)
  self.assertFalse(doc["final_targets_opened"])
if __name__=="__main__":unittest.main(verbosity=2)
