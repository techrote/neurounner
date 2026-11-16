"""Model-free adversarial G09 vs G06 source geometry tests."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from g09_smol_visual_structure_v1 import compare,blob
G09={"version":2,"family":"greebles","components":[
 {"kind":"panel","box":[7,13,78,19]},{"kind":"panel","box":[15,42,61,17]},{"kind":"panel","box":[34,70,52,18]}]}
G06={"version":2,"family":"greebles","components":[
 {"kind":"panel","box":[8,66,22,18]},{"kind":"panel","box":[32,48,19,20]},{"kind":"panel","box":[70,12,20,22]}]}
class Tests(unittest.TestCase):
 def test_positive_with_real_geometry(self):
    x=compare(G09,G06)
    self.assertEqual((x["sprocket_box_cells"],x["smol_box_cells"]),(3455,1216))
    self.assertEqual(x["new_train_groups"],0)
    self.assertFalse(x["globally_distinct"])
 def test_json_component_order_invariant(self):
    a=copy.deepcopy(G09);b=copy.deepcopy(G06)
    a["components"].reverse();b["components"].reverse()
    self.assertEqual(compare(a,b),compare(G09,G06))
 def test_reject_g09_cassette(self):
    with self.assertRaisesRegex(ValueError,"G09"):compare(G06,G06)
 def test_reject_g06_wide_band_alias(self):
    with self.assertRaisesRegex(ValueError,"SMOL"):compare(G09,G09)
 def test_reject_g09_left_step_change(self):
    a=copy.deepcopy(G09);a["components"][1]["box"][0]=7
    with self.assertRaisesRegex(ValueError,"G09"):compare(a,G06)
 def test_reject_g06_step_direction(self):
    a=copy.deepcopy(G06);a["components"][0]["box"][0]=74
    with self.assertRaisesRegex(ValueError,"SMOL"):compare(G09,a)
 def test_reject_outside_canvas(self):
    a=copy.deepcopy(G09);a["components"][0]["box"][0]=90
    with self.assertRaisesRegex(ValueError,"out-of-bounds"):compare(a,G06)
 def test_reject_noninteger(self):
    a=copy.deepcopy(G09);a["components"][0]["box"][2]=78.0
    with self.assertRaisesRegex(ValueError,"invalid panel"):compare(a,G06)
 def test_reject_extra_panel(self):
    a=copy.deepcopy(G09);a["components"].append(a["components"][0])
    with self.assertRaisesRegex(ValueError,"three-panel"):compare(a,G06)
 def test_git_blob_fixture(self):
    self.assertEqual(blob(b"test\n"),"9daeafb9864cf43055ae93beb0afd6c7d144bfa4")
if __name__=="__main__":unittest.main()
