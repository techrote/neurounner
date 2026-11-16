import tempfile
from pathlib import Path
import unittest

from experiments.driv3r.frankenlayers.patch_runtime import FLOWS, patch, TARGET

class FrankenLayerTests(unittest.TestCase):
    def test_three_yolo_flows_are_exact(self):
        self.assertEqual(FLOWS["greyhound"],(0,1,2,3,4,6,8,10,14,18,20,22,23))
        self.assertEqual(FLOWS["scrambled_eggs"],(0,1,2,3,4,5,7,13,9,15,11,6,10,8,12,14,20,21,22,23))
        self.assertEqual(FLOWS["branchwitch"],(0,1,2,3,4,5,6,7,9,10,11,10,16,17,18,19,20,21,22,23))
    def test_greyhound_removes_almost_half_the_backbone(self):
        self.assertEqual(len(FLOWS["greyhound"]),13)
        self.assertEqual(len(set(FLOWS["greyhound"])),13)
    def test_scrambled_is_reordered_not_duplicated(self):
        self.assertEqual(len(FLOWS["scrambled_eggs"]),20)
        self.assertEqual(len(set(FLOWS["scrambled_eggs"])),20)
        self.assertNotEqual(FLOWS["scrambled_eggs"],tuple(sorted(FLOWS["scrambled_eggs"])))
    def test_branchwitch_executes_layer_10_twice(self):
        self.assertEqual(FLOWS["branchwitch"].count(10),2)
        self.assertEqual(len(FLOWS["branchwitch"]),20)
        self.assertEqual(len(set(FLOWS["branchwitch"])),19)
    def test_patch_is_fail_closed_and_decision_only(self):
        source='''#include "models.h"\n#include "../llama-memory-hybrid-iswa.h"\n\nvoid f() {\n    for (int il = 0; il < n_layer; ++il) {\n        auto * prev_cur = cur;\n        if (il == n_layer - 1 && inp_out_ids) {\n            cur = x;\n        }\n    }\n}\n'''
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/TARGET; p.parent.mkdir(parents=True); p.write_text(source)
            r=patch(root)
            text=p.read_text()
            self.assertIn("CYBERSAND_D1_FRANKEN_FLOW",text)
            self.assertIn("if constexpr (decision)",text)
            self.assertIn("#include <cstdlib>",text)
            self.assertIn("d1_step + 1 == d1_flow.size()",text)
            self.assertNotIn(r"{\\n", text)
            self.assertFalse(r["q8_weights_modified"])
            with self.assertRaises(RuntimeError):
                patch(root)

if __name__=="__main__":
    unittest.main()
