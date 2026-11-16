"""Cheap source-only safety tests, no model downloads or game changes."""
import tempfile
from pathlib import Path
import unittest

from experiments.driv3r.frankenlayers_lab.flows import (
    LAB_FLOWS, WAVE1_FLOWS, WAVE2_FLOWS, WAVE3_FLOWS, WAVE5_FLOWS, WAVE6_FLOWS, WAVE14_FLOWS, WAVE15_FLOWS, validate_flows,
)
from experiments.driv3r.frankenlayers_lab.patch_lab import ATTN_GUARD, TARGET, patch

class FrankenLabTests(unittest.TestCase):
    def test_immutable_first_wave_and_new_tail_mutants(self):
        validate_flows()
        self.assertEqual({k:len(v) for k,v in LAB_FLOWS.items()},
            {"skele6":6,"aero6":6,"skele4":4,"aero4":4,"tailspin4":4,"needle4":4,
             "tailflip4":4,"echo4":4,"aero5_no20":5,"aero5_no23":5,
             "aero4_no10":4,"aero5_tailflip":5,"aero4_no2":4,"aero4_bridge":4,
             "aero4_no2_tailflip":4,"aero5_no2_echo10":5})
        self.assertEqual(WAVE1_FLOWS["skele4"], (0,4,18,23))
        self.assertEqual(WAVE1_FLOWS["aero4"], (0,4,20,23))
        self.assertEqual(WAVE2_FLOWS["tailspin4"], (0,4,18,20))
        self.assertEqual(WAVE2_FLOWS["needle4"], (0,4,18,22))
        self.assertEqual(WAVE3_FLOWS["tailflip4"], (0,4,20,18))
        self.assertEqual(WAVE3_FLOWS["echo4"], (0,4,18,4))
        self.assertEqual(WAVE3_FLOWS["aero5_no20"], (0,2,4,10,23))
        self.assertEqual(WAVE3_FLOWS["aero5_no23"], (0,2,4,10,20))
        self.assertEqual(WAVE5_FLOWS["aero4_no10"],(0,2,4,20))
        self.assertEqual(WAVE5_FLOWS["aero5_tailflip"],(0,2,4,20,10))
        self.assertEqual(WAVE6_FLOWS["aero4_no2"],(0,4,10,20))
        self.assertEqual(WAVE14_FLOWS["aero4_bridge"],(0,2,10,20))
        self.assertEqual(WAVE15_FLOWS["aero4_no2_tailflip"],(0,4,20,10))
        self.assertEqual(WAVE15_FLOWS["aero5_no2_echo10"],(0,4,10,20,10))
        self.assertNotEqual(WAVE6_FLOWS["aero4_no2"],WAVE5_FLOWS["aero4_no10"])
        self.assertEqual(len(set(WAVE3_FLOWS["echo4"])),3)
        self.assertNotEqual(tuple(sorted(WAVE3_FLOWS["tailflip4"])),WAVE3_FLOWS["tailflip4"])
        self.assertTrue(all(layers[0]==0 for layers in LAB_FLOWS.values()))
        self.assertEqual(len(set(LAB_FLOWS.values())), len(LAB_FLOWS))

    def test_invalid_or_historically_relabelled_flows_rejected(self):
        for name in LAB_FLOWS:
            mutated=dict(LAB_FLOWS)
            mutated[name]=(0,4,4,23) if len(mutated[name])==4 else (0,4,5,6,23,24)
            with self.subTest(name=name),self.assertRaises(ValueError):
                validate_flows(mutated)
            mutated=dict(LAB_FLOWS)
            original=list(mutated[name])
            original[1] = original[1]+1
            mutated[name]=tuple(original)
            with self.subTest(name=name,kind="tampered"),self.assertRaises(ValueError):
                validate_flows(mutated)

    def test_wave4_quant_candidates_source_qualified(self):
        from experiments.driv3r.frankenlayers_lab.infer_quant import QUANT_MODES
        self.assertEqual(set(QUANT_MODES), {"aero6", "skele6", "aero5_no23", "tailflip4", "echo4", "aero4_no10"})
        self.assertEqual(LAB_FLOWS["aero5_no23"], (0, 2, 4, 10, 20))
        self.assertEqual(LAB_FLOWS["tailflip4"], (0, 4, 20, 18))

    def test_wave3_nonmonotonic_and_duplicate_guard(self):
        mutated=dict(LAB_FLOWS)
        mutated["needle4"] = WAVE3_FLOWS["echo4"]
        with self.assertRaises(ValueError):
            validate_flows(mutated)
        mutated=dict(LAB_FLOWS)
        mutated["echo4"] = (0,4,18,18)
        with self.assertRaises(ValueError):
            validate_flows(mutated)
        mutated=dict(LAB_FLOWS)
        mutated["tailflip4"] = (0,4,18,20)
        with self.assertRaises(ValueError):
            validate_flows(mutated)
        mutated=dict(LAB_FLOWS)
        mutated["aero5_tailflip"] = (0,2,4,10,20)
        with self.assertRaises(ValueError):
            validate_flows(mutated)
        mutated=dict(LAB_FLOWS)
        mutated["aero4_no2"] = (0,2,4,20)
        with self.assertRaises(ValueError):
            validate_flows(mutated)

    def test_exact_source_patch_and_attention_guard(self):
        original = """#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void f() {
    for (int il = 0; il < n_layer; ++il) {
        auto * prev_cur = cur;
        if (il == n_layer - 1 && inp_out_ids) {
            cur = x;
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/TARGET
            p.parent.mkdir(parents=True)
            p.write_text(original)
            receipt=patch(root)
            output=p.read_text()
            for name,layers in LAB_FLOWS.items():
                self.assertIn(f'mode == "{name}"',output)
                self.assertIn("{"+",".join(map(str,layers))+"}",output)
            self.assertEqual(output.count(ATTN_GUARD),1)
            self.assertIn("!hparams.is_recr(source_il)",output)
            self.assertIn("!d1_has_full || !d1_has_sliding",output)
            self.assertIn("if constexpr (decision)",output)
            self.assertIn("d1_step + 1 == d1_flow.size()",output)
            self.assertFalse(receipt["weights_modified"])
            self.assertFalse(receipt["game_code_modified"])
            with self.assertRaises(RuntimeError):
                patch(root)

if __name__=="__main__":
    unittest.main()
