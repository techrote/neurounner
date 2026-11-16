#!/usr/bin/env python3
"""Model-free regression for non-admitted Greeble/Architecture v3 studies."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from threefamily_v3 import STUDIES, HOST_SEEDS, program_for
from compiler import (
    CompilerError, canonical_json, parse_program, compile_program, structural_signature,
)

class ThreeFamilyV3Tests(unittest.TestCase):
    def test_exact_structural_composition_count_and_distinct_signature(self):
        self.assertEqual(set(STUDIES), {"G08", "G09", "A07", "A08"})
        self.assertEqual([STUDIES[k]["family"] for k in STUDIES], [
            "greebles", "greebles", "architecture", "architecture",
        ])
        signatures = []
        for name in STUDIES:
            p = program_for(name)
            self.assertEqual(parse_program(canonical_json(p)), p)
            signatures.append(canonical_json(structural_signature(p)))
        self.assertEqual(len(set(signatures)), 4)

    def test_two_seed_views_per_composition_remain_proposals(self):
        for name in STUDIES:
            for seed in HOST_SEEDS:
                with self.subTest(name=name, seed=seed):
                    c = compile_program(program_for(name), seed=seed)
                    self.assertTrue(c["structural_preflight"]["passed"])
                    self.assertTrue(c["proposal_only"])
                    self.assertFalse(c["native_qualified"])
                    self.assertLessEqual(c["operation_count"], 512)
                    self.assertGreater(c["estimated_cell_work"], 0)

    def test_greeble_negative_space_and_hierarchy(self):
        for name, count in (("G08", 5), ("G09", 3)):
            c = compile_program(program_for(name), seed=HOST_SEEDS[0])
            m = c["structural_preflight"]["metrics"]
            self.assertEqual(m["panel_count"], count)
            self.assertGreaterEqual(m["negative_space_percent"], 60)
            self.assertGreaterEqual(m["detail_kinds"], 3)

    def test_architecture_horizontal_side_elevation_and_route(self):
        for name, count in (("A07", 1), ("A08", 3)):
            p = program_for(name)
            for part in p["components"]:
                self.assertIn(part["kind"], {"room", "platform", "support"})
                if part["kind"] == "room":
                    self.assertTrue(all(o["side"] in {"left", "right"} for o in part["openings"]))
            for seed in HOST_SEEDS:
                c = compile_program(p, seed=seed)
                m = c["structural_preflight"]["metrics"]
                self.assertEqual(m["room_count"], count)
                self.assertTrue(m["route_requested"])
                self.assertTrue(m["clearance_route_passed"])
                self.assertEqual(m["clearance_cells"], 3)

    def test_invalid_panel_overlap_fails_closed(self):
        p = program_for("G08")
        p["components"][1]["box"] = list(p["components"][0]["box"])
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0])

    def test_invalid_room_door_crossing_corner_fails_closed(self):
        p = program_for("A08")
        p["components"][0]["openings"][0]["offset"] = 66
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0])

    def test_outside_host_authorization_fails_closed(self):
        p = program_for("A07")
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0], material=58, allowed_materials=(3,))

if __name__ == "__main__":
    unittest.main()
