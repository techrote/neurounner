#!/usr/bin/env python3
"""SPROCKET A09: independently conceived model-free/source-only tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compiler import (CompilerError, canonical_json, compile_program,
                      parse_program, structural_signature)
from architecture_revision_v1 import (PROGRAM, HOST_SEEDS, program_for,
                                       original_v3_program)

class ArchitectureRevisionV1Tests(unittest.TestCase):
    def test_exact_new_two_volume_open_courtyard_grammar(self):
        p = program_for()
        self.assertEqual(parse_program(canonical_json(p)), p)
        self.assertEqual(p["version"], 2)
        self.assertEqual(p["family"], "architecture")
        self.assertEqual(p["requirements"]["clearance"], 3)
        self.assertEqual([c["kind"] for c in p["components"]],
                         ["room", "room", "platform", "support",
                          "support", "platform"])
        self.assertEqual([[o["side"] for o in c["openings"]]
                          for c in p["components"][:2]],
                         [["left", "right", "top"],
                          ["left", "right", "bottom"]])
        lower, elevated = p["components"][:2]
        self.assertGreater(lower["box"][1], elevated["box"][1])
        self.assertLess(lower["box"][0]+lower["box"][2],
                        elevated["box"][0])

    def test_two_seed_compilation_is_proposal_not_native_success(self):
        for seed in HOST_SEEDS:
            with self.subTest(seed=seed):
                c = compile_program(program_for(), seed=seed)
                m = c["structural_preflight"]["metrics"]
                self.assertTrue(c["structural_preflight"]["passed"])
                self.assertTrue(c["proposal_only"])
                self.assertFalse(c["native_qualified"])
                self.assertEqual((m["room_count"], m["opening_count"]), (2, 6))
                self.assertTrue(m["route_requested"])
                self.assertTrue(m["clearance_route_passed"])
                self.assertEqual(m["clearance_cells"], 3)
                self.assertLessEqual(c["operation_count"], 512)
                self.assertLessEqual(c["estimated_cell_work"], 100000)

    def test_no_relabel_of_older_architecture_research(self):
        current = canonical_json(structural_signature(program_for()))
        for name in ("A07", "A08"):
            self.assertNotEqual(current, canonical_json(
                structural_signature(original_v3_program(name))))

    def test_canonical_source_is_copy_on_read(self):
        edited = program_for()
        edited["components"][0]["box"][0] = 94
        self.assertNotEqual(edited, PROGRAM)
        self.assertEqual(program_for(), PROGRAM)

    def test_invalid_doorway_crossing_corner_refused(self):
        p = program_for()
        p["components"][0]["openings"][2]["offset"] = 28
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0])

    def test_unauthorised_material_refused(self):
        with self.assertRaises(CompilerError):
            compile_program(program_for(), seed=HOST_SEEDS[0],
                            material=58, allowed_materials=(3,))

    def test_bounded_capacity_not_silently_overridden(self):
        with self.assertRaises(CompilerError):
            compile_program(program_for(), seed=HOST_SEEDS[0],
                            max_cell_work=1)

    def test_invalid_program_request_refused(self):
        with self.assertRaises(ValueError):
            program_for("A08")

if __name__ == "__main__":
    unittest.main()
