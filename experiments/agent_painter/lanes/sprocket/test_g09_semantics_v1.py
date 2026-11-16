#!/usr/bin/env python3
"""Adversarial spatial-hierarchy regression, separate from curriculum admission."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from compiler import CompilerError, canonical_json, parse_program
from threefamily_v3 import program_for
from g09_semantics_v1 import check_hierarchy


class G09HierarchyTests(unittest.TestCase):
    def test_original_three_bands_pass(self):
        w = check_hierarchy(program_for("G09"))
        self.assertEqual(w["bands"], 3)
        self.assertEqual(w["widths"], [78, 61, 52])
        self.assertEqual(w["left_margins"], [7, 15, 34])
        self.assertEqual(w["inter_band_empty_rows"], [10, 11])
        self.assertFalse(w["global_independence_proven"])

    def test_json_component_order_irrelevant(self):
        p = program_for("G09")
        p["components"].reverse()
        self.assertEqual(check_hierarchy(p), check_hierarchy(program_for("G09")))

    def test_reject_diagonal_cassette_same_count_and_details(self):
        p = program_for("G09")
        p["components"][0]["box"][2] = 30
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "wide horizontal"):
            check_hierarchy(p)

    def test_reject_too_short_panels(self):
        p = program_for("G09")
        p["components"][2]["box"][2] = 39
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "substantial"):
            check_hierarchy(p)

    def test_reject_compressed_service_channel(self):
        p = program_for("G09")
        p["components"][1]["box"][1] = 35  # non-overlapping but only 3 empty rows
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "service gaps"):
            check_hierarchy(p)

    def test_reject_nonstaggered_left_margin(self):
        p = program_for("G09")
        p["components"][1]["box"][0] = 9
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "left margin"):
            check_hierarchy(p)

    def test_reject_nonshrinking_band_sequence(self):
        p = program_for("G09")
        p["components"][1]["box"][2] = 79  # end=94: schema-valid
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "shorter"):
            check_hierarchy(p)

    def test_reject_repeated_detail(self):
        p = program_for("G09")
        p["components"][1]["detail"] = p["components"][0]["detail"]
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "detail"):
            check_hierarchy(p)

    def test_reject_two_panel_shortcut(self):
        p = program_for("G09")
        p["components"].pop()
        parse_program(canonical_json(p))
        with self.assertRaisesRegex(ValueError, "three greeble"):
            check_hierarchy(p)

    def test_compiler_still_rejects_schema_corruption(self):
        p = program_for("G09")
        p["components"][0]["unreviewed_extra_field"] = True
        with self.assertRaises(CompilerError):
            check_hierarchy(p)


if __name__ == "__main__":
    unittest.main()
