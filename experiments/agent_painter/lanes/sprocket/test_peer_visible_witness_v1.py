#!/usr/bin/env python3
"""Offline adversarial tests for pinned, proposal-only SPROCKET peer witnesses."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peer_visible_witness_v1 import (
    SPROCKET_V3_BLOB, architecture, checked_pinned_json, git_blob_sha, panels,
)
from threefamily_v3 import program_for

class PinnedPeerWitnessTests(unittest.TestCase):
    def test_original_sprocket_program_blob_is_unchanged(self):
        source = Path(__file__).with_name("threefamily_v3.py").read_bytes()
        self.assertEqual(git_blob_sha(source), SPROCKET_V3_BLOB)

    def test_refuses_changed_peer_bytes_before_parsing(self):
        raw = json.dumps({"candidates": []}).encode()
        good = git_blob_sha(raw)
        self.assertEqual(checked_pinned_json(raw, good), {"candidates": []})
        with self.assertRaisesRegex(ValueError, "drift"):
            checked_pinned_json(raw + b" ", good)
        with self.assertRaisesRegex(ValueError, "drift"):
            checked_pinned_json(raw, "0" * 40)

    def test_refuses_non_object_peer_document(self):
        raw = b"[]"
        with self.assertRaisesRegex(ValueError, "object"):
            checked_pinned_json(raw, git_blob_sha(raw))

    def test_g08_five_panel_vertical_band_hierarchy(self):
        m = panels(program_for("G08"))
        self.assertEqual(m["count"], 5)
        self.assertEqual(m["top_y_bands"], [2, 2, 1])
        self.assertEqual(len(set(m["detail_vocabulary"])), 5)

    def test_g09_three_long_bands_not_three_compact_islands(self):
        m = panels(program_for("G09"))
        self.assertEqual(m["count"], 3)
        self.assertEqual(m["wide_panels_w_gt_2h"], 3)
        self.assertEqual(m["top_y_bands"], [1, 1, 1])

    def test_a07_is_one_hall_with_interior_lower_supported_decks(self):
        m = architecture(program_for("A07"))
        self.assertEqual(m["room_count"], 1)
        self.assertEqual(m["opening_sides"], {"left": 1, "right": 1})
        self.assertEqual(m["component_kinds"]["platform"], 2)
        self.assertEqual(m["component_kinds"]["support"], 2)
        self.assertEqual(m["platform_main_room_vertical_relations"],
                         ["within_vertical_room_extent"] * 2)
        self.assertEqual(m["route_axis"], "horizontal")
        self.assertEqual(m["clearance"], 3)

    def test_a08_facing_doors_and_repeated_room_motif_is_not_approved(self):
        m = architecture(program_for("A08"))
        self.assertEqual(m["room_count"], 3)
        self.assertEqual(m["opening_sides"], {"left": 3, "right": 3})
        self.assertEqual(m["route_axis"], "horizontal")
        self.assertEqual(m["component_kinds"], {"room": 3})

if __name__ == "__main__":
    unittest.main()
