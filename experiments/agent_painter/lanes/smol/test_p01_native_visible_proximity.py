#!/usr/bin/env python3
"""Model-free negative controls for the read-only P01 native-visible audit."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p01_native_visible_proximity as audit


class NativeVisibleAuditTests(unittest.TestCase):
    def test_material_crop_and_domain(self):
        import struct
        raw = struct.pack("<9216i", *([58, 0, 58] + [0] * 9213))
        self.assertEqual(audit.cells_from_i32(raw), {0, 2})
        with self.assertRaisesRegex(ValueError, "crop"):
            audit.cells_from_i32(raw[:-4])
        with self.assertRaisesRegex(ValueError, "material ID"):
            audit.cells_from_i32(struct.pack("<9216i", *([59] + [0] * 9215)))

    def test_mask_identical_and_one_offset(self):
        cells = {10, 11, 12, 96 + 10}
        same = audit.compare(cells, cells)
        self.assertEqual(same["same_coordinate_iou"], 1)
        self.assertEqual(same["candidate_near_visible_radius3_hits"], 4)
        shifted = {x + 96 for x in cells}
        moved = audit.compare(cells, shifted)
        self.assertLess(moved["same_coordinate_iou"], 1)
        self.assertEqual(moved["candidate_near_visible_radius3_hits"], 4)

    def test_radius_preserves_world_edges(self):
        a = {0}
        b = {95}
        self.assertEqual(audit.chebyshev_coverage(a, b), (0, 1))
        self.assertEqual(audit.chebyshev_coverage({95}, {96}), (0, 1))
        self.assertEqual(audit.chebyshev_coverage({0}, {3}), (1, 1))
        self.assertEqual(audit.chebyshev_coverage({0}, {4}), (0, 1))

    def test_refuse_empty_or_unbounded(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            audit.compare(set(), {1})
        with self.assertRaisesRegex(ValueError, "unbounded"):
            audit.chebyshev_coverage({1}, {1}, 10)

    def test_only_expected_visible_evidence_paths(self):
        self.assertEqual(audit.safe_evidence("evidence/original.materials.i32le"),
                         "corpus/evidence/original.materials.i32le")
        for bad in ("", "final.jsonl", "evidence/../final.jsonl",
                    "../evidence/file", "/evidence/file",
                    "evidence/nested/file", "evidence\\original", "evidence/"):
            with self.subTest(path=bad):
                with self.assertRaises(ValueError):
                    audit.safe_evidence(bad)

    def test_whole_original_zip_identity_required(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "made-up.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("corpus/train.jsonl", "invalid")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                audit.check_archive(f, audit.CORPUS_ZIP)

    def test_refuse_duplicate_archive_members(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "duplicate.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("a", "one")
                z.writestr("a", "two")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit.check_archive(f, hashlib.sha256(f.read_bytes()).hexdigest())

    def test_refuse_unpinned_visible_manifest_before_opening_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "corpus.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("corpus/admitted-corpus.json", '{"source_sha":"fake"}')
                z.writestr("corpus/final.jsonl", "do-not-open")
            with zipfile.ZipFile(f) as z:
                with self.assertRaisesRegex(ValueError, "manifest drift"):
                    audit.visible_pipework(z)

    def test_actual_authored_P01_tree_not_three_riser_label(self):
        src = Path(__file__).with_name("curriculum_proposals_v2.json")
        p = next(x["program"] for x in json.loads(src.read_text())["proposals"]
                 if x["id"] == audit.CANDIDATE)
        tree = audit.canonical_tree(p)
        self.assertIsNotNone(tree)
        self.assertEqual(len(p["components"][0]["nodes"]), 7)
        self.assertEqual(len(p["components"][0]["edges"]), 6)
        self.assertEqual(p["components"][0]["diameter"], 3)


if __name__ == "__main__":
    unittest.main()
