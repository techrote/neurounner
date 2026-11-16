"""Adversarial and original-archive tests for the SMOL P01/R10 peer audit."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from p01_r10_original_peer import audit, archive, r10_native, R10_ZIP
from p01_native_visible_proximity import compare, chebyshev_coverage


class P01R10PeerTests(unittest.TestCase):
    def test_identical_and_directional_metrics(self):
        same = compare({0, 1}, {0, 1})
        self.assertEqual(same["same_coordinate_iou"], 1)
        self.assertEqual(same["candidate_near_visible_radius3_hits"], 2)
        self.assertEqual(compare({0}, {1, 2})["same_coordinate_iou"], 0)
        self.assertEqual(compare({0}, {1, 2})["candidate_near_visible_radius3_denominator"], 1)
        self.assertEqual(compare({0}, {1, 2})["visible_near_candidate_radius3_denominator"], 2)

    def test_no_row_boundary_wrap(self):
        self.assertEqual(chebyshev_coverage({95}, {96}, 1)[0], 0)
        self.assertEqual(chebyshev_coverage({95}, {191}, 1)[0], 1)

    def test_bad_zip_hash_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("some.txt", "not an original native archive")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                archive(zpath, R10_ZIP)

    @unittest.skipUnless(os.environ.get("SMOL_P01_NATIVE") and
                         os.environ.get("SPROCKET_R10_NATIVE"),
                         "hosted original artifacts not present")
    def test_both_original_native_views_and_final_seal(self):
        p = Path(os.environ["SMOL_P01_NATIVE"])
        r = Path(os.environ["SPROCKET_R10_NATIVE"])
        base = Path(__file__).resolve().parent
        original_read = zipfile.ZipFile.read
        names = []
        def guard(z, member, *args, **kw):
            names.append(str(member))
            self.assertNotIn("final", str(member).lower())
            return original_read(z, member, *args, **kw)
        with patch.object(zipfile.ZipFile, "read", guard):
            result = audit(p, r, base / "curriculum_proposals_v2.json",
                           base / "P01_SHARED_ADMISSION_REVIEW_OFFER.json")
        self.assertEqual(len(result["comparisons"]), 2)
        self.assertEqual([v["seed"] for v in result["comparisons"]], [104729, 130363])
        self.assertEqual([v["acknowledged_material_cells"]
                          for v in result["original_R10_receipts"]], [1724, 1724])
        self.assertTrue(all(v["candidate_cells"] == 596 and v["visible_cells"] == 1724
                            for v in result["comparisons"]))
        self.assertEqual(result["new_shared_train_admissions"], 0)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertFalse(result["final_access"])
        self.assertTrue(names)

    @unittest.skipUnless(os.environ.get("SPROCKET_R10_NATIVE"),
                         "hosted original R10 artifact not present")
    def test_R10_native_verification_cannot_accept_case_loss(self):
        with archive(Path(os.environ["SPROCKET_R10_NATIVE"]), R10_ZIP) as z:
            original = z.read
            def missing(name, *args, **kwargs):
                if str(name).endswith("s130363.materials.i32le"):
                    raise KeyError("second seed omitted")
                return original(name, *args, **kwargs)
            with patch.object(z, "read", side_effect=missing):
                with self.assertRaises(KeyError):
                    r10_native(z)


if __name__ == "__main__":
    unittest.main()
