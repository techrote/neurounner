"""No-model adversarial regressions for SMOL's native-material neighbour witness."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import native_mask_peer_audit as mod

MARKERS = {
    "smol": ("smol_g06_stepped_absence", "smol_p01_nested_branch_tree", "smol_a01_unequal_alcove_hall"),
    "sprocket": ("sprocket-G09",),
    "void": ("void-v5-A02",),
}
KEY = {
    "smol_g06_stepped_absence": "G06",
    "smol_p01_nested_branch_tree": "P01",
    "smol_a01_unequal_alcove_hall": "A01",
    "sprocket-G09": "G09",
    "void-v5-A02": "A02-v5",
}


def fixture(path: Path, family: str, *, illegal=False, duplicate=False, count_extra=False):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for marker in MARKERS[family]:
            for j, seed in enumerate(mod.SEEDS):
                count = mod.KNOWN_CELLS[KEY[marker]][j] + (1 if count_extra else 0)
                values = [58] * count + [0] * (9216 - count)
                if illegal:
                    values[0] = 59
                name = f"native/{marker}-s{seed}.materials.i32le"
                z.writestr(name, struct.pack("<9216i", *values))
                if duplicate:
                    z.writestr("duplicate/" + name, struct.pack("<9216i", *values))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaskPeerAuditTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.paths = {key: self.root / (key + ".zip") for key in MARKERS}
        self.hashes = {key: fixture(self.paths[key], key) for key in MARKERS}

    def check(self):
        with patch.dict(mod.ZIP_SHA256, self.hashes):
            return mod.audit(self.paths["smol"], self.paths["sprocket"], self.paths["void"])

    def test_positive_two_seed_distinct_design_accounting(self):
        result = self.check()
        self.assertEqual(result["independent_structures"], 5)
        self.assertEqual(result["physical_seed_views_read"], 10)
        self.assertEqual(len(result["same_seed_peer_pairs"]), 4)
        self.assertEqual(result["shared_train_admission"], "HOLD")
        self.assertEqual(result["new_optimizer_steps"], 0)
        self.assertFalse(result["global_structural_independence_proven"])
        self.assertFalse(result["final_targets_opened"])

    def test_modified_original_archive_rejected(self):
        with self.paths["smol"].open("ab") as f:
            f.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "archive SHA256 mismatch"):
            self.check()

    def test_wrong_material_id_rejected(self):
        self.hashes["sprocket"] = fixture(self.paths["sprocket"], "sprocket", illegal=True)
        with self.assertRaisesRegex(ValueError, "material domain"):
            self.check()

    def test_duplicate_member_refused(self):
        self.hashes["smol"] = fixture(self.paths["smol"], "smol", duplicate=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check()

    def test_bad_cell_count_refused(self):
        self.hashes["void"] = fixture(self.paths["void"], "void", count_extra=True)
        with self.assertRaisesRegex(ValueError, "cell count"):
            self.check()

    def test_missing_seed_refused(self):
        with zipfile.ZipFile(self.paths["void"], "w") as z:
            z.writestr("native/void-v5-A02-s104729.materials.i32le",
                       struct.pack("<9216i", *([58]*662 + [0]*(9216-662))))
        self.hashes["void"] = hashlib.sha256(self.paths["void"].read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check()

    def test_iou_math_exposes_differences(self):
        result = mod.compare({0, 1, 2}, {1, 2, 3})
        self.assertEqual(result["same_coordinate_iou"], 0.5)
        self.assertEqual(result["only_first_cells"], 1)
        self.assertEqual(result["only_second_cells"], 1)


if __name__ == "__main__":
    unittest.main()
