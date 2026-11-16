"""Fail-closed, network-free source regressions for pinned A02 v5 spatial witness."""
import unittest
import json
import hashlib
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from a02_v5_spatial_comparison import (need, mask, iou, proximity_fraction,
                                       compare, visible_architecture, checked_zip, SEEDS, SPLITS)


class A02V5SpatialTests(unittest.TestCase):
    def test_native_little_endian_96_square(self):
        import struct
        raw = struct.pack("<9216i", *([0] * 9215 + [58]))
        self.assertEqual(mask(raw), {9215})
        with self.assertRaisesRegex(ValueError, "dimensions"):
            mask(raw[:-4])

    def test_reject_other_material_ids(self):
        import struct
        with self.assertRaisesRegex(ValueError, "material ID"):
            mask(struct.pack("<9216i", *([0] * 9215 + [59])))

    def test_iou_disjoint_and_equal(self):
        self.assertEqual(iou({1, 2}, {3}), 0)
        self.assertEqual(iou({1, 2}, {1, 2}), 1)
        self.assertAlmostEqual(iou({1, 2}, {2, 3}), 1 / 3)

    def test_iou_cannot_accept_empty_mask(self):
        with self.assertRaises(ValueError):
            iou(set(), {1})

    def test_proximity_nonwrapping_at_row_edge(self):
        # Adjacent flat-index cells 95/96 are far apart horizontally in a 96-wide crop.
        self.assertEqual(proximity_fraction({95}, {96}, radius=1), 0)
        self.assertEqual(proximity_fraction({95}, {94}, radius=1), 1)
        self.assertEqual(proximity_fraction({95}, {94}, radius=0), 0)

    def test_reject_invalid_proximity_radius(self):
        for radius in (-1, 9, 1.5, True):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                proximity_fraction({0}, {0}, radius)

    def test_hold_final_sealed_without_reading_zip(self):
        class NoReads:
            def read(self, name):
                raise AssertionError("not permitted to read a sealed target")
        with self.assertRaisesRegex(ValueError, "sealed/unknown"):
            visible_architecture(NoReads(), "FINAL", {})

    def test_v5_superset_does_not_imply_monotone_iou(self):
        # The original 138-cell addition may lower IoU against a peer that
        # overlaps only v4, while raising it against a peer in the added region.
        # Geometry/split identity, not a metric improvement, is the invariant.
        four = set(range(524))
        five = set(range(662))
        peers = [(split, group, seed, cells, "a" * 64, "b" * 64, "c" * 64)
                 for split, group, cells in (("TRAIN", "overlaps-original", {0, 1}),
                                             ("TRAIN", "second-train", {0, 1}),
                                             ("DEVELOPMENT", "overlaps-addition", {650, 651}),
                                             ("DEVELOPMENT", "second-dev", {650, 651}))
                 for seed in SEEDS]
        result = compare(four, five, peers)
        self.assertEqual(len(result), 8)
        self.assertLess(result[0]["v5_iou"], result[0]["v4_iou"])
        self.assertGreater(result[4]["v5_iou"], result[4]["v4_iou"])
        self.assertEqual(result[0]["split"], "TRAIN")
        self.assertEqual(result[4]["split"], "DEVELOPMENT")
        self.assertEqual([r["reference_seed"] for r in result[:2]], list(SEEDS))
        with self.assertRaisesRegex(ValueError, "138-cell"):
            compare(four, five - {0} | {700}, peers)

    def test_missing_second_seed_is_fail_closed(self):
        """Twelve original rows and six groups are insufficient if a host seed is lost."""
        groups = ["arch-A", "arch-B", "other-C", "other-D", "other-E", "other-F"]
        rows = [{"group_id": group, "family": "architecture", "seed": seed,
                 "id": f"{group}-{seed}"} for group in groups[:2] for seed in SEEDS]
        rows.extend({"group_id": groups[2 + (i % 4)], "family": "pipework",
                     "seed": SEEDS[i % 2], "id": f"other-{i}"} for i in range(8))
        rows[3]["seed"] = SEEDS[0]  # duplicate (arch-B, seed0); omit seed1
        payload = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in rows)
        digest = hashlib.sha256(payload).hexdigest()

        class VisibleOnly:
            def read(self, name):
                if name != "corpus/train.jsonl":
                    raise AssertionError("do not open native/FINAL when a split is incomplete")
                return payload

        manifest = {"source_sha": "633920373a3d406a2ede61ce4137c324f76be23b",
                    "split_files": {"TRAIN": {"sha256": digest, "count": 12}},
                    "structural_groups": {"TRAIN": groups}}
        with patch.dict(SPLITS, {"TRAIN": digest}):
            with self.assertRaisesRegex(ValueError, "both native seeds"):
                visible_architecture(VisibleOnly(), "TRAIN", manifest)

    def test_compare_rejects_missing_or_duplicated_reference_seed(self):
        four, five = set(range(524)), set(range(662))
        peers = [(split, group, seed, {5}, "a" * 64, "b" * 64, "c" * 64)
                 for split, group in (("TRAIN", "A"), ("TRAIN", "B"),
                                      ("DEVELOPMENT", "C"), ("DEVELOPMENT", "D"))
                 for seed in SEEDS]
        with self.assertRaisesRegex(ValueError, "eight original"):
            compare(four, five, peers[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            compare(four, five, peers[:-1] + [peers[0]])

    def test_reject_nonoriginal_zip(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "fake.zip"
            file.write_bytes(b"not a native research artifact")
            with self.assertRaisesRegex(ValueError, "whole original ZIP hash"):
                checked_zip(file, "0" * 64)

    def test_bool_is_not_integer_radius(self):
        with self.assertRaises(ValueError):
            proximity_fraction({0}, {0}, radius=True)


if __name__ == "__main__":
    unittest.main()
