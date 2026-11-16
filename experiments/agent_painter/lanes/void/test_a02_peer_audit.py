#!/usr/bin/env python3
"""Fail-closed controls for the original-byte Architecture spatial witness."""
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from a02_peer_audit import checked_zip, metrics, native_mask


class A02PeerAuditTests(unittest.TestCase):
    def test_iou_exact_and_disjoint(self):
        self.assertEqual(metrics({1, 2, 3}, {1, 2, 3})["same_coordinate_iou"], 1)
        self.assertEqual(metrics({1, 2, 3}, {4, 5})["same_coordinate_iou"], 0)

    def test_bad_original_zip_refused_before_any_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "untrusted.zip"
            p.write_bytes(b"ZIP bytes not original")
            with self.assertRaisesRegex(ValueError, "wrong original ZIP bytes"):
                checked_zip(p, "public")

    def test_non_native_material_refused(self):
        import struct
        with io.BytesIO() as data:
            with zipfile.ZipFile(data, "w") as z:
                cells = [0] * 9216
                cells[13] = 75
                z.writestr("native.i32le", struct.pack("<9216i", *cells))
            data.seek(0)
            with zipfile.ZipFile(data) as z:
                with self.assertRaisesRegex(ValueError, "unexpected material ID"):
                    native_mask(z, "native.i32le")


if __name__ == "__main__":
    unittest.main(verbosity=2)
