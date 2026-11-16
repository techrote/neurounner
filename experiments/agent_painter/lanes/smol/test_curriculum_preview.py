"""Adversarial no-admission checks for the SMOL compiler-only review images."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_preview import LABEL, png_bytes


class CompilerPreviewTests(unittest.TestCase):
    def test_png_lossless_geometry_and_determinism(self):
        grid = [0, 58, 58, 0]
        data = png_bytes(grid, size=2)
        self.assertEqual(data, png_bytes(grid, size=2))
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        off = 8
        chunks = []
        while off < len(data):
            length = struct.unpack_from(">I", data, off)[0]
            tag = data[off+4:off+8]
            payload = data[off+8:off+8+length]
            checksum = struct.unpack_from(">I", data, off+8+length)[0]
            self.assertEqual(checksum, zlib.crc32(tag+payload) & 0xffffffff)
            chunks.append((tag, payload))
            off += length + 12
        self.assertEqual([x[0] for x in chunks], [b"IHDR", b"IDAT", b"IEND"])
        pixels = zlib.decompress(chunks[1][1])
        self.assertEqual(pixels, bytes([0, 28,30,34,224,224,224,
                                        0, 224,224,224,28,30,34]))
        self.assertEqual(struct.unpack_from(">II", chunks[0][1]), (2, 2))

    def test_reject_nonbinary_grid(self):
        for grid in ([0, 58, 1, 0], [0, 58, -1, 0], [0, 58, True, 0],
                     [0, 58, 0], [0, 58, "58", 0]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                png_bytes(grid, size=2)

    def test_label_refuses_native_or_training_claim(self):
        self.assertIn("not native painting", LABEL)
        self.assertIn("not a trained model", LABEL)
        self.assertIn("not admitted TRAIN", LABEL)


if __name__ == "__main__":
    unittest.main()
