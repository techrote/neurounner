"""Synthetic-only PNG chunk-order fixtures for numeric guidance ingestion.

PNG import is an authoring step, never material permission, visual-model input
or native evidence. All adversarial chunks have valid CRCs to isolate ordering.
"""
import struct
import unittest
import zlib

from experiments.driv3r.stage_d.guidance.field import decode_png8, rgba_png

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))


def _pieces(png: bytes) -> list[tuple[bytes, bytes]]:
    assert png.startswith(_SIGNATURE)
    pos, parts = len(_SIGNATURE), []
    while pos < len(png):
        size = struct.unpack_from(">I", png, pos)[0]
        parts.append((png[pos + 4:pos + 8], png[pos + 8:pos + 8 + size]))
        pos += size + 12
    assert pos == len(png)
    return parts


def _assemble(parts: list[tuple[bytes, bytes]]) -> bytes:
    return _SIGNATURE + b"".join(_chunk(tag, data) for tag, data in parts)


class GuidancePngStructureTests(unittest.TestCase):
    def setUp(self):
        self.original = rgba_png(
            2, 1, bytes((255, 255, 255, 255, 0, 0, 0, 128)))
        self.parts = _pieces(self.original)
        self.assertEqual([tag for tag, _ in self.parts],
                         [b"IHDR", b"IDAT", b"IEND"])

    def test_multiple_consecutive_idat_chunks_remain_valid(self):
        head, (_, compressed), end = self.parts
        split = len(compressed) // 2
        variant = _assemble([head, (b"IDAT", compressed[:split]),
                             (b"IDAT", compressed[split:]), end])
        self.assertEqual(decode_png8(variant), decode_png8(self.original))

    def test_ancillary_before_ihdr_rejected_even_with_valid_crc(self):
        png = _assemble([(b"tEXt", b"keyword\x00value"), *self.parts])
        with self.assertRaisesRegex(ValueError, "IHDR must be the first"):
            decode_png8(png)

    def test_ancillary_between_idat_chunks_rejected_even_with_valid_crc(self):
        head, (_, compressed), end = self.parts
        split = len(compressed) // 2
        png = _assemble([head, (b"IDAT", compressed[:split]),
                         (b"tEXt", b"keyword\x00value"),
                         (b"IDAT", compressed[split:]), end])
        with self.assertRaisesRegex(ValueError, "IDAT chunks must be consecutive"):
            decode_png8(png)

    def test_trailing_ancillary_after_final_idat_is_valid(self):
        head, data, end = self.parts
        png = _assemble([head, data, (b"tEXt", b"keyword\x00value"), end])
        self.assertEqual(decode_png8(png), decode_png8(self.original))


if __name__ == "__main__":
    unittest.main()
