"""D-1B standalone soft-guidance *research* field. Never writes native materials.

No shared D-COMMON imports until its exact contract is owner-accepted.
All positions are world-aligned, top-left origin, +x right, +y down.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import struct
import zlib

MAX_CELLS = 4_194_304
PNG_SIG = b"\x89PNG\r\n\x1a\n"
ENCODING = "u16le_higher_is_better_zero_is_neutral"
CONVERSION = "png8_rec709_luma_times_alpha_round_u16_v1"


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _within(value: int, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


@dataclass(frozen=True)
class GuidanceField:
    """Canonical row-major 16-bit soft *preference*, not a native permission mask."""
    width: int
    height: int
    values: tuple[int, ...]
    x: int = 0
    y: int = 0

    def __post_init__(self) -> None:
        _within(self.width, 1, MAX_CELLS, "width")
        _within(self.height, 1, MAX_CELLS, "height")
        if self.width * self.height > MAX_CELLS:
            raise ValueError("field exceeds bounded cell budget")
        _within(self.x, 0, 2**31 - 1, "x")
        _within(self.y, 0, 2**31 - 1, "y")
        if type(self.values) is not tuple:
            raise ValueError("guidance scalar storage must be an immutable tuple")
        if self.x + self.width > 2**31 or self.y + self.height > 2**31:
            raise ValueError("field frame exceeds bounded world coordinates")
        if len(self.values) != self.width * self.height:
            raise ValueError("field scalar count must equal width * height")
        for v in self.values:
            _within(v, 0, 65535, "scalar")

    def at(self, x: int, y: int) -> int:
        if not (self.x <= x < self.x + self.width and self.y <= y < self.y + self.height):
            raise IndexError("world point outside field")
        return self.values[(y - self.y) * self.width + (x - self.x)]

    def raw(self) -> bytes:
        return struct.pack(f"<{len(self.values)}H", *self.values)

    @property
    def sha256(self) -> str:
        return _digest(self.raw())

    @classmethod
    def from_raw(cls, width: int, height: int, raw: bytes, x: int = 0, y: int = 0) -> GuidanceField:
        _within(width, 1, MAX_CELLS, "width")
        _within(height, 1, MAX_CELLS, "height")
        if width * height > MAX_CELLS or len(raw) != 2 * width * height:
            raise ValueError("u16le byte count must equal 2 * width * height")
        return cls(width, height, struct.unpack(f"<{width * height}H", raw), x, y)


def radial_airbrush(width: int, height: int, *, center_x: float, center_y: float,
                    radius_cells: float, hardness: float = 0.35,
                    gradient_length: float = 0.8, x: int = 0, y: int = 0) -> GuidanceField:
    """Deterministic numeric radial field; density/seed affect only the separate preview.

    hardness=1 yields a nearly hard disk. gradient_length scales falloff width.
    Values are sampled at cell centres in declared world coordinates.
    """
    if not all(math.isfinite(t) for t in (center_x, center_y, radius_cells, hardness, gradient_length)):
        raise ValueError("airbrush parameters must be finite")
    if radius_cells <= 0 or not 0 <= hardness <= 1 or not 0 <= gradient_length <= 1:
        raise ValueError("invalid radial airbrush parameters")
    _within(width, 1, MAX_CELLS, "width")
    _within(height, 1, MAX_CELLS, "height")
    if width * height > MAX_CELLS:
        raise ValueError("field exceeds bounded cell budget")
    transition = max(0.000001, (1 - hardness) * gradient_length)
    plateau = 1 - transition
    values = []
    for row in range(height):
        for col in range(width):
            r = math.hypot(x + col + 0.5 - center_x, y + row + 0.5 - center_y) / radius_cells
            strength = 1.0 if r <= plateau else (0.0 if r >= 1 else (1 - r) / transition)
            values.append(int(strength * 65535 + 0.5))
    return GuidanceField(width, height, tuple(values), x, y)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff)


def rgba_png(width: int, height: int, rgba: bytes) -> bytes:
    """Minimal deterministic non-interlaced RGBA8 PNG with CRCs; no Pillow required."""
    if width < 1 or height < 1 or width * height > MAX_CELLS or len(rgba) != width * height * 4:
        raise ValueError("invalid RGBA dimensions/data")
    scan = b"".join(b"\x00" + rgba[row * width * 4:(row + 1) * width * 4] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return PNG_SIG + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(scan, 9)) + _chunk(b"IEND", b"")


def decode_png8(png: bytes) -> tuple[int, int, bytes]:
    """Strict PNG8 G/GA/RGB/RGBA decoder with filters 0-4, no palette/interlace.

    Return RGBA bytes. Reject unsupported encodings rather than silently changing H.
    """
    if not png.startswith(PNG_SIG):
        raise ValueError("invalid PNG signature")
    pos, header, data, ended = len(PNG_SIG), None, bytearray(), False
    seen_idat, closed_idat = False, False
    while pos + 12 <= len(png):
        length = struct.unpack_from(">I", png, pos)[0]
        end = pos + 12 + length
        if end > len(png) or length > 128 * 1024 * 1024:
            raise ValueError("truncated/oversized PNG chunk")
        tag, part = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + length]
        expected = struct.unpack_from(">I", png, pos + 8 + length)[0]
        if expected != (zlib.crc32(tag + part) & 0xffffffff):
            raise ValueError("PNG CRC mismatch")
        pos = end
        # PNG requires IHDR first and a single contiguous IDAT run. Allowing
        # ancillary chunks to split IDAT can produce nonportable imports.
        if header is None and tag != b"IHDR":
            raise ValueError("PNG IHDR must be the first chunk")
        if tag == b"IHDR":
            if header is not None or length != 13:
                raise ValueError("duplicate/invalid PNG header")
            width, height, depth, mode, compression, filtering, interlace = struct.unpack(">IIBBBBB", part)
            if depth != 8 or mode not in (0, 2, 4, 6) or compression or filtering or interlace:
                raise ValueError("only non-interlaced PNG8 G/GA/RGB/RGBA supported")
            if width < 1 or height < 1 or width * height > MAX_CELLS:
                raise ValueError("PNG exceeds bounded dimensions")
            header = (width, height, mode)
        elif tag == b"IDAT":
            if closed_idat:
                raise ValueError("PNG IDAT chunks must be consecutive")
            seen_idat = True
            data.extend(part)
            if len(data) > 128 * 1024 * 1024:
                raise ValueError("PNG compressed payload too large")
        elif tag == b"IEND":
            if header is None or part or end != len(png):
                raise ValueError("invalid PNG end")
            ended = True
            break
        else:
            if seen_idat:
                closed_idat = True
            if tag == b"tRNS" or (tag and not (tag[0] & 32)):
                raise ValueError("unsupported PNG critical/transparency chunk")
    if not ended or header is None or not data:
        raise ValueError("incomplete PNG")
    width, height, mode = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[mode]
    stride = width * channels
    expected_bytes = height * (stride + 1)
    decompressor = zlib.decompressobj()
    decoded = decompressor.decompress(bytes(data), expected_bytes + 1)
    if len(decoded) != expected_bytes or not decompressor.eof or decompressor.unconsumed_tail or decompressor.unused_data:
        raise ValueError("PNG decompressed byte length mismatch")
    out = bytearray(width * height * 4)
    prior = bytearray(stride)
    for y in range(height):
        off = y * (stride + 1)
        filt, raw = decoded[off], bytearray(decoded[off + 1:off + stride + 1])
        if filt not in (0, 1, 2, 3, 4):
            raise ValueError("unsupported PNG filter")
        for i in range(stride):
            left = raw[i - channels] if i >= channels else 0
            above = prior[i]
            upper_left = prior[i - channels] if i >= channels else 0
            if filt == 1:
                raw[i] = (raw[i] + left) & 255
            elif filt == 2:
                raw[i] = (raw[i] + above) & 255
            elif filt == 3:
                raw[i] = (raw[i] + ((left + above) // 2)) & 255
            elif filt == 4:
                p = left + above - upper_left
                d = (abs(p - left), abs(p - above), abs(p - upper_left))
                predictor = (left, above, upper_left)[d.index(min(d))]
                raw[i] = (raw[i] + predictor) & 255
        for x in range(width):
            pixel = raw[x * channels:(x + 1) * channels]
            if mode == 0:
                r = g = b = pixel[0]
                a = 255
            elif mode == 2:
                r, g, b = pixel
                a = 255
            elif mode == 4:
                r = g = b = pixel[0]
                a = pixel[1]
            else:
                r, g, b, a = pixel
            index = (y * width + x) * 4
            out[index:index + 4] = bytes((r, g, b, a))
        prior = raw
    return width, height, bytes(out)


def import_png8(png: bytes, x: int = 0, y: int = 0) -> GuidanceField:
    """Pinned conversion: rec.709 RGB luminance * alpha, nearest unsigned u16.

    RGB colours are converted to luminance; transparent pixels mean neutral.
    PNG is only an imported source, not evidence of graphical model perception.
    """
    width, height, rgba = decode_png8(png)
    vals = []
    for i in range(0, len(rgba), 4):
        r, g, b, alpha = rgba[i:i + 4]
        score = (0.2126 * r + 0.7152 * g + 0.0722 * b) * alpha / 255
        vals.append(int(score * 65535 / 255 + 0.5))
    return GuidanceField(width, height, tuple(vals), x, y)


def speckle_png(field: GuidanceField, *, seed: int, centre_density: float = 0.8) -> bytes:
    """Separate *visualisation*: stochastic gaps never replace scalar values."""
    _within(seed, 0, 2**32 - 1, "seed")
    if not math.isfinite(centre_density) or not 0 <= centre_density <= 1:
        raise ValueError("centre_density must be in [0,1]")
    rng = random.Random(seed)
    pixels = bytearray()
    for scalar in field.values:
        on = rng.random() < centre_density * scalar / 65535
        pixels.extend((255, 255, 255, 255 if on else 0))
    return rgba_png(field.width, field.height, bytes(pixels))


def export_bundle(prefix: Path, field: GuidanceField, *, seed: int, centre_density: float,
                  source_png_sha256: str | None = None,
                  authoring_parameters: dict | None = None) -> dict:
    """Save raw numeric values + independent PNG preview + manifest with content hashes."""
    prefix = Path(prefix)
    numeric = field.raw()
    preview = speckle_png(field, seed=seed, centre_density=centre_density)
    field_path = prefix.with_suffix(".u16le")
    image_path = prefix.with_suffix(".png")
    manifest_path = prefix.with_suffix(".json")
    field_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "cybersand.driv3r.guidance-bundle/0",
        "source_kind": ("imported_png" if source_png_sha256 else
                        ("airbrush" if authoring_parameters is not None else "manual_scalar")),
        "encoding": ENCODING,
        "frame": [field.x, field.y, field.width, field.height],
        "values": {"file": field_path.name, "bytes": len(numeric), "sha256": _digest(numeric)},
        "display": {"file": image_path.name, "bytes": len(preview), "sha256": _digest(preview),
                    "kind": "seeded_speckle_not_numeric_truth"},
        "seed": seed,
        "centre_density": centre_density,
        "conversion": CONVERSION if source_png_sha256 else None,
        "source_png_sha256": source_png_sha256,
        "authoring_parameters": authoring_parameters,
        "native_material_authority": False,
        "contract_status": "pre_m1_d1b_local_not_d_common_conformant"
    }
    field_path.write_bytes(numeric)
    image_path.write_bytes(preview)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_bundle(prefix: Path, *, expected_manifest_sha256: str | None = None) -> tuple[GuidanceField, dict]:
    """Load numeric/preview artifacts; optionally require an external manifest pin.

    Raw and preview hashes alone cannot authenticate editable manifest metadata.
    The caller must retain the CLI-emitted manifest SHA-256 independently when
    authoring provenance must be source-fenced.
    """
    prefix = Path(prefix)
    manifest_bytes = prefix.with_suffix(".json").read_bytes()
    if expected_manifest_sha256 is not None:
        valid = (type(expected_manifest_sha256) is str and
                 len(expected_manifest_sha256) == 64 and
                 all(c in "0123456789abcdef" for c in expected_manifest_sha256))
        if not valid or _digest(manifest_bytes) != expected_manifest_sha256:
            raise ValueError("bundle manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("format") != "cybersand.driv3r.guidance-bundle/0" or manifest.get("native_material_authority") is not False:
        raise ValueError("unknown/unsafe guidance bundle")
    x, y, w, h = manifest["frame"]
    field_path, display_path = prefix.with_suffix(".u16le"), prefix.with_suffix(".png")
    for desc, path in ((manifest["values"], field_path), (manifest["display"], display_path)):
        if desc["file"] != path.name:
            raise ValueError("bundle file name mismatch")
        content = path.read_bytes()
        if len(content) != desc["bytes"] or _digest(content) != desc["sha256"]:
            raise ValueError("bundle size/hash mismatch")
    if manifest["encoding"] != ENCODING or manifest["display"]["kind"] != "seeded_speckle_not_numeric_truth":
        raise ValueError("unsupported bundle encoding or preview")
    field = GuidanceField.from_raw(w, h, field_path.read_bytes(), x, y)
    expected = speckle_png(field, seed=manifest["seed"], centre_density=manifest["centre_density"])
    if decode_png8(expected) != decode_png8(display_path.read_bytes()):
        raise ValueError("bundle preview pixels are inconsistent with numeric field and render seed")
    return field, manifest


@dataclass(frozen=True)
class Hysteresis:
    exit_below: int
    reenter_at_or_above: int

    def __post_init__(self) -> None:
        _within(self.exit_below, 0, 65535, "exit_below")
        _within(self.reenter_at_or_above, 0, 65535, "reenter_at_or_above")
        if self.exit_below >= self.reenter_at_or_above:
            raise ValueError("exit must be strictly lower than reentry")

    def assess(self, value: int, engaged: bool) -> tuple[bool, str]:
        """Return updated flag and a *proposal only*; no native write/undo implied."""
        _within(value, 0, 65535, "guidance value")
        if type(engaged) is not bool:
            raise ValueError("engaged must be a boolean")
        if engaged and value < self.exit_below:
            return False, "propose_recovery"
        if not engaged and value >= self.reenter_at_or_above:
            return True, "reentered"
        return engaged, "no_transition"
