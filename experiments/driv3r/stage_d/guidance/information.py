"""Pre-M1 deterministic typed-channel and information-accounting pilot.

Only constructs synthetic observation records: no D-COMMON adapter, native
permission, brush/undo dispatch, GPU/model call, or image-conditioned claim.
All model-visible samples are bounded explicit tiles, not stippled PNG pixels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import re
import struct
from typing import Mapping

from .geometry_guard import validate_host_geometry

PROTOCOL = "cybersand.driv3r.d1b-channel-shadow/0"
MAX_TILE_CELLS = 4096
# Deliberately narrow host-geometry pilot vocabulary; additional fields need
# explicit experiment revision, not an unreviewed covert guidance channel.
GEOMETRY_KEYS = frozenset({"clearance_cells", "selected_wall_id", "target_xy",
                           "cursor_xy", "signed_distance_cells", "blocked_mask_sha256",
                           "visited_mask_sha256", "obstacle_count", "target_progress_cells"})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
# This is an experimental consumer registration, NOT the unaccepted contract.
REGISTERED = {
    "positive_scalar/v1": ("preference", "u16le"),
    "soft_attraction/v1": ("preference", "u16le"),
    "soft_avoidance/v1": ("preference", "u16le"),
    "signed_preference/v1": ("preference", "i16le"),
    "signed_distance/v1": ("geometry", "i16le"),
    "flow_vector_xy/v1": ("geometry", "vec2_i16le"),
}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _int(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _hashable(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=True, allow_nan=False,
                             sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


@dataclass(frozen=True)
class Channel:
    """A signed/unsigned explicit numeric tile and its independently hashed source."""
    semantic_id: str
    signal_class: str
    encoding: str
    x: int
    y: int
    width: int
    height: int
    samples: bytes
    producer_revision: str
    config_sha256: str
    seed: int

    def __post_init__(self) -> None:
        _require(self.semantic_id in REGISTERED, "unregistered channel semantics")
        _require((self.signal_class, self.encoding) == REGISTERED[self.semantic_id],
                 "channel semantic/class/encoding disagreement")
        _require(_int(self.x, 0, 2**31 - 1) and _int(self.y, 0, 2**31 - 1),
                 "invalid world tile origin")
        _require(_int(self.width, 1, MAX_TILE_CELLS) and
                 _int(self.height, 1, MAX_TILE_CELLS) and
                 self.width * self.height <= MAX_TILE_CELLS,
                 "model-facing tile exceeds bounded cells")
        _require(self.x + self.width <= 2**31 and self.y + self.height <= 2**31,
                 "world tile overflows coordinate range")
        _require(type(self.samples) is bytes and
                 len(self.samples) == self.width * self.height * (4 if self.encoding == "vec2_i16le" else 2),
                 "channel encoding/byte length mismatch")
        _require(type(self.producer_revision) is str and COMMIT.fullmatch(self.producer_revision) is not None,
                 "producer_revision must be an exact source commit")
        _require(type(self.config_sha256) is str and SHA256.fullmatch(self.config_sha256) is not None,
                 "config_sha256 must be pinned")
        _require(_int(self.seed, 0, 2**32 - 1), "seed must be an unsigned 32-bit integer")

    def descriptor(self, *, with_values: bool) -> dict:
        count = self.width * self.height * (2 if self.encoding == "vec2_i16le" else 1)
        fmt = "H" if self.encoding == "u16le" else "h"
        meta = {
            "semantic_id": self.semantic_id, "signal_class": self.signal_class,
            "encoding": self.encoding, "frame": [self.x, self.y, self.width, self.height],
            "sample_sha256": _digest(self.samples), "sample_bytes": len(self.samples),
            "producer_revision": self.producer_revision,
            "config_sha256": self.config_sha256, "seed": self.seed,
        }
        if with_values:
            vals = struct.unpack("<" + fmt * count, self.samples)
            meta["values"] = ([list(vals[i:i + 2]) for i in range(0, count, 2)]
                              if self.encoding == "vec2_i16le" else list(vals))
        return meta


def _catalog(candidates: list[dict]) -> tuple[list[dict], str]:
    """Never derive legal candidates from guidance; accept strict host declarations."""
    _require(type(candidates) is list and 1 <= len(candidates) <= 128,
             "host must supply 1..128 legal candidate records")
    ids = set()
    result = []
    for item in candidates:
        _require(type(item) is dict and set(item) ==
                 {"id", "destination_xy", "swept_xy", "native_limit_cells"},
                 "legal candidate contains unsupported fields (including presentation scores)")
        cid = item["id"]
        _require(type(cid) is str and 1 <= len(cid) <= 96 and cid not in ids,
                 "duplicate or invalid legal candidate id")
        ids.add(cid)
        dest = item["destination_xy"]
        swept = item["swept_xy"]
        _require(type(dest) is list and len(dest) == 2 and
                 all(_int(v, 0, 2**31 - 1) for v in dest), "invalid candidate destination")
        _require(type(swept) is list and len(swept) <= 4096 and
                 all(type(p) is list and len(p) == 2 and
                     all(_int(v, 0, 2**31 - 1) for v in p) for p in swept),
                 "invalid candidate swept footprint")
        _require(len({tuple(p) for p in swept}) == len(swept),
                 "repeated swept footprint cell")
        _require(_int(item["native_limit_cells"], 0, 4096) and
                 len(swept) <= item["native_limit_cells"],
                 "candidate swept footprint exceeds supplied native limit")
        result.append({"id": cid, "destination_xy": list(dest),
                       "swept_xy": sorted((list(p) for p in swept)),
                       "native_limit_cells": item["native_limit_cells"]})
    # Catalog identity is independent of display order and footprint order.
    digest = _hashable(sorted(result, key=lambda c: c["id"]))
    return result, digest


def make_shadow_arms(*, snapshot_id: str, world_sha256: str,
                     source_revision: str, legal_candidates: list[dict],
                     geometry: Mapping, channels: list[Channel]) -> dict[str, dict]:
    """Build A1/A2/A3 identical-state ablations and separate audit records.

    All three arms see the exact same host-supplied geometry/legal actions.
    A1: mandatory scalar only; A2: scalar plus registered extra preferences;
    A3: scalar plus registered geometric metrics. Ignored channels and hashes
    exist ONLY in the audit record, never in the model-facing observation.
    """
    _require(type(snapshot_id) is str and 1 <= len(snapshot_id) <= 96,
             "invalid snapshot id")
    _require(type(world_sha256) is str and SHA256.fullmatch(world_sha256) is not None,
             "world_sha256 must be pinned")
    _require(type(source_revision) is str and COMMIT.fullmatch(source_revision) is not None,
             "source_revision must be pinned")
    _require(isinstance(geometry, Mapping), "geometry must be host-supplied mapping")
    _require(type(channels) is list and 1 <= len(channels) <= 9 and
             all(type(c) is Channel for c in channels),
             "core channel and at most eight optional typed channels required")
    _require(channels[0].semantic_id == "positive_scalar/v1" and
             all(c.semantic_id != "positive_scalar/v1" for c in channels[1:]) and
             len({c.semantic_id for c in channels}) == len(channels),
             "unique mandatory positive scalar core required")
    first = channels[0]
    _require(all((c.x, c.y, c.width, c.height) ==
                 (first.x, first.y, first.width, first.height) for c in channels),
             "paired pilot requires identical tile coordinates")
    catalog, legal_sha = _catalog(legal_candidates)
    # Key allowlisting alone does not prevent hidden guidance/critic payloads
    # from being smuggled into a legitimate host geometry attribute.
    geometry_copy = json.loads(json.dumps(validate_host_geometry(geometry),
                                          allow_nan=False, sort_keys=True))
    result = {}
    # Each arm is independently frozen at construction. Shared mutable catalog
    # or geometry objects would allow one consumer to alter the other controls
    # while their recorded model-view SHA-256 values remain stale.
    for arm in ("A1", "A2", "A3"):
        used = [c for c in channels if c is first or
                (arm == "A2" and c.signal_class == "preference") or
                (arm == "A3" and c.signal_class == "geometry")]
        unused = [c for c in channels if c not in used]
        view = {
            "protocol": PROTOCOL, "evidence_class": "synthetic_pre_m1",
            "snapshot_id": snapshot_id, "world_sha256": world_sha256,
            "source_revision": source_revision,
            "legal_candidate_set_sha256": legal_sha,
            "host_geometry": deepcopy(geometry_copy),
            "legal_candidates": deepcopy(catalog),
            "guidance": [c.descriptor(with_values=True) for c in used],
        }
        audit = {
            "protocol": PROTOCOL, "arm": arm,
            "legal_candidate_set_sha256": legal_sha,
            "model_view_sha256": _hashable(view),
            "used_channels": [c.descriptor(with_values=False) for c in used],
            "ignored_channels": [c.descriptor(with_values=False) for c in unused],
            "unregistered_channels_used": [],
            "native_execution_verified": False,
            "native_material_authority": False,
        }
        result[arm] = {"model_view": view, "audit_only": audit}
    return result
