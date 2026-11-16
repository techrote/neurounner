"""Strict lane-local types for host geometry in synthetic D-1B observations.

A name allowlist alone cannot prevent a guidance/critic payload from being
smuggled into geometry under a legitimate key. This guards value *shape*, not
truth or host authority; only a future native host can authenticate geometry.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
GEOMETRY_KEYS = frozenset({
    "clearance_cells", "selected_wall_id", "target_xy", "cursor_xy",
    "signed_distance_cells", "blocked_mask_sha256", "visited_mask_sha256",
    "obstacle_count", "target_progress_cells",
})
MAX_COUNT = 2**31 - 1
MAX_CLEARANCE_SAMPLES = 32


def _int(value: object, low: int, high: int = MAX_COUNT) -> bool:
    return type(value) is int and low <= value <= high


def validate_host_geometry(geometry: Mapping) -> dict:
    """Check a bounded, declared geometry vocabulary and return an isolated copy.

    clearance_cells accepts either a scalar or a short numeric sample list
    for compatibility with preexisting two-sample pilot fixtures. Neither may
    include arbitrary strings, objects, NaN, bools or nested metadata. Hash
    fields are strictly SHA-256, not a freeform string carrying extra channels.
    """
    if not isinstance(geometry, Mapping) or not all(
        type(k) is str and k in GEOMETRY_KEYS for k in geometry
    ):
        raise ValueError("unregistered host geometry key could leak guidance/critic")
    out = {}
    for key, value in geometry.items():
        if key == "clearance_cells":
            if _int(value, 0):
                out[key] = value
            elif (type(value) is list and 1 <= len(value) <= MAX_CLEARANCE_SAMPLES and
                  all(_int(v, 0) for v in value)):
                out[key] = list(value)
            else:
                raise ValueError("invalid clearance_cells geometry shape")
        elif key in ("target_xy", "cursor_xy"):
            if type(value) is not list or len(value) != 2 or not all(_int(v, 0) for v in value):
                raise ValueError(f"invalid {key} geometry coordinates")
            out[key] = list(value)
        elif key == "selected_wall_id":
            if type(value) is not str or IDENTIFIER.fullmatch(value) is None:
                raise ValueError("invalid selected_wall_id geometry identifier")
            out[key] = value
        elif key in ("blocked_mask_sha256", "visited_mask_sha256"):
            if type(value) is not str or SHA256.fullmatch(value) is None:
                raise ValueError(f"invalid {key} geometry digest")
            out[key] = value
        elif key == "signed_distance_cells":
            if not _int(value, -MAX_COUNT, MAX_COUNT):
                raise ValueError("invalid signed_distance_cells geometry value")
            out[key] = value
        elif key == "target_progress_cells":
            if not _int(value, -MAX_COUNT, MAX_COUNT):
                raise ValueError("invalid target_progress_cells geometry value")
            out[key] = value
        elif key == "obstacle_count":
            if not _int(value, 0, MAX_COUNT):
                raise ValueError("invalid obstacle_count geometry value")
            out[key] = value
    return out
