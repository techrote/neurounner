"""Bounded D-1B synthetic trajectory audit for soft-guidance *research*.

Only considers authored u16 guidance and source-hash-bound hypothetical cursor
movements. No candidate is admitted, no material is written, no receipt or native
World is verified, and supplied progress is NOT an independently measured fact.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re
import struct

PROTOCOL = "cybersand.driv3r.d1b-trajectory-audit/0"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
STEP_KEYS = frozenset({"decision_index", "action", "path_xy",
                       "progress_before", "progress_after"})
MAX_FIELD_CELLS = 4_194_304


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _xy(value: object) -> bool:
    return (type(value) is list and len(value) == 2 and
            all(_integer(v, 0, 2**31 - 1) for v in value))


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def audit_synthetic_trajectory(*, guidance_u16le: bytes, width: int, height: int,
                               origin_xy: list[int], guidance_sha256: str,
                               scenario_sha256: str, source_revision: str,
                               initial_xy: list[int], steps: list[dict],
                               exit_below: int, reenter_at_or_above: int,
                               initial_engaged: bool = True,
                               max_decisions: int = 36,
                               max_segment_cells: int = 8) -> dict:
    """Audit short four-connected cursor paths against an immutable numeric field.

    progress fields are authored *synthetic* objective units. Productive guidance
    contact additionally requires positive progress AND a first-time cell visit;
    sitting on a high-H pixel or cycling over it cannot inflate this metric.
    Nothing here evaluates host permission, safety or actual causal benefit.
    """
    _require(_integer(width, 1, MAX_FIELD_CELLS) and
             _integer(height, 1, MAX_FIELD_CELLS) and
             width * height <= MAX_FIELD_CELLS, "invalid guidance dimensions")
    _require(_xy(origin_xy) and origin_xy[0] + width <= 2**31 and
             origin_xy[1] + height <= 2**31, "invalid guidance origin")
    _require(type(guidance_u16le) is bytes and
             len(guidance_u16le) == width * height * 2,
             "invalid u16le field bytes")
    _require(type(guidance_sha256) is str and SHA256.fullmatch(guidance_sha256) is not None and
             sha256(guidance_u16le).hexdigest() == guidance_sha256,
             "guidance bytes/hash mismatch")
    _require(type(scenario_sha256) is str and SHA256.fullmatch(scenario_sha256) is not None and
             type(source_revision) is str and COMMIT.fullmatch(source_revision) is not None,
             "scenario and source revision must be pinned")
    _require(_integer(exit_below, 0, 65535) and
             _integer(reenter_at_or_above, 0, 65535) and
             exit_below < reenter_at_or_above, "invalid hysteresis thresholds")
    _require(type(initial_engaged) is bool, "initial engagement must be boolean")
    _require(_integer(max_decisions, 1, 144) and _integer(max_segment_cells, 1, 8),
             "invalid bounded decision/macro policy")
    _require(_xy(initial_xy), "invalid initial cursor")
    _require(type(steps) is list and 1 <= len(steps) <= max_decisions,
             "invalid bounded trajectory length")

    def field_value(xy: list[int]) -> int:
        x, y = xy
        _require(origin_xy[0] <= x < origin_xy[0] + width and
                 origin_xy[1] <= y < origin_xy[1] + height,
                 "cursor/sample is outside numeric field")
        index = (y - origin_xy[1]) * width + (x - origin_xy[0])
        return struct.unpack_from("<H", guidance_u16le, index * 2)[0]

    field_value(initial_xy)
    cursor = list(initial_xy)
    engaged = initial_engaged
    seen: set[tuple[int, int]] = {tuple(cursor)}
    entries = []
    transitions = {"propose_recovery": 0, "reentered": 0}
    unique_guidance_sum = 0
    productive_guidance_sum = 0
    productive_new_cells = 0
    positive_field_count = sum(v > 0 for (v,) in struct.iter_unpack("<H", guidance_u16le))
    distinct_positive_cells = int(field_value(cursor) > 0)
    path_samples = revisits = zero_progress_moves = 0
    declared_positive_progress = declared_negative_progress = 0
    first_progress = last_progress = None
    stopped = False
    for expected_index, step in enumerate(steps, start=1):
        _require(not stopped, "STOP is terminal")
        _require(type(step) is dict and set(step) == STEP_KEYS,
                 "invalid synthetic decision keys")
        _require(step["decision_index"] == expected_index and
                 type(step["decision_index"]) is int,
                 "decision indices must be contiguous and one-based")
        action = step["action"]
        _require(action in ("MOVE", "STOP") and type(action) is str,
                 "invalid synthetic action")
        path = step["path_xy"]
        _require(type(path) is list, "path_xy must be a list")
        _require((action == "MOVE" and 1 <= len(path) <= max_segment_cells) or
                 (action == "STOP" and not path), "action/path length mismatch")
        before, after = step["progress_before"], step["progress_after"]
        _require(_integer(before, -1_000_000, 1_000_000) and
                 _integer(after, -1_000_000, 1_000_000),
                 "invalid supplied objective progress")
        _require(last_progress is None or before == last_progress,
                 "synthetic progress continuity broken")
        if first_progress is None:
            first_progress = before
        delta = after - before
        _require(action != "STOP" or delta == 0,
                 "STOP may not claim movement progress")
        last_progress = after
        declared_positive_progress += max(0, delta)
        declared_negative_progress += max(0, -delta)
        if action == "MOVE" and delta <= 0:
            zero_progress_moves += 1
        new_cells = 0
        new_positive = 0
        for point in path:
            _require(_xy(point), "invalid cursor path coordinate")
            _require(abs(point[0] - cursor[0]) + abs(point[1] - cursor[1]) == 1,
                     "non-four-connected or teleporting cursor path")
            scalar = field_value(point)
            cursor = list(point)
            path_samples += 1
            key = tuple(point)
            if key in seen:
                revisits += 1
            else:
                seen.add(key)
                new_cells += 1
                unique_guidance_sum += scalar
                if scalar > 0:
                    distinct_positive_cells += 1
                if delta > 0:
                    new_positive += 1
                    productive_guidance_sum += scalar
                    productive_new_cells += 1
            if engaged and scalar < exit_below:
                engaged = False
                transitions["propose_recovery"] += 1
            elif not engaged and scalar >= reenter_at_or_above:
                engaged = True
                transitions["reentered"] += 1
        entries.append({"decision_index": expected_index, "action": action,
                        "end_xy": list(cursor), "new_cells": new_cells,
                        "new_cells_on_positive_progress": new_positive,
                        "declared_progress_delta": delta,
                        "engaged_after": engaged})
        stopped = action == "STOP"
    return {
        "protocol": PROTOCOL, "evidence_class": "synthetic_only_not_native",
        "native_material_authority": False, "native_execution_verified": False,
        "native_undo_executions": 0, "model_inference_performed": False,
        "source_revision": source_revision, "scenario_sha256": scenario_sha256,
        "guidance_sha256": guidance_sha256,
        "guidance_frame": [*origin_xy, width, height],
        "input_trajectory_sha256": _hash(steps),
        "hysteresis": {"exit_below": exit_below,
                       "reenter_at_or_above": reenter_at_or_above,
                       "initial_engaged": initial_engaged},
        "decision_count": len(entries), "terminal_stop": stopped,
        "path_samples": path_samples, "revisited_samples": revisits,
        "distinct_visited_cells_including_start": len(seen),
        "positive_guidance_cells_in_field": positive_field_count,
        "distinct_positive_guidance_cells_visited": distinct_positive_cells,
        "new_cell_guidance_sum_u16_excluding_start": unique_guidance_sum,
        "productive_new_cells": productive_new_cells,
        "productive_guidance_sum_u16": productive_guidance_sum,
        "synthetic_declared_net_progress": last_progress - first_progress,
        "synthetic_declared_positive_progress": declared_positive_progress,
        "synthetic_declared_negative_progress": declared_negative_progress,
        "no_positive_progress_moves": zero_progress_moves,
        "hysteresis_events": transitions, "engaged_at_end": engaged,
        "decisions": entries,
        "limitation": "authored synthetic progress and u16 cursor samples only; no native safety, receipt, model or causal success"
    }
