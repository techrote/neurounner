"""D-1B read-only, synthetic line-worm guidance-response diagnostics.

Unlike an A/B trial, these are authored cursor traces, not native/model runs.
A neutral/free arm must never receive the nonzero evaluation preference field.
Numerical guidance is an audit-only *reference* for comparing two traces;
no legal actions, materials, permissions or native receipts are inferred.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re
import struct

PROTOCOL = "cybersand.driv3r.d1b-worm-trace-audit/2"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REV = re.compile(r"[0-9a-f]{40}\Z")
_DIRECTIONS = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


def _assert(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _int(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _xy(value: object) -> bool:
    return type(value) is list and len(value) == 2 and all(_int(x, 0, 2**31 - 1) for x in value)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def audit_synthetic_worm_pair(*, guidance_u16le: bytes, width: int, height: int,
                              origin_xy: list[int], guidance_sha256: str,
                              initial_xy: list[int], initial_world_sha256: str,
                              scenario_sha256: str, source_revision: str, seed: int,
                              free_trace: list[dict], soft_trace: list[dict],
                              max_decisions: int = 36, high_threshold: int = 32768) -> dict:
    """Compare authored traces against one *withheld* scalar evaluation field.

    Free and soft episodes share initial geometry/seed but are allowed different
    resulting paths. We explicitly distinguish each arm's model-visible scalar:
    the free arm's is all-neutral, while the soft arm's is the input field.
    This has NO binding to real model inputs or native receipts. No visual/art
    quality, causal steerability or native legality can be concluded.
    """
    _assert(_int(width, 1, 4_194_304) and _int(height, 1, 4_194_304) and
            width * height <= 4_194_304, "invalid frame dimensions")
    _assert(_xy(origin_xy) and origin_xy[0] + width <= 2**31 and
            origin_xy[1] + height <= 2**31, "invalid frame origin")
    _assert(type(guidance_u16le) is bytes and len(guidance_u16le) == width * height * 2,
            "invalid 16-bit field byte count")
    _assert(type(guidance_sha256) is str and _HASH.fullmatch(guidance_sha256) is not None and
            sha256(guidance_u16le).hexdigest() == guidance_sha256,
            "reference guidance hash mismatch")
    _assert(type(initial_world_sha256) is str and _HASH.fullmatch(initial_world_sha256) is not None and
            type(scenario_sha256) is str and _HASH.fullmatch(scenario_sha256) is not None and
            type(source_revision) is str and _REV.fullmatch(source_revision) is not None,
            "source/World/scenario must be source-fenced")
    _assert(_xy(initial_xy) and _int(seed, 0, 2**32 - 1), "invalid start or seed")
    _assert(_int(max_decisions, 1, 36) and _int(high_threshold, 1, 65535),
            "invalid decision cap or threshold")

    def sample(point: list[int]) -> int:
        x, y = point
        _assert(origin_xy[0] <= x < origin_xy[0] + width and
                origin_xy[1] <= y < origin_xy[1] + height, "cursor outside evaluation field")
        return struct.unpack_from("<H", guidance_u16le,
                                  2 * ((y - origin_xy[1]) * width + x - origin_xy[0]))[0]

    sample(initial_xy)

    def evaluate(trace: list[dict]) -> dict:
        _assert(type(trace) is list and 1 <= len(trace) <= max_decisions,
                "invalid bounded trace length")
        cursor = initial_xy[:]
        first_heading = prev_heading = None
        seen = {tuple(cursor)}
        # Pen-up motion is a cursor visit, not a pen-down intent contact.
        ink_seen = set()
        ink_total = ink_unique_total = ink_high_moves = ink_high_unique = 0
        moving_samples = painted_intent = lifted_moves = 0
        low_count = high_count = repeated = turns = reversals = left = right = 0
        h_total = h_unique = 0
        trace_term = False
        for number, step in enumerate(trace, 1):
            _assert(type(step) is dict and set(step) ==
                    {"decision_index", "action", "to_xy", "pen_down"},
                    "malformed decision record")
            _assert(type(step["decision_index"]) is int and step["decision_index"] == number,
                    "noncontiguous decision number")
            action, target, pen = step["action"], step["to_xy"], step["pen_down"]
            _assert(type(action) is str and action in ("MOVE", "STOP") and type(pen) is bool,
                    "invalid action or pen state")
            if action == "STOP":
                _assert(target == [] and pen is False and number == len(trace),
                        "STOP must be terminal with no movement or pen")
                trace_term = True
                continue
            _assert(_xy(target), "invalid target")
            dx, dy = target[0] - cursor[0], target[1] - cursor[1]
            _assert((dx, dy) in _DIRECTIONS, "teleport or non-four-connected motion")
            value = sample(target)
            moving_samples += 1
            painted_intent += int(pen)
            lifted_moves += int(not pen)
            h_total += value
            high_count += int(value >= high_threshold)
            low_count += int(value < high_threshold)
            key = tuple(target)
            if key in seen:
                repeated += 1
            else:
                seen.add(key)
                h_unique += value
            if pen:
                ink_total += value
                ink_high_moves += int(value >= high_threshold)
                # Count distinct intended ink destinations, not cursor visits.
                if key not in ink_seen:
                    ink_seen.add(key)
                    ink_unique_total += value
                    ink_high_unique += int(value >= high_threshold)
            heading = _DIRECTIONS[(dx, dy)]
            if first_heading is None:
                first_heading = heading
            if prev_heading is not None:
                change = (heading - prev_heading) % 4
                if change:
                    turns += 1
                    if change == 1:
                        right += 1
                    elif change == 3:
                        left += 1
                    else:
                        reversals += 1
            prev_heading = heading
            cursor = target[:]
        return {
            "trajectory_sha256": _digest(trace),
            "decisions": len(trace), "terminal_stop": trace_term,
            "ended_by_decision_budget": not trace_term and len(trace) == max_decisions,
            "truncated_before_budget_without_stop": not trace_term and len(trace) < max_decisions,
            "movement_cells": moving_samples, "pen_down_intent_moves": painted_intent,
            "brush_lifted_moves": lifted_moves, "visited_distinct_including_start": len(seen),
            "revisits": repeated, "turns": turns, "left_turns": left,
            "right_turns": right, "reversals": reversals,
            "high_reference_contact_moves": high_count,
            "low_reference_contact_moves": low_count,
            "mean_reference_guidance_all_moves_u16": h_total / moving_samples if moving_samples else None,
            "mean_reference_guidance_new_cells_u16":
                h_unique / (len(seen) - 1) if len(seen) > 1 else None,
            "pen_down_intent_reference": {
                "unique_target_cells": len(ink_seen),
                "revisited_target_moves": painted_intent - len(ink_seen),
                "high_contact_moves": ink_high_moves,
                "high_contact_unique_cells": ink_high_unique,
                "mean_all_pen_down_moves_u16":
                    ink_total / painted_intent if painted_intent else None,
                "mean_unique_pen_down_target_cells_u16":
                    ink_unique_total / len(ink_seen) if ink_seen else None,
            },
            "last_xy": cursor,
        }

    free = evaluate(free_trace)
    soft = evaluate(soft_trace)
    both_observed = (free["movement_cells"] > 0 and soft["movement_cells"] > 0)
    free_ink = free["pen_down_intent_reference"]["mean_all_pen_down_moves_u16"]
    soft_ink = soft["pen_down_intent_reference"]["mean_all_pen_down_moves_u16"]
    return {
        "protocol": PROTOCOL, "evidence_class": "synthetic_authored_cursor_traces_only",
        "native_material_authority": False, "native_receipts_verified": False,
        "native_undo_executions": 0, "model_inference_performed": False,
        "source_revision": source_revision, "scenario_sha256": scenario_sha256,
        "initial_world_sha256": initial_world_sha256, "seed": seed,
        "initial_xy": initial_xy[:], "reference_guidance_sha256": guidance_sha256,
        "frame": [*origin_xy, width, height], "max_decisions": max_decisions,
        "high_threshold_u16": high_threshold,
        "arm_visible_guidance_sha256": {
            "free": sha256(bytes(len(guidance_u16le))).hexdigest(),
            "soft": guidance_sha256,
        },
        "arms": {"free": free, "soft": soft},
        "matched_movement_counts": free["movement_cells"] == soft["movement_cells"],
        "matched_pen_down_intent_counts":
            free["pen_down_intent_moves"] == soft["pen_down_intent_moves"],
        "posthoc_soft_minus_free_mean_reference_pen_down_intent_u16":
            soft_ink - free_ink if free_ink is not None and soft_ink is not None else None,
        "posthoc_soft_minus_free_mean_reference_guidance_u16":
            (soft["mean_reference_guidance_all_moves_u16"] -
             free["mean_reference_guidance_all_moves_u16"]) if both_observed else None,
        "interpretation": "posthoc preference/contact and line-morphology comparison only; "
                          "pen-down intent is not paint or verified native receipt; "
                          "unequal move/ink budgets are descriptive not matched outcomes; "
                          "no inferred model exposure, native permission, painting or causal effect",
    }
