"""Read-only morphology of synthetic one-cell worm pen intentions.

Every result describes intended pen-down destination cells, *not* actual native
materials. Cursor visits, pen-up transfers and synthetic STOP never deposit ink.
The caller must first validate legal movement/ordering with worm_trace.py.
"""
from __future__ import annotations

from hashlib import sha256
import json

PROTOCOL = "cybersand.driv3r.d1b-worm-intent-morphology/1"
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _seal(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def analyze_validated_trace(trace: list[dict]) -> dict:
    """Summarize a pre-validated cardinal single-cell trace; no native claims.

    No implicit paint at the initial cursor or while the pen is lifted. Repeated
    intended contacts retain their order in the input seal, but only distinct
    cells contribute to the static ink-graph topology. Touching separate pen
    runs form one connected component regardless of stroke chronology.
    """
    if type(trace) is not list or not 1 <= len(trace) <= 36:
        raise ValueError("morphology expects a bounded prevalidated trace")
    seen: set[tuple[int, int]] = set()
    ordered_ink: list[list[int]] = []
    stroke_lengths: list[int] = []
    active = 0
    for index, item in enumerate(trace, 1):
        if type(item) is not dict or set(item) != {"decision_index", "action", "to_xy", "pen_down"}:
            raise ValueError("morphology expects canonical trace records")
        if type(item["decision_index"]) is not int or item["decision_index"] != index or \
           item["action"] not in ("MOVE", "STOP") or type(item["pen_down"]) is not bool:
            raise ValueError("invalid morphology trace control")
        if item["action"] == "STOP":
            if item["to_xy"] != [] or item["pen_down"] or index != len(trace):
                raise ValueError("STOP must be terminal with lifted brush")
        elif type(item["to_xy"]) is not list or len(item["to_xy"]) != 2 or \
             any(type(v) is not int or v < 0 or v >= 2**31 for v in item["to_xy"]):
            raise ValueError("invalid morphology destination")
        if item["action"] == "MOVE" and item["pen_down"]:
            target = tuple(item["to_xy"])
            seen.add(target)
            ordered_ink.append(list(target))
            active += 1
        elif active:
            stroke_lengths.append(active)
            active = 0
    if active:
        stroke_lengths.append(active)

    # Static 4-neighbour topology of the DISTINCT *intended* destination cells.
    # This is intentionally separate from the chronology of brush-lifted runs.
    edges = 0
    endpoints = junctions = isolates = 0
    for x, y in seen:
        degree = sum((x + dx, y + dy) in seen for dx, dy in _NEIGHBORS)
        edges += degree
        endpoints += degree == 1
        junctions += degree >= 3
        isolates += degree == 0
    edges //= 2
    unvisited = set(seen)
    components = 0
    while unvisited:
        components += 1
        frontier = [unvisited.pop()]
        while frontier:
            x, y = frontier.pop()
            for dx, dy in _NEIGHBORS:
                adjacent = (x + dx, y + dy)
                if adjacent in unvisited:
                    unvisited.remove(adjacent)
                    frontier.append(adjacent)
    xs = [p[0] for p in seen]
    ys = [p[1] for p in seen]
    return {
        "protocol": PROTOCOL,
        "evidence_class": "synthetic_intended_pen_destinations_only",
        "validated_trace_sha256": _seal(trace),
        "ordered_ink_intent_sha256": _seal(ordered_ink),
        "intended_contact_moves": len(ordered_ink),
        "distinct_intended_cells": len(seen),
        "pen_down_runs": len(stroke_lengths),
        "longest_pen_down_run": max(stroke_lengths, default=0),
        "ink_components_four_connected": components,
        "ink_adjacencies_four_connected": edges,
        "ink_endpoints": endpoints,
        "ink_junctions_degree_3plus": junctions,
        "ink_isolated_cells": isolates,
        "ink_cycle_rank": edges - len(seen) + components,
        "ink_bbox_inclusive_xyxy": [min(xs), min(ys), max(xs), max(ys)] if seen else None,
        "native_material_verified": False,
        "model_behavior_verified": False,
    }


def compare_validated_traces(free_trace: list[dict], soft_trace: list[dict]) -> dict:
    """Report first apparent control and pen-intent divergences; not causal."""
    free = analyze_validated_trace(free_trace)
    soft = analyze_validated_trace(soft_trace)
    first_control = first_cursor = first_ink = None
    for i in range(max(len(free_trace), len(soft_trace))):
        f = free_trace[i] if i < len(free_trace) else None
        s = soft_trace[i] if i < len(soft_trace) else None
        if first_control is None and f != s:
            first_control = i + 1
        # STOP versus a missing terminal record is a control/length difference,
        # not a cursor movement; only MOVE destinations count as cursor choices.
        fc = tuple(f["to_xy"]) if f is not None and f["action"] == "MOVE" else None
        sc = tuple(s["to_xy"]) if s is not None and s["action"] == "MOVE" else None
        if first_cursor is None and fc != sc:
            first_cursor = i + 1
        fi = tuple(f["to_xy"]) if f and f["action"] == "MOVE" and f["pen_down"] else None
        si = tuple(s["to_xy"]) if s and s["action"] == "MOVE" and s["pen_down"] else None
        if first_ink is None and fi != si:
            first_ink = i + 1
    return {
        "protocol": PROTOCOL,
        "evidence_class": "synthetic_prevalidated_pair_only",
        "arms": {"free": free, "soft": soft},
        "first_different_control_index": first_control,
        "first_different_cursor_choice_index": first_cursor,
        "first_different_pen_down_destination_index": first_ink,
        "same_intent_cell_count": free["distinct_intended_cells"] == soft["distinct_intended_cells"],
        "native_material_verified": False,
        "model_behavior_verified": False,
        "interpretation": "Intended one-cell contacts only; no verified native material, image aesthetics, or causal steering",
    }
