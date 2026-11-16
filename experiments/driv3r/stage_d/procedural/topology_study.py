"""D-2 deterministic material-topology counterfactual, DEVELOPMENT SYNTHETIC ONLY.

Reserve some of the same original painting-cell budget, use host-legal *synthetic*
cardinal brush-up travel and add the cheapest boundary-to-boundary material
bridges. Never infer native cursor travel, native receipts, or material Undo.
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path
import struct
import zlib

from recovery_study import (NEIGHBORS, cell_digest, components, digest, encoded,
                            read_episode, run_arm, shortest_route)

VERSION = "d2.synthetic-topology-bridge/1"
INK = bytes((232, 222, 199, 255))
EMPTY = bytes((24, 33, 43, 255))


def material_from_episode(episode):
    """Reconstruct and check all synthetic material steps, not just the last hash."""
    cells = set()
    for event in episode["events"]:
        if event["kind"] == "PAINT_SYNTHETIC":
            point = tuple(event["point"])
            assert len(point) == 2 and point not in cells
            cells.add(point)
            assert cell_digest(cells) == event["material_after_sha256"]
        elif event["kind"] == "BRUSH_UP_REENTRY":
            assert event["material_before_sha256"] == cell_digest(cells)
            assert event["material_after_sha256"] == cell_digest(cells)
    assert cell_digest(cells) == episode["material_sha256"]
    return cells


def groups(cells):
    """Stable 4-neighbour material components for reproducible bridge selection."""
    unseen = set(cells)
    output = []
    while unseen:
        point = min(unseen)
        stack = [point]
        group = {point}
        unseen.remove(point)
        while stack:
            x, y = stack.pop()
            for dx, dy in NEIGHBORS:
                neighbor = (x + dx, y + dy)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    group.add(neighbor)
                    stack.append(neighbor)
        output.append(group)
    return sorted(output, key=lambda part: (-len(part), min(part)))


def cheapest_link(material, permitted, max_new_cells, paint_allowed=None):
    """Least new material cells, then hops, then stable coordinates.

    All traversed points must be permitted. An existing material cell costs zero;
    painting previously empty points costs one. Searches are bounded to the
    supplied permitted mask, not an unconstrained World.
    """
    if max_new_cells < 0:
        raise ValueError("negative bridge budget")
    parts = groups(material)
    if len(parts) <= 1:
        return None
    assert material <= permitted
    # The brush may travel brush-up through permitted cells; new paint may be
    # restricted by an independent aesthetic/guidance choice, never conflated
    # with a native permission mask. Already painted cells remain traversable.
    destinations = permitted if paint_allowed is None else (material | set(paint_allowed)) & permitted
    best = None
    for part in parts:
        remaining = material - part
        queue = []
        costs = {}
        parent = {}
        for point in sorted(part):
            costs[point] = (0, 0)
            parent[point] = None
            heapq.heappush(queue, (0, 0, point))
        while queue:
            new, steps, point = heapq.heappop(queue)
            if costs[point] != (new, steps):
                continue
            if point in remaining:
                path = [point]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                rank = (new, steps, tuple(path))
                if best is None or rank < best[0]:
                    best = (rank, path)
                break
            if new > max_new_cells or (best is not None and (new, steps) >= best[0][:2]):
                continue
            for dx, dy in NEIGHBORS:
                neighbor = (point[0] + dx, point[1] + dy)
                if neighbor not in destinations:
                    continue
                cost = (new + int(neighbor not in material), steps + 1)
                if cost[0] > max_new_cells:
                    continue
                if neighbor not in costs or cost < costs[neighbor]:
                    costs[neighbor] = cost
                    parent[neighbor] = point
                    heapq.heappush(queue, (cost[0], cost[1], neighbor))
    if best is None:
        return None
    return best[1]


def bridge_prefix(prefix, permitted, capacity, max_route_hops=192, paint_allowed=None):
    """Replay brush-up transport and material bridging with explicit cost bounds."""
    if capacity < 0 or max_route_hops < 0:
        raise ValueError("invalid capacity or hop bound")
    material = material_from_episode(prefix)
    assert material <= permitted
    paintable = permitted if paint_allowed is None else set(paint_allowed) & permitted
    paint_events = [e for e in prefix["events"] if e["kind"] == "PAINT_SYNTHETIC"]
    cursor = tuple(paint_events[-1]["point"]) if paint_events else None
    start_hash = cell_digest(material)
    events = []
    edits = 0
    travel = 0
    terminal = "ALREADY_CONNECTED" if len(groups(material)) <= 1 else "CONNECTED"
    while len(groups(material)) > 1:
        path = cheapest_link(material, permitted, capacity - edits, paint_allowed=paintable)
        if path is None:
            terminal = "NO_BUDGETED_PERMITTED_BRIDGE"
            break
        assert path[0] in material and path[-1] in material
        start = path[0]
        if cursor is None:
            terminal = "NO_PAINTED_CURSOR"
            break
        route = shortest_route(cursor, start, permitted, max_route_hops)
        if route is None:
            terminal = "NO_BRUSH_UP_ROUTE"
            events.append({"kind": "RECOVERY_REFUSED", "reason": terminal,
                           "material_sha256": cell_digest(material)})
            break
        before = cell_digest(material)
        events.append({"kind": "BRUSH_UP_ROUTE", "origin": list(cursor),
                       "destination": list(start), "route": [list(p) for p in route],
                       "material_before_sha256": before, "material_after_sha256": before,
                       "synthetic_cardinal_route_verified": True,
                       "native_route_verified": False, "native_receipt": None})
        travel += len(route) - 1
        for point in path:
            assert point in permitted
            previous = cell_digest(material)
            changed = int(point not in material)
            assert not changed or point in paintable
            material.add(point)
            edits += changed
            assert edits <= capacity
            events.append({"kind": "BRIDGE_STROKE_SYNTHETIC", "point": list(point),
                           "actual_changed_cells": changed,
                           "material_before_sha256": previous,
                           "material_after_sha256": cell_digest(material),
                           "native_receipt": None})
        cursor = path[-1]
    sizes = components(material)
    return {"qualification": "DEVELOPMENT_SYNTHETIC_ONLY", "native_replay_verified": False,
            "native_route_verified": False, "native_undo_supported": False,
            "terminal": terminal, "prefix_material_sha256": start_hash,
            "material_sha256": cell_digest(material), "material_cells": len(material),
            "new_material_cells": edits, "components": len(sizes),
            "largest_component_fraction": round(sizes[0] / len(material), 6) if sizes else 0,
            "synthetic_brush_up_hops": travel,
            "event_sha256": digest(encoded(events)), "events": events}, material


def png_material(cells, width, height):
    """Minimal deterministic PNG with the same palette as the D-2 original views."""
    assert 1 <= width <= 256 and 1 <= height <= 256
    assert all(0 <= x < width and 0 <= y < height for x, y in cells)
    rows = []
    for y in range(height):
        rows.append(b"\x00" + b"".join(INK if (x, y) in cells else EMPTY
                                         for x in range(width)))
    def chunk(tag, raw):
        return (struct.pack(">I", len(raw)) + tag + raw +
                struct.pack(">I", zlib.crc32(tag + raw) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def compare_case(folder, reserve=24, permitted=None, max_route_hops=192):
    """Same original inputs; compare matched changed-cell budget, not stroke count."""
    recipe, curve, scalar, custody = read_episode(folder)
    width, height = recipe["canvas"]
    original_budget = recipe["steps"]
    if not 0 < reserve < original_budget:
        raise ValueError("reserve must be inside the original painting budget")
    mask = (set((x, y) for y in range(height) for x in range(width))
            if permitted is None else set(permitted))
    assert all(0 <= x < width and 0 <= y < height for x, y in mask)
    prefix_recipe = dict(recipe, steps=original_budget - reserve)
    prefix = run_arm(prefix_recipe, curve, scalar, "routed_future", allowed=mask,
                     max_route_hops=max_route_hops)
    initial = material_from_episode(prefix)
    bridged, final = bridge_prefix(prefix, mask, reserve, max_route_hops)
    # The original scalar guidance is *not* a hard native permission. Evaluate
    # both deliberately overriding it to connect topology and preserving it as
    # a strict additional painting restriction; do not hide this tradeoff.
    strict_allowed = {p for p in mask if scalar[p] >= recipe["exit_below"]}
    strict, strict_cells = bridge_prefix(prefix, mask, reserve, max_route_hops,
                                         paint_allowed=strict_allowed)
    crossing = [e["point"] for e in bridged["events"]
                if e["kind"] == "BRIDGE_STROKE_SYNTHETIC" and
                e["actual_changed_cells"] and scalar[tuple(e["point"])] < recipe["exit_below"]]
    assert bridged["new_material_cells"] >= len(crossing)
    equal_recipe = dict(recipe, steps=len(final))
    matched = run_arm(equal_recipe, curve, scalar, "routed_future", allowed=mask,
                      max_route_hops=max_route_hops)
    matched_cells = material_from_episode(matched)
    original = run_arm(recipe, curve, scalar, "routed_future", allowed=mask,
                       max_route_hops=max_route_hops)
    original_cells = material_from_episode(original)
    # Failure cases may stop before the requested budget: never mislabel them as
    # genuinely matched if one arm exhausted its legal options earlier.
    is_matched = len(matched_cells) == len(final)
    assert initial <= final and len(final) <= original_budget
    if is_matched:
        assert initial <= matched_cells
    return ({"case": Path(folder).name, "recipe_sha256": digest(encoded(recipe)),
             "input_custody": custody, "original_steps_budget": original_budget,
             "reserved_bridge_budget": reserve, "permitted_mask_sha256": cell_digest(mask),
             "match_equal_actual_cell_count": is_matched,
             "soft_guidance_exit_below": recipe["exit_below"],
             "soft_guidance_crossings_count": len(crossing),
             "soft_guidance_crossings": crossing,
             "interpretation": "soft-guidance bridge override is NOT host permission; strict arm forbids below-exit new paint",
             "arms": {
                 "reference_curve": {key: original[key] for key in (
                     "material_cells", "material_sha256", "components", "routed_brush_up_hops", "terminal")},
                 "prefix_curve": {key: prefix[key] for key in (
                     "material_cells", "material_sha256", "components", "routed_brush_up_hops", "terminal")},
                 "equal_cells_curve": {key: matched[key] for key in (
                     "material_cells", "material_sha256", "components", "routed_brush_up_hops", "terminal")},
                 "bridged": bridged,
                 "bridge_guidance_strict": strict,
             }}, {"reference_curve": original_cells, "prefix_curve": initial,
                  "equal_cells_curve": matched_cells, "bridged": final,
                  "bridge_guidance_strict": strict_cells}, (width, height))


def study(corpus, output, reserve=24):
    """Write original synthetic comparison views and hashed deterministic report."""
    corpus, output = Path(corpus), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for folder in sorted(corpus.iterdir()):
        if not folder.is_dir() or not (folder / "recipe.json").exists():
            continue
        import json
        if json.loads((folder / "recipe.json").read_bytes())["family"] != "guided_coverage":
            continue
        record, cellsets, dimensions = compare_case(folder, reserve=reserve)
        directory = output / record["case"]
        directory.mkdir(exist_ok=True)
        views = {}
        for name, cells in cellsets.items():
            payload = png_material(cells, *dimensions)
            destination = directory / (name + ".png")
            destination.write_bytes(payload)
            views[name] = {"path": f"{record['case']}/{name}.png", "bytes": len(payload),
                           "sha256": digest(payload), "cells_sha256": cell_digest(cells)}
        record["original_views"] = {
            name: digest((folder / name).read_bytes()) for name in
            ("before.png", "midpoint.png", "after.png")}
        record["counterfactual_views"] = views
        items.append(record)
    assert items
    result = {"version": VERSION, "qualification": "DEVELOPMENT_SYNTHETIC_ONLY",
              "native_replay_verified": False, "native": None, "model": None,
              "no_native_undo_or_inference": True,
              "scope": "same original guidance/geometry, conditional offline synthetic continuation",
              "cases": items}
    raw = encoded(result) + b"\n"
    (output / "topology-study.json").write_bytes(raw)
    return result, digest(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserve", type=int, default=24)
    args = parser.parse_args()
    outcome, checksum = study(args.corpus, args.out, args.reserve)
    print(encoded({"cases": len(outcome["cases"]), "report_sha256": checksum,
                   "qualification": outcome["qualification"]}).decode())


if __name__ == "__main__":
    main()
