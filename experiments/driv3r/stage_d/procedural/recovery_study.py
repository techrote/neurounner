"""D-2 offline, matched guidance-recovery study; synthetic material ONLY.

Consumes immutable generated D-2 development episodes. No native IO, inference,
physical controller claim, training, or executable Actions workflow.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import struct

VERSION = "d2.synthetic-route-study/1"
NEIGHBORS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ARMS = ("virtual_future", "routed_future", "tile_restart", "stop")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def cell_digest(cells):
    return digest(encoded(sorted([list(p) for p in cells])))


def components(cells):
    remaining = set(cells)
    sizes = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = [seed]
        size = 0
        while queue:
            x, y = queue.pop()
            size += 1
            for dx, dy in NEIGHBORS:
                q = (x + dx, y + dy)
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def shortest_route(start, end, allowed, max_hops):
    """Bounded brush-UP four-neighbour route; no diagonals or implicit teleport."""
    if start not in allowed or end not in allowed or max_hops < 0:
        return None
    frontier = deque([start])
    parents = {start: None}
    depths = {start: 0}
    while frontier:
        point = frontier.popleft()
        if point == end:
            path = [end]
            while parents[path[-1]] is not None:
                path.append(parents[path[-1]])
            return list(reversed(path))
        if depths[point] >= max_hops:
            continue
        for dx, dy in NEIGHBORS:
            nxt = point[0] + dx, point[1] + dy
            if nxt in allowed and nxt not in parents:
                parents[nxt] = point
                depths[nxt] = depths[point] + 1
                frontier.append(nxt)
    return None


def read_episode(folder):
    """Verify every original episode artifact before using its synthetic state."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    assert manifest["qualification"] == "OFFLINE_SYNTHETIC_ONLY"
    assert manifest["native_replay_verified"] is False
    assert manifest["model_decisions"] == 0
    for name, spec in manifest["artifacts"].items():
        raw = (folder / name).read_bytes()
        assert spec == {"bytes": len(raw), "sha256": digest(raw)}, name
    recipe = json.loads((folder / "recipe.json").read_bytes())
    assert recipe["version"] == "d2.offline/1"
    assert recipe["family"] == "guided_coverage"
    assert recipe["brush_radius"] == 0  # no silent changed-footprint comparison
    w, h = recipe["canvas"]
    assert 24 <= w <= 256 and 24 <= h <= 256
    geometry = json.loads((folder / "geometry.json").read_bytes())
    curve = [tuple(p) for p in geometry["ordered_curve"]]
    assert len(curve) == len(set(curve)) and curve
    assert all(0 <= x < w and 0 <= y < h for x, y in curve)
    assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
               for a, b in zip(curve, curve[1:]))
    field = (folder / "guidance.u16le").read_bytes()
    assert len(field) == w * h * 2
    values = struct.unpack("<" + "H" * (w*h), field)
    scalar = {(x, y): values[y*w+x] for y in range(h) for x in range(w)}
    return recipe, curve, scalar, {"manifest_sha256": digest((folder/"manifest.json").read_bytes()),
                                   "original_events_sha256": digest((folder/"events.json").read_bytes()),
                                   "guidance_sha256": digest(field)}


def run_arm(recipe, curve, scalar, strategy, allowed=None, max_route_hops=192):
    """Replay matched curve/guidance with distinct synthetic recovery policies."""
    assert strategy in ARMS
    w, h = recipe["canvas"]
    permitted = set((x, y) for y in range(h) for x in range(w)) if allowed is None else set(allowed)
    assert all(0 <= x < w and 0 <= y < h for x, y in permitted)
    assert max_route_hops <= w*h and max_route_hops >= 0
    low, high = recipe["exit_below"], recipe["reenter_at"]
    assert 0 <= low < high <= 65535
    budget = recipe["steps"]
    assert 1 <= budget <= 2048
    material, events = set(), []
    i, cursor = 0, None
    productive = 0
    recovery_spans = []
    recovered_hops = 0
    last_reentry_step = None
    terminal = "CURVE_COMPLETE"
    while i < len(curve) and len(material) < budget:
        point = curve[i]
        if point not in permitted:
            terminal = "PROTECTED_OR_BLOCKED_PAINT_CELL"
            events.append({"kind":"REFUSAL","index":i,"point":list(point),"reason":terminal})
            break
        if scalar[point] < low:
            candidates = [j for j in range(i+1, min(i+33, len(curve)-3))
                          if scalar[curve[j]] >= high and
                          all(q not in material and q in permitted and scalar[q] >= low
                              for q in curve[j:j+4])]
            if strategy == "tile_restart":
                # A new local 4x4-curve-tile boundary, never an arbitrary jump.
                candidates = [j for j in candidates if j % 16 == 0]
                candidates.sort(key=lambda j: (abs(curve[j][0]-point[0]) +
                                                abs(curve[j][1]-point[1]),j))
            if last_reentry_step is not None:
                recovery_spans.append(len(material)-last_reentry_step)
                last_reentry_step = None
            if strategy == "stop":
                terminal = "STOP_ON_LOW_GUIDANCE"
                events.append({"kind":"STOP","index":i,"reason":terminal})
                break
            chosen, route = None, None
            for j in candidates:
                if strategy == "virtual_future":
                    chosen = j
                    break
                # Start of a session without a painted point uses the initial
                # curve position, which is itself subject to the same legality.
                start = cursor if cursor is not None else curve[i]
                proposal = shortest_route(start, curve[j], permitted, max_route_hops)
                if proposal is not None:
                    chosen, route = j, proposal
                    break
            if chosen is None:
                terminal = "NO_VERIFIED_REENTRY" if strategy != "virtual_future" else "NO_FUTURE_REENTRY"
                events.append({"kind":"RECOVERY_FAILED","index":i,"reason":terminal,
                               "candidate_count":len(candidates)})
                break
            before = cell_digest(material)
            if route is not None:
                assert route[0] == (cursor if cursor is not None else curve[i])
                assert route[-1] == curve[chosen]
                assert all(a in permitted for a in route)
                assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1
                           for a,b in zip(route,route[1:]))
                recovered_hops += len(route)-1
            events.append({"kind":"BRUSH_UP_REENTRY", "strategy":strategy,
                           "from_index":i,"to_index":chosen,
                           "from_cursor":list(cursor) if cursor else None,
                           "route": [list(p) for p in route] if route is not None else None,
                           "route_verified_synthetic":route is not None,
                           "native_route_verified":False,
                           "material_before_sha256":before,"material_after_sha256":cell_digest(material)})
            cursor = curve[chosen]
            i = chosen
            last_reentry_step = len(material)
            continue
        if point in material:
            terminal = "REVISIT_REFUSED"
            events.append({"kind":"REFUSAL","index":i,"reason":terminal})
            break
        material.add(point)
        cursor = point
        productive += 1
        events.append({"kind":"PAINT_SYNTHETIC","index":i,"point":list(point),
                       "material_after_sha256":cell_digest(material)})
        i += 1
    if last_reentry_step is not None:
        recovery_spans.append(len(material)-last_reentry_step)
    if len(material) == budget:
        terminal = "BUDGET_EXHAUSTED"
    sizes = components(material)
    seeks = [e for e in events if e["kind"] == "BRUSH_UP_REENTRY"]
    return {"strategy":strategy,"qualification":"DEVELOPMENT_SYNTHETIC_ONLY",
            "native_replay_verified":False,"native_route_verified":False,
            "terminal":terminal,"strokes":productive,"material_cells":len(material),
            "material_sha256":cell_digest(material),"components":len(sizes),
            "largest_component":sizes[0] if sizes else 0,
            "largest_component_fraction":round(sizes[0]/len(material),6) if sizes else 0,
            "reentries":len(seeks),"routed_brush_up_hops":recovered_hops,
            "productive_steps_after_reentry":recovery_spans,
            "event_sha256":digest(encoded(events)),"events":events}


def study(corpus):
    results = []
    for folder in sorted(Path(corpus).iterdir()):
        if not folder.is_dir() or not (folder / "manifest.json").exists():
            continue
        if json.loads((folder/"recipe.json").read_bytes())["family"] != "guided_coverage":
            continue
        recipe, curve, scalar, custody = read_episode(folder)
        arms = {arm:run_arm(recipe,curve,scalar,arm) for arm in ARMS}
        assert len({x["material_sha256"] for x in arms.values()}) >= 1
        results.append({"case":folder.name,"recipe_sha256":digest(encoded(recipe)),
                        "input_custody":custody,"arms":arms})
    assert results
    return {"version":VERSION,"qualification":"DEVELOPMENT_SYNTHETIC_ONLY",
            "native":None,"model":None,"native_replay_verified":False,
            "route_semantics":"synthetic four-neighbour in-bounds allowed-mask only; NOT host travel",
            "matched_cases":results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",required=True)
    ap.add_argument("--out",required=True)
    args = ap.parse_args()
    result = study(args.corpus)
    output = Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    raw = encoded(result)
    output.write_bytes(raw)
    print(encoded({"output":str(output),"sha256":digest(raw),
                   "cases":len(result["matched_cases"]),"qualification":result["qualification"]}).decode())


if __name__ == "__main__":
    main()
