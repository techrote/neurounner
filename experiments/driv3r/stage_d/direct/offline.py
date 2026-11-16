"""Deterministic, offline D-1A spatial-control fixtures and baselines.

No native World, d1 inference, model observations or D-COMMON conformance is
claimed here. This isolated module cannot authorize material writes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
from math import hypot
from pathlib import Path
import random
from typing import Any

VERSION = "d1a-offline/1"
Point = tuple[int, int]
DIRECTIONS: tuple[tuple[str, Point], ...] = (
    ("E", (1, 0)), ("N", (0, -1)), ("S", (0, 1)), ("W", (-1, 0))
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def horizontal(y: int, x0: int, x1: int) -> set[Point]:
    return {(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)}


def vertical(x: int, y0: int, y1: int) -> set[Point]:
    return {(x, y) for y in range(min(y0, y1), max(y0, y1) + 1)}


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    split: str
    task: str
    family: str
    width: int
    height: int
    start: Point
    target: Point
    solid: frozenset[Point]
    material_z: frozenset[Point]
    protected: frozenset[Point]
    selected_wall: frozenset[Point]
    wall_profile: tuple[int, ...]
    offset: int
    budget: int = 36

    @property
    def forbidden(self) -> frozenset[Point]:
        return self.solid | self.material_z | self.protected

    def on_grid(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def legal(self, p: Point) -> bool:
        return self.on_grid(p) and p not in self.forbidden

    def source(self) -> dict[str, Any]:
        return {
            "version": VERSION, "fixture_id": self.fixture_id, "split": self.split,
            "task": self.task, "family": self.family, "width": self.width,
            "height": self.height, "start": self.start, "target": self.target,
            "solid": sorted(self.solid), "material_z": sorted(self.material_z),
            "protected": sorted(self.protected), "selected_wall": sorted(self.selected_wall),
            "wall_profile": self.wall_profile, "offset": self.offset, "budget": self.budget,
        }

    @property
    def sha256(self) -> str:
        return digest(self.source())


def wall_fixture(family: str, split: str) -> Fixture:
    # A single identified wall with recesses/protrusions, plus a distinct decoy.
    # Profile deltas are at most one horizontal cell between rows; fill links.
    profiles = {
        "step": [10]*7 + [11, 12, 13, 14] + [14]*6 + [13, 12, 11, 10] + [10]*9,
        "recess": [12]*5 + [11, 10, 9, 8] + [8]*7 + [9, 10, 11, 12] + [12]*10,
        "bulge": [9]*6 + [10, 11, 12, 13, 14] + [14]*6 +
                 [13, 12, 11, 10, 9] + [9]*8,
        "switchback": [11]*3 + [12, 13, 14, 15, 16] + [16]*4 +
                      [15, 14, 13, 12, 11, 10, 9] + [9]*4 +
                      [10, 11, 12, 13] + [13]*3,
        "double_notch": [13]*4 + [12, 11, 10, 9] + [9]*4 +
                        [10, 11, 12, 13, 14, 15] + [15]*3 +
                        [14, 13, 12, 11] + [11]*4,
    }
    profile = tuple(profiles[family])
    y0 = 2
    wall: set[Point] = set()
    for i, x in enumerate(profile):
        y = y0+i
        wall.add((x,y))
        if i:
            wall.update(horizontal(y, profile[i-1], x))
    # Decoys are explicitly NOT selected wall; near enough to mislead nearest-wall policies.
    decoy = vertical(34, 6, min(25, y0 + len(profile) - 3))
    solid = wall | decoy
    start = (profile[0] + 8, y0)
    target = (profile[-1] + 8, y0 + len(profile) - 1)
    fixture = Fixture(
        f"wall_{family}_v1", split, "wall_offset", family, 46, 36,
        start, target, frozenset(solid), frozenset(), frozenset(),
        frozenset(wall), profile, 8
    )
    check_fixture(fixture)
    return fixture


def routing_fixture(family: str, split: str) -> Fixture:
    # Material Z is forbidden by the host in later native execution; this
    # synthetic grid has exactly the same strict blocking invariant.
    obstacles: dict[str, set[Point]] = {
        "gate": vertical(11, 0, 9) | vertical(11, 12, 20),
        "slalom": vertical(9, 0, 8) | vertical(18, 10, 20),
        "hook": vertical(12, 2, 13) | horizontal(13, 12, 20),
        "culdesac": horizontal(4, 8, 18) | vertical(8, 4, 12) | vertical(18, 4, 12),
        "offset_gates": (vertical(9, 0, 7) | vertical(9, 10, 20) |
                         vertical(19, 0, 10) | vertical(19, 13, 20)),
        "nested_corners": (horizontal(6, 5, 16) | vertical(16, 6, 11) |
                           horizontal(14, 11, 22) | vertical(11, 11, 14)),
    }
    pairs = {
        "gate": ((2, 4), (25, 4)),
        "slalom": ((2, 10), (25, 10)),
        "hook": ((2, 6), (25, 6)),
        "culdesac": ((2, 9), (25, 9)),
        "offset_gates": ((2, 8), (25, 8)),
        "nested_corners": ((2, 10), (25, 10)),
    }
    start, target = pairs[family]
    # Additional protected islands test that legal checks apply to all blocker types.
    protected = frozenset({(27, 16), (27, 17)})
    fixture = Fixture(
        f"route_{family}_v1", split, "avoid_z", family, 29, 22,
        start, target, frozenset(), frozenset(obstacles[family]),
        protected, frozenset(), (), 0
    )
    check_fixture(fixture)
    return fixture


def check_fixture(f: Fixture) -> None:
    if f.split not in {"development", "heldout"} or f.budget != 36:
        raise ValueError("Invalid frozen split or decision budget")
    if not f.legal(f.start) or not f.legal(f.target):
        raise ValueError(f"Blocked start/target: {f.fixture_id}")
    if any(not f.on_grid(p) for p in f.forbidden):
        raise ValueError(f"Out-of-bounds material: {f.fixture_id}")
    if f.task == "wall_offset" and (len(f.wall_profile) < 20 or not f.selected_wall):
        raise ValueError("Missing independently identified wall")
    if not shortest_path(f, f.start, f.target):
        raise ValueError(f"Unreachable goal: {f.fixture_id}")


def fixtures() -> tuple[Fixture, ...]:
    # Structural family, not seed, determines split. Never revise FINAL after
    # observing model outcomes; v2 instead requires a separate manifest.
    return (
        wall_fixture("step", "development"),
        wall_fixture("recess", "development"),
        wall_fixture("bulge", "development"),
        routing_fixture("gate", "development"),
        routing_fixture("slalom", "development"),
        routing_fixture("hook", "development"),
        routing_fixture("culdesac", "development"),
        wall_fixture("switchback", "heldout"),
        wall_fixture("double_notch", "heldout"),
        routing_fixture("offset_gates", "heldout"),
        routing_fixture("nested_corners", "heldout"),
    )


def wall_distance(f: Fixture, p: Point) -> float:
    if not f.selected_wall:
        raise ValueError("No selected wall in routing task")
    # Distances to the *identified* wall; other walls are not eligible.
    return min(hypot(p[0]-x, p[1]-y) for x,y in f.selected_wall)


def deviation(f: Fixture, p: Point) -> float:
    distance = wall_distance(f,p)
    return max(0.0, 6.0-distance, distance-10.0)


def neighbors(f: Fixture, p: Point) -> list[Point]:
    return [
        (p[0]+dx, p[1]+dy) for _,(dx,dy) in DIRECTIONS
        if f.legal((p[0]+dx, p[1]+dy))
    ]


def shortest_path(f: Fixture, start: Point, goal: Point, contour: bool = False) -> list[Point]:
    """A* with reproducible tie-break; contour mode prefers the selected band."""
    if not f.legal(start) or not f.legal(goal):
        return []
    frontier: list[tuple[float,int,int,Point]] = []
    serial = 0
    h = lambda p: abs(goal[0]-p[0])+abs(goal[1]-p[1])
    heapq.heappush(frontier,(h(start),0,serial,start))
    gscore: dict[Point,float] = {start:0.0}
    parent: dict[Point,Point] = {}
    while frontier:
        score, _, _, point = heapq.heappop(frontier)
        if score > gscore[point] + h(point) + 1e-9:
            continue
        if point == goal:
            path = [point]
            while point in parent:
                point = parent[point]
                path.append(point)
            return path[::-1]
        for neighbor in neighbors(f,point):
            # Every candidate is a cardinal step. Native later checks swept cells.
            step = 1.0 + (2.0 * deviation(f,neighbor) if contour else 0.0)
            candidate = gscore[point] + step
            if candidate + 1e-9 < gscore.get(neighbor,float("inf")):
                gscore[neighbor] = candidate
                parent[neighbor] = point
                serial += 1
                heapq.heappush(frontier,(candidate+h(neighbor),serial,serial,neighbor))
    return []


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    path: tuple[Point,...]
    stop: bool

    def source(self) -> dict[str, Any]:
        return {"candidate_id":self.candidate_id, "path":self.path, "stop":self.stop}


def candidates(f: Fixture, pos: Point, lengths: tuple[int,...] = (1,)) -> tuple[Candidate,...]:
    options = [Candidate("STOP",(),True)]
    for name,(dx,dy) in DIRECTIONS:
        for length in lengths:
            if length < 1 or length > 8:
                raise ValueError("Unbounded macro")
            swept = tuple((pos[0]+dx*i,pos[1]+dy*i) for i in range(1,length+1))
            if all(f.legal(p) for p in swept):
                options.append(Candidate(name+str(length),swept,False))
    return tuple(options)


def select_candidate(f: Fixture, pos: Point, options: tuple[Candidate,...],
                     policy: str, rng: random.Random) -> Candidate:
    if policy == "random_legal":
        return rng.choice(options)  # STOP is present; no hidden policy advantage.
    if policy not in {"astar", "contour_astar"}:
        raise ValueError(f"Unknown policy {policy}")
    if pos == f.target:
        return options[0]
    path = shortest_path(f,pos,f.target,contour=(policy=="contour_astar" and f.task=="wall_offset"))
    if len(path) < 2:
        return options[0]
    # A macro is admissible only if its *entire* swept trajectory is a
    # prefix of the newly computed reference path, never endpoint-only.
    matching = [c for c in options if not c.stop and
                list(c.path) == path[1:1+len(c.path)]]
    if not matching:
        raise AssertionError("A* proposed no admitted first step")
    return max(matching,key=lambda c: len(c.path))


def observation_stub(f: Fixture, position: Point, path: list[Point],
                     options: tuple[Candidate,...], decision: int) -> dict[str,Any]:
    """Nonconforming offline data staging view; actual native fences are NULL."""
    return {
        "adapter":VERSION, "authority":"offline_geometry_only",
        "fixture_sha256":f.sha256, "native_world_ref":None,
        "native_receipt_ref":None, "native_material_sha256":None,
        "position":position, "target":f.target, "selected_wall_id":
            f.fixture_id+"/selected-wall" if f.selected_wall else None,
        "selected_wall_distance":round(wall_distance(f,position),6)
            if f.selected_wall else None,
        "decision":decision, "budget_remaining":f.budget-decision,
        "path_sha256":digest(path),
        "candidate_set_sha256":digest([c.source() for c in options]),
        "candidates":[c.source() for c in options],
    }


def score_episode(f: Fixture, path: list[Point], attempts: int, stop: str) -> dict[str,Any]:
    # Full segment cells are in path, so both retraces and distances account for macros.
    visited: set[Point] = set()
    retraces = 0
    for p in path:
        if p in visited:
            retraces += 1
        visited.add(p)
    optimal = shortest_path(f,f.start,f.target)
    arrived = path[-1] == f.target
    result: dict[str,Any] = {
        "arrived":arrived,"stop_reason":stop,"false_stop":stop=="policy_stop" and not arrived,
        "attempted_decisions":attempts,"cursor_cell_steps":len(path)-1,
        "native_edit_operations":0,"native_receipts":0,"native_readback":"not_performed",
        "collisions":sum(not f.legal(p) for p in path),
        "protected_violations":sum(p in f.protected for p in path),
        "material_z_violations":sum(p in f.material_z for p in path),
        "retraced_cells":retraces,"unique_visited_cells":len(visited),
        "reference_astar_steps":len(optimal)-1,
        "route_inefficiency":round((len(path)-1)/(len(optimal)-1),6)
            if arrived and len(optimal)>1 else None,
    }
    if f.task == "wall_offset":
        distances = [wall_distance(f,p) for p in path]
        deviations = [deviation(f,p) for p in path]
        first_y=f.start[1]; last_y=f.target[1]
        result.update({
            "selected_wall_id":f.fixture_id+"/selected-wall",
            "route_progress_fraction":round(
                max(0.0,min(1.0,(max(p[1] for p in path)-first_y)/(last_y-first_y))),6),
            "fraction_cells_within_6_10":round(
                sum(d==0 for d in deviations)/len(deviations),6),
            "mean_offset_deviation":round(sum(deviations)/len(deviations),6),
            "max_offset_deviation":round(max(deviations),6),
            "mean_selected_wall_distance":round(sum(distances)/len(distances),6),
        })
    return result


def run_episode(f: Fixture, policy: str, seed: int = 0,
                lengths: tuple[int,...] = (1,)) -> dict[str,Any]:
    """One frozen offline episode. No inferred native success claims."""
    rng = random.Random(seed)
    pos = f.start
    path: list[Point] = [pos]
    events: list[dict[str,Any]] = []
    stop = "budget_exhausted"
    for decision in range(f.budget):
        if pos == f.target:
            stop = "arrived"
            break
        options = candidates(f,pos,lengths)
        observed = observation_stub(f,pos,path,options,decision)
        choice = select_candidate(f,pos,options,policy,rng)
        # Verify ID binding, source ordering and entire swept segment before committing.
        if choice not in options or any(not f.legal(p) for p in choice.path):
            raise AssertionError("Invalid host candidate or swept segment")
        if choice.stop:
            stop = "policy_stop"
        else:
            path.extend(choice.path)
            pos = choice.path[-1]
        events.append({
            "seq":decision, "observation":observed, "choice_id":choice.candidate_id,
            "simulated_post_position":pos, "native_receipt_ref":None,
            "simulated_path_sha256":digest(path),
        })
        if choice.stop:
            break
    else:
        if pos == f.target:
            stop = "arrived"
    return {
        "experiment_version":VERSION, "fixture":f.fixture_id,
        "fixture_sha256":f.sha256, "split":f.split, "policy":policy, "seed":seed,
        "macro_lengths":lengths, "source_kind":"offline_simulation_only",
        "native_qualified":False, "sequential_native_receipts":0,
        "path":path, "events":events,
        "metrics":score_episode(f,path,len(events),stop),
    }


def manifest() -> dict[str,Any]:
    catalog = fixtures()
    return {
        "version":VERSION,"split_rule":"distinct hardcoded topology families; never seed-only",
        "default_evaluation_split":"development",
        "fixtures":[{"fixture_id":f.fixture_id,"split":f.split,"sha256":f.sha256,
                     "task":f.task,"family":f.family,"budget":f.budget} for f in catalog],
        "catalog_sha256":digest([f.source() for f in catalog]),
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",action="store_true")
    parser.add_argument("--policy",choices=("astar","contour_astar","random_legal"),default="astar")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--macro",action="store_true",help="Separate 1/2/4/8 segment arm")
    parser.add_argument("--output",type=Path,help="Output JSONL path (otherwise print summary)")
    args=parser.parse_args()
    if args.manifest:
        print(json.dumps(manifest(),indent=2,sort_keys=True))
        return
    results = [run_episode(f,args.policy,args.seed,(1,2,4,8) if args.macro else (1,))
               for f in fixtures() if f.split=="development"]
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("w",encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result,sort_keys=True)+"\n")
    else:
        print(json.dumps([{"fixture":r["fixture"],"fixture_sha256":r["fixture_sha256"],
                           "metrics":r["metrics"]} for r in results],indent=2,sort_keys=True))


if __name__=="__main__":
    main()
