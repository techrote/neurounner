"""Synthetic, host-authoritative 96x96 CyberSand-style one-cell ink worlds.

Inspired by the read-only M3/M4 native legal menu. This is NOT native PAINT,
not the M1 evidence, and does not extend the native 64 logical-move limit.
A synthetic HISTORY return crosses only own ink; real native acknowledgement
and World epoch checks would be required for host execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Iterable

VERSION = "driv3r-long-horizon-world/2"
HOST_REF = "techrote/cybersand@313b836b3c9ce2c60e4659e3763df140f84347a6:experiments/driv3r/m3_model/short.py"
GRID = 96
EMPTY, SOLID, MATERIAL_Z, INK = 0, 1, 37, 58
MOVE = {"MOVE_N": (0, -1), "MOVE_E": (1, 0), "MOVE_S": (0, 1), "MOVE_W": (-1, 0)}
ALLOWED = tuple(sorted((*MOVE, "BACKTRACK", "RELOCATE_START", "STOP")))
TRAIN_FAMILIES = ("pickets", "ridges", "islands", "zigzag")
HELDOUT_FAMILIES = ("archipelago", "staggered_vault")
PILOT_EXCLUDED_FAMILIES = ("rooms", "funnel") # earlier design-inspected; NOT blind holdout
MODES = ("wall", "route", "motif")


def digest(o: object) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Spec:
    split: str
    family: str
    seed: int
    mode: str
    challenge: str = "normal"

    def __post_init__(self):
        if self.split not in ("train", "development", "heldout"):
            raise ValueError("Invalid split")
        if self.family not in (HELDOUT_FAMILIES if self.split == "heldout" else TRAIN_FAMILIES):
            raise ValueError("Structural family leaks into wrong split")
        if self.challenge not in ("normal", "trap_corridor"):
            raise ValueError("Unknown challenge")
        if self.mode not in MODES or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid mode/seed")
        if self.split == "train" and not 1000 <= self.seed < 2000:
            raise ValueError("Training seeds must be partitioned")
        if self.split == "development" and not 3000 <= self.seed < 4000:
            raise ValueError("Development seeds must be partitioned")
        if self.split == "heldout" and not 9000 <= self.seed < 10000:
            raise ValueError("Heldout seeds must be partitioned")

    @property
    def ident(self) -> str:
        return digest([VERSION, self.split, self.family, self.seed, self.mode, self.challenge])[:24]


def specs(split: str, n_per_family: int = 3) -> list[Spec]:
    if not 1 <= n_per_family <= 30:
        raise ValueError("Number of worlds out of range")
    families = HELDOUT_FAMILIES if split == "heldout" else TRAIN_FAMILIES
    base = {"train": 1000, "development": 3000, "heldout": 9000}[split]
    return [Spec(split, family, base + fi * 100 + j, MODES[(fi + j) % len(MODES)])
            for fi, family in enumerate(families) for j in range(n_per_family)]


def _base_materials(spec: Spec) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    rng = random.Random(spec.seed)
    materials: dict[tuple[int, int], int] = {}
    protected: set[tuple[int, int]] = set()

    def line(points: Iterable[tuple[int, int]], kind: int = SOLID):
        for p in points:
            x, y = p
            if 5 <= x < GRID - 5 and 5 <= y < GRID - 5:
                materials[p] = kind

    if spec.family == "pickets":
        for x in range(15, 88, 12):
            gap = rng.randint(29, 65)
            line(((x, y) for y in range(12, 83) if abs(y-gap) > 3),
                 MATERIAL_Z if x % 3 == 0 else SOLID)
    elif spec.family == "ridges":
        for phase in (0, 1, 2):
            yoff = (phase - 1) * 22
            line(((x, int(48 + yoff + 5 * math.sin((x + spec.seed % 17) / 6 + phase)))
                  for x in range(9, 87)), SOLID if phase != 1 else MATERIAL_Z)
    elif spec.family == "islands":
        for _ in range(19):
            x, y = rng.randrange(9, 86), rng.randrange(9, 86)
            w, h = rng.randint(2, 7), rng.randint(2, 5)
            line(((xx, yy) for xx in range(x, x+w) for yy in range(y, y+h)),
                 MATERIAL_Z if rng.random() < .3 else SOLID)
    elif spec.family == "zigzag":
        for start in (12, 34, 63):
            x = start
            y = 13
            while y < 82:
                line(((xx, y) for xx in range(x, x+10)), MATERIAL_Z if start == 34 else SOLID)
                line(((x+9, yy) for yy in range(y, y+9)))
                x = 13 if x > 50 else 68
                y += 15
    elif spec.family == "archipelago":
        # New, structurally disjoint final test: isolated boulder islands.
        for row,y0 in enumerate(range(15,83,13)):
            for x0 in range(13+(row%2)*6,85,14):
                rx = rng.randint(2,5); ry = rng.randint(2,5)
                line(((x,y) for x in range(x0,x0+rx) for y in range(y0,y0+ry)),
                     MATERIAL_Z if (row+x0)%3 else SOLID)
    elif spec.family == "staggered_vault":
        # Final-test vault banks with staggered, obstructed passage gaps.
        for row,y0 in enumerate((13,27,43,62,78)):
            gap = (19,65,30,73,42)[row] + spec.seed % 4
            line(((x,y0) for x in range(10,87) if not gap <= x < gap+7),
                 MATERIAL_Z if row%2 else SOLID)
            if row<4:
                line(((gap+3,y) for y in range(y0+2,min(y0+7,86))))
    elif spec.family == "rooms":
        for x0,y0,x1,y1 in ((12,12,40,39),(56,12,84,40),(17,56,79,83)):
            line(((x,y0) for x in range(x0,x1)))
            line(((x,y1) for x in range(x0,x1)))
            line(((x0,y) for y in range(y0,y1)))
            line(((x1,y) for y in range(y0,y1)))
            for p in list(materials):
                if p[0] in (x0, x1) and p[1] in range(y0+10, y0+15):
                    materials.pop(p, None)
    elif spec.family == "funnel":
        for dx in range(-35, 36):
            if abs(dx) < 4:
                continue
            spread = max(4, 29 - abs(dx)//2)
            line([(48+dx,48+spread),(48+dx,48-spread)], MATERIAL_Z if dx % 3 else SOLID)
        for y in range(18, 80, 8):
            line(((26,yy) for yy in range(y,y+3)))
            line(((71,yy) for yy in range(y,y+3)))

    # A source-identified protected exclusion zone: NEVER a soft guidance value.
    px = 18 + (spec.seed % 5) * 11
    py = 20 + ((spec.seed // 7) % 4) * 13
    protected.update((x,y) for x in range(px,px+4) for y in range(py,py+4))
    # The start is always truly Empty and offers immediate cardinal movement.
    center = (48,48)
    for x in range(44,53):
        for y in range(44,53):
            materials.pop((x,y), None)
            protected.discard((x,y))
    if spec.challenge == "trap_corridor":
        # Purpose-built forced recovery after a plausibly attractive bad move.
        # Start retains other exits; the eastern six-cell tunnel is a dead end.
        for x in range(49,56):
            for y in (47,49): materials[(x,y)] = MATERIAL_Z
        materials[(55,48)] = SOLID
        for x in range(49,55):
            materials.pop((x,48),None)
            protected.discard((x,48))
        # One end-of-tunnel protected cell to test hard exclusion.
        protected.add((55,48))
    return materials, protected


@dataclass
class World:
    spec: Spec
    materials: dict[tuple[int,int], int]
    protected: set[tuple[int,int]]
    cursor: tuple[int,int] = (48,48)
    heading: str = "E"
    path: list[tuple[int,int]] = field(default_factory=lambda: [(48,48)])
    ink: set[tuple[int,int]] = field(default_factory=set)
    steps: int = 0
    logical_moves: int = 0
    recoveries: int = 0
    stopped: bool = False
    last_action: str = ""
    decisions: list[dict] = field(default_factory=list)

    @classmethod
    def create(cls, spec: Spec) -> "World":
        base, protected = _base_materials(spec)
        return cls(spec, base, protected)

    def base_sha(self) -> str:
        return digest([VERSION, self.spec.ident,
                       [[*p, v] for p,v in sorted(self.materials.items())],
                       sorted(self.protected)])

    def state_sha(self) -> str:
        return digest([self.base_sha(), sorted(self.ink), self.cursor, self.heading,
                       self.path, self.steps, self.recoveries, self.stopped])

    def occupied(self, pos: tuple[int,int]) -> bool:
        x,y = pos
        return (not 0 <= x < GRID or not 0 <= y < GRID or pos in self.protected
                or pos in self.materials or pos in self.ink)

    def menu(self, max_recoveries: int = 12) -> list[str]:
        if self.stopped:
            return []
        legal = [a for a,(dx,dy) in MOVE.items()
                 if not self.occupied((self.cursor[0]+dx,self.cursor[1]+dy))]
        if self.recoveries < max_recoveries and len(self.path)>1:
            prev = self.path[-2]
            if abs(prev[0]-self.cursor[0])+abs(prev[1]-self.cursor[1]) == 1:
                legal.append("BACKTRACK")
            if len(self.path)>5 and self.cursor != self.path[0]:
                legal.append("RELOCATE_START")
        legal.append("STOP")  # never suppress STOP to engineer 128-turn continuation
        return sorted(legal)

    def apply(self, action: str, max_recoveries: int = 12) -> dict:
        legal = self.menu(max_recoveries)
        if action not in legal:
            raise ValueError(f"Rejected non-host-legal action {action}, offered {legal}")
        before = self.cursor
        pre = self.state_sha()
        paint: list[list[int]] = []
        if action in MOVE:
            dx,dy = MOVE[action]
            target = (before[0]+dx,before[1]+dy)
            assert not self.occupied(target)
            self.ink.add(target)
            self.cursor = target
            self.heading = action[-1]
            self.path.append(target)
            self.logical_moves += 1
            paint.append(list(target))
        elif action == "BACKTRACK":
            self.path.pop()
            self.cursor = self.path[-1]
            self.recoveries += 1
            self.logical_moves += 1
        elif action == "RELOCATE_START":
            # Synthetic brush-up return along owned history, not an unbounded
            # real native teleport or a grant to exceed the M3 64-move ceiling.
            self.logical_moves += len(self.path)-1
            self.path = [self.path[0]]
            self.cursor = self.path[0]
            self.recoveries += 1
        elif action == "STOP":
            self.stopped = True
        self.steps += 1
        self.last_action = action
        record = {"index": self.steps-1, "pre_sha256": pre, "legal": legal,
                  "chosen": action, "cursor_before": list(before),
                  "cursor_after": list(self.cursor), "painted": paint,
                  "ink_count": len(self.ink), "logical_moves": self.logical_moves,
                  "recoveries": self.recoveries, "post_sha256": self.state_sha(),
                  "native_receipt": None}
        self.decisions.append(record)
        return record

    def metrics(self, horizon: int) -> dict:
        moves = [x["chosen"] for x in self.decisions if x["chosen"].startswith("MOVE_")]
        turns = sum(a != b for a,b in zip(moves,moves[1:]))
        branches = sum(1 for p in self.ink if sum((p[0]+dx,p[1]+dy) in self.ink
                        for dx,dy in MOVE.values()) >= 3)
        return {"decisions": self.steps, "horizon": horizon,
                "unique_painted_cells": len(self.ink), "logical_cursor_moves": self.logical_moves,
                "recoveries": self.recoveries, "turns": turns, "branch_cells": branches,
                "stopped_early": self.stopped and self.steps < horizon,
                "stop_reason": "policy_stop" if self.stopped else "budget_exhausted",
                "illegal_host_actions": 0,
                "protected_cell_edits": len(self.ink.intersection(self.protected)),
                "non_empty_cell_edits": len(self.ink.intersection(self.materials)),
                "native_receipts": 0}


def replay(spec: Spec, transcript: list[dict], max_recoveries: int = 12) -> World:
    """Independently re-execute every claimed menu, state, geometry, material edit."""
    w = World.create(spec)
    for event in transcript:
        if event.get("index") != w.steps or event.get("pre_sha256") != w.state_sha():
            raise ValueError("Replay source/ordering/state mismatch")
        offered = w.menu(max_recoveries)
        if event.get("legal") != offered or event.get("chosen") not in offered:
            raise ValueError("Replay altered host-legal menu or action")
        if w.apply(event["chosen"], max_recoveries) != event:
            raise ValueError("Replay original material, cursor or receipt mismatch")
    return w
