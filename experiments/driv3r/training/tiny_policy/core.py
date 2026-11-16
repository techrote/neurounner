"""Isolated *synthetic* CyberSand-like 96x96 host-legal worm world.

No native authority is claimed. The action IDs are the Stage-A direct-drive
UP/DOWN/LEFT/RIGHT/STOP vocabulary; this is not a Workbench adapter.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

import numpy as np

SIZE = 96
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "STOP")
DELTA = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
FAMILIES = ("Greebles", "Pipework", "Architecture")
FEATURES = 3 * 7 * 7 + 4 + 3 + 4 + 4  # blocked/visited/protected; heading, family, step, local stats


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class World:
    seed: int
    family: str
    budget: int = 96
    blocked: np.ndarray = field(init=False, repr=False)
    protected: np.ndarray = field(init=False, repr=False)
    visited: np.ndarray = field(init=False, repr=False)
    x: int = 48
    y: int = 48
    heading: int = 3
    path: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    dead_ends: int = 0
    revisit_landings: int = 0
    stopped: bool = False

    def __post_init__(self):
        if self.family not in FAMILIES or not 0 <= self.seed < 2**32 or not 1 <= self.budget <= 128:
            raise ValueError("Invalid family, seed or bounded budget")
        r = random.Random(self.seed)
        self.blocked = np.zeros((SIZE, SIZE), dtype=np.bool_)
        self.protected = np.zeros((SIZE, SIZE), dtype=np.bool_)
        self.visited = np.zeros((SIZE, SIZE), dtype=np.bool_)
        self.blocked[0, :] = self.blocked[-1, :] = True
        self.blocked[:, 0] = self.blocked[:, -1] = True
        for _ in range(29):
            x, y = r.randrange(3, 90), r.randrange(3, 90)
            w, h = r.randrange(2, 12), r.randrange(2, 11)
            # Reserve independent central initial space; geometry is paired across policies.
            if x <= 54 and x + w >= 42 and y <= 54 and y + h >= 42:
                continue
            self.blocked[y:y+h, x:x+w] = True
        for _ in range(6):
            x, y = r.randrange(8, 85), r.randrange(8, 85)
            if 40 <= x <= 56 and 40 <= y <= 56:
                continue
            self.protected[y:y+3, x:x+4] = True
        self.path.append((self.x, self.y))
        self.visited[self.y, self.x] = True

    def legal(self) -> tuple[int, ...]:
        if self.stopped or len(self.actions) >= self.budget:
            return ()
        out = []
        for i, (dx, dy) in enumerate(DELTA[:4]):
            x, y = self.x + dx, self.y + dy
            if 0 <= x < SIZE and 0 <= y < SIZE and not self.blocked[y, x] and not self.protected[y, x]:
                out.append(i)
        return (*out, 4)  # Host-owned STOP always available, not forced as a model hallucination.

    def observe(self) -> np.ndarray:
        v = np.zeros((3, 7, 7), np.float32)
        for iy, yy in enumerate(range(self.y-3, self.y+4)):
            for ix, xx in enumerate(range(self.x-3, self.x+4)):
                if not (0 <= xx < SIZE and 0 <= yy < SIZE):
                    v[0, iy, ix] = 1
                else:
                    v[0, iy, ix] = float(self.blocked[yy, xx])
                    v[1, iy, ix] = float(self.visited[yy, xx])
                    v[2, iy, ix] = float(self.protected[yy, xx])
        head = np.eye(4, dtype=np.float32)[self.heading]
        fam = np.eye(3, dtype=np.float32)[FAMILIES.index(self.family)]
        t = len(self.actions)
        local = np.array([t / self.budget, min(self.dead_ends, 12)/12,
                          min(self.revisit_landings, 12)/12, len(self.path) / (self.budget+1)], np.float32)
        geom = np.array([self.x / 95, self.y / 95, float(np.mean(v[0])), float(np.mean(v[1]))], np.float32)
        return np.concatenate([v.ravel(), head, fam, local, geom])

    def apply(self, action: int) -> dict:
        if type(action) is not int or action not in self.legal():
            raise ValueError(f"Host rejected illegal action {action!r}")
        before = (self.x, self.y)
        prev_visit = int(self.visited[self.y, self.x])
        if action == 4:
            self.stopped = True
        else:
            dx, dy = DELTA[action]
            self.x, self.y = self.x+dx, self.y+dy
            prev_visit = int(self.visited[self.y, self.x])
            if prev_visit and self.path.count((self.x, self.y)) > 0:
                self.revisit_landings += 1
            self.visited[self.y, self.x] = True
            self.heading = action
            self.path.append((self.x, self.y))
        self.actions.append(ACTIONS[action])
        if 4 not in self.legal() and not self.stopped:  # Budget is exhausted.
            self.stopped = True
        # Count true locally trapped positions, regardless of STOP choice.
        if not any(i != 4 and (self.x+DELTA[i][0], self.y+DELTA[i][1]) not in self.path
                   for i in self.legal()):
            self.dead_ends += 1
        receipt = {"turn":len(self.actions)-1, "before":list(before), "action":ACTIONS[action],
                   "after":[self.x,self.y], "was_retrace": bool(prev_visit) if action != 4 else False,
                   "state_sha256":self.state_sha256()}
        return receipt

    def branch_nodes(self) -> int:
        """Degree >=3 in the *actually traversed edge graph*, not adjacent cells."""
        graph={}
        for src,dst in zip(self.path,self.path[1:]):
            graph.setdefault(src,set()).add(dst)
            graph.setdefault(dst,set()).add(src)
        return sum(len(neighbors)>=3 for neighbors in graph.values())

    def state_sha256(self) -> str:
        # Binds full obstacle/protection/visited arrays, not just path coordinates.
        h=hashlib.sha256()
        for a in (self.blocked, self.protected, self.visited):
            h.update(a.tobytes())
        h.update(json.dumps([self.seed,self.family,self.x,self.y,self.heading,self.actions]).encode())
        return h.hexdigest()


def teacher(world: World, rng: random.Random) -> int:
    """Small, intentionally imperfect synthetic geometric teacher (not original native TRAIN)."""
    moves = [a for a in world.legal() if a != 4]
    if not moves:
        return 4
    t = len(world.actions)
    scores=[]
    for a in moves:
        dx,dy=DELTA[a]
        x,y=world.x+dx,world.y+dy
        repeated=bool(world.visited[y,x])
        straight=a==world.heading
        turn=(a-world.heading)%4  # Discrete style signal, not an optimized path search.
        nearby=sum(bool(world.blocked[y+ey,x+ex] or world.protected[y+ey,x+ex]) for ex,ey in DELTA[:4])
        if world.family == "Greebles":
            s=1.9 * (not repeated) + 0.7 * (turn in (1,3)) + 0.6 * (t%9 in (0,1) and repeated)
        elif world.family == "Pipework":
            s=2.1*(not repeated)+1.3*straight+0.35*nearby-0.2*(t%13==0)
        else:
            s=2.2*(not repeated)+1.5*(straight if (t//7)%2==0 else turn in (1,3)) + 0.3*(a in (0,3))
        scores.append((s+rng.uniform(-0.35,0.35),a))
    return max(scores)[1]


def choose_random(world: World, rng: random.Random) -> int:
    moves=[a for a in world.legal() if a!=4]
    return rng.choice(moves) if moves else 4


def choose_wall(world: World, rng: random.Random) -> int:
    legal=set(world.legal())
    for a in (world.heading, (world.heading+1)%4, (world.heading+3)%4, (world.heading+2)%4):
        if a in legal: return a
    return 4


def run(world: World, selector, rng_seed: int):
    rng=random.Random(rng_seed)
    events=[]
    for _ in range(world.budget):
        legal=world.legal()
        if not legal: break
        before=world.state_sha256()
        choice=int(selector(world,rng))
        if choice not in legal: raise ValueError("Policy attempted illegal action")
        row={"before_sha256":before,"legal":[ACTIONS[a] for a in legal], "feature_sha256":hashlib.sha256(world.observe().tobytes()).hexdigest()}
        row.update(world.apply(choice))
        events.append(row)
        if world.stopped:break
    return {"world_seed":world.seed,"family":world.family,"budget":world.budget,
            "controller":getattr(selector,"__name__",type(selector).__name__),
            "rng_seed":rng_seed,"events":events,"unique_cells":int(world.visited.sum()),
            "retrace_steps":sum(x["was_retrace"] for x in events),"revisit_landings":world.revisit_landings,
            "branch_nodes":world.branch_nodes(),
            "dead_end_ticks":world.dead_ends,"stop":world.actions[-1]=="STOP" if world.actions else False,
            "final_state_sha256":world.state_sha256()}


def replay(report: dict):
    w=World(report["world_seed"],report["family"],report["budget"])
    for row in report["events"]:
        if w.state_sha256() != row["before_sha256"]:
            raise ValueError("Replay state digest mismatch")
        if [ACTIONS[x] for x in w.legal()] != row["legal"]:
            raise ValueError("Replay legal options mismatch")
        if hashlib.sha256(w.observe().tobytes()).hexdigest() != row["feature_sha256"]:
            raise ValueError("Replay feature digest mismatch")
        try:
            chosen=ACTIONS.index(row["action"])
        except ValueError as exc:
            raise ValueError("Unknown replay action") from exc
        got=w.apply(chosen)
        if got != {k:row[k] for k in got}:
            raise ValueError("Replay action or receipt mismatch")
    if (report["unique_cells"] != int(w.visited.sum()) or report["final_state_sha256"] != w.state_sha256()
            or report["branch_nodes"] != w.branch_nodes() or report["retrace_steps"] != w.revisit_landings):
        raise ValueError("Replay terminal state mismatch")
    return w


def raster(world: World, scale: int = 4):
    from PIL import Image
    a=np.full((SIZE,SIZE,3), (238,235,226), dtype=np.uint8)
    a[world.blocked]=[65,65,67]
    a[world.protected]=[180,75,84]
    c={"Greebles":(43,121,119),"Pipework":(36,104,189),"Architecture":(177,107,35)}[world.family]
    a[world.visited]=c
    a[world.y,world.x]=[255,211,71]
    return Image.fromarray(a,mode="RGB").resize((SIZE*scale,SIZE*scale),Image.Resampling.NEAREST)
