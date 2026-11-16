"""Bounded synthetic line-worm sketchbook. No native calls or model decisions.

Aesthetic descriptors are observations, not a quality function to optimize.
The three policies share their exact original canvas, guide, seed, movement
budget, pen cadence and hard permissions. All trajectories are saved, including
unproductive or ugly ones. This module is deliberately independent of the
frozen D-COMMON contract and preregistered D-2 confirmation policies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

VERSION = "d2.line-worm-development/1"
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
POLICIES = ("stubborn", "random_legal", "wayward")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class RNG:
    def __init__(self, seed):
        self.state = (seed & 0xffffffff) or 0x9e3779b9

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        self.state = x & 0xffffffff
        return self.state


def field(w, h, seed):
    # Exact integer-valued scalar guidance: a soft preference, never permission.
    return {(x, y): round(32767 + 24000 * math.sin(x * .11 + seed * .07)
                            * math.cos(y * .13 - seed * .03))
            for y in range(h) for x in range(w)}


def environment(w, h, scene):
    permitted = {(x, y) for x in range(2, w-2) for y in range(2, h-2)}
    if scene == "slats":
        for x in (w//3, 2*w//3):
            permitted -= {(x, y) for y in range(2, h-2)
                          if y not in range(h//3, h//3 + 7) and
                             y not in range(2*h//3, 2*h//3 + 7)}
    elif scene == "ring":
        cx, cy = w//2, h//2
        permitted -= {(x, y) for (x, y) in permitted
                      if 13 <= max(abs(x-cx), abs(y-cy)) <= 15
                      and not (abs(x-cx) <= 2 or abs(y-cy) <= 2)}
    elif scene != "open":
        raise ValueError("unregistered scene")
    return permitted


def recipe(scene, seed, policy, steps=192):
    return dict(version=VERSION, canvas=[80, 80], scene=scene,
                seed=seed, policy=policy, steps=steps,
                origin=list({"open": (28, 38), "slats": (23, 42), "ring": (22, 36)}[scene]), initial_heading=0,
                pen_rhythm=dict(period=37, lifted=4))


def simulate(spec):
    assert spec["version"] == VERSION
    assert spec["policy"] in POLICIES
    w, h = spec["canvas"]
    assert 32 <= w <= 128 and 32 <= h <= 128
    assert isinstance(spec["seed"], int) and 0 <= spec["seed"] < 2**32
    assert isinstance(spec["steps"], int) and 1 <= spec["steps"] <= 512
    origin = tuple(spec["origin"])
    permitted = environment(w, h, spec["scene"])
    assert origin in permitted
    guide = field(w, h, spec["seed"])
    pos, direction, rng = origin, spec["initial_heading"], RNG(spec["seed"])
    assert direction in range(4)
    trail, paint, events, counts = [pos], set(), [], {pos: 1}
    pen_period, pen_lift = spec["pen_rhythm"]["period"], spec["pen_rhythm"]["lifted"]
    assert 5 <= pen_period <= 128 and 0 <= pen_lift < pen_period
    stuck = 0
    for tick in range(spec["steps"]):
        legal = [(d, (pos[0]+v[0], pos[1]+v[1]))
                 for d, v in enumerate(DIRS)
                 if (pos[0]+v[0], pos[1]+v[1]) in permitted]
        if not legal:
            events.append(dict(tick=tick, kind="STOP", reason="no cardinal permitted move"))
            break
        # Every policy receives the same legal catalogue. Neither the guide nor
        # the controller is allowed to broaden permissions.
        catalogue = [[d, list(p)] for d, p in legal]
        if spec["policy"] == "stubborn":
            chosen = min(legal, key=lambda a: ((a[0]-direction) % 4 != 0,
                                                counts.get(a[1], 0), a[0]))
        elif spec["policy"] == "random_legal":
            chosen = legal[rng.next() % len(legal)]
        else:
            # A biased, intentionally imperfect controller: retain direction,
            # wander toward high soft guidance, sometimes favor a sharp turn,
            # and avoid repeatedly drilling the same pixel. No A* oracle.
            phase = (tick // 13 + spec["seed"]) % 4
            noise = {d: rng.next() % 73 for d in range(4)}
            def score(candidate):
                d, q = candidate
                bend = (d - direction) % 4
                heading_bonus = (42 if bend == 0 else 12 if bend in (1, 3) else -12)
                oddity = 40 if d == phase and tick % 29 < 9 else 0
                return (guide[q] // 1024 + heading_bonus + oddity
                        - 28 * min(3, counts.get(q, 0)) + noise[d])
            chosen = max(legal, key=lambda a: (score(a), -a[0]))
        new_direction, target = chosen
        down = tick % pen_period >= pen_lift
        before = digest(canonical(sorted(map(list, paint))))
        if down:
            paint.add(target)
        after = digest(canonical(sorted(map(list, paint))))
        events.append(dict(tick=tick, kind="STEP", from_cursor=list(pos),
                           to_cursor=list(target), heading=new_direction,
                           pen="down" if down else "up", guide_u16=guide[target],
                           legal_catalogue=catalogue,
                           changed=before != after,
                           material_before_sha256=before,
                           material_after_sha256=after,
                           native_receipt=None, model_decision=None))
        pos = target
        direction = new_direction
        counts[pos] = counts.get(pos, 0) + 1
        trail.append(pos)
        stuck = stuck+1 if counts[pos] >= 5 else 0
        if stuck >= 8:
            events.append(dict(tick=tick, kind="STOP", reason="repeated local oscillation"))
            break
    # Logical cursor movements and pen-up events are NOT material undo.
    events.append(dict(tick=len(trail)-1, kind="UNDO_MATERIAL_EDIT_UNAVAILABLE",
                       native_receipt=None, executed=False))
    vectors = [tuple(b[i]-a[i] for i in range(2))
               for a, b in zip(trail, trail[1:])]
    turns = sum(a != b for a, b in zip(vectors, vectors[1:]))
    uturns = sum(a == (-b[0], -b[1]) for a, b in zip(vectors, vectors[1:]))
    box = ([min(p[0] for p in paint), min(p[1] for p in paint),
            max(p[0] for p in paint), max(p[1] for p in paint)] if paint else None)
    return dict(recipe=spec, events=events,
                material=paint, trail=trail,
                metrics=dict(steps=len(vectors), painted_cells=len(paint),
                             brush_down_steps=sum(e.get("pen") == "down" for e in events),
                             turns=turns, uturns=uturns,
                             repeat_fraction_ppm=round(1000000 * (len(trail)-len(set(trail)))/len(trail)),
                             turn_fraction_ppm=round(1000000*turns/max(1, len(vectors)-1)),
                             painted_bbox=box,
                             guidance_mean_u16=(round(sum(guide[tuple(e["to_cursor"])]
                               for e in events if e["kind"] == "STEP")/len(vectors))
                                                      if vectors else None),
                             stop_reason=next((e["reason"] for e in events
                                               if e["kind"] == "STOP"), "budget"),
                             material_sha256=digest(canonical(sorted(map(list, paint)))),
                             trajectory_sha256=digest(canonical(trail))),
                permitted=permitted, guide=guide)


def rgba_png(w, h, pixels):
    raw = b"".join(b"\0" + b"".join(pixels(x, y) for x in range(w)) for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data))+tag+data+struct.pack(">I", zlib.crc32(tag+data)&0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w,h,8,6,0,0,0)) + chunk(b"IDAT", zlib.compress(raw,9)) + chunk(b"IEND",b"")


def artifact_bytes(spec, outcome):
    w, h = spec["canvas"]
    permitted, ink = outcome["permitted"], outcome["material"]
    def pixels(x, y):
        if (x, y) not in permitted:
            return bytes((77, 69, 86, 255))
        return bytes((230, 217, 184, 255)) if (x, y) in ink else bytes((22, 32, 43, 255))
    return {"recipe.json": canonical(spec),
            "trajectory.json": canonical(outcome["trail"]),
            "events.json": canonical(outcome["events"]),
            "geometry-before.png": rgba_png(w, h, lambda x,y:
                bytes((77,69,86,255)) if (x,y) not in permitted else bytes((22,32,43,255))),
            "material-after.png": rgba_png(w, h, pixels),
            "guidance.u16le": b"".join(struct.pack("<H", outcome["guide"][x,y])
                                     for y in range(h) for x in range(w))}


def replay(folder):
    """Independently regenerate every synthetic artifact from recorded recipe/source."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    assert manifest["version"] == VERSION
    assert manifest["qualification"] == "DEVELOPMENT_SYNTHETIC_ONLY"
    assert manifest["native"] is None and manifest["model"] is None
    assert manifest["native_replay_verified"] is False
    assert manifest["source_python_sha256"] == digest(Path(__file__).read_bytes())
    spec = json.loads((folder / "recipe.json").read_bytes())
    outcome = simulate(spec)
    expected = artifact_bytes(spec, outcome)
    assert manifest["recipe_sha256"] == digest(expected["recipe.json"])
    assert manifest["metrics"] == outcome["metrics"]
    assert manifest["artifacts"] == {
        name: {"bytes": len(data), "sha256": digest(data)}
        for name, data in expected.items()}
    assert {p.name for p in folder.iterdir()} == set(expected) | {"manifest.json"}
    for name, data in expected.items():
        assert (folder / name).read_bytes() == data, name
    return True


def run(out, specs):
    out = Path(out)
    records = []
    names = set()
    for spec in specs:
        outcome = simulate(spec)
        name = f"{spec['scene']}-seed{spec['seed']}-{spec['policy']}"
        assert name not in names
        names.add(name)
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        blobs = artifact_bytes(spec, outcome)
        for filename, blob in blobs.items():
            (folder / filename).write_bytes(blob)
        manifest = dict(version=VERSION, qualification="DEVELOPMENT_SYNTHETIC_ONLY",
                        source_python_sha256=digest(Path(__file__).read_bytes()),
                        native=None, model=None, native_replay_verified=False,
                        recipe_sha256=digest(blobs["recipe.json"]),
                        artifacts={filename:{"bytes":len(data),"sha256":digest(data)}
                                   for filename,data in blobs.items()},
                        metrics=outcome["metrics"])
        (folder/"manifest.json").write_bytes(canonical(manifest))
        assert replay(folder)
        records.append(dict(case=name, **manifest))
    result=dict(version=VERSION, cases=records,
                note="Synthetic sketch descriptors; NOT aesthetics scores or native evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out/"index.json").write_bytes(canonical(result))
    return result


def default_specs():
    return [recipe(scene, seed, policy) for scene in ("open", "slats", "ring")
            for seed in (23, 79) for policy in POLICIES]


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a=p.parse_args()
    print(digest(canonical(run(a.out, default_specs()))))
