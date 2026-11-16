"""Offline matched scent ablation for the D-2 wayward line worm.

No model/native calls. Frozen v1 wayward scoring is intentionally reproduced with
the soft u16 guide field as the primary intervention. Explicit gain arms also
alter the guide weight; hard geometry, RNG, action catalogue and pen rhythm
are unchanged. V1 attract parity
is asserted in tests. All outputs are DEVELOPMENT-only, never native receipts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from worm_sketch import (DIRS, RNG, VERSION as BASE_VERSION, canonical, digest,
                         environment, field, recipe, simulate, rgba_png)

VERSION = "d2.worm-scent-ablation/1"
ARMS = ("neutral", "attract", "attract_x4", "attract_x16", "repel", "mirror_x")
SCENES = ("open", "slats", "ring")
SEEDS = (23, 79)


def guide_for(w, h, seed, arm):
    if arm not in ARMS:
        raise ValueError("unregistered guide arm")
    original = field(w, h, seed)
    if arm == "neutral":
        return {(x, y): 32768 for y in range(h) for x in range(w)}
    if arm in ("attract", "attract_x4", "attract_x16"):
        return original
    if arm == "repel":
        return {(x, y): 65535 - original[x, y] for y in range(h) for x in range(w)}
    return {(x, y): original[w - 1 - x, y] for y in range(h) for x in range(w)}


def run_one(scene, seed, arm, steps=192):
    if scene not in SCENES or arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered development scenario")
    if type(steps) is not int or not 1 <= steps <= 512:
        raise ValueError("steps must be an integer between 1 and 512")
    spec = recipe(scene, seed, "wayward", steps)
    w, h = spec["canvas"]
    allowed = environment(w, h, scene)
    guide = guide_for(w, h, seed, arm)
    pos = tuple(spec["origin"])
    heading = spec["initial_heading"]
    pen_period = spec["pen_rhythm"]["period"]
    pen_lift = spec["pen_rhythm"]["lifted"]
    rng = RNG(seed)
    visited = {pos: 1}
    material = set()
    trail = [pos]
    events = []
    stuck = 0
    for tick in range(steps):
        legal = [(d, (pos[0]+v[0], pos[1]+v[1]))
                 for d, v in enumerate(DIRS)
                 if (pos[0]+v[0], pos[1]+v[1]) in allowed]
        if not legal:
            events.append(dict(tick=tick, kind="STOP", reason="no cardinal permitted move"))
            break
        phase = (tick // 13 + seed) % 4
        noise = {d: rng.next() % 73 for d in range(4)}
        def score(candidate):
            direction, point = candidate
            bend = (direction - heading) % 4
            heading_bonus = 42 if bend == 0 else 12 if bend in (1, 3) else -12
            oddity = 40 if direction == phase and tick % 29 < 9 else 0
            gain = 16 if arm == "attract_x16" else 4 if arm == "attract_x4" else 1
            return (gain * (guide[point] // 1024) + heading_bonus + oddity
                    - 28 * min(3, visited.get(point, 0)) + noise[direction])
        direction, target = max(legal, key=lambda a: (score(a), -a[0]))
        before = digest(canonical(sorted(map(list, material))))
        down = tick % pen_period >= pen_lift
        if down:
            material.add(target)
        after = digest(canonical(sorted(map(list, material))))
        events.append(dict(tick=tick, kind="STEP", from_cursor=list(pos),
                           to_cursor=list(target), heading=direction,
                           guide_u16=guide[target],
                           legal_catalogue=[[d, list(p)] for d, p in legal],
                           pen="down" if down else "up", changed=before != after,
                           material_before_sha256=before,
                           material_after_sha256=after,
                           native_receipt=None, model_decision=None))
        pos, heading = target, direction
        trail.append(pos)
        visited[pos] = visited.get(pos, 0) + 1
        stuck = stuck+1 if visited[pos] >= 5 else 0
        if stuck >= 8:
            events.append(dict(tick=tick, kind="STOP", reason="repeated local oscillation"))
            break
    events.append(dict(tick=len(trail)-1, kind="UNDO_MATERIAL_EDIT_UNAVAILABLE",
                       native_receipt=None, executed=False))
    moves = [e for e in events if e["kind"] == "STEP"]
    advantage = round(sum(guide[tuple(e["to_cursor"])] -
                          sum(guide[tuple(item[1])] for item in e["legal_catalogue"])
                          // len(e["legal_catalogue"]) for e in moves)/len(moves)) if moves else 0
    return dict(spec=dict(version=VERSION, baseline=BASE_VERSION, scene=scene,
                          seed=seed, arm=arm, steps=steps,
                          guidance_gain=16 if arm == "attract_x16" else 4 if arm == "attract_x4" else 1),
                trail=trail, events=events, material=material,
                guide=guide, permitted=allowed,
                metrics=dict(steps=len(moves), cells=len(material),
                             turns=sum(a != b for a, b in zip(
                                 [e["heading"] for e in moves],
                                 [e["heading"] for e in moves][1:])),
                             repeats=len(trail)-len(set(trail)),
                             guide_advantage_u16=advantage,
                             stop=next((e["reason"] for e in events if e["kind"] == "STOP"), "budget"),
                             material_sha256=digest(canonical(sorted(map(list, material)))),
                             trajectory_sha256=digest(canonical(trail))))


def evidence_bytes(outcome):
    spec = outcome["spec"]
    w, h = 80, 80
    material = outcome["material"]
    permitted = outcome["permitted"]
    guide = outcome["guide"]
    return {
        "recipe.json": canonical(spec),
        "events.json": canonical(outcome["events"]),
        "trajectory.json": canonical(outcome["trail"]),
        "metrics.json": canonical(outcome["metrics"]),
        "guide-u16.json": canonical([guide[x, y] for y in range(h) for x in range(w)]),
        "before.png": rgba_png(w, h, lambda x, y:
                               bytes((77, 69, 86, 255)) if (x, y) not in permitted
                               else bytes((22, 32, 43, 255))),
        "after.png": rgba_png(w, h, lambda x, y:
                              bytes((77, 69, 86, 255)) if (x, y) not in permitted
                              else bytes((230, 217, 184, 255)) if (x, y) in material
                              else bytes((22, 32, 43, 255)))
    }


def pair_summary(a, b):
    aa, bb = [tuple(p) for p in a["trail"]], [tuple(p) for p in b["trail"]]
    first = next((i for i in range(min(len(aa), len(bb))) if aa[i] != bb[i]), None)
    if first is None and len(aa) != len(bb):
        first = min(len(aa), len(bb))
    overlap = len(a["material"] & b["material"])
    union = len(a["material"] | b["material"])
    return dict(first_divergence_step=first,
                material_jaccard_ppm=round(1_000_000 * overlap/union) if union else 1_000_000,
                same_hard_mask=a["permitted"] == b["permitted"],
                same_initial_cursor=aa[0] == bb[0])


def corpus(root):
    root = Path(root)
    summaries = []
    for scene in SCENES:
        for seed in SEEDS:
            arms = {arm: run_one(scene, seed, arm) for arm in ARMS}
            original = simulate(recipe(scene, seed, "wayward"))
            assert arms["attract"]["trail"] == original["trail"]
            assert arms["attract"]["material"] == original["material"]
            baseline = arms["neutral"]
            for arm, outcome in arms.items():
                case = f"{scene}-seed{seed}-{arm}"
                folder = root / case
                folder.mkdir(parents=True, exist_ok=True)
                blobs = evidence_bytes(outcome)
                for name, data in blobs.items():
                    (folder / name).write_bytes(data)
                manifest = {"qualification": "DEVELOPMENT_SYNTHETIC_ONLY",
                            "native": None, "model": None, "native_replay_verified": False,
                            "source_sha256": digest(Path(__file__).read_bytes()),
                            "baseline_sha256": digest(Path(__import__("worm_sketch").__file__).read_bytes()),
                            "artifacts": {name: digest(data) for name, data in blobs.items()}}
                (folder / "manifest.json").write_bytes(canonical(manifest))
                summaries.append(dict(case=case, arm=arm, scene=scene, seed=seed,
                                      metrics=outcome["metrics"],
                                      versus_neutral=pair_summary(baseline, outcome),
                                      manifest_sha256=digest(canonical(manifest))))
    report = dict(version=VERSION, qualification="DEVELOPMENT_SYNTHETIC_ONLY",
                  native=None, model=None, outcomes=summaries)
    (root / "index.json").write_bytes(canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(digest(canonical(corpus(args.out))))
