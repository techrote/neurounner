"""D-2 synthetic loop-escape ablation for the wayward drawing worm.

DEVELOPMENT only: no model, no native World, no host history or receipts. The
frozen worm_sketch and worm_scent_study source files are imported unchanged.
All arms share guidance, hard legality, cursor, pen cadence and tick-indexed
xorshift noise. Teleportation and material Undo are never simulated as painting.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from worm_sketch import DIRS, RNG, canonical, digest, environment, rgba_png, recipe
from worm_scent_study import SCENES, SEEDS, guide_for, run_one

VERSION = "d2.worm-hysteresis-escape/1"
ARMS = ("strong_x16", "fixed_x1", "hysteresis", "always_novel", "sham_hysteresis")
WINDOW = 12
MIN_UNIQUE = 10
MIN_ESCAPE_TICKS = 4
EXIT_NEW_CELLS = 5
MAX_ESCAPE_TICKS = 24
COOLDOWN_TICKS = 8


def run_one_escape(scene: str, seed: int, arm: str, steps: int = 192):
    if scene not in SCENES or seed not in SEEDS or arm not in ARMS:
        raise ValueError("unregistered synthetic study scenario")
    if type(steps) is not int or not (1 <= steps <= 512):
        raise ValueError("steps outside bounded integer range")
    spec = recipe(scene, seed, "wayward", steps)
    w, h = spec["canvas"]
    allowed = environment(w, h, scene)
    guide = guide_for(w, h, seed, "attract")
    pos = tuple(spec["origin"])
    heading = spec["initial_heading"]
    period, lift = spec["pen_rhythm"]["period"], spec["pen_rhythm"]["lifted"]
    rng = RNG(seed)
    visits = {pos: 1}
    recent = deque([pos], maxlen=WINDOW)
    trail, material, events = [pos], set(), []
    active = False
    escape_start = -1
    cooldown_until = 0
    fresh_during_escape = set()
    triggers = releases = 0
    legacy_stuck = 0

    for tick in range(steps):
        legal = [(d, (pos[0] + vec[0], pos[1] + vec[1]))
                 for d, vec in enumerate(DIRS)
                 if (pos[0] + vec[0], pos[1] + vec[1]) in allowed]
        if not legal:
            events.append(dict(tick=tick, kind="STOP", reason="no cardinal permitted move"))
            break
        # Trigger at the beginning of a decision, using ONLY prior observations.
        unique_recent = len(set(recent))
        can_trigger = (len(recent) == WINDOW and unique_recent <= MIN_UNIQUE
                       and tick >= cooldown_until)
        if arm in ("hysteresis", "sham_hysteresis") and not active and can_trigger:
            active = True
            escape_start = tick
            fresh_during_escape = set()
            triggers += 1
            events.append(dict(tick=tick, kind="ESCAPE_ENTER", recent_unique=unique_recent,
                               window=WINDOW, trigger="recent revisit density",
                               native_receipt=None, material_changed=False))
        if active and tick > escape_start:
            elapsed = tick - escape_start
            exit_due_to_novelty = elapsed >= MIN_ESCAPE_TICKS and len(fresh_during_escape) >= EXIT_NEW_CELLS
            exit_due_to_timeout = elapsed >= MAX_ESCAPE_TICKS
            if exit_due_to_novelty or exit_due_to_timeout:
                active = False
                cooldown_until = tick + COOLDOWN_TICKS
                releases += 1
                events.append(dict(tick=tick, kind="ESCAPE_EXIT",
                                   reason="novelty" if exit_due_to_novelty else "timeout",
                                   elapsed=elapsed, novel_cells=len(fresh_during_escape),
                                   native_receipt=None, material_changed=False))
        phase = (tick // 13 + seed) % 4
        # Exactly four noise draws, as in frozen strong/weak scent arms.
        noise = {d: rng.next() % 73 for d in range(4)}
        escape_weight = active and arm == "hysteresis"
        guidance_gain = 1 if (arm == "fixed_x1" or escape_weight or arm == "always_novel") else 16
        def score(candidate):
            d, q = candidate
            bend = (d - heading) % 4
            heading_bonus = 42 if bend == 0 else 12 if bend in (1, 3) else -12
            oddity = 40 if d == phase and tick % 29 < 9 else 0
            normal_penalty = 28 * min(3, visits.get(q, 0))
            # Only the modified policy uses extra novelty during an escape.
            # It does not invent a waypoint/teleport or erase prior material.
            extra_penalty = 0
            if escape_weight or arm == "always_novel":
                extra_penalty = 120 * min(3, visits.get(q, 0))
            return (guidance_gain * (guide[q] // 1024) + heading_bonus + oddity
                    - normal_penalty - extra_penalty + noise[d])
        direction, target = max(legal, key=lambda a: (score(a), -a[0]))
        # All arms receive identical legal candidate sets and host-only permissions.
        before = digest(canonical(sorted(map(list, material))))
        pen_down = tick % period >= lift
        was_unvisited = target not in visits
        if pen_down:
            material.add(target)
        after = digest(canonical(sorted(map(list, material))))
        events.append(dict(tick=tick, kind="STEP", from_cursor=list(pos),
                           to_cursor=list(target), heading=direction,
                           pen="down" if pen_down else "up",
                           legal_catalogue=[[d, list(p)] for d, p in legal],
                           guide_u16=guide[target], guidance_gain=guidance_gain,
                           escape_mode=bool(escape_weight),
                           recent_unique_before=unique_recent,
                           first_visit=was_unvisited, changed=before != after,
                           material_before_sha256=before, material_after_sha256=after,
                           native_receipt=None, model_decision=None))
        pos, heading = target, direction
        trail.append(pos)
        recent.append(pos)
        if was_unvisited and active:
            fresh_during_escape.add(pos)
        visits[pos] = visits.get(pos, 0) + 1
        legacy_stuck = legacy_stuck + 1 if visits[pos] >= 5 else 0
        if legacy_stuck >= 8:
            events.append(dict(tick=tick, kind="STOP", reason="repeated local oscillation"))
            break

    # A named unavailable reversal is never an actual synthetic Undo.
    events.append(dict(tick=len(trail)-1, kind="UNDO_MATERIAL_EDIT_UNAVAILABLE",
                       executed=False, native_receipt=None))
    moves = [e for e in events if e["kind"] == "STEP"]
    headings = [e["heading"] for e in moves]
    m = dict(steps=len(moves), painted_cells=len(material),
             unique_cursor_cells=len(set(trail)),
             first_visits=sum(e["first_visit"] for e in moves),
             turns=sum(a != b for a, b in zip(headings, headings[1:])),
             repeats=len(trail)-len(set(trail)),
             escape_triggers=triggers, escape_exits=releases,
             escape_steps=sum(e["escape_mode"] for e in moves),
             stop=next((e["reason"] for e in events if e["kind"] == "STOP"), "budget"),
             trajectory_sha256=digest(canonical(trail)),
             material_sha256=digest(canonical(sorted(map(list, material)))))
    return dict(spec=dict(version=VERSION, scene=scene, seed=seed, arm=arm,
                          steps=steps, window=WINDOW, min_unique=MIN_UNIQUE,
                          min_escape_ticks=MIN_ESCAPE_TICKS,
                          exit_new_cells=EXIT_NEW_CELLS,
                          max_escape_ticks=MAX_ESCAPE_TICKS,
                          cooldown_ticks=COOLDOWN_TICKS),
                material=material, trail=trail, permitted=allowed, guide=guide,
                events=events, metrics=m)


def evidence(outcome):
    w, h = 80, 80
    spec, guide = outcome["spec"], outcome["guide"]
    allowed, material = outcome["permitted"], outcome["material"]
    return {
        "recipe.json": canonical(spec),
        "trajectory.json": canonical(outcome["trail"]),
        "events.json": canonical(outcome["events"]),
        "metrics.json": canonical(outcome["metrics"]),
        "guidance-u16.json": canonical([guide[x, y] for y in range(h) for x in range(w)]),
        "before.png": rgba_png(w, h, lambda x, y:
           bytes((77, 69, 86, 255)) if (x, y) not in allowed else bytes((22, 32, 43, 255))),
        "after.png": rgba_png(w, h, lambda x, y:
           bytes((77, 69, 86, 255)) if (x, y) not in allowed
           else bytes((230, 217, 184, 255)) if (x, y) in material
           else bytes((22, 32, 43, 255)))
    }


def verify_folder(folder: Path):
    import json
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    spec = json.loads((folder / "recipe.json").read_bytes())
    if manifest["qualification"] != "DEVELOPMENT_SYNTHETIC_ONLY" or manifest["native_replay_verified"]:
        raise ValueError("invalid research qualification")
    if manifest["native"] is not None or manifest["model"] is not None:
        raise ValueError("synthetic experiment cannot contain native/model status")
    if manifest["source_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("research source drift")
    for filename, module in (("worm_sketch.py", "worm_sketch"),
                             ("worm_scent_study.py", "worm_scent_study")):
        if manifest["dependencies"][filename] != digest(Path(__import__(module).__file__).read_bytes()):
            raise ValueError("baseline source drift")
    out = run_one_escape(spec["scene"], spec["seed"], spec["arm"], spec["steps"])
    blobs = evidence(out)
    if out["spec"] != spec or manifest["metrics"] != out["metrics"]:
        raise ValueError("recipe/results mismatch")
    if set(blobs) | {"manifest.json"} != {p.name for p in folder.iterdir()}:
        raise ValueError("missing or extra original artifacts")
    if manifest["artifacts"] != {key: digest(value) for key, value in blobs.items()}:
        raise ValueError("artifact hash mismatch")
    for name, blob in blobs.items():
        if (folder / name).read_bytes() != blob:
            raise ValueError("original artifact tamper: " + name)
    return True


def corpus(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dependencies = {name: digest(Path(__import__(name[:-3]).__file__).read_bytes())
                    for name in ("worm_sketch.py", "worm_scent_study.py")}
    records = []
    for scene in SCENES:
        for seed in SEEDS:
            for arm in ARMS:
                out = run_one_escape(scene, seed, arm)
                folder = root / f"{scene}-seed{seed}-{arm}"
                folder.mkdir(parents=True, exist_ok=True)
                blobs = evidence(out)
                for name, content in blobs.items():
                    (folder / name).write_bytes(content)
                manifest = dict(qualification="DEVELOPMENT_SYNTHETIC_ONLY",
                                native=None, model=None, native_replay_verified=False,
                                source_sha256=digest(Path(__file__).read_bytes()),
                                dependencies=dependencies,
                                artifacts={k: digest(v) for k, v in blobs.items()},
                                metrics=out["metrics"])
                (folder / "manifest.json").write_bytes(canonical(manifest))
                assert verify_folder(folder)
                records.append(dict(scene=scene, seed=seed, arm=arm,
                                    case=folder.name, metrics=out["metrics"],
                                    manifest_sha256=digest(canonical(manifest))))
    report = dict(version=VERSION, qualification="DEVELOPMENT_SYNTHETIC_ONLY",
                  note="No native/model execution; development seeds, not held-out evidence",
                  outcomes=records)
    (root / "index.json").write_bytes(canonical(report))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(digest(canonical(corpus(Path(a.out)))))
