#!/usr/bin/env python3
"""WORM-001 SPROCKET support v2: executed W0 action examples, new bounded trails.

This is a model-free counterbaseline, NOT model decisions, real native materials,
legacy Art TRAIN, or the sealed FINAL partition. Only the pinned common W0 host
makes legal moves, constructs observations, owns RNG, bounds, receipts and PNGs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curriculum as old

# The current stacked #614 parent contains W0 before the shepherd-added
# observation callback. Reuse exact source-pinned W0 v2 on a separate read-only
# file rather than edit common, guess a transport, or simulate W0 locally.
W0_COMMIT = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_SHA256 = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
def load_pinned_w0():
    name = os.environ.get("SPROCKET_W0_PINNED_PATH")
    if not name:
        raise RuntimeError("SPROCKET_W0_PINNED_PATH required: download W0 source commit exactly")
    path = Path(name).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != W0_SHA256:
        raise ValueError("pinned W0 source missing or modified")
    spec = importlib.util.spec_from_file_location("worm_w0_pinned_support_v2", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned W0")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "selector" not in module.run.__code__.co_varnames:
        raise ValueError("pinned W0 callback API absent")
    if tuple(module.ACTIONS) != tuple(old.ACTIONS):
        raise ValueError("W0 action vocabulary mismatch")
    return module

worm = load_pinned_w0()

VERSION = "sprocket-support-observed-v2"
POLICIES = ("groove", "contour", "corridor")
TRAIN_WORLDS = (301, 307, 311, 313)
DEV_WORLDS = (1301, 1303)
GALLERY_WORLDS = (211, 223)
FAMILIES = tuple(worm.FAMILIES)
ACTIONS = tuple(worm.ACTIONS)
SIZE = 96
DATA_BUDGET = 96
GALLERY_BUDGET = 96

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")

def sha(value):
    return hashlib.sha256(value).hexdigest()

def choose(obs, policy, budget):
    """Strictly local deterministic policy; no hidden state, future lookahead or teacher."""
    old.validate_observation(obs)
    if policy not in POLICIES or not 1 <= budget <= 128:
        raise ValueError("unbounded/unknown policy")
    tick = obs["tick"]
    if tick >= budget - 1:
        return "STOP"  # voluntarily stop before exhaustion
    legal = set(obs["legal"])
    heading = obs["heading"]
    grid = obs["local_5x5"]
    fresh = []
    for action, turn in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1)):
        dx, dy = worm.DIRECTIONS[(heading + turn) % 4]
        if action in legal and grid[2 + dy][2 + dx] == ".":
            fresh.append(action)
    if not fresh:
        return "RETRACE" if "RETRACE" in legal else "STOP"
    # A branch is legal only when W0 itself found >=2 fresh host moves.
    # Family conditioning changes preferences but never changes host legality.
    family = obs["family"]
    if policy == "groove":
        pitch = (5, 7, 9)[FAMILIES.index(family)]
        if "BRANCH" in legal and tick > 0 and tick % pitch == 0:
            return "BRANCH"
        variant = (tick // 3 + obs["x"] + 2 * obs["y"]) % 4
        desired = (("LEFT", "STRAIGHT", "RIGHT"), ("RIGHT", "LEFT", "STRAIGHT"),
                   ("STRAIGHT", "LEFT", "RIGHT"), ("LEFT", "RIGHT", "STRAIGHT"))[variant]
    elif policy == "contour":
        if "BRANCH" in legal and tick > 0 and tick % (13 if family == "pipework" else 17) == 0:
            return "BRANCH"
        forward = "STRAIGHT" in fresh
        # The local 5x5 is orientation-independent. Favour a visible obstacle
        # to the right if present, or a left-handed boundary when free.
        rdx, rdy = worm.DIRECTIONS[(heading + 1) % 4]
        wall_right = grid[2 + rdy][2 + rdx] == "#"
        desired = (("STRAIGHT", "LEFT", "RIGHT") if forward and wall_right else
                   ("RIGHT", "STRAIGHT", "LEFT") if family == "greebles" else
                   ("LEFT", "STRAIGHT", "RIGHT"))
    else:
        run_length = (7 if family == "architecture" else
                      13 if family == "pipework" else 5)
        if "BRANCH" in legal and tick > 0 and tick % (2 * run_length) == 0:
            return "BRANCH"
        phase = tick % run_length
        desired = (("LEFT", "STRAIGHT", "RIGHT") if phase == run_length - 1 else
                   ("STRAIGHT", "RIGHT", "LEFT"))
    return next(action for action in desired if action in fresh)

def rollout(seed, family, policy, steps=DATA_BUDGET):
    if policy not in POLICIES or family not in FAMILIES or not 1 <= steps <= 128:
        raise ValueError("invalid model-free rollout")
    record = worm.run(seed, "callback", family, steps, SIZE,
                      selector=lambda obs: choose(obs, policy, steps))
    if record["steps_executed"] != steps or record["metrics"]["invalid_raw_choices"]:
        raise AssertionError("host refused/shortened local policy")
    if any(e["raw_choice"] != e["applied"] or not e["raw_legal"] for e in record["events"]):
        raise AssertionError("raw and applied policy differ")
    if record["events"][-1]["applied"] != "STOP":
        raise AssertionError("no bounded model-free STOP")
    return record

def trace_stats(record):
    counts = Counter(e["raw_choice"] for e in record["events"])
    dead_ends = 0
    blocked_forward = 0
    for e in record["events"]:
        obs = e["observation"]
        if obs["local_5x5"][2][3] == "#":
            blocked_forward += 1
        fresh = []
        for action, turn in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1)):
            dx, dy = worm.DIRECTIONS[(obs["heading"] + turn) % 4]
            if action in obs["legal"] and obs["local_5x5"][2+dy][2+dx] == ".":
                fresh.append(action)
        if not fresh and "RETRACE" in obs["legal"]:
            dead_ends += 1
    return {"actions": {a: counts[a] for a in ACTIONS},
            "dead_end_or_exhausted_local_fresh": dead_ends,
            "east_neighbor_blocked": blocked_forward,
            "unique_cells": record["metrics"]["unique_trail_cells"],
            "branches": record["metrics"]["branches"],
            "retraces": record["metrics"]["retraces"],
            "revisits": record["metrics"]["total_revisit_contacts"]}

def make_observed(split, seeds):
    rows = []
    duplicates = conflicts = 0
    seen = {}
    for seed in seeds:
        for family in FAMILIES:
            for policy in POLICIES:
                record = rollout(seed, family, policy)
                for event in record["events"]:
                    obs = event["observation"]
                    target = event["raw_choice"]
                    if target not in obs["legal"]:
                        raise AssertionError("illegal W0-observed action")
                    key = sha(canonical(obs))
                    if key in seen:
                        duplicates += 1
                        conflicts += int(seen[key] != target)
                        continue
                    seen[key] = target
                    rows.append({"schema": VERSION, "split": split,
                                 "kind": "actual-W0-callback-observation-not-native",
                                 "group": "world-seed-" + str(seed),
                                 "world_seed": seed, "policy": policy,
                                 "family": family, "observation": obs,
                                 "observation_sha256": key, "target": target})
    return rows, {"duplicate_observations_dropped": duplicates,
                  "conflicting_policy_targets_dropped": conflicts}

def write_dataset(out):
    train, td = make_observed("TRAIN", TRAIN_WORLDS)
    dev, dd = make_observed("DEVELOPMENT", DEV_WORLDS)
    ta = {r["observation_sha256"] for r in train}
    # Seed-disjoint worlds can expose *identical local W0 observations*, notably
    # the protected unobstructed starting area. Remove these from held-out DEV
    # rather than silently labeling identical inputs as independent samples.
    previous_dev = len(dev)
    dev = [r for r in dev if r["observation_sha256"] not in ta]
    cross_split_removed = previous_dev - len(dev)
    da = {r["observation_sha256"] for r in dev}
    if ta & da:
        raise AssertionError("cross-split W0 observation overlap")
    if set(TRAIN_WORLDS) & set(DEV_WORLDS) or (
        set(TRAIN_WORLDS + DEV_WORLDS) & set(old.TRAIN_SEEDS + old.DEV_SEEDS)):
        raise AssertionError("world group overlap with prior synthetic curriculum")
    out.mkdir(parents=True, exist_ok=True)
    result = {"version": VERSION, "world_size": SIZE, "budget": DATA_BUDGET,
              "policies": list(POLICIES), "families": list(FAMILIES),
              "train_world_seeds": TRAIN_WORLDS, "development_world_seeds": DEV_WORLDS,
              "source_code_sha256": sha(Path(__file__).read_bytes()),
              "w0_host": "read-only externally pinned W0 callback source",
              "w0_source_commit": W0_COMMIT, "w0_source_sha256": W0_SHA256,
              "kind": "observed W0 synthetic action policy, NOT model/native/FINAL",
              "split_observation_intersection": 0,
              "group_count": {"TRAIN": len(TRAIN_WORLDS), "DEVELOPMENT": len(DEV_WORLDS)},
              "deduplication": {"TRAIN": td, "DEVELOPMENT": dd},
              "identical_local_observations_removed_from_development": cross_split_removed}
    for label, rows in (("TRAIN", train), ("DEVELOPMENT", dev)):
        payload = b"".join(canonical(r) + b"\n" for r in rows)
        name = label.lower() + ".jsonl"
        (out / name).write_bytes(payload)
        result[label] = {
            "rows": len(rows), "sha256": sha(payload),
            "by_family": dict(Counter(r["family"] for r in rows)),
            "by_target": {a: sum(r["target"] == a for r in rows) for a in ACTIONS}}
        if not rows or any(r["target"] not in r["observation"]["legal"] for r in rows):
            raise AssertionError("empty/illegal W0 dataset")
    (out / "manifest.json").write_bytes(canonical(result) + b"\n")
    return result

def write_gallery(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in GALLERY_WORLDS:
        for family in FAMILIES:
            for policy in ("random", "wall", "fidget") + POLICIES:
                record = (rollout(seed, family, policy, GALLERY_BUDGET)
                          if policy in POLICIES else
                          worm.run(seed, policy, family, GALLERY_BUDGET, SIZE))
                stem = "%s-%s-%d" % (family, policy, seed)
                png, data = worm.png_bytes(record), canonical(record) + b"\n"
                (out / (stem + ".png")).write_bytes(png)
                (out / (stem + ".json")).write_bytes(data)
                rows.append({"id": stem, "world_seed": seed, "family": family,
                             "policy": policy, "metrics": record["metrics"],
                             "analysis": trace_stats(record),
                             "sha256_png": sha(png), "sha256_json": sha(data)})
    result = {"version": VERSION, "model_optimizer_steps": 0,
              "kind": "actual W0 Python PNG, no learned policy or native materials",
              "world_seed_groups": list(GALLERY_WORLDS),
              "family_labels": list(FAMILIES), "matched_steps": GALLERY_BUDGET,
              "gallery_count": len(rows), "entries": rows}
    (out / "manifest.json").write_bytes(canonical(result) + b"\n")
    return result

def selftest():
    if ACTIONS != ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP"):
        raise AssertionError("W0 action contract changed")
    if set(TRAIN_WORLDS) & set(DEV_WORLDS):
        raise AssertionError("world split collision")
    # Replay the exact source-generated output: full JSON, complete raw choices,
    # and every original W0 material pixel. No intermediary model or repairs.
    for seed in (211, 223):
        for family in FAMILIES:
            for policy in POLICIES:
                one = rollout(seed, family, policy, 48)
                two = rollout(seed, family, policy, 48)
                actions = [e["raw_choice"] for e in one["events"]]
                reproduced = worm.run(seed, "replay", family, 48, SIZE, actions)
                check = dict(one)
                check["mode"] = "replay"
                if check != reproduced or one != two:
                    raise AssertionError("W0 callback/full-state replay mismatch")
                if worm.png_bytes(one) != worm.png_bytes(reproduced):
                    raise AssertionError("W0 original PNG pixel replay mismatch")
                if trace_stats(one)["actions"]["STOP"] != 1:
                    raise AssertionError("STOP not present")
    print("SELFTEST: 18/18 repeated deterministic W0 callback/full-state/PNG replays PASS")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selftest-only", action="store_true")
    a = p.parse_args()
    selftest()
    if not a.selftest_only:
        data = write_dataset(a.output / "dataset")
        gallery = write_gallery(a.output / "gallery")
        print(json.dumps({"dataset": data, "gallery_count": gallery["gallery_count"]},
                         sort_keys=True))

if __name__ == "__main__":
    main()
