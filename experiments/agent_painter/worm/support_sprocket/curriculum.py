#!/usr/bin/env python3
"""WORM-001 SPROCKET-support: source-faithful synthetic decisions and 96x96 galleries.

No model, optimizer, native cell write, sealed FINAL input or hidden JSON repair.
The shepherd-owned W0 engine remains the sole legality/step/replay authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "common"
if not (COMMON / "worm.py").is_file():
    raise RuntimeError("W0 common/worm.py is required; run from the stacked WORM branch")
sys.path.insert(0, str(COMMON))
import worm  # noqa: E402 -- exact sibling common source, not copied/reimplemented

VERSION = "sprocket-support-teacher-v0"
ACTIONS = tuple(worm.ACTIONS)
FAMILIES = tuple(worm.FAMILIES)
TRAIN_SEEDS = (101, 107, 113, 127, 131, 137)
DEV_SEEDS = (1009, 1013, 1019)
GALLERY_SEEDS = (2111, 2129)
OBS_MODES = ("random", "wall", "fidget")
GRID_SIZE = 96
DATA_STEPS = 120
GALLERY_STEPS = 128


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def validate_observation(obs: dict) -> None:
    if not isinstance(obs, dict) or obs.get("family") not in FAMILIES:
        raise ValueError("missing/invalid family")
    if not isinstance(obs.get("tick"), int) or not 0 <= obs["tick"] <= 4096:
        raise ValueError("invalid tick")
    if obs.get("heading") not in range(4):
        raise ValueError("invalid heading")
    grid = obs.get("local_5x5")
    if not isinstance(grid, list) or len(grid) != 5 or any(
        not isinstance(row, str) or len(row) != 5 or set(row) - set(".#+@") for row in grid
    ) or grid[2][2] != "@":
        raise ValueError("invalid local 5x5")
    legal = obs.get("legal")
    if not isinstance(legal, list) or not legal or len(legal) != len(set(legal)) or (
        set(legal) - set(ACTIONS)
    ) or "STOP" not in legal:
        raise ValueError("invalid legal action list")


def teacher_choice(obs: dict) -> str:
    """Small deterministic local teacher; a synthetic *target*, not a trained model."""
    validate_observation(obs)
    legal = obs["legal"]
    tick, heading, family = obs["tick"], obs["heading"], obs["family"]
    if tick >= 116:
        return "STOP"

    fresh = []
    for action, rotation in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1)):
        direction = (heading + rotation) % 4
        dx, dy = worm.DIRECTIONS[direction]
        if action in legal and obs["local_5x5"][2 + dy][2 + dx] == ".":
            fresh.append(action)
    if not fresh:
        return "RETRACE" if "RETRACE" in legal else "STOP"

    # Family-specific local habits, not a hard-coded complete architecture.
    if family == "greebles":
        preference = ("LEFT", "RIGHT", "STRAIGHT") if tick % 6 in (1, 2, 4) else (
            "STRAIGHT", "RIGHT", "LEFT"
        )
        branch_mod = 9
    elif family == "pipework":
        preference = ("RIGHT", "STRAIGHT", "LEFT") if tick % 15 == 14 else (
            "STRAIGHT", "LEFT", "RIGHT"
        )
        branch_mod = 17
    else:
        preference = ("LEFT", "STRAIGHT", "RIGHT") if tick % 12 == 11 else (
            "STRAIGHT", "RIGHT", "LEFT"
        )
        branch_mod = 13
    if "BRANCH" in legal and tick > 0 and tick % branch_mod == 0:
        return "BRANCH"
    return next(action for action in preference if action in fresh)


def synthetic_fixture(family: str, case: str) -> dict:
    """Clearly constructed counterfactual observations; never native admissions."""
    if family not in FAMILIES:
        raise ValueError("invalid fixture family")
    base = {
        "x": 34, "y": 41, "heading": 0,
        "local_5x5": [".....", ".....", "..@..", ".....", "....."],
        "legal": ["STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP"],
        "last_actions": ["STRAIGHT"], "family": family, "tick": 9,
    }
    if case == "junction":
        base["tick"] = {"greebles": 9, "pipework": 17, "architecture": 13}[family]
    elif case == "blocked":
        base["local_5x5"] = ["#####", "#####", "##@##", "#####", "#####"]
        base["legal"] = ["RETRACE", "STOP"]
    elif case == "terminal":
        base["local_5x5"] = ["#####", "#####", "##@##", "#####", "#####"]
        base["legal"] = ["STOP"]
    elif case == "deadline":
        base["tick"] = 116
    else:
        raise ValueError("unknown fixture case")
    return base


def make_row(obs: dict, seed: int | None, group: str, source: str) -> dict:
    action = teacher_choice(obs)
    return {
        "version": VERSION,
        "source": source,
        "group": group,
        "world_seed": seed,
        "observation_sha256": digest(obs),
        "observation": obs,
        "input": "WORM-001 local steering. Choose exactly one legal uppercase action; "
                 "no explanations. OBS=" + canonical(obs).decode("ascii") + "\nACTION=",
        "target": action,
    }


def make_dataset() -> tuple[list[dict], list[dict], dict]:
    """Split by whole obstacle-world seed, then remove exact observation overlaps."""
    output: dict[str, list[dict]] = {"TRAIN": [], "DEVELOPMENT": []}
    seen: dict[str, str] = {}
    dropped = {"TRAIN": 0, "DEVELOPMENT": 0}
    for split, seeds in (("TRAIN", TRAIN_SEEDS), ("DEVELOPMENT", DEV_SEEDS)):
        for seed in seeds:
            for family in FAMILIES:
                for mode in OBS_MODES:
                    record = worm.run(seed, mode, family, DATA_STEPS, GRID_SIZE)
                    for i, event in enumerate(record["events"]):
                        if i % 3 != 0:
                            continue
                        obs = event["observation"]
                        key = digest(obs)
                        if key in seen:
                            dropped[split] += 1
                            continue
                        seen[key] = split
                        output[split].append(
                            make_row(obs, seed, "world-%d" % seed, "w0-%s-observation" % mode)
                        )
        # Constructed rare-action coverage is TRAIN-only; held-out DEV contains real W0 states.
        if split == "TRAIN":
            for family in FAMILIES:
                for case in ("junction", "blocked", "terminal", "deadline"):
                    obs = synthetic_fixture(family, case)
                    key = digest(obs)
                    if key not in seen:
                        seen[key] = split
                        output[split].append(
                            make_row(obs, None, "fixture-%s-%s" % (family, case),
                                     "constructed-not-native")
                        )
    # Strict output and entire-world structural split checks.
    a = {r["observation_sha256"] for r in output["TRAIN"]}
    b = {r["observation_sha256"] for r in output["DEVELOPMENT"]}
    if a & b:
        raise AssertionError("exact observation leakage")
    train_worlds = {r["world_seed"] for r in output["TRAIN"] if r["world_seed"] is not None}
    dev_worlds = {r["world_seed"] for r in output["DEVELOPMENT"] if r["world_seed"] is not None}
    if train_worlds & dev_worlds:
        raise AssertionError("world-seed leakage")
    for split in output:
        if not output[split] or any(
            row["target"] not in row["observation"]["legal"] for row in output[split]
        ):
            raise AssertionError("empty/illegal synthetic dataset")
    report = {
        "version": VERSION, "kind": "synthetic-W0-observations-not-native",
        "contract": "cybersand.worm.observation-action/0-experimental",
        "grid_size": GRID_SIZE, "observation_modes": list(OBS_MODES),
        "sample_stride": 3, "step_budget": DATA_STEPS,
        "train_world_seeds": list(TRAIN_SEEDS),
        "development_world_seeds": list(DEV_SEEDS),
        "gallery_world_seeds": list(GALLERY_SEEDS),
        "counts": {k: len(v) for k, v in output.items()},
        "exact_duplicate_observations_discarded": dropped,
        "action_histograms": {
            split: {a: sum(r["target"] == a for r in output[split]) for a in ACTIONS}
            for split in output
        },
        "family_histograms": {
            split: {f: sum(r["observation"]["family"] == f for r in output[split])
                    for f in FAMILIES}
            for split in output
        },
        "limitations": [
            "World seeds are groups; rows and policy views are correlated",
            "Rule-derived targets are not model outcomes or original native TRAIN",
            "Exact-observation dedupe is not proof of semantic generalisation",
            "DEVELOPMENT is a synthetic holdout, not Art v2 DEV60 or sealed FINAL",
        ],
    }
    return output["TRAIN"], output["DEVELOPMENT"], report


def teacher_rollout(seed: int, family: str, steps: int = GALLERY_STEPS) -> dict:
    """Closed-loop observation→teacher→host; prefix replay costs O(steps^2) but is bounded."""
    if not 0 <= steps <= GALLERY_STEPS:
        raise ValueError("overlong support replay")
    actions: list[str] = []
    for tick in range(steps):
        # W0 itself reconstructs the *true* legal observation after the prefix.
        probe = worm.run(seed, "replay", family, tick + 1, GRID_SIZE, actions + ["STOP"])
        if len(probe["events"]) != tick + 1:
            raise AssertionError("shortened prefix replay")
        obs = probe["events"][-1]["observation"]
        action = teacher_choice(obs)
        if action not in obs["legal"]:
            raise AssertionError("teacher returned illegal action")
        actions.append(action)
        if action == "STOP":
            break
    record = worm.run(seed, "replay", family, steps, GRID_SIZE, actions)
    if [e["raw_choice"] for e in record["events"]] != actions:
        raise AssertionError("closed-loop replay mismatch")
    record["policy_source"] = VERSION
    record["action_sequence_sha256"] = digest(actions)
    return record


def write_dataset(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train, dev, report = make_dataset()
    for name, rows in (("train", train), ("development", dev)):
        payload = b"".join(canonical(row) + b"\n" for row in rows)
        (out / (name + ".jsonl")).write_bytes(payload)
        report[name + "_sha256"] = hashlib.sha256(payload).hexdigest()
    (out / "dataset-manifest.json").write_bytes(canonical(report) + b"\n")
    return report


def write_gallery(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in GALLERY_SEEDS:
        for family in FAMILIES:
            for mode in OBS_MODES + ("support_teacher",):
                record = (teacher_rollout(seed, family) if mode == "support_teacher"
                          else worm.run(seed, mode, family, GALLERY_STEPS, GRID_SIZE))
                stem = "%s-%s-%d" % (family, mode, seed)
                j = canonical(record) + b"\n"
                png = worm.png_bytes(record)
                (out / (stem + ".json")).write_bytes(j)
                (out / (stem + ".png")).write_bytes(png)
                entries.append({
                    "id": stem, "seed": seed, "family": family, "policy": mode,
                    "source": "W0-executed-model-free",
                    "steps": record["steps_executed"], "metrics": record["metrics"],
                    "sha256_json": hashlib.sha256(j).hexdigest(),
                    "sha256_png": hashlib.sha256(png).hexdigest(),
                })
    result = {
        "version": VERSION, "contract": "cybersand.worm.observation-action/0-experimental",
        "kind": "source-faithful-python-W0-not-native-not-model",
        "paired_world_seeds": list(GALLERY_SEEDS),
        "same_grid_step_and_obstacle_seed_for_each_policy": True,
        "requested_step_budget": GALLERY_STEPS,
        "entries": entries,
    }
    (out / "gallery-manifest.json").write_bytes(canonical(result) + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--gallery-only", action="store_true")
    options = parser.parse_args()
    if options.dataset_only and options.gallery_only:
        parser.error("choose at most one mode")
    if not options.gallery_only:
        report = write_dataset(options.output / "dataset")
        print("DATASET " + json.dumps(report, sort_keys=True))
    if not options.dataset_only:
        report = write_gallery(options.output / "gallery")
        summary = [
            {"id": e["id"], "steps": e["steps"], "metrics": e["metrics"]}
            for e in report["entries"]
        ]
        print("GALLERY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
