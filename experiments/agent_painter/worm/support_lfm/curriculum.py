#!/usr/bin/env python3
"""WORM-001 LFM support: reproducible local-decision curriculum and paired galleries.

NO model training/inference and NO native/Godot application. The imported W0 host
is authoritative for obstacles, legal actions, STOP and rasterization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))
import worm  # W0 shared contract; this file never modifies it

VERSION = "worm-lfm-local-decisions-v2"
COMMON_SOURCE_SHA = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
TRAIN_SEEDS = tuple(range(1101, 1113))
DEV_SEEDS = tuple(range(2101, 2107))
SOURCE_MODES = ("wall", "fidget")
EVAL_POLICIES = ("random", "wall", "oracle")
PROMPT_VERSION = "lfm-worm-text-action-v1"


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def fresh_directions(observation: dict) -> list[str]:
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    heading = observation["heading"]
    grid = observation["local_5x5"]
    fresh = []
    for action, turn in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1)):
        if action not in observation["legal"]:
            continue
        dx, dy = directions[(heading + turn) % 4]
        if grid[2 + dy][2 + dx] == ".":
            fresh.append(action)
    return fresh


def oracle(observation: dict) -> str:
    """Deliberately imperfect, family-biased supervised teacher, never an AI model."""
    fresh = fresh_directions(observation)
    legal = observation["legal"]
    tick = observation["tick"]
    family = observation["family"]
    if len(fresh) >= 2 and "BRANCH" in legal and tick % 13 == 5:
        return "BRANCH"
    if fresh:
        if family == "greebles":
            order = ("LEFT", "RIGHT", "STRAIGHT") if tick % 2 else ("RIGHT", "LEFT", "STRAIGHT")
        elif family == "pipework":
            order = ("STRAIGHT", "RIGHT", "LEFT")
        else:
            order = ("STRAIGHT", "LEFT", "RIGHT") if tick % 17 < 12 else ("RIGHT", "STRAIGHT", "LEFT")
        return next(a for a in order if a in fresh)
    if "RETRACE" in legal:
        return "RETRACE"
    return "STOP"


def prompt(observation: dict) -> str:
    # Do not silently adopt a model-specific chat template. LFM primary owns its
    # official processor/EOS and can wrap this plain source text into its framing.
    state = {k: observation[k] for k in (
        "family", "heading", "local_5x5", "legal", "last_actions", "tick")}
    return (
        "WORM-001 local steering. Return exactly one uppercase action from "
        "STRAIGHT LEFT RIGHT BRANCH RETRACE STOP. Obey the LEGAL list; "
        "no scene JSON, explanations, material edits or control commands.\n"
        "Observation " + json.dumps(state, sort_keys=True, separators=(",", ":")) + "\nAction:"
    )



# Independently labelled LOCAL counterfactual motifs, NOT W0 world executions.
# They deliberately expose missing STOP/blocked-front/recovery classes without
# inventing fake Godot cells or mistaking rotated patterns for new world groups.
ADVERSE_MOTIFS = {
    "train": (
        ("front_wall_two_exits", (".....", ".....", "..@#.", ".....", "....."), True),
        ("sealed_with_backtrack", (".....", "..#..", ".#@#.", "..#..", "....."), True),
        ("sealed_no_backtrack", (".###.", ".###.", ".#@#.", ".###.", ".###."), False),
        ("visited_arms", (".....", "..+..", "..@+.", "..+..", "....."), True),
        ("single_straight_corridor", ("..#..", "..#..", "..@..", "..#..", "..#.."), True),
        ("open_three_way", (".....", ".....", "..@..", ".....", "....."), True),
    ),
    "development": (
        ("left_only_offset_corner", (".....", ".....", "..@#.", "..#..", "....."), True),
        ("right_only_offset_corner", (".....", "..#..", "..@#.", ".....", "....."), True),
        ("visited_backtracking_pocket", (".....", "..+..", "..@+.", "..#..", "....."), True),
        ("closed_crook_stop", ("..#..", "..#..", "..@#.", "..#..", "..#.."), False),
    ),
}


def adverse_rows(split: str) -> list[dict]:
    """Local synthetic counterexamples with their own non-native provenance."""
    if split not in ADVERSE_MOTIFS:
        raise ValueError("unknown adversarial split")
    result = []
    for motif, grid, has_history in ADVERSE_MOTIFS[split]:
        assert len(grid) == 5 and all(len(r) == 5 for r in grid)
        assert grid[2][2] == "@"
        for family in worm.FAMILIES:
            # Four tick formulations share ONE motif group; not independent scenes.
            for tick in (5, 6, 18, 19):
                legal = []
                for name, yy, xx in (("STRAIGHT", 2, 3), ("LEFT", 1, 2), ("RIGHT", 3, 2)):
                    if grid[yy][xx] != "#":
                        legal.append(name)
                fresh = sum(grid[yy][xx] == "." for yy, xx in ((2, 3), (1, 2), (3, 2)))
                if fresh >= 2:
                    legal.append("BRANCH")
                if has_history:
                    legal.append("RETRACE")
                legal.append("STOP")
                obs = {
                    "x": 48, "y": 48, "heading": 0,
                    "local_5x5": list(grid), "legal": legal,
                    "last_actions": ["STRAIGHT"] if has_history else [],
                    "family": family, "tick": tick,
                }
                label = oracle(obs)
                assert label in legal
                result.append({
                    "version": "worm-lfm-adverse-local-v1",
                    "split": split,
                    "world_group": "counterfactual-motif-" + split + "-" + motif,
                    "id": f"counterfactual-{split}-{motif}-{family}-{tick}",
                    "source_policy": "none-synthetic-local-observation",
                    "source_contract": "synthetic-local-from-w0-vocabulary-not-world-replay",
                    "observation_source": "counterfactual-no-world-or-native-receipt",
                    "observation": obs,
                    "prompt": prompt(obs),
                    "target_action": label,
                    "target_is_teacher": True,
                    "host_legal": legal,
                })
    return result


def rows(split: str, steps: int = 32) -> list[dict]:
    if split not in ("train", "development") or not 1 <= steps <= 512:
        raise ValueError("bad split or steps")
    seeds = TRAIN_SEEDS if split == "train" else DEV_SEEDS
    result = []
    for seed in seeds:
        for family in worm.FAMILIES:
            for mode in SOURCE_MODES:
                record = worm.run(seed=seed, mode=mode, family=family, steps=steps)
                for tick, event in enumerate(record["events"]):
                    obs = event["observation"]
                    label = oracle(obs)
                    assert label in obs["legal"]
                    result.append({
                        "version": VERSION,
                        "split": split,
                        "world_group": f"obstacle-seed-{seed}",
                        "id": f"{split}-{seed}-{family}-{mode}-{tick:03d}",
                        "source_policy": mode,
                        "source_contract": record["contract"],
                        "observation": obs,
                        "prompt": prompt(obs),
                        "target_action": label,
                        "target_is_teacher": True,
                        "host_legal": obs["legal"],
                    })
    return result



def split_distinct_actions() -> tuple[list[dict], list[dict], dict]:
    """V2: literal prompt-disjoint, duplicate-free local supervision.

    V1's disjoint world seeds were insufficient: different worlds can expose
    precisely the same local 5x5/heading/history prompt. Preserve that negative
    v1 artifact; never quietly treat identical prompt bytes as held-out.
    Keep first deterministic source row for each prompt, with labels verified.
    """
    raw_train, raw_dev = rows("train"), rows("development")
    dev_map: dict[str, dict] = {}
    for row in raw_dev:
        old = dev_map.get(row["prompt"])
        if old is not None and old["target_action"] != row["target_action"]:
            raise ValueError("conflicting DEVELOPMENT label for identical prompt")
        dev_map.setdefault(row["prompt"], row)
    train_map: dict[str, dict] = {}
    overlap_groups: set[str] = set()
    discarded_overlapping = 0
    for row in raw_train:
        old = train_map.get(row["prompt"])
        if old is not None and old["target_action"] != row["target_action"]:
            raise ValueError("conflicting TRAIN label for identical prompt")
        if row["prompt"] in dev_map:
            overlap_groups.add(row["world_group"])
            discarded_overlapping += 1
            continue
        train_map.setdefault(row["prompt"], row)
    if set(train_map).intersection(dev_map):
        raise ValueError("leaked literal prompt between TRAIN and DEVELOPMENT")
    output_train, output_dev = list(train_map.values()), list(dev_map.values())
    if not output_train or not output_dev:
        raise ValueError("empty deduplicated split")
    audit = {
        "raw_train_rows": len(raw_train),
        "raw_development_rows": len(raw_dev),
        "train_rows": len(output_train),
        "development_rows": len(output_dev),
        "removed_train_for_exact_development_prompt": discarded_overlapping,
        "distinct_overlapping_prompt_count": len(
            {r["prompt"] for r in raw_train} & set(dev_map)),
        "removed_duplicate_train_rows_after_collision_filter":
            len(raw_train) - discarded_overlapping - len(output_train),
        "removed_duplicate_development_rows": len(raw_dev) - len(output_dev),
        "literal_cross_split_prompt_overlap": 0,
        "retained_train_worlds": sorted({r["world_group"] for r in output_train}),
        "retained_development_worlds": sorted({r["world_group"] for r in output_dev}),
        "overlap_affected_train_world_groups": sorted(overlap_groups),
    }
    return output_train, output_dev, audit


def checked_splits(train: list[dict], dev: list[dict]) -> None:
    if not train or not dev:
        raise ValueError("empty split")
    if set(TRAIN_SEEDS) & set(DEV_SEEDS):
        raise ValueError("leaky split seed")
    ids = [item["id"] for item in train + dev]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate decision id")
    train_worlds = {r["world_group"] for r in train}
    dev_worlds = {r["world_group"] for r in dev}
    if train_worlds & dev_worlds:
        raise ValueError("world group leaked")
    for item in train + dev:
        if item["target_action"] not in item["host_legal"]:
            raise ValueError("illegal teacher action")
        if item["observation"]["family"] not in worm.FAMILIES:
            raise ValueError("unknown family")
        if item["target_action"] not in worm.ACTIONS:
            raise ValueError("unknown vocabulary")
    # Separate obstacle-world seeds do NOT guarantee unseen model input bytes.
    # This guard also includes counterfactual adverse rows supplied to callers.
    literal_overlap = {r["prompt"] for r in train} & {r["prompt"] for r in dev}
    if literal_overlap:
        raise ValueError(f"literal prompt TRAIN/DEVELOPMENT overlap: {len(literal_overlap)}")


def rollout(seed: int, family: str, steps: int, choose: Callable[[dict], str]) -> dict:
    """Live closed-loop W0 callback, with host-owned legality and raw receipts.

    Shepherd W0 callback receives a deepcopy of each current observation.
    No teacher forcing, raw-output rewriting, hidden retry or precomputed action
    list is involved. The host still owns STOP, obstacles and legal actions.
    """
    if not 1 <= steps <= 512 or family not in worm.FAMILIES:
        raise ValueError("rollout limit")
    return worm.run(seed=seed, family=family, mode="callback", steps=steps,
                    selector=choose)


def statistics(record: dict) -> dict:
    m = record["metrics"]
    events = record["events"]
    applied = [e["applied"] for e in events]
    raw = [e["raw_choice"] for e in events]
    longest = 0
    runlen = 0
    previous = None
    for action in applied:
        runlen = runlen + 1 if action == previous else 1
        longest = max(longest, runlen)
        previous = action
    counts = Counter(applied)
    raw_counts = Counter(raw)
    n = max(len(applied), 1)
    entropy = -sum((v/n)*math.log2(v/n) for v in counts.values())
    raw_entropy = -sum((v/n)*math.log2(v/n) for v in raw_counts.values())
    raw_streak = 0
    raw_max_streak = 0
    previous_raw = None
    for choice in raw:
        raw_streak = raw_streak + 1 if choice == previous_raw else 1
        raw_max_streak = max(raw_max_streak, raw_streak)
        previous_raw = choice
    no_fresh = sum(not fresh_directions(e["observation"]) for e in events)
    recoveries = sum(
        not fresh_directions(e["observation"]) and e["applied"] == "RETRACE"
        for e in events
    )
    return {
        "decision_count": len(events),
        "invalid_raw_count": m["invalid_raw_choices"],
        "invalid_raw_rate": round(m["invalid_raw_choices"] / n, 5),
        "raw_action_counts": dict(sorted(raw_counts.items())),
        "raw_action_entropy_bits": round(raw_entropy, 5),
        "max_same_raw_streak": raw_max_streak,
        "raw_straight_fraction": round(raw_counts["STRAIGHT"] / n, 5),
        "host_fallback_count": sum(not e["raw_legal"] for e in events),
        "unique_cells": m["unique_trail_cells"],
        "coverage_free_fraction": round(
            m["unique_trail_cells"] / (record["size"] ** 2 - len(record["obstacles"])), 6),
        "novelty_rate": round(m["new_steps"] / n, 5),
        "revisits": m["total_revisit_contacts"],
        "branches": m["branches"],
        "retrace_events": m["retraces"],
        "blocked_or_visited_decisions": no_fresh,
        "recoveries": recoveries,
        "max_same_action_streak": longest,
        "action_entropy_bits": round(entropy, 5),
        "stopped_by_host": m["terminated_by_stop"],
    }


def evaluate_policy(name: str, choose: Callable[[dict], str],
                    seeds: tuple[int, ...] = (2101, 2102),
                    steps: int = 64) -> list[tuple[dict, dict]]:
    """External LFM primary may supply a live choose(observation)->raw_string.

    Never supply target_action from DEVELOPMENT rows to the policy. The host
    still refuses illegal actions; receipts retain the original raw string.
    """
    if not name or name in ("random", "wall") or not 1 <= steps <= 512:
        raise ValueError("bad policy name or budget")
    if any(s not in DEV_SEEDS for s in seeds):
        raise ValueError("only held-out DEVELOPMENT obstacle seed allowed")
    return [(rollout(seed, family, steps, choose), {"policy": name})
            for seed in seeds for family in worm.FAMILIES]


def save_gallery(out: Path, steps: int = 240,
                 seeds: tuple[int, ...] = (2101, 2102)) -> dict:
    if any(s not in DEV_SEEDS for s in seeds):
        raise ValueError("gallery must be DEV world seeds")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    reports = []
    for family in worm.FAMILIES:
        for seed in seeds:
            for policy in EVAL_POLICIES:
                record = (rollout(seed, family, steps, oracle) if policy == "oracle"
                          else worm.run(seed, policy, family, steps))
                name = f"{family}-{seed}-{policy}"
                trace = canonical(record)
                png = worm.png_bytes(record)
                (out / f"{name}.json").write_bytes(trace)
                (out / f"{name}.png").write_bytes(png)
                scores = statistics(record)
                reports.append({"id": name, "family": family, "seed": seed,
                                "policy": policy, "metrics": scores})
                files.extend((
                    {"file": f"{name}.json", "sha256": hashlib.sha256(trace).hexdigest()},
                    {"file": f"{name}.png", "sha256": hashlib.sha256(png).hexdigest()},
                ))
        # Two deliberately bad raw controllers per theme; do not treat as trained.
        seed = seeds[0]
        for name, controller in (
            ("invalid-teleport", lambda obs: "TELEPORT"),
            ("repetition-left", lambda obs: "LEFT"),
        ):
            record = rollout(seed, family, steps, controller)
            identity = f"{family}-{seed}-failure-{name}"
            trace = canonical(record)
            png = worm.png_bytes(record)
            (out / f"{identity}.json").write_bytes(trace)
            (out / f"{identity}.png").write_bytes(png)
            reports.append({"id": identity, "family": family, "seed": seed,
                            "policy": "failure-" + name, "metrics": statistics(record)})
            files.extend((
                {"file": f"{identity}.json", "sha256": hashlib.sha256(trace).hexdigest()},
                {"file": f"{identity}.png", "sha256": hashlib.sha256(png).hexdigest()},
            ))
    summary = {
        "version": VERSION, "contract": "cybersand.worm.observation-action/0-experimental",
        "host_source": COMMON_SOURCE_SHA, "native": False, "model_inference": False,
        "steps": steps, "seeds": list(seeds), "files": files, "reports": reports,
        "metrics_version": "worm-lfm-raw-applied-metrics-v1",
        "scientific_note": "No model improvement claim; controls are non-learned. "
                           "Judge aesthetics from original PNGs independently of metrics.",
    }
    (out / "summary.json").write_bytes(canonical(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    if not 1 <= args.steps <= 512:
        parser.error("steps must be 1..512")
    args.output.mkdir(parents=True, exist_ok=True)
    train, dev, dedup_audit = split_distinct_actions()
    adverse_train, adverse_dev = adverse_rows("train"), adverse_rows("development")
    checked_splits(train + adverse_train, dev + adverse_dev)
    hashes = {}
    for name, data in (("train-v2.jsonl", train), ("development-v2.jsonl", dev),
                       ("adverse-train.jsonl", adverse_train),
                       ("adverse-development.jsonl", adverse_dev)):
        raw = b"".join(canonical(r) for r in data)
        (args.output / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    gallery = save_gallery(args.output / "gallery", steps=args.steps)
    manifest = {
        "version": VERSION, "prompt_version": PROMPT_VERSION,
        "host_source": COMMON_SOURCE_SHA, "train_world_seeds": list(TRAIN_SEEDS),
        "development_world_seeds": list(DEV_SEEDS), "train_rows": len(train),
        "development_rows": len(dev), "dedup_audit": dedup_audit,
        "adverse_train_rows": len(adverse_train),
        "adverse_development_rows": len(adverse_dev), "hashes": hashes,
        "gallery_png_count": sum(f["file"].endswith(".png") for f in gallery["files"]),
        "gallery_manifest_sha256": hashlib.sha256(canonical(gallery)).hexdigest(),
        "only_train_can_optimize": True, "no_native_execution": True,
        "no_model_inference": True,
    }
    (args.output / "manifest.json").write_bytes(canonical(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
