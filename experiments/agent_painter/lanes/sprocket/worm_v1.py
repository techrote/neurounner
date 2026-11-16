#!/usr/bin/env python3
"""WORM-001 SPROCKET v1: W0-observation -> one tiny legal action. No native authority.

The shepherd's pinned W0 stdlib walker is source data, not silently vendored into this
lane. Synthetic TRAIN and held-out observation seeds are disjoint; neither is Art v2.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

W0_COMMIT = "2be9fd02ee525f2eac999555d4e7392a1dbc739b"
W0_PATH = "experiments/agent_painter/worm/common/worm.py"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
TRAIN_SEEDS = (101, 103, 107, 109, 113, 127)
EVAL_SEEDS = (211, 223)
SYSTEM = ("You are a tiny procedural drawing worm. Respond with exactly one action: "
          "STRAIGHT, LEFT, RIGHT, BRANCH, RETRACE, or STOP. "
          "Use only the given legal moves. No explanation, JSON or scene program.")
SCHEMA = "cybersand.worm.sprocket-steering-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_w0(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("pinned shepherd W0 source missing")
    spec = importlib.util.spec_from_file_location("_worm_w0_frozen", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ACTIONS != ACTIONS or module.FAMILIES != FAMILIES:
        raise ValueError("unexpected W0 action vocabulary or families")
    return module


def prompt(observation):
    legal = observation["legal"]
    if not legal or any(a not in ACTIONS for a in legal) or len(set(legal)) != len(legal):
        raise ValueError("W0 legal actions malformed")
    if observation["family"] not in FAMILIES:
        raise ValueError("W0 family malformed")
    if len(observation["local_5x5"]) != 5 or any(len(x) != 5 or any(c not in "#.+@" for c in x)
                                                      for x in observation["local_5x5"]):
        raise ValueError("W0 5x5 malformed")
    lines = [
        "Family=" + observation["family"],
        "Heading=" + str(observation["heading"]),
        "Neighbourhood=" + "/".join(observation["local_5x5"]),
        "Legal=" + ",".join(legal),
        "Recent=" + (",".join(observation["last_actions"]) or "-"),
        "Action:",
    ]
    return "\n".join(lines)


def make_rows(w0, seeds, max_per_trace=24):
    rows = []
    for seed in seeds:
        for family in FAMILIES:
            for mode in ("wall", "fidget"):
                record = w0.run(seed, mode, family, steps=72)
                for tick, event in enumerate(record["events"]):
                    if tick % 3 or tick // 3 >= max_per_trace:
                        continue
                    obs = event["observation"]
                    choice = event["applied"]
                    if choice not in obs["legal"] or choice not in ACTIONS:
                        raise ValueError("nonlegal teacher action")
                    rows.append({"id": f"{family}-{seed}-{mode}-{tick:03d}", "family": family,
                                 "seed": seed, "prompt": prompt(obs), "completion": choice,
                                 "legal": obs["legal"], "observation": obs})
                # A bounded, legal stop demonstration, independent of whether W0 stopped.
                last = record["events"][-1]["observation"]
                rows.append({"id": f"{family}-{seed}-{mode}-stop", "family": family,
                             "seed": seed, "prompt": prompt(last), "completion": "STOP",
                             "legal": last["legal"], "observation": last})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate synthetic IDs")
    if set(seeds) & (set(EVAL_SEEDS) if tuple(seeds) == TRAIN_SEEDS else set(TRAIN_SEEDS)):
        raise ValueError("structural seed leakage")
    return rows


def build_dataset(w0, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows = make_rows(w0, TRAIN_SEEDS)
    heldout = make_rows(w0, EVAL_SEEDS, max_per_trace=6)
    if {r["seed"] for r in rows} & {r["seed"] for r in heldout}:
        raise ValueError("TRAIN/EVAL seed collision")
    paths = {}
    for label, subset in (("train", rows), ("eval", heldout)):
        data = "".join(canonical(row) + "\n" for row in subset).encode("utf-8")
        path = destination / (label + ".jsonl")
        path.write_bytes(data)
        paths[label] = {"sha256": sha_bytes(data), "rows": len(subset),
                        "seeds": sorted({r["seed"] for r in subset}),
                        "actions": dict(Counter(r["completion"] for r in subset))}
    return paths


def verify_dataset(directory):
    directory = Path(directory)
    data = {}
    for name, expected_seeds in (("train", TRAIN_SEEDS), ("eval", EVAL_SEEDS)):
        path = directory / (name + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if {r["seed"] for r in rows} != set(expected_seeds):
            raise ValueError("invalid synthetic seed split")
        if any(r["completion"] not in r["legal"] or prompt(r["observation"]) != r["prompt"] for r in rows):
            raise ValueError("synthetic observation/teacher mismatch")
        data[name] = rows
    if {r["id"] for r in data["train"]} & {r["id"] for r in data["eval"]}:
        raise ValueError("TRAIN/EVAL ID leakage")
    return data


def sanity():
    class Dummy:
        pass
    obs = {"family": "pipework", "heading": 0, "local_5x5": [".....", ".....", "..@..", ".....", "....."],
           "legal": ["STRAIGHT", "LEFT", "RIGHT", "STOP"], "last_actions": []}
    assert "Action:" in prompt(obs)
    try:
        prompt(dict(obs, legal=["TELEPORT"]))
    except ValueError:
        pass
    else:
        raise AssertionError("illegal teacher action not refused")
    assert not set(TRAIN_SEEDS) & set(EVAL_SEEDS)


if __name__ == "__main__":
    sanity()
    print("SPROCKET WORM v1 synthetic contract PASS")
