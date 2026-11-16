#!/usr/bin/env python3
"""WORM-001 SPROCKET v3: pinned support curriculum, prompt-disjoint tiny actions.

Explicitly separate from the legacy full-scene Art-v2 TRAIN/DEV/FINAL partitions.
The real W0 walker and support teacher are imported from exact source snapshots;
neither implementation is copied, patched or given host authority here.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import worm_v1 as base

W0_COMMIT = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
SUPPORT_COMMIT = "2b6534ac21542805f4bda123f67e9624d21e39d7"
W0_BLOB = "8979b6f0296e81051194bacc907992d38e006605"
SUPPORT_BLOB = "4f879502921e26d526a8976f79954ffa636c1d4a"
SCHEMA = "cybersand.worm.sprocket-steering-v3"
SYSTEM = base.SYSTEM
ACTIONS = base.ACTIONS
FAMILIES = base.FAMILIES
EVAL_SEEDS = (211, 223)

def prompt(observation):
    """Full observation makes tick, position and local action affordances visible.

    v1/v2 omitted tick and position; that caused literal prompt/target collisions.
    """
    if observation.get("family") not in FAMILIES:
        raise ValueError("WORM family mismatch")
    if not isinstance(observation.get("tick"), int) or observation["tick"] < 0:
        raise ValueError("WORM tick missing")
    legal = observation.get("legal")
    if not legal or "STOP" not in legal or len(legal) != len(set(legal)) or set(legal) - set(ACTIONS):
        raise ValueError("invalid host legal actions")
    base.prompt(observation)  # Additional adversarial 5x5/last-action checks.
    return "Local WORM steering; select only one host-legal action.\nOBS=" + json.dumps(
        observation, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\nACTION="

def load_support(path, w0_path):
    path, w0_path = Path(path).resolve(), Path(w0_path).resolve()
    expected_w0 = path.parent.parent / "common" / "worm.py"
    if expected_w0.resolve() != w0_path or not expected_w0.is_file():
        raise ValueError("support curriculum must use exactly separately pinned W0")
    sys.path.insert(0, str(expected_w0.parent))
    spec = importlib.util.spec_from_file_location("_sprocket_support_pinned_v0", path)
    if spec is None or spec.loader is None:
        raise ValueError("support curriculum missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.worm.__file__).resolve() != w0_path:
        raise ValueError("support imported wrong W0")
    if tuple(module.ACTIONS) != ACTIONS or tuple(module.FAMILIES) != FAMILIES:
        raise ValueError("unexpected support action/family vocabulary")
    return module

def dataset(support, dest):
    train, dev, source_manifest = support.make_dataset()
    output = {}
    for name, source_rows in (("train", train), ("eval", dev)):
        rows = []
        seen = set()
        for source in source_rows:
            obs = source["observation"]
            target = source["target"]
            if target not in obs["legal"] or target not in ACTIONS:
                raise ValueError("illegal synthetic target")
            text = prompt(obs)
            if text in seen:
                raise ValueError("duplicate prompt in a WORM split")
            seen.add(text)
            ident = hashlib.sha256(text.encode()).hexdigest()
            rows.append({"id": name + "-" + ident, "prompt": text,
                         "completion": target, "family": obs["family"],
                         "seed": source["world_seed"], "legal": obs["legal"],
                         "observation": obs, "group": source["group"],
                         "support_source": source["source"]})
        output[name] = rows
    train_prompts = {r["prompt"] for r in output["train"]}
    eval_prompts = {r["prompt"] for r in output["eval"]}
    if train_prompts & eval_prompts:
        raise ValueError("literal TRAIN/EVAL leakage after model prompt rendering")
    seeds_a = {r["seed"] for r in output["train"] if r["seed"] is not None}
    seeds_b = {r["seed"] for r in output["eval"] if r["seed"] is not None}
    if seeds_a & seeds_b:
        raise ValueError("WORM world-seed group overlap")
    if len(output["train"]) < 100 or len(output["eval"]) < 20:
        raise ValueError("unexpectedly small WORM support dataset")
    for family in FAMILIES:
        for action in ACTIONS:
            if not any(r["family"] == family and r["completion"] == action for r in output["train"]):
                raise ValueError("missing TRAIN family/action pair " + family + "/" + action)
    dest.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": SCHEMA + "/dataset", "support_version": source_manifest["version"],
               "support_source_groups": source_manifest["train_world_seeds"],
               "support_dev_groups": source_manifest["development_world_seeds"],
               "literal_prompt_intersection": 0, "support_manifest": source_manifest}
    for name, rows in output.items():
        raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()
        (dest / (name + ".jsonl")).write_bytes(raw)
        receipt[name] = {"rows": len(rows), "sha256": hashlib.sha256(raw).hexdigest(),
                         "by_action": dict(Counter(r["completion"] for r in rows)),
                         "by_family": dict(Counter(r["family"] for r in rows))}
    (dest / "manifest.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return output, receipt

def probes(rows, count=18):
    """Precommit one novel held-out observation per available family/action pair."""
    chosen = []
    for family in FAMILIES:
        for action in ACTIONS:
            matches = [r for r in rows if r["family"] == family and r["completion"] == action]
            if matches:
                chosen.append(min(matches, key=lambda r: r["id"]))
    return chosen[:count]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--w0", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = load_support(args.support, args.w0)
    _, receipt = dataset(module, args.output)
    print(json.dumps({k: receipt[k] for k in ("train", "eval", "literal_prompt_intersection")},
                     sort_keys=True))
