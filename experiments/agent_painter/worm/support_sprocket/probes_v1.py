#!/usr/bin/env python3
"""SPROCKET support-only WORM v1: split-checked synthetic action probes; no optimizer."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import curriculum as c

ACTIONS = c.ACTIONS
FAMILIES = c.FAMILIES
TRAIN_VARIANTS = (0, 1, 2)
DEV_VARIANTS = (10, 11)
VERSION = "worm-sprocket-local-six-action-v1"


def local(family: str, action: str, split: str, variant: int) -> dict:
    """Constructed, internally legal LOCAL states; never claimed W0-world observations."""
    if family not in FAMILIES or action not in ACTIONS:
        raise ValueError("unknown family or action")
    if split not in ("TRAIN", "DEVELOPMENT") or variant not in (
            TRAIN_VARIANTS if split == "TRAIN" else DEV_VARIANTS):
        raise ValueError("invalid split or variant")
    fresh = {
        "STRAIGHT": ("STRAIGHT",), "LEFT": ("LEFT",),
        "RIGHT": ("RIGHT",), "BRANCH": ("STRAIGHT", "LEFT", "RIGHT"),
        "RETRACE": (), "STOP": (),
    }[action]
    history = action != "STOP"
    tick = 12 + variant % 9
    if action == "BRANCH":
        tick = {"greebles": 9, "pipework": 17, "architecture": 13}[family] * (1 + variant % 4)
    if action == "STOP" and variant % 2:
        fresh, history, tick = ("STRAIGHT", "LEFT"), True, 116
    grid = [list("#####") for _ in range(5)]
    grid[2][2] = "@"
    for move in fresh:
        dx, dy = {"STRAIGHT": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[move]
        grid[2+dy][2+dx] = "."
    legal = [a for a in ("STRAIGHT", "LEFT", "RIGHT") if a in fresh]
    if len(fresh) >= 2 and history:
        legal.append("BRANCH")
    if history:
        legal.append("RETRACE")
    legal.append("STOP")
    obs = {
        "x": 26 + variant + (20 if split == "DEVELOPMENT" else 0),
        "y": 29 + variant % 7 + 5 * FAMILIES.index(family),
        "heading": 0, "local_5x5": ["".join(r) for r in grid],
        "legal": legal, "last_actions": ["STRAIGHT"] if history else [],
        "family": family, "tick": tick,
    }
    c.validate_observation(obs)
    if c.teacher_choice(obs) != action:
        raise AssertionError("local case cannot produce requested legal teacher action")
    visible = {a for a, (dx, dy) in {
        "STRAIGHT": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)
    }.items() if grid[2+dy][2+dx] == "."}
    if visible != set(legal) & {"STRAIGHT", "LEFT", "RIGHT"}:
        raise AssertionError("counterfactual legal mask inconsistent")
    if ("BRANCH" in legal) != (history and len(visible) >= 2):
        raise AssertionError("counterfactual BRANCH inconsistent")
    return obs


def probes(split: str) -> list[dict]:
    variants = TRAIN_VARIANTS if split == "TRAIN" else DEV_VARIANTS
    if split not in ("TRAIN", "DEVELOPMENT"):
        raise ValueError("unknown split")
    output = []
    for family in FAMILIES:
        for action in ACTIONS:
            for variant in variants:
                obs = local(family, action, split, variant)
                row = c.make_row(obs, None, "constructed-%s-%s-%s-%d" % (
                    split.lower(), family, action.lower(), variant),
                    "counterfactual-local-not-world-not-native")
                if row["target"] != action:
                    raise AssertionError("target mismatch")
                output.append(row)
    return output


def verify_prompts(train: list[dict], dev: list[dict]) -> dict:
    seen = []
    for split, rows in (("TRAIN", train), ("DEVELOPMENT", dev)):
        labels = {}
        for row in rows:
            inp, target = row["input"], row["target"]
            if not inp.endswith("\nACTION=") or target not in row["observation"]["legal"]:
                raise AssertionError("illegal/malformed " + split)
            if inp in labels and labels[inp] != target:
                raise AssertionError("conflicting labels for same literal model input")
            labels[inp] = target
        seen.append(set(labels))
    if seen[0] & seen[1]:
        raise AssertionError("literal TRAIN/DEV prompt leakage")
    return {"train_rows": len(train), "development_rows": len(dev),
            "unique_train_prompts": len(seen[0]), "unique_development_prompts": len(seen[1]),
            "shared_literal_model_prompts": 0}


def balanced_view(world_train: list[dict], constructed_train: list[dict]) -> list[dict]:
    """Three TRAIN-only rows/family/action, prefer two real W0-observed rows."""
    result = []
    for family in FAMILIES:
        for action in ACTIONS:
            real = sorted((r for r in world_train if r["target"] == action and
                           r["observation"]["family"] == family and r["world_seed"] is not None),
                          key=lambda r: r["observation_sha256"])
            synthetic = sorted((r for r in constructed_train if r["target"] == action and
                                r["observation"]["family"] == family),
                               key=lambda r: r["observation_sha256"])
            selected = (real[:2] + synthetic)[:3]
            if len(selected) != 3:
                raise AssertionError("insufficient family/action examples")
            result.extend(selected)
    if len(result) != 54 or len(set(r["input"] for r in result)) != 54:
        raise AssertionError("balanced view repeats literal model prompts")
    return result


def summarize(rows: list[dict]) -> dict:
    return {
        "actions": {a: sum(r["target"] == a for r in rows) for a in ACTIONS},
        "family_action": {f: {a: sum(r["target"] == a and
                                  r["observation"]["family"] == f for r in rows)
                              for a in ACTIONS} for f in FAMILIES},
        "sources": dict(sorted(Counter(r["source"] for r in rows).items())),
    }


def save_jsonl(path: Path, rows: list[dict]) -> str:
    raw = b"".join(c.canonical(r) + b"\n" for r in rows)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def write_probe_set(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train, dev, previous = c.make_dataset()
    ct, cd = probes("TRAIN"), probes("DEVELOPMENT")
    checked = verify_prompts(train + ct, dev + cd)
    selection = balanced_view(train, ct)
    verify_prompts(selection, dev + cd)
    datasets = {"constructed_train": (ct, "constructed-train.jsonl"),
                "constructed_development": (cd, "constructed-development.jsonl"),
                "balanced_train_view": (selection, "balanced-train-view.jsonl")}
    hashes = {name: save_jsonl(out / filename, rows)
              for name, (rows, filename) in datasets.items()}
    report = {
        "version": VERSION, "kind": "W0-plus-counterfactual-local-not-native-not-model",
        "contract": "cybersand.worm.observation-action/0-experimental",
        "train_variants": list(TRAIN_VARIANTS), "development_variants": list(DEV_VARIANTS),
        "world_train_seeds": previous["train_world_seeds"],
        "world_dev_seeds": previous["development_world_seeds"],
        "original_w0_rows": {"TRAIN": len(train), "DEVELOPMENT": len(dev)},
        "full_literal_model_prompt_check": checked,
        "datasets": {n: {"rows": len(rows), "histogram": summarize(rows),
                         "sha256": hashes[n], "filename": filename}
                     for n, (rows, filename) in datasets.items()},
        "sources_sha256": {
            "W0": hashlib.sha256(Path(c.worm.__file__).read_bytes()).hexdigest(),
            "support_v0": hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Counterfactual local masks are not real W0 world observations",
            "Variant coordinates are not independent obstacle structures",
            "Split by literal prompt does not imply semantic generalization",
            "Balanced view is a proposed TRAIN sampler, NOT a model optimizer receipt",
            "Original v0 corpus unchanged; no FINAL, native cells or Art v2 admission",
        ],
    }
    (out / "manifest.json").write_bytes(c.canonical(report) + b"\n")
    return report


def write_right_stress(out: Path) -> dict:
    """Original W0 replay PNGs of synthetic RIGHT-only input, not trained SPROCKET."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in c.GALLERY_SEEDS:
        for family in FAMILIES:
            rec = c.worm.run(seed, "replay", family, c.GALLERY_STEPS, c.GRID_SIZE,
                             ["RIGHT"] * c.GALLERY_STEPS)
            rec["policy_source"] = "constant-RIGHT-control-NOT-trained-model"
            stem = "right-only-%s-%d" % (family, seed)
            raw, png = c.canonical(rec) + b"\n", c.worm.png_bytes(rec)
            (out / (stem + ".json")).write_bytes(raw)
            (out / (stem + ".png")).write_bytes(png)
            rows.append({
                "id": stem, "seed": seed, "family": family,
                "steps": rec["steps_executed"], "metrics": rec["metrics"],
                "raw_actions": dict(sorted(Counter(e["raw_choice"] for e in rec["events"]).items())),
                "applied_actions": dict(sorted(Counter(e["applied"] for e in rec["events"]).items())),
                "json_sha256": hashlib.sha256(raw).hexdigest(),
                "png_sha256": hashlib.sha256(png).hexdigest(),
            })
    manifest = {"version": VERSION, "not_model_or_native": True,
                "seeds": list(c.GALLERY_SEEDS), "budget": c.GALLERY_STEPS, "rows": rows}
    (out / "manifest.json").write_bytes(c.canonical(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    output = ap.parse_args().output
    print(json.dumps({"probes": write_probe_set(output / "probes"),
                      "right_control": write_right_stress(output / "right-control")},
                     sort_keys=True))
