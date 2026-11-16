#!/usr/bin/env python3
"""Independent, read-only SPROCKET S2 learned-logit/host-RNG and W0 replay audit.

Inputs are immutable original S2 artifact #14169781247 and shepherd W0 source.
Do NOT mistake score sampling for greedy EOS5, optimizer training or Godot painting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import zipfile

ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = (211, 223)
MODES = ("random", "wall", "fidget", "sprocket-scored-t1", "sprocket-scored-t3")
ARTIFACT_SHA = "014818c9cb44826aafb6bb0124228addcc0a602f3ff379b14b284ae17ddfb811"
W0_SHA = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
PRODUCER_SHA = "d0d34ab031461aa38e15b24a64bc6c1fb64cb882"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def draw(seed: int, tick: int) -> float:
    key = f"612:sprocket:score-crn-s2:{seed}:{tick}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / (1 << 64)


def sample(logits: dict, legal: list, temperature: float, u: float):
    assert set(logits) == set(ACTIONS)
    assert legal and len(legal) == len(set(legal)) and set(legal) <= set(ACTIONS)
    assert temperature > 0 and 0 <= u < 1
    active = [a for a in ACTIONS if a in legal]
    scaled = [float(logits[a]) / temperature for a in active]
    assert all(math.isfinite(x) for x in scaled)
    peak = max(scaled)
    weights = [math.exp(x - peak) for x in scaled]
    probabilities = {a: 0.0 for a in ACTIONS}
    for a, w in zip(active, weights):
        probabilities[a] = w / sum(weights)
    cumulative = 0.0
    selected = active[-1]
    for a in active:
        cumulative += probabilities[a]
        if u < cumulative:
            selected = a
            break
    uniform = active[min(int(u * len(active)), len(active) - 1)]
    return selected, uniform, probabilities


def original_zip(path: Path):
    assert digest(path.read_bytes()) == ARTIFACT_SHA, "original S2 ZIP SHA mismatch"
    z = zipfile.ZipFile(path)
    assert z.testzip() is None, "original ZIP CRC failed"
    members = {n for n in z.namelist() if not n.endswith("/")}
    inventory = {}
    for line in z.read("SHA256SUMS.txt").decode().splitlines():
        expected, name = line.split("  ", 1)
        if name.startswith("./"):
            name = name[2:]
        assert len(expected) == 64 and name not in inventory and name in members
        inventory[name] = expected
    assert set(inventory) == members - {"SHA256SUMS.txt"}, "missing/extra member"
    for name, sha in inventory.items():
        assert digest(z.read(name)) == sha, f"original member changed: {name}"
    assert z.read("source-sha.txt").decode().strip() == PRODUCER_SHA
    return z


def pinned_w0(path: Path):
    assert digest(path.read_bytes()) == W0_SHA, "untrusted W0 source"
    spec = importlib.util.spec_from_file_location("worm_w0_source_pinned", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(archive: Path, w0_path: Path) -> dict:
    z = original_zip(archive)
    w0 = pinned_w0(w0_path)
    meta = json.loads(z.read("results/comparison.json"))
    assert meta["schema"] == "cybersand.worm.sprocket-live-crn-ablation-s2"
    assert meta["w0_commit"] == "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
    assert meta["training_source"] == "c14795efe978dadeaa9e0da60f6273e18ab17b4d"
    assert meta["model"] == "kandivault/sprocket-500m"
    assert meta["revision"] == "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert meta["actual_optimizer_steps_this_job"] == 0
    assert meta["model_rollouts"] == 12 and meta["controls"] == 18
    assert meta["completed"] is True and meta["steps_per_trajectory"] == 24
    assert meta["seed_groups"] == list(SEEDS) and meta["families"] == list(FAMILIES)
    assert meta["native"] is False and meta["sealed_final_opened"] is False
    rows = {(r["family"], r["seed"], r["mode"]): r for r in meta["rows"]}
    cases = {(f, s, m) for f in FAMILIES for s in SEEDS for m in MODES}
    assert set(rows) == cases and len(meta["rows"]) == 30
    original = {}
    receipt_count = raw_left = paired_changes = 0
    for family, seed, mode in sorted(cases):
        path = f"results/png-and-json/{family}-{mode}-{seed}"
        rec = json.loads(z.read(path + ".json"))
        png = z.read(path + ".png")
        assert (rec["family"], rec["seed"], rec["mode"]) == (family, seed, mode)
        assert rec["steps_executed"] <= 24
        assert len(rec["events"]) == rec["steps_executed"]
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert struct.unpack(">II", png[16:24]) == (384, 384)
        assert rec["metrics"] == rows[family, seed, mode]["metrics"]
        points = [tuple(p) for p in rec["trajectory"]]
        sx, sy = points[0]
        xs, ys = zip(*points)
        assert rows[family, seed, mode]["spatial"] == {
            "unique_cells": len(set(points)),
            "bbox_width": max(xs) - min(xs) + 1,
            "bbox_height": max(ys) - min(ys) + 1,
            "max_manhattan_from_start": max(abs(x-sx) + abs(y-sy) for x, y in points)
        }
        actions = [event["raw_choice"] for event in rec["events"]]
        replay = w0.run(seed, "replay", family, steps=24, raw_actions=actions)
        for key in ("events", "metrics", "trajectory", "obstacles", "branch_points"):
            assert replay[key] == rec[key], f"{path}: W0 replay mismatch in {key}"
        assert w0.png_bytes(replay) == png, f"{path}: pixel-exact W0 PNG mismatch"
        receipts = rec.get("score_receipts", [])
        learned = mode.startswith("sprocket-scored-")
        assert len(receipts) == len(rec["events"]) if learned else not receipts
        if learned:
            temp = 1.0 if mode.endswith("t1") else 3.0
            assert rows[family, seed, mode]["sampled_decisions"] == len(receipts)
            for i, r in enumerate(receipts):
                assert r["tick"] == i and r["temperature"] == temp
                assert r["is_model_logit_conditioned"] is True
                assert r["is_eos5_completion"] is False and r["is_greedy_decoding"] is False
                assert r["host_legal"] == rec["events"][i]["observation"]["legal"]
                assert r["selected_action"] == actions[i] == r["applied_by_host"]
                assert r["selected_host_legal"] is True
                u = draw(seed, i)
                assert abs(u - r["host_uniform_draw"]) < 1e-15
                chosen, flat, probs = sample(r["full_six_token_logits"], r["host_legal"], temp, u)
                assert chosen == r["selected_action"] and flat == r["paired_uniform_same_observation"]
                for a in ACTIONS:
                    assert abs(probs[a] - r["masked_probs"][a]) < 1e-10
                assert (chosen != flat) == r["paired_differs_from_uniform"]
                assert r["raw_unmasked_top_action"] == max(ACTIONS, key=lambda a: r["full_six_token_logits"][a])
                raw_left += r["raw_unmasked_top_action"] == "LEFT"
                paired_changes += chosen != flat
                receipt_count += 1
            assert rows[family, seed, mode]["paired_diff_from_uniform"] == sum(
                r["paired_differs_from_uniform"] for r in receipts)
        original[family, seed, mode] = (digest(png), tuple(actions))
    assert receipt_count == meta["paired_uniform_action_count"] == 135
    assert raw_left == 135
    family_equal = []
    for seed in SEEDS:
        for mode in MODES:
            values = [original[family, seed, mode] for family in FAMILIES]
            family_equal.append({"seed": seed, "mode": mode,
                 "identical_png": len({v[0] for v in values}) == 1,
                 "identical_actions": len({v[1] for v in values}) == 1})
    model_rows = [r for r in meta["rows"] if r["model"]]
    return {
       "schema": "cybersand.worm.sprocket-s2-independent-audit-v1",
       "producer_sha": PRODUCER_SHA, "original_zip_sha256": ARTIFACT_SHA,
       "crc_verified": len(z.namelist()), "sha256_members_verified": len(z.namelist())-1,
       "W0_full_json_and_pixel_exact_replays": 30, "real_model_score_decisions": receipt_count,
       "all_raw_logit_top_LEFT": raw_left, "model_rollouts": 12, "matched_controls": 18,
       "same_observation_model_vs_uniform_action_differences": paired_changes,
       "early_STOP": {str(t): sum(r["early_stop"] for r in model_rows if r["temperature"] == t)
                      for t in (1.0, 3.0)},
       "mean_unique_cells": {m: sum(r["spatial"]["unique_cells"] for r in meta["rows"]
                                   if r["mode"] == m)/6 for m in MODES},
       "family_comparisons": family_equal,
       "trained_PNGs_identical_within_seed_temperature_across_three_families":
           all(r["identical_png"] for r in family_equal if r["mode"].startswith("sprocket-scored-")),
       "optimizer_updates": 0, "native_Godot_paintings": 0,
       "limitations": ["two independent obstacle worlds; three family-labelled views each",
           "samples are NOT greedy model completions", "the uniform action pair is at the learned observation, not a diverged closed-loop uniform rollout"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--trusted-w0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.zip, args.trusted_w0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("family_comparisons","limitations")},sort_keys=True))


if __name__ == "__main__":
    main()
