#!/usr/bin/env python3
"""Research-only paired-geometry intervention on sparse d1 game policies.

Keep the W0 family label, seed, prompt, source weights and action contract
identical within each pair. Only host-side synthetic obstacle cells differ.
This detects *possible* geometry sensitivity, not successful navigation.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.frankenlayers import infer as paired
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows
from experiments.driv3r.frankenlayers_lab.infer_quant import validate_quant_receipt

SEED = 2718
STEPS = 36
SIZE = 96
FAMILY = "architecture"
PAIRS = {
    "forward_gate": ("east_open", "east_blocked"),
    "side_obstacle": ("north_blocked", "south_blocked"),
    "distant_gate": ("far_open", "far_blocked"),
}
ALL_VARIANTS = tuple(x for pair in PAIRS.values() for x in pair)
MODES = ("aero4_no10", "aero5_no23", "aero4_no2")


def variant_obstacles(variant: str, size: int = SIZE) -> set[tuple[int, int]]:
    if variant not in ALL_VARIANTS or size != SIZE:
        raise ValueError("Unsupported intervention")
    y = size // 2
    return {
        "east_open": set(),
        "east_blocked": {(9, y)},
        "north_blocked": {(7, y - 1), (8, y - 1)},
        "south_blocked": {(7, y + 1), (8, y + 1)},
        "far_open": set(),
        "far_blocked": {(14, y)},
    }[variant]


def intervention_world(baseline: set[tuple[int, int]], variant: str, size: int = SIZE):
    """Clear exactly the same local area before inserting either intervention."""
    blocks = set(baseline)
    c = size // 2
    blocks.difference_update((x, y) for x in range(4, 32)
                             for y in range(c - 12, c + 13))
    blocks.update(variant_obstacles(variant, size))
    if (7, c) in blocks:
        raise AssertionError("Spawn obstructed")
    if not all(0 <= x < size and 0 <= y < size for x, y in blocks):
        raise AssertionError("Out-of-bounds intervention")
    return blocks


class CausalHost:
    def __init__(self, w0, variant: str):
        if variant not in ALL_VARIANTS:
            raise ValueError("Unregistered intervention")
        self.w0 = w0
        self.variant = variant

    def run(self, seed, *, mode, family, steps, selector):
        if (seed, mode, family, steps) != (SEED, "callback", FAMILY, STEPS):
            raise ValueError("Paired causal run contract changed")
        original = self.w0.initial_world

        def initial_world(seed_arg, size=SIZE):
            return intervention_world(original(seed_arg, size), self.variant, size)

        self.w0.initial_world = initial_world
        try:
            record = self.w0.run(seed=seed, mode=mode, family=family, steps=steps,
                                 size=SIZE, selector=selector)
        finally:
            self.w0.initial_world = original
        record["research_intervention"] = {
            "schema": "frankenlayers-paired-geometry/1",
            "variant": self.variant,
            "fixed_family": FAMILY,
            "seed": SEED,
            "synthetic_only": True,
            "native_game": False,
        }
        return record

    def png_bytes(self, record):
        return self.w0.png_bytes(record)


def first_difference(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def pair_diagnostic(a: dict, b: dict) -> dict:
    # These records are the *actual* model requests, not inferred from a PNG.
    obs_a = [r["observation"] for r in a["calls"]]
    obs_b = [r["observation"] for r in b["calls"]]
    actions_a = a["arm"]["experiments"][0]["raw_choices"]
    actions_b = b["arm"]["experiments"][0]["raw_choices"]
    first_observation = first_difference(obs_a, obs_b)
    first_action = first_difference(actions_a, actions_b)
    return {
        "first_observation_difference": first_observation,
        "first_action_difference": first_action,
        "initial_observation_differs": bool(obs_a and obs_b and obs_a[0] != obs_b[0]),
        "geometry_exposure_observed": first_observation is not None,
        "action_divergence_after_observation": (
            first_action is not None and first_observation is not None
            and first_action >= first_observation),
        "unexplained_early_action_divergence": (
            first_action is not None and
            (first_observation is None or first_action < first_observation)),
        "decisions": [len(actions_a), len(actions_b)],
        "new_cells": [a["arm"]["experiments"][0]["metrics"]["new_steps"],
                      b["arm"]["experiments"][0]["metrics"]["new_steps"]],
        "early_stop": [bool(a["arm"]["experiments"][0]["metrics"]["terminated_by_stop"]),
                       bool(b["arm"]["experiments"][0]["metrics"]["terminated_by_stop"])],
    }


def run_variant(binary: Path, model: Path, w0, mode: str,
                root: Path, seed: int, variant: str):
    old_steps, old_families = paired.STEPS, paired.FAMILIES
    try:
        paired.STEPS, paired.FAMILIES = STEPS, (FAMILY,)
        path = root / variant
        arm = paired.run_arm(binary, model, CausalHost(w0, variant), mode, path, seed)
    finally:
        paired.STEPS, paired.FAMILIES = old_steps, old_families
    calls = json.loads((path / f"{mode}-{FAMILY}-choices.json").read_text())
    return {"arm": arm, "calls": calls}


def run_all(binary, model, w0, mode, root, seed):
    runs = {v: run_variant(binary, model, w0, mode, root, seed, v)
            for v in ALL_VARIANTS}
    diagnostics = {name: pair_diagnostic(runs[left], runs[right])
                   for name, (left, right) in PAIRS.items()}
    arms = [v["arm"] for v in runs.values()]
    episodes = [a["experiments"][0] for a in arms]
    counts = {key: sum(a["choice_counts"].get(key, 0) for a in arms)
              for key in paired.ACTION_DESCRIPTIONS}
    return {
        "mode": mode,
        "variants": list(ALL_VARIANTS),
        "pairs": diagnostics,
        "episodes": [
            {"variant": variant, "decisions": run["arm"]["model_decisions"],
             "metrics": run["arm"]["experiments"][0]["metrics"],
             "raw_choices": run["arm"]["experiments"][0]["raw_choices"]}
            for variant, run in runs.items()
        ],
        "summary": {
            "decisions": sum(a["model_decisions"] for a in arms),
            "new_cells": sum(e["metrics"]["new_steps"] for e in episodes),
            "revisits": sum(e["metrics"]["total_revisit_contacts"] for e in episodes),
            "early_stop_episodes": sum(bool(e["metrics"]["terminated_by_stop"])
                                       for e in episodes),
            "invalid_host_choices": sum(e["metrics"]["invalid_raw_choices"]
                                        for e in episodes),
            "geometry_exposed_pairs": sum(v["geometry_exposure_observed"]
                                          for v in diagnostics.values()),
            "observation_followed_by_action_divergence_pairs": sum(
                v["action_divergence_after_observation"] for v in diagnostics.values()),
            "unexplained_early_divergence_pairs": sum(
                v["unexplained_early_action_divergence"] for v in diagnostics.values()),
            "choice_counts": counts,
            "median_of_variant_p50_seconds": statistics.median(
                a["p50_seconds"] for a in arms if a["p50_seconds"] is not None),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--model-q8", type=Path)
    p.add_argument("--model-q4", type=Path)
    p.add_argument("--quant-receipt", type=Path)
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.seed != SEED or a.output.exists():
        raise ValueError("Held-out intervention seed changed or evidence would be overwritten")
    validate_flows()
    if any(k in FLOWS for k in LAB_FLOWS):
        raise ValueError("Lab flow collides with frozen parent")
    FLOWS.update(LAB_FLOWS)
    q4 = a.model_q4 is not None
    if (a.mode == "aero4_no2" and q4) or (a.mode != "aero4_no2" and not q4):
        raise ValueError("Specimen/quant selection does not match wave-9 request")
    q8_model = a.model_q8 if q4 else a.model
    if q8_model is None or sha256_file(q8_model) != PINS["model"]["sha256"]:
        raise ValueError("Q8 donor pin mismatch")
    if q4:
        if a.quant_receipt is None:
            raise ValueError("Missing Q4 conversion receipt")
        validate_quant_receipt(json.loads(a.quant_receipt.read_text()), q8_model, a.model_q4)
    w0 = paired.load_w0(a.w0)
    # Every specimen compares all six variants against same-runner stock.
    q8_stock = run_all(a.binary, q8_model, w0, "stock",
                       a.output / "q8-stock", a.seed)
    q4_stock = (run_all(a.binary, a.model_q4, w0, "stock",
                        a.output / "q4-stock", a.seed) if q4 else None)
    mutant = run_all(a.binary, a.model_q4 if q4 else q8_model,
                     w0, a.mode, a.output / "mutant", a.seed)
    base = q8_stock["summary"]["median_of_variant_p50_seconds"]
    current = mutant["summary"]["median_of_variant_p50_seconds"]
    report = {
        "schema": "frankenlayers-matched-geometry-counterfactual/1",
        "source_sha": a.source_sha,
        "mode": a.mode, "seed": a.seed, "fixed_family": FAMILY,
        "steps_per_variant": STEPS, "pairs": PAIRS,
        "immutable_w0_sha256": sha256_file(a.w0),
        "immutable_q8_sha256": PINS["model"]["sha256"],
        "q4_sha256": sha256_file(a.model_q4) if q4 else None,
        "q8_stock": q8_stock, "q4_stock": q4_stock, "mutant": mutant,
        "speedup_vs_q8_stock_p50_of_p50": base / current if current else None,
        "real_inference": True, "real_training": False,
        "game_code_modified": False, "native_workbench": False,
        "limitations": "Pair labels, seed and prompts held constant. Only synthetic geometry "
                       "varies. Behavioural divergence is not proof of competent navigation "
                       "or a causal internal representation. Server decoding may be nondeterministic.",
    }
    paired.write_json(a.output / "comparison.json", report)
    print(json.dumps({
        "mode": a.mode, "speedup": report["speedup_vs_q8_stock_p50_of_p50"],
        "summary": mutant["summary"],
        "pair_diagnostics": mutant["pairs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
