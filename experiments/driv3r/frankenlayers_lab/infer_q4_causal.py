#!/usr/bin/env python3
"""Held-out matched-geometry Q4_K_M triad for sparse d1 CyberSand-style policies.

Research-only W0 host. Model calibration uses disjoint synthetic game prompts;
these six obstacle interventions are strictly evaluation-only. All three arms
run on the *same* CPU runner and pinned patched llama.cpp runtime.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.frankenlayers import infer as paired
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers_lab.counterfactual import (
    FAMILY, PAIRS, SEED, STEPS, run_all,
)
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows
from experiments.driv3r.frankenlayers_lab.infer_quant import validate_quant_receipt

MODES = ("aero4_no2", "aero4_bridge", "aero4_no2_tailflip", "aero5_no2_echo10")

ACTION_COUNT = len(paired.ACTION_DESCRIPTIONS)
MAX_DECISIONS = STEPS * 2 * len(PAIRS)


def pareto_axis(arm: dict) -> dict:
    """Keep independent objectives separate; do not bury STOPs in a speed score."""
    s = arm["summary"]
    choices = s["choice_counts"]
    decisions = s["decisions"]
    entropy = -sum(
        (n / decisions) * math.log2(n / decisions)
        for n in choices.values() if n and decisions
    ) if decisions else 0.0
    exposed = s["geometry_exposed_pairs"]
    return {
        "decisions": decisions,
        "decision_budget": MAX_DECISIONS,
        "legal_choice_fraction": (
            (decisions - s["invalid_host_choices"]) / decisions
            if decisions else None
        ),
        "new_cells": s["new_cells"],
        "new_cells_per_decision": s["new_cells"] / decisions if decisions else None,
        "new_cells_per_budget": s["new_cells"] / MAX_DECISIONS,
        "early_stop_episodes": s["early_stop_episodes"],
        "exposed_geometry_pairs": exposed,
        "responded_exposed_pairs": s["observation_followed_by_action_divergence_pairs"],
        "unexplained_early_divergence_pairs": s["unexplained_early_divergence_pairs"],
        "action_entropy_bits": entropy,
        "normalised_action_entropy": entropy / math.log2(ACTION_COUNT),
        "p50_of_variant_p50_seconds": s["median_of_variant_p50_seconds"],
    }


def make_report(source_sha: str, mode: str, receipt: dict, w0_sha256: str,
                q8_stock: dict, q4_stock: dict, mutant: dict) -> dict:
    axes = {
        "q8_stock": pareto_axis(q8_stock),
        "q4_stock": pareto_axis(q4_stock),
        "q4_mutant": pareto_axis(mutant),
    }
    fast = axes["q4_mutant"]["p50_of_variant_p50_seconds"]
    def ratio(x):
        return x / fast if x is not None and fast else None
    return {
        "schema": "frankenlayers-q4-matched-geometry/1",
        "source_sha": source_sha, "mode": mode, "seed": SEED,
        "flow": list(LAB_FLOWS[mode]),
        "fixed_family": FAMILY, "steps_per_variant": STEPS,
        "interventions": PAIRS,
        "quant_receipt": receipt, "immutable_w0_sha256": w0_sha256,
        "immutable_q8_sha256": PINS["model"]["sha256"],
        "arms": {"q8_stock": q8_stock, "q4_stock": q4_stock, "q4_mutant": mutant},
        "pareto_axes": axes,
        "ratio_vs_q8_stock_per_decision": ratio(axes["q8_stock"]["p50_of_variant_p50_seconds"]),
        "ratio_vs_q4_stock_per_decision": ratio(axes["q4_stock"]["p50_of_variant_p50_seconds"]),
        "real_inference": True, "real_training": False,
        "native_workbench": False, "game_code_modified": False,
        "limitations": (
            "Fixed-label paired synthetic W0 obstacle interventions only; Q4_K_M is lossy "
            "requantisation from immutable Q8, not FP16. Observation-followed-by-action divergence "
            "is evidence of geometry sensitivity, not goal competence or model-internal "
            "causality. Per-decision speed ratios do not imply whole-episode speed."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model-q8", type=Path, required=True)
    p.add_argument("--model-q4", type=Path, required=True)
    p.add_argument("--quant-receipt", type=Path, required=True)
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    if a.seed != SEED or a.output.exists():
        raise ValueError("Held-out seed changed or evidence would be overwritten")
    validate_flows()
    if any(name in FLOWS for name in LAB_FLOWS):
        raise ValueError("Lab flow collides with pinned parent mode registry")
    receipt = json.loads(a.quant_receipt.read_text())
    validate_quant_receipt(receipt, a.model_q8, a.model_q4)
    FLOWS.update(LAB_FLOWS)
    w0 = paired.load_w0(a.w0)
    a.output.mkdir(parents=True)

    # One identical sequence of six geometry variants per arm, on the same
    # machine/runtime/seed. Q4 stock isolates quantisation from surgery.
    q8_stock = run_all(a.binary, a.model_q8, w0, "stock",
                       a.output / "q8-stock", a.seed)
    q4_stock = run_all(a.binary, a.model_q4, w0, "stock",
                        a.output / "q4-stock", a.seed)
    mutant = run_all(a.binary, a.model_q4, w0, a.mode,
                     a.output / "q4-mutant", a.seed)
    report = make_report(a.source_sha, a.mode, receipt, sha256_file(a.w0),
                         q8_stock, q4_stock, mutant)
    paired.write_json(a.output / "comparison.json", report)
    print(json.dumps({
        "mode": a.mode, "seed": SEED,
        "q4_mutant_pareto": report["pareto_axes"]["q4_mutant"],
        "q8_stock_pareto": report["pareto_axes"]["q8_stock"],
        "q4_stock_pareto": report["pareto_axes"]["q4_stock"],
        "speedup_vs_q8": report["ratio_vs_q8_stock_per_decision"],
        "speedup_vs_q4": report["ratio_vs_q4_stock_per_decision"],
        "pairs": mutant["pairs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
