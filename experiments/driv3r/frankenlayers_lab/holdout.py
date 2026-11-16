#!/usr/bin/env python3
"""Research-only held-out W0 geometry stress test for sparse d1 game policies.

This is NOT CyberSand native gameplay. It keeps the source-pinned W0 legal-action
implementation unchanged and overlays synthetic obstacles in memory only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.driv3r.backend import PINS, sha256_file
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
from experiments.driv3r.frankenlayers import infer as paired
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows
from experiments.driv3r.frankenlayers_lab.infer_quant import validate_quant_receipt

# New, deliberately held-out seed; none of the six exploration waves used it.
HOLDOUT_SEED = 1741
STEPS = 72
SIZE = 96
CASES = {
    "greebles": "cul_de_sac",
    "pipework": "alternating_gates",
    "architecture": "protected_islands",
}
ELIGIBLE = ("aero4_no10", "aero4_no2", "aero5_no23", "aerohound")


def overlay_obstacles(base: set[tuple[int, int]], family: str, size: int) -> set[tuple[int, int]]:
    """Scenario geometry; blocked=host-impassable, never game material semantics."""
    if family not in CASES or size != SIZE:
        raise ValueError("Unsupported held-out challenge")
    center = size // 2
    blocks = set(base)
    # Remove baseline random obstruction only in the controlled challenge area.
    blocks.difference_update((x, y) for x in range(4, 33)
                             for y in range(center - 16, center + 17))
    if family == "greebles":
        # Short tunnel with a blind east wall and narrow north/south escape routes.
        blocks.update((x, center - 2) for x in range(9, 15))
        blocks.update((x, center + 2) for x in range(9, 15))
        blocks.update((14, y) for y in range(center - 2, center + 3))
    elif family == "pipework":
        # Three offset gates. A fixed heading policy reaches blockages.
        for x, gate in ((12, center + 7), (19, center - 7), (26, center + 6)):
            blocks.update((x, y) for y in range(center - 10, center + 11) if y != gate)
    else:
        # Protected/material-like islands are represented by blocked host cells.
        # This does NOT test real CyberSand material reactions.
        blocks.update((x, y) for x in range(11, 15) for y in range(center - 3, center + 4))
        blocks.update((x, y) for x in range(19, 24) for y in range(center + 2, center + 8))
        blocks.update((x, y) for x in range(23, 28) for y in range(center - 9, center - 5))
    if (7, center) in blocks:
        raise AssertionError("Challenge spawn obstructed")
    if any(not (0 <= x < size and 0 <= y < size) for x, y in blocks):
        raise AssertionError("Challenge obstacle out of bounds")
    return blocks


class ChallengeHost:
    """Run pinned W0 unchanged except a temporary in-memory initial_world overlay."""
    def __init__(self, w0):
        self.w0 = w0

    def run(self, seed, *, mode, family, steps, selector):
        # The pinned parent infer.run_arm() passes seed positionally. Mirror
        # that exact call contract; all remaining arguments stay keyword-only.
        if seed != HOLDOUT_SEED or steps != STEPS or mode != "callback":
            raise ValueError("Held-out episode contract changed")
        original = self.w0.initial_world

        def initial_world(seed_arg, size=SIZE):
            return overlay_obstacles(original(seed_arg, size), family, size)

        self.w0.initial_world = initial_world
        try:
            record = self.w0.run(seed=seed, mode=mode, family=family, steps=steps,
                                 size=SIZE, selector=selector)
        finally:
            self.w0.initial_world = original
        record["research_challenge"] = {
            "label": CASES[family],
            "schema": "frankenlayers-heldout-geometry/1",
            "host_changes": "initial obstacle overlay only; same W0 host legality",
            "native_game": False,
            "true_material_reactions": False,
        }
        return record

    def png_bytes(self, record):
        return self.w0.png_bytes(record)


def summarize(arm: dict) -> dict:
    cases = arm["experiments"]
    decisions = sum(e["decisions"] for e in cases)
    new = sum(e["metrics"]["new_steps"] for e in cases)
    stopped = sum(bool(e["metrics"]["terminated_by_stop"]) for e in cases)
    revisits = sum(e["metrics"]["total_revisit_contacts"] for e in cases)
    # Scene-specific action sequences are required; mere choice-type variety
    # from a fixed attractor is not equivalent to environmental sensitivity.
    distinct_sequences = len({tuple(e["raw_choices"]) for e in cases})
    return {
        "decisions": decisions,
        "episodes": len(cases),
        "full_length_episodes": sum(e["decisions"] == STEPS for e in cases),
        "early_stop_episodes": stopped,
        "new_steps": new,
        "new_steps_per_decision": new / decisions if decisions else None,
        "revisit_contacts": revisits,
        "distinct_world_action_sequences": distinct_sequences,
        "legal_choice_count": arm["valid_decisions"],
        "invalid_host_choices": sum(e["metrics"]["invalid_raw_choices"] for e in cases),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path)          # Q8 comparison
    p.add_argument("--model-q8", type=Path)       # Q4 three-way comparison
    p.add_argument("--model-q4", type=Path)
    p.add_argument("--quant-receipt", type=Path)
    p.add_argument("--w0", type=Path, required=True)
    p.add_argument("--mode", choices=ELIGIBLE, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--source-sha", required=True)
    a = p.parse_args()
    if a.seed != HOLDOUT_SEED:
        raise ValueError("Hold-out seed must remain independent from W0 exploration")
    validate_flows()
    if any(name in FLOWS for name in LAB_FLOWS):
        raise RuntimeError("Phase2/lab flow name collision")
    FLOWS.update(LAB_FLOWS)
    q4 = a.model_q4 is not None
    if a.mode == "aerohound" and q4:
        raise ValueError("No unqualified eight-layer Q4 mixing")
    if a.output.exists():
        raise RuntimeError("Refusing to overwrite original evidence")
    q8_path = a.model_q8 if q4 else a.model
    if q8_path is None or sha256_file(q8_path) != PINS["model"]["sha256"]:
        raise ValueError("Immutable Q8 model is missing/altered")
    if q4:
        if a.quant_receipt is None:
            raise ValueError("Q4 receipt required")
        validate_quant_receipt(json.loads(a.quant_receipt.read_text()), q8_path, a.model_q4)
    w0 = ChallengeHost(paired.load_w0(a.w0))
    old_steps, old_families = paired.STEPS, paired.FAMILIES
    try:
        paired.STEPS, paired.FAMILIES = STEPS, tuple(CASES)
        q8_stock = paired.run_arm(a.binary, q8_path, w0, "stock", a.output / "q8-stock", a.seed)
        q4_stock = (paired.run_arm(a.binary, a.model_q4, w0, "stock", a.output / "q4-stock", a.seed)
                    if q4 else None)
        mutant_model = a.model_q4 if q4 else q8_path
        mutant = paired.run_arm(a.binary, mutant_model, w0, a.mode, a.output / "mutant", a.seed)
    finally:
        paired.STEPS, paired.FAMILIES = old_steps, old_families
    stock_p50, mutant_p50 = q8_stock["p50_seconds"], mutant["p50_seconds"]
    report = {
        "schema": "cybersand-frankenlayers-heldout-geometry/1",
        "source_sha": a.source_sha, "mode": a.mode, "seed": a.seed, "steps": STEPS,
        "case_mapping": CASES, "w0_sha256": sha256_file(a.w0),
        "q8_sha256": PINS["model"]["sha256"], "q4": q4,
        "q4_model_sha256": sha256_file(a.model_q4) if q4 else None,
        "q8_stock": q8_stock, "q4_stock": q4_stock, "mutant": mutant,
        "summary": {
            "q8_stock": summarize(q8_stock),
            "q4_stock": summarize(q4_stock) if q4_stock else None,
            "mutant": summarize(mutant),
        },
        "speedup_p50_vs_q8_stock": stock_p50 / mutant_p50 if mutant_p50 else None,
        "speedup_p50_vs_q4_stock": (
            q4_stock["p50_seconds"] / mutant_p50 if q4_stock and mutant_p50 else None),
        "real_model_inference": True, "training": False, "native_workbench": False,
        "game_code_modified": False,
        "limits": "Synthetic W0 obstacle overlays; protected cells only as impassable masks. "
                  "No real material chemistry, goal completion or native simulation.",
    }
    paired.write_json(a.output / "comparison.json", report)
    print(json.dumps({
        "mode": a.mode, "q4": q4, "mutant_p50": mutant_p50,
        "speedup": report["speedup_p50_vs_q8_stock"],
        "mutant_summary": report["summary"]["mutant"],
        "choice_counts": mutant["choice_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
