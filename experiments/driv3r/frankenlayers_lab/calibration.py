#!/usr/bin/env python3
"""Deterministic model-only game-prompt calibration for IQ3_XS; never use evaluation worlds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROWS = 96
CHUNKS = 8
MATERIALS = ("EMPTY", "SAND", "STONE", "WATER", "METAL", "GLASS", "GAS")
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
SEEDS = (101, 103, 107, 109)  # deliberately disjoint from evaluation seeds 570/571

def build_calibration() -> str:
    lines = []
    for i in range(ROWS):
        seed = SEEDS[i % len(SEEDS)]
        grid = []
        for y in range(5):
            row = []
            for x in range(5):
                value = (x * 7 + y * 11 + i * 13 + seed) % 23
                row.append("P" if value == 0 else "#" if value < 5
                           else MATERIALS[(value + i) % len(MATERIALS)][0])
            grid.append("".join(row))
        # Exercise protected cells and bounded legal menus, never introduce an illegal action.
        legal = [a for j, a in enumerate(ACTIONS) if a == "STOP" or
                 (i + seed + 3 * j) % 7 not in (0, 1)]
        state = {
            "identity": "small slightly deranged CyberSand drawing worm",
            "objective": "continue a legal and distinctive stroke across materials; avoid protected cells",
            "seed": seed, "family": "CALIBRATION_ONLY", "tick": i % 32,
            "x": (3 * i + 1) % 29, "y": (5 * i + 3) % 23,
            "heading": ("N", "E", "S", "W")[i % 4],
            "local_5x5": grid, "last_actions": [ACTIONS[(i + j) % 5] for j in range(i % 4)],
            "legal_actions": legal, "protected_cells": "P", "blocked_cells": "#",
            "painting_material": MATERIALS[(i + 2) % len(MATERIALS)],
        }
        question = {"worm_move": {"type": "choice", "instructions":
                    "Choose one legal action. Branch, retrace and recover are permitted; never enter protected cells.",
                    "criteria": {a: a + " is available" for a in legal}}}
        lines.append("<|startoftext|><|im_start|>user\n" +
                     json.dumps({"state": state, "question": question},
                                separators=(",", ":"), sort_keys=True) +
                     "\n<|im_end|>")
    return "\n".join(lines) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Calibration text is immutable; refusing overwrite")
    data = build_calibration()
    assert len(data.encode()) > 15_000 and data.count("<|im_start|>") == ROWS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data.encode("utf-8"))
    print(json.dumps({"schema": "frankenlayers-iq3-game-calibration/1",
                      "rows": ROWS, "chunks": CHUNKS, "seeds": SEEDS,
                      "sha256": hashlib.sha256(data.encode()).hexdigest(),
                      "bytes": len(data.encode()), "heldout_leakage": False},
                     sort_keys=True))

if __name__ == "__main__":
    main()
