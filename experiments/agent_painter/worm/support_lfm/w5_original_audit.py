#!/usr/bin/env python3
"""WORM LFM W4->W5 comparison using ORIGINAL primary first-pass PNGs.

No model inference, optimizer, target repair or native/Godot calls here.
Replay remains owned by pinned primary_gallery_audit.py/W0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from primary_gallery_audit import audit
from curriculum import canonical, worm

FAMILIES = tuple(worm.FAMILIES)
POLICIES = ("model", "random", "wall", "oracle")
SEEDS = (2101, 2102)
BUDGET = 48


def gallery_manifest(folder: Path) -> dict:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("missing/link gallery")
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.is_symlink())
    if len(files) != 54 or len(list(folder.iterdir())) != 54:
        raise ValueError("expected precisely 54 original gallery files")
    lines = [p.name + "\t" + hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    return {"count": len(files),
            "sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()}


def compare_reports(before: list[dict], after: list[dict]) -> dict:
    if len(before) != 2 or len(after) != 2:
        raise ValueError("requires two exact independent obstacle seeds per version")
    cases = []
    seen = set()
    family_images = {"W4": {}, "W5": {}}
    for version, reports in (("W4", before), ("W5", after)):
        if [r.get("seed") for r in reports] != list(SEEDS):
            raise ValueError("unmatched or duplicate world seeds")
        for report in reports:
            if (report.get("schema") != "worm-lfm-primary-gallery-audit-v2"
                    or report.get("budget") != BUDGET
                    or len(report.get("metrics", [])) != 12
                    or len(report.get("matched_pairs", [])) != 9):
                raise ValueError("incompatible source-fenced gallery report")
            by_key = {(q["family"], q["policy"]): q for q in report["metrics"]}
            if len(by_key) != 12 or set(by_key) != {
                    (f, p) for f in FAMILIES for p in POLICIES}:
                raise ValueError("missing or duplicate family/policy evidence")
            for fam in FAMILIES:
                model = by_key[fam, "model"]
                if model["raw_decisions"] != BUDGET or model["genuine_eos"] != BUDGET:
                    raise ValueError("incomplete original first-pass output")
                family_images[version].setdefault(report["seed"], []).append(
                    model["original_png_sha256"])
                k = (version, report["seed"], fam)
                if k in seen:
                    raise ValueError("duplicate model result")
                seen.add(k)
    for old, new in zip(before, after):
        a = {(q["family"], q["policy"]): q for q in old["metrics"]}
        b = {(q["family"], q["policy"]): q for q in new["metrics"]}
        seed = old["seed"]
        for fam in FAMILIES:
            # Distinct policy runs MUST share exactly the same W0 baseline world
            # between W4/W5. A changed obstacle/seed/control invalidates comparison.
            for control in ("random", "wall", "oracle"):
                if (a[fam, control]["original_png_sha256"] !=
                        b[fam, control]["original_png_sha256"] or
                        a[fam, control]["metrics"] != b[fam, control]["metrics"]):
                    raise ValueError("W4/W5 comparison control drift")
            m4 = a[fam, "model"]; m5 = b[fam, "model"]
            cases.append({
                "seed": seed, "family": fam, "decisions_each": BUDGET,
                "W4_unique_cells": m4["metrics"]["unique_cells"],
                "W5_unique_cells": m5["metrics"]["unique_cells"],
                "delta_unique_cells": m5["metrics"]["unique_cells"] - m4["metrics"]["unique_cells"],
                "random_unique_cells": b[fam, "random"]["metrics"]["unique_cells"],
                "wall_unique_cells": b[fam, "wall"]["metrics"]["unique_cells"],
                "W4_invalid_raw": m4["raw_rejected_or_missing_eos"],
                "W5_invalid_raw": m5["raw_rejected_or_missing_eos"],
                "W4_fallbacks": m4["metrics"]["host_fallback_count"],
                "W5_fallbacks": m5["metrics"]["host_fallback_count"],
                "W4_revisits": m4["metrics"]["revisits"],
                "W5_revisits": m5["metrics"]["revisits"],
                "W4_raw_action_counts": m4["metrics"]["raw_action_counts"],
                "W5_raw_action_counts": m5["metrics"]["raw_action_counts"],
                "W4_png_sha256": m4["original_png_sha256"],
                "W5_png_sha256": m5["original_png_sha256"],
            })
    return {
        "schema": "worm-lfm-w4-w5-original-two-world-comparison-v1",
        "native": False, "optimizer_updates_in_support": 0,
        "world_seeds": list(SEEDS), "family_requests_per_world": len(FAMILIES),
        "per_policy_budget": BUDGET, "model_decisions_per_version": 6 * BUDGET,
        "cases": cases,
        "family_model_pngs_identical": {
            v: {str(seed): len(set(hashes)) == 1
                for seed, hashes in by_seed.items()}
            for v, by_seed in family_images.items()},
        "worlds_not_independent_by_family": True,
        "note": "Model policy first-pass/PNG custody is from original producers. "
                "Every original host trace and PNG is replayed by W0 audit. "
                "Coverage is not an aesthetic score; no native Godot cell readback."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w4-gallery", type=Path, required=True)
    parser.add_argument("--w5-gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    w4 = [audit(args.w4_gallery, seed=s, budget=BUDGET) for s in SEEDS]
    w5 = [audit(args.w5_gallery, seed=s, budget=BUDGET) for s in SEEDS]
    output = compare_reports(w4, w5)
    output["gallery_manifest"] = {
        "W4": gallery_manifest(args.w4_gallery),
        "W5": gallery_manifest(args.w5_gallery)}
    args.out.write_bytes(canonical(output))
    print(json.dumps({"schema": output["schema"], "cases": len(output["cases"]),
                      "world_seeds": len(SEEDS), "model_decisions_per_version": 288},
                     sort_keys=True))


if __name__ == "__main__":
    main()
