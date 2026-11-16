#!/usr/bin/env python3
"""SPROCKET G09 local band-hierarchy witness, never TRAIN admission.

This validates one native-positive hand-authored proposal, not the independence
of its curriculum group and not an untuned/trained-model output. No FINAL access.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PAINTER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER_ROOT))

from compiler import canonical_json, compile_program, parse_program
from threefamily_v3 import HOST_SEEDS, evaluate, program_for

SOURCE_V3_BLOB = "b3ac98a8cd06158b39a2f58776fab845046f245f"


def check_hierarchy(program: dict) -> dict:
    """Fail closed on diagonal cassettes, compressed gaps or lost band ordering."""
    parse_program(canonical_json(program))  # enforce unchanged common v2 schema
    if program["family"] != "greebles" or len(program["components"]) != 3:
        raise ValueError("G09 requires exactly three greeble bands")
    panels = program["components"]
    if any(p["kind"] != "panel" for p in panels):
        raise ValueError("G09 requires only panels")
    if len({p["detail"] for p in panels}) != 3:
        raise ValueError("G09 requires three separate detail types")
    # Semantic order is spatial, not JSON component-array order.
    boxes = sorted((tuple(p["box"]) for p in panels), key=lambda b: b[1])
    if any(w <= 2 * h for _, _, w, h in boxes):
        raise ValueError("G09 panels must be wide horizontal bands, not cassettes")
    if any(w < 40 for _, _, w, _ in boxes):
        raise ValueError("G09 requires substantial horizontal panel spans")
    lefts = [x for x, _, _, _ in boxes]
    widths = [w for _, _, w, _ in boxes]
    gaps = [boxes[i + 1][1] - (boxes[i][1] + boxes[i][3])
            for i in range(2)]
    if any(g < 8 for g in gaps):
        raise ValueError("G09 requires two broad, clear horizontal service gaps")
    if not (lefts[0] <= 10
            and lefts[1] - lefts[0] >= 5
            and lefts[2] - lefts[1] >= 5):
        raise ValueError("G09 requires a stepped, strictly increasing left margin")
    if not (widths[0] > widths[1] > widths[2]):
        raise ValueError("G09 requires successively shorter descending bands")
    if boxes[0][1] < 5 or boxes[-1][1] + boxes[-1][3] > 90:
        raise ValueError("G09 requires bounded framing and lower service margin")
    return {
        "bands": 3, "row_top_y": [b[1] for b in boxes],
        "left_margins": lefts, "widths": widths,
        "heights": [b[3] for b in boxes],
        "inter_band_empty_rows": gaps,
        "distinct_details": 3,
        "orientation": "three-wide-horizontal-bands",
        "single_design": True, "global_independence_proven": False,
    }


def qualify(corpus: Path) -> dict:
    # Existing strict source/compiler audit revalidates the original public
    # native source's TRAIN/DEVELOPMENT partition, never opens FINAL targets.
    original = evaluate(corpus)
    if original["candidate_compositions"] != 4 or original["shared_train_groups_admitted"]:
        raise ValueError("immutable prospective group/split contract changed")
    p = program_for("G09")
    hierarchy = check_hierarchy(p)
    views = []
    for seed in HOST_SEEDS:
        c = compile_program(p, seed=seed)
        m = c["structural_preflight"]["metrics"]
        if not (c["proposal_only"] is True and c["native_qualified"] is False
                and m["panel_count"] == 3 and m["detail_kinds"] == 3
                and m["negative_space_percent"] >= 60
                and c["operation_count"] <= 512
                and c["estimated_cell_work"] <= 100000):
            raise ValueError("G09 strict compilation/safety/negative-space refusal")
        views.append({
            "host_seed": seed, "operations_sha256": c["operations_sha256"],
            "operation_count": c["operation_count"],
            "negative_space_percent": m["negative_space_percent"],
        })
    return {
        "schema": "cybersand.sprocket-g09-spatial-hierarchy-source-witness/1",
        "disposition": "LOCAL_SEMANTIC_WITNESS_ONLY_GLOBAL_REVIEW_HOLD",
        "source_v3_git_blob_sha": SOURCE_V3_BLOB,
        "native_producer_sha": "63ee8b2b561068a72e0652e54b6e4193cb47d5e6",
        "native_original_run": 43587600137,
        "native_original_artifact": 14111800549,
        "original_native_manifest_sha256": original["original_corpus_manifest_sha256"],
        "original_train_sha256": original["original_train_sha256"],
        "original_development_sha256": original["original_development_sha256"],
        "compiler_sha256": original["compiler_sha256"],
        "program_spatial_witness": hierarchy,
        "compiled_host_seed_views": views,
        "model_generated_views": 0,
        "new_native_views": 0, "optimizer_updates": 0,
        "new_shared_train_groups": 0, "final_targets_opened": False,
        "next_gate": "Independent global comparison of G09 to SMOL G06 and original visible TRAIN/DEVELOPMENT; no unilateral TRAIN admission.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite of immutable one-shot evidence")
    result = qualify(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"result": result["disposition"], "new_train": 0,
                      "new_native": 0, "verified_compiled_views": len(result["compiled_host_seed_views"])}))


if __name__ == "__main__":
    main()
