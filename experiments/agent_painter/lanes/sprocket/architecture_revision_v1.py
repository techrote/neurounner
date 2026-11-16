#!/usr/bin/env python3
"""SPROCKET A09: independently authored orthographic split-height courtyard.

Source-compiler research ONLY. No model generation, no native admission,
no optimizer update. FINAL content is not opened, hashed as a member, or read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

PAINTER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER_ROOT))
from compiler import (canonical_json, parse_program, compile_program,
                      structural_signature)
from proposal_pipework import (
    HOST_SEEDS, NATIVE_SOURCE, NATIVE_MANIFEST_SHA, ORIGINAL_TRAIN_SHA,
    ORIGINAL_DEV_SHA, ORIGINAL_COMPILER_SHA, file_sha,
    public_split_signatures,
)
from threefamily_v3 import program_for as original_v3_program

IDENTIFIER = "A09"
DESCRIPTION = (
    "Strict perpendicular side elevation of an asymmetric split-height "
    "service courtyard: a low left workshop with two face-on side doors "
    "and a roof hatch, a raised right equipment bay with staggered side "
    "openings and an underside hatch, an exposed cantilevered service "
    "deck on two columns, and a short separate yard threshold. "
    "Maintain deliberate open air between the volumes. No three-bay "
    "repeated facade, single perimeter hall, bridge/tower clone, or "
    "isometric platform surfaces."
)
# Concept and layout are authored from an asymmetric lower-workshop /
# upper-service-pod grammar, NOT selected by optimizing against held-out
# DEVELOPMENT masks or coordinates. The two seeds remain one structure.
PROGRAM = {
    "version": 2, "family": "architecture", "variation": 1,
    "components": [
        {"kind": "room", "box": [8, 43, 31, 40], "thickness": 2,
         "openings": [
             {"side": "left", "offset": 20, "size": 10},
             {"side": "right", "offset": 20, "size": 10},
             {"side": "top", "offset": 9, "size": 9},
         ]},
        {"kind": "room", "box": [55, 10, 29, 38], "thickness": 2,
         "openings": [
             {"side": "left", "offset": 20, "size": 9},
             {"side": "right", "offset": 9, "size": 10},
             {"side": "bottom", "offset": 10, "size": 8},
         ]},
        {"kind": "platform", "box": [50, 51, 39, 3]},
        {"kind": "support", "box": [58, 54, 3, 26]},
        {"kind": "support", "box": [79, 54, 3, 26]},
        {"kind": "platform", "box": [39, 78, 15, 3]},
    ],
    # A lower-workshop interior to open-yard passage; 3x3 square-cell
    # free-space connectivity only, never gravity/jump/climb playability.
    "requirements": {
        "route": [[17, 68], [47, 68]],
        "clearance": 3,
        "region": [2, 2, 92, 92],
    },
}

def program_for(identifier: str = IDENTIFIER) -> dict:
    if identifier != IDENTIFIER:
        raise ValueError("unknown independently versioned A09 composition")
    return deepcopy(PROGRAM)

def evaluate(corpus: Path) -> dict:
    if file_sha(PAINTER_ROOT / "compiler.py") != ORIGINAL_COMPILER_SHA:
        raise ValueError("qualified common v2 compiler source changed")
    known = public_split_signatures(corpus)  # TRAIN/DEVELOPMENT only
    program = program_for()
    if parse_program(canonical_json(program)) != program:
        raise ValueError("strict canonical program parse changed A09")
    sig = canonical_json(structural_signature(program))
    if sig in known:
        raise ValueError("A09 collides with original visible structural signature")
    if sig in {
        canonical_json(structural_signature(original_v3_program(name)))
        for name in ("A07", "A08")
    }:
        raise ValueError("A09 relabels previous SPROCKET Architecture signature")
    components = program["components"]
    if ([c["kind"] for c in components] !=
        ["room", "room", "platform", "support", "support", "platform"]):
        raise ValueError("A09 asymmetrical two-volume grammar changed")
    if components[0]["box"][1] <= components[1]["box"][1]:
        raise ValueError("A09 low left workshop / elevated right bay inverted")
    if components[0]["box"][0] + components[0]["box"][2] >= components[1]["box"][0]:
        raise ValueError("A09 must retain an open courtyard between volumes")
    if [[o["side"] for o in c["openings"]] for c in components[:2]] != [
        ["left", "right", "top"], ["left", "right", "bottom"]
    ]:
        raise ValueError("A09 roof/underside hatch grammar changed")
    views = []
    for seed in HOST_SEEDS:
        c = compile_program(program, seed=seed, material=58,
                            allowed_materials=(58,), max_operations=512,
                            max_cell_work=100000)
        m = c["structural_preflight"]["metrics"]
        if (not c["structural_preflight"]["passed"]
            or c["family"] != "architecture"
            or not c["proposal_only"] or c["native_qualified"]
            or m["room_count"] != 2 or m["opening_count"] != 6
            or not m["route_requested"] or not m["clearance_route_passed"]
            or m["clearance_cells"] != 3):
            raise ValueError("A09 source compiler/clearance proof rejected")
        views.append({
            "seed": seed,
            "operations_sha256": c["operations_sha256"],
            "operation_count": c["operation_count"],
            "estimated_cell_work": c["estimated_cell_work"],
            "metrics": m,
            "native_positive": False,
        })
    return {
        "schema": "cybersand.sprocket-a09-architecture-source-screen/1",
        "disposition": "NEW_CONCEPT_COMPILER_ONLY_GLOBAL_REVIEW_PENDING",
        "candidate_id": IDENTIFIER,
        "description": DESCRIPTION,
        "source_proposal": "human-authored-spatial-revision-not-model-output",
        "native_source": NATIVE_SOURCE,
        "original_corpus_manifest_sha256": NATIVE_MANIFEST_SHA,
        "original_train_sha256": ORIGINAL_TRAIN_SHA,
        "original_development_sha256": ORIGINAL_DEV_SHA,
        "compiler_sha256": ORIGINAL_COMPILER_SHA,
        "program_sha256": sha256(canonical_json(program).encode()).hexdigest(),
        "signature_sha256": sha256(sig.encode()).hexdigest(),
        "design_count": 1,
        "compiler_seed_views": 2,
        "views": views,
        "native_executions": 0,
        "new_shared_train_groups": 0,
        "model_training_steps": 0,
        "model_generated_native_views": 0,
        "sealed_final_members_accessed": 0,
        "next_gate": "Independent whole-peer global source/silhouette review and separately source-built two-seed native Art evidence before shepherd-controlled TRAIN admission.",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("immutable source screen output already exists")
    report = evaluate(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": report["disposition"],
                      "designs": report["design_count"],
                      "compiler_views": report["compiler_seed_views"],
                      "native": 0, "TRAIN": 0}, sort_keys=True))

if __name__ == "__main__":
    main()
