#!/usr/bin/env python3
"""SPROCKET v3 three-family research: novel Greeble/Architecture proposals only.

Source-compiler proposals are NOT admitted TRAIN, native Art, or model outputs.
This never reads FINAL and never modifies the original corpus.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

PAINTER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER_ROOT))
from compiler import canonical_json, parse_program, compile_program, structural_signature
from proposal_pipework import (
    HOST_SEEDS, NATIVE_SOURCE, NATIVE_MANIFEST_SHA, ORIGINAL_TRAIN_SHA,
    ORIGINAL_DEV_SHA, ORIGINAL_COMPILER_SHA, file_sha, public_split_signatures,
)

STUDIES = {
    "G08": {
        "description": "Five unequal manufacturing cassettes in a stepped stagger; deliberately open lower-right void; varied ribs/vent/grille/plates/inset detailing.",
        "family": "greebles",
        "program": {
            "version": 2, "family": "greebles", "variation": 0,
            "components": [
                {"kind": "panel", "box": [8, 8, 35, 23], "detail": "vent", "density": 2, "border": 1, "fasteners": True},
                {"kind": "panel", "box": [56, 8, 26, 22], "detail": "inset", "density": 2, "border": 1, "fasteners": True},
                {"kind": "panel", "box": [9, 39, 23, 21], "detail": "grille", "density": 2, "border": 1, "fasteners": True},
                {"kind": "panel", "box": [44, 39, 38, 22], "detail": "plates", "density": 3, "border": 1, "fasteners": True},
                {"kind": "panel", "box": [11, 69, 41, 19], "detail": "ribs", "density": 2, "border": 1, "fasteners": True},
            ],
            "requirements": {"negative_space_min_percent": 60},
        },
    },
    "G09": {
        "description": "Three wide descending manufacturing bands of unequal lengths and left margins with uninterrupted horizontal service gaps; no central spine or four-around-void motif.",
        "family": "greebles",
        "program": {
            "version": 2, "family": "greebles", "variation": 0,
            "components": [
                {"kind": "panel", "box": [7, 13, 78, 19], "detail": "ribs", "density": 3, "border": 1, "fasteners": True},
                {"kind": "panel", "box": [15, 42, 61, 17], "detail": "vent", "density": 2, "border": 1, "fasteners": True},
                {"kind": "panel", "box": [34, 70, 52, 18], "detail": "plates", "density": 3, "border": 1, "fasteners": True},
            ],
            "requirements": {"negative_space_min_percent": 60},
        },
    },
    "A07": {
        "description": "One flat orthographic hall with two separate low mezzanine platforms on vertical posts, clear high side-door corridor and geometric three-cell route; not stairs, bridge, or paired rooms.",
        "family": "architecture",
        "program": {
            "version": 2, "family": "architecture", "variation": 0,
            "components": [
                {"kind": "room", "box": [7, 10, 82, 78], "thickness": 2, "openings": [
                    {"side": "left", "offset": 35, "size": 17},
                    {"side": "right", "offset": 35, "size": 17},
                ]},
                {"kind": "platform", "box": [19, 66, 21, 3]},
                {"kind": "platform", "box": [54, 66, 22, 3]},
                {"kind": "support", "box": [28, 69, 3, 16]},
                {"kind": "support", "box": [63, 69, 3, 16]},
            ],
            "requirements": {"route": [[13, 55], [82, 55]], "clearance": 3, "region": [0, 0, 96, 96]},
        },
    },
    "A08": {
        "description": "Three discrete side-elevation rooms in an explicit ground-level linear passage, each with two face-on side doors and continuous three-cell-clear horizontal corridor; no elevated bridge.",
        "family": "architecture",
        "program": {
            "version": 2, "family": "architecture", "variation": 0,
            "components": [
                {"kind": "room", "box": [5, 19, 28, 68], "thickness": 2, "openings": [
                    {"side": "left", "offset": 37, "size": 15},
                    {"side": "right", "offset": 37, "size": 15},
                ]},
                {"kind": "room", "box": [34, 19, 28, 68], "thickness": 2, "openings": [
                    {"side": "left", "offset": 37, "size": 15},
                    {"side": "right", "offset": 37, "size": 15},
                ]},
                {"kind": "room", "box": [63, 19, 28, 68], "thickness": 2, "openings": [
                    {"side": "left", "offset": 37, "size": 15},
                    {"side": "right", "offset": 37, "size": 15},
                ]},
            ],
            "requirements": {"route": [[12, 63], [83, 63]], "clearance": 3, "region": [0, 0, 96, 96]},
        },
    },
}

def program_for(identifier: str) -> dict:
    return deepcopy(STUDIES[identifier]["program"])

def evaluate(corpus: Path) -> dict:
    if file_sha(PAINTER_ROOT / "compiler.py") != ORIGINAL_COMPILER_SHA:
        raise ValueError("frozen strict compiler source changed")
    # Verifies original corpus manifest/source and only original TRAIN/DEVELOPMENT
    # split bytes. The sealed FINAL file is intentionally never accessed.
    known = public_split_signatures(corpus)
    seen = set()
    candidates = []
    for identifier, row in STUDIES.items():
        program = program_for(identifier)
        if parse_program(canonical_json(program)) != program:
            raise ValueError(f"{identifier}: strict schema round trip failed")
        signature = canonical_json(structural_signature(program))
        if signature in known:
            raise ValueError(f"{identifier}: original TRAIN/DEVELOPMENT structural-signature collision")
        if signature in seen:
            raise ValueError(f"{identifier}: within-stage structural-signature duplicate")
        seen.add(signature)
        views = []
        for seed in HOST_SEEDS:
            compiled = compile_program(program, seed=seed)
            if compiled["family"] != row["family"] or not compiled["structural_preflight"]["passed"]:
                raise ValueError(f"{identifier}: semantic screen failure")
            if compiled["proposal_only"] is not True or compiled["native_qualified"] is not False:
                raise ValueError(f"{identifier}: compiler improperly claims native admission")
            if compiled["operation_count"] > 512 or compiled["estimated_cell_work"] > 100000:
                raise ValueError(f"{identifier}: bounded operation violation")
            m = compiled["structural_preflight"]["metrics"]
            if row["family"] == "architecture":
                if m["room_count"] not in (1, 3) or m["route_requested"] is not True or m["clearance_route_passed"] is not True:
                    raise ValueError(f"{identifier}: orthographic clearance or room-group witness failed")
            elif m["panel_count"] not in (3, 5) or m["detail_kinds"] < 3:
                raise ValueError(f"{identifier}: panel diversity witness failed")
            views.append({
                "host_seed": seed, "operations_sha256": compiled["operations_sha256"],
                "operation_count": compiled["operation_count"],
                "estimated_cell_work": compiled["estimated_cell_work"],
                "structural_metrics": m,
            })
        candidates.append({
            "id": identifier, "family": row["family"], "description": row["description"],
            "program_sha256": sha256(canonical_json(program).encode()).hexdigest(),
            "structural_signature_sha256": sha256(signature.encode()).hexdigest(),
            "views": views, "compiler_only": True,
            "native_positive": False, "shared_train_admitted": False,
        })
    return {
        "schema": "cybersand.sprocket-threefamily-prospective-source-screen/1",
        "status": "COMPILER_ONLY_REVIEW_PENDING_NO_TRAIN",
        "original_native_source_sha": NATIVE_SOURCE,
        "original_corpus_manifest_sha256": NATIVE_MANIFEST_SHA,
        "original_train_sha256": ORIGINAL_TRAIN_SHA,
        "original_development_sha256": ORIGINAL_DEV_SHA,
        "compiler_sha256": ORIGINAL_COMPILER_SHA,
        "native_cases_executed": 0, "training_updates": 0,
        "model_generated_views": 0, "final_targets_opened": False,
        "candidate_compositions": len(candidates),
        "compiler_seed_views": sum(len(x["views"]) for x in candidates),
        "shared_train_groups_admitted": 0,
        "candidate_groups": candidates,
        "next_gate": "Independent cross-lane/spatial/near-heldout review, then source-built two-seed Art readback/PNG receipts, then shepherd's explicit split admission.",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite of research screening evidence")
    report = evaluate(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "compositions": report["candidate_compositions"], "seed_views": report["compiler_seed_views"], "native": 0, "TRAIN": 0}, sort_keys=True))

if __name__ == "__main__":
    main()
