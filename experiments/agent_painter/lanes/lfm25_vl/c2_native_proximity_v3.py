#!/usr/bin/env python3
"""C2c source-only width-3 pipe topology and original visible-native near-neighbour screen.

No FINAL inspection, model call, native Art, new TRAIN admission, hard numeric
approval threshold or change to C2/C2b. Reviewer/shepherd decide split safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import c2_stepped_research_v2 as v2
import contrastive_preflight as legacy

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "c2_offset_proposal_v3.json"
STATUS = "REVIEW_ONLY_SOURCE_SCREEN_NO_NATIVE_NO_TRAIN"
SCHEMA = "cybersand.lfm25_vl.c2-offset-research/3"
C2B_SHA = "100dd9b188c2b2c96ab5d3680a929606b1dfd69f40cb524730f4da8915b739b2"
FOCUS = (0, 0, 96, 96)


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def load_proposal(path: Path = PROPOSAL) -> dict:
    require(path.is_file() and not path.is_symlink(), "ordinary proposal file required")
    doc = json.loads(path.read_text(encoding="utf-8"))
    require(set(doc) == {
        "schema", "status", "revision_of", "id", "original_c2_source_blob",
        "c2b_proposal_sha256", "original_native_source", "original_manifest_sha256",
        "host_seeds", "request", "witness", "program"
    }, "unreviewed C2c document shape")
    require(doc["schema"] == SCHEMA and doc["status"] == STATUS,
            "C2c is research only, never native/TRAIN")
    require(doc["revision_of"] == "lfm-c2b-offset-stepped-two-cross-six-terminal-network"
            and doc["id"] == "lfm-c2c-wide-asymmetric-bridge-two-cross-six-terminal-network",
            "C2c identity or lineage changed")
    require(doc["original_c2_source_blob"] == v2.LEGACY_BLOB
            and doc["c2b_proposal_sha256"] == C2B_SHA
            and legacy.sha(v2.PROPOSAL) == C2B_SHA,
            "prior research identity changed")
    require(doc["original_native_source"] == legacy.SOURCE
            and doc["original_manifest_sha256"] == legacy.MANIFEST
            and doc["host_seeds"] == list(legacy.SEEDS),
            "original native corpus/host-seed identity changed")
    require(isinstance(doc["request"], str) and len(doc["request"]) >= 70
            and isinstance(doc["witness"], str) and len(doc["witness"]) >= 70,
            "substantive textual review specification required")
    return doc


def centreline(program: dict) -> set[tuple[int, int]]:
    part = program["components"][0]
    pixels = set()
    for a, b in part["edges"]:
        x1, y1 = part["nodes"][a]
        x2, y2 = part["nodes"][b]
        require(x1 == x2 or y1 == y2, "nonorthogonal pipe edge")
        if x1 == x2:
            pixels.update((x1, y) for y in range(min(y1, y2), max(y1, y2) + 1))
        else:
            pixels.update((x, y1) for x in range(min(x1, x2), max(x1, x2) + 1))
    require(bool(pixels) and all(0 <= x < 96 and 0 <= y < 96 for x, y in pixels),
            "out-of-focus or empty centreline")
    return pixels


def coverage(centres: set[tuple[int, int]], native: frozenset[tuple[int, int]],
             radius: int = 3) -> dict:
    require(centres and native and radius == 3, "nonempty original mask/radius 3 required")
    near = sum(any((x + dx, y + dy) in native
                   for dx in range(-radius, radius + 1)
                   for dy in range(-radius, radius + 1))
               for x, y in centres)
    exact = len(centres & native)
    return {
        "candidate_centreline_pixels": len(centres),
        "within_chebyshev_3_count": near,
        "within_chebyshev_3_percent": round(100 * near / len(centres), 4),
        "exact_centreline_to_native_pixels": exact,
        "comparator_native_material_cells": len(native),
    }


def visible_native_masks(corpus: Path):
    """Use explicit SHA-verified native readbacks from ORIGINAL TRAIN/DEVELOPMENT only."""
    legacy.original_visible_splits(corpus)  # Verifies source633, manifest and both split hashes.
    out = []
    for split in ("TRAIN", "DEVELOPMENT"):
        rows = (corpus / (split.lower() + ".jsonl")).read_text(encoding="utf-8").splitlines()
        for line in rows:
            row = json.loads(line)
            if row["family"] != "pipework":
                continue
            ident, seed = row["id"], row["seed"]
            require(re.fullmatch(r"pipework-[a-z0-9_]+-v1-s[0-9]+", ident) is not None
                    and seed in legacy.SEEDS and ident.endswith("-s" + str(seed)),
                    "unexpected group/case path")
            admission = row["native_admission"]
            require(row["native_qualified"] is True and admission["state"] == "PASS"
                    and admission["verified"] is True
                    and admission["source_sha"] == legacy.SOURCE,
                    "original native admission not positive")
            paths = admission["paths"]
            require(set(paths) == {"png", "readback", "receipt"},
                    "native evidence custody shape mismatch")
            require(paths["readback"] == "evidence/" + ident + ".materials.i32le"
                    and paths["receipt"] == "evidence/" + ident + ".receipt.json",
                    "native readback path not original")
            raw_path = corpus / paths["readback"]
            receipt_path = corpus / paths["receipt"]
            require(raw_path.is_file() and not raw_path.is_symlink()
                    and receipt_path.is_file() and not receipt_path.is_symlink(),
                    "original evidence missing/symlink")
            raw, receipt_bytes = raw_path.read_bytes(), receipt_path.read_bytes()
            raw_sha = hashlib.sha256(raw).hexdigest()
            require(raw_sha == admission["readback_sha256"]
                    and hashlib.sha256(receipt_bytes).hexdigest() == admission["receipt_sha256"]
                    and len(raw) == 96 * 96 * 4,
                    "original native grid or receipt hash mismatch")
            receipt = json.loads(receipt_bytes)
            values = struct.unpack("<9216i", raw)
            require(set(values).issubset({0, 58}) and receipt["passed"] is True
                    and int(receipt["seed"]) == seed
                    and receipt["crop_i32le_sha256"] == raw_sha,
                    "native material/receipt inconsistent")
            native = frozenset((i % 96, i // 96) for i, val in enumerate(values) if val == 58)
            require(len(native) == receipt["filled_cells"]
                    and len(native) == admission["acknowledged_cells"],
                    "native cell denominator inconsistent")
            out.append((split, row["group_id"], seed, native, raw_sha))
    require(len(out) == 8
            and {split: len([r for r in out if r[0] == split])
                 for split in ("TRAIN", "DEVELOPMENT")} == {"TRAIN": 4, "DEVELOPMENT": 4}
            and len({(r[0], r[1]) for r in out}) == 4,
            "expected precisely two structural Pipework groups per visible split, two seeds each")
    return out


def screen(corpus: Path | None = None) -> dict:
    doc = load_proposal()
    sem = v2.validate_stepped_structure(doc["program"])
    older = v2.read_document()
    # Same required two-cross/two-elbow topology: structural signature matching
    # C2b is expected, not evidence of identical geometry or global uniqueness.
    new_lines = centreline(doc["program"])
    older_lines = centreline(older["program"])
    require(new_lines != older_lines, "C2c duplicates C2b centreline geometry")
    comparisons = []
    if corpus is not None:
        for split, group, seed, native, raw_sha in visible_native_masks(corpus):
            comparisons.append({
                "split": split, "group_id": group, "seed": seed,
                "original_material_sha256": raw_sha,
                "c2b": coverage(older_lines, native),
                "c2c": coverage(new_lines, native),
            })
        known = legacy.original_visible_splits(corpus)
        require(all(legacy.signature(doc["program"]) not in known[s]
                    for s in ("TRAIN", "DEVELOPMENT")),
                "C2c duplicates a frozen visible structural signature")
    compiled = []
    for seed in legacy.SEEDS:
        built = legacy.compiler.compile_program(doc["program"], seed=seed,
                     focus=FOCUS, material=58, allowed_materials=(58,))
        grid = legacy.compiler.rasterize_operations(built["operations"], FOCUS)
        verification = legacy.compiler.verify_structure(built, grid)
        require(built["structural_preflight"]["passed"] and verification["passed"]
                and not built["native_qualified"] and built["proposal_only"]
                and len(built["semantic"]["endpoints"]) == 6
                and len(built["semantic"]["junctions"]) == 2
                and len(built["semantic"]["diameter_probes"]) == 9,
                "two-seed compiler-only topology/diameter failure")
        compiled.append({
            "seed": seed, "program_sha256": built["program_sha256"],
            "operations_sha256": built["operations_sha256"],
            "operation_count": built["operation_count"],
            "compiler_unique_cells": built["unique_cells"],
            "native_qualified": False,
        })
    return {
        "status": "C2C_COMPILER_AND_VISIBLE_NATIVE_PROXIMITY_REVIEW_ONLY",
        "source_proposal_sha256": legacy.sha(PROPOSAL),
        "original_native_source": legacy.SOURCE,
        "original_manifest_sha256": legacy.MANIFEST,
        "c2b_proposal_sha256": C2B_SHA,
        "structure": sem, "centreline_denominator_c2b": len(older_lines),
        "centreline_denominator_c2c": len(new_lines),
        "comparison_metric": "directional candidate centreline pixel within Chebyshev<=3 of original native material58",
        "comparisons": comparisons,
        "numeric_auto_accept_threshold": None,
        "review_disposition": "PENDING_INDEPENDENT_REVIEW_AND_SHEPHERD",
        "seed_compilation": compiled,
        "opened_splits": ["TRAIN", "DEVELOPMENT"] if corpus is not None else [],
        "final_targets_opened": False,
        "native_executions": 0, "model_calls": 0,
        "admitted_train_groups": 0, "optimizer_updates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = screen(args.corpus)
    except Exception as exc:
        args.receipt.write_text(json.dumps({
            "status": "FAIL_NO_ADMISSION", "error": repr(exc),
            "final_targets_opened": False, "native_executions": 0,
            "admitted_train_groups": 0, "optimizer_updates": 0,
        }, indent=2) + "\n", encoding="utf-8")
        raise
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "comparators": len(result["comparisons"]),
                      "compiler_seed_cases": len(result["seed_compilation"])}))


if __name__ == "__main__":
    main()
