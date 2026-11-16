#!/usr/bin/env python3
"""C2b offset-step source review only; no model, FINAL, native Art or TRAIN admission.

The legacy C2 research artifact and original FAILED native producer remain immutable.
Read ONLY source633 visible TRAIN/DEVELOPMENT original rows when given --corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import contrastive_preflight as legacy

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "c2_stepped_proposal_v2.json"
SCHEMA = "cybersand.lfm25_vl.c2-stepped-research/2"
STATUS = "REVIEW_ONLY_NOT_NATIVE_NOT_TRAIN_ADMITTED"
LEGACY_C2 = "lfm-c2-dual-cross-six-terminal-network"
LEGACY_BLOB = "226ce959c68a9e2649e88fe15c10713437cf82be"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_document(path: Path = PROPOSAL) -> dict:
    require(path.is_file() and not path.is_symlink(), "proposal must be an ordinary file")
    source = path.read_bytes()
    doc = json.loads(source)
    require(type(doc) is dict and set(doc) == {
        "schema", "status", "revision_of", "id", "original_c2_source_blob",
        "original_native_source", "original_manifest_sha256", "host_seeds",
        "request", "witness", "program"}, "C2b document shape changed")
    require(doc["schema"] == SCHEMA and doc["status"] == STATUS,
            "C2b is review only; no approval, training or native claim")
    require(doc["revision_of"] == LEGACY_C2 and
            doc["original_c2_source_blob"] == LEGACY_BLOB and
            doc["original_native_source"] == legacy.SOURCE and
            doc["original_manifest_sha256"] == legacy.MANIFEST and
            doc["host_seeds"] == list(legacy.SEEDS), "immutable source/split custody mismatch")
    require(doc["id"] == "lfm-c2b-offset-stepped-two-cross-six-terminal-network"
            and isinstance(doc["request"], str) and len(doc["request"]) >= 50
            and isinstance(doc["witness"], str) and len(doc["witness"]) >= 50,
            "identified independent revision and substantive requirements required")
    return doc


def validate_stepped_structure(program: dict) -> dict:
    legacy.compiler.validate_program(program)
    require(program["family"] == "pipework" and program["version"] == 2 and
            program["variation"] == 0 and
            program["requirements"] == {"connected": True, "cycles": 0} and
            len(program["components"]) == 1, "unexpected C2b family/requirements")
    part = program["components"][0]
    require(part["kind"] == "pipe_network" and part["diameter"] == 3 and
            part["fittings"] is True, "C2b requires nominal diameter3 and collars")
    nodes, edges = part["nodes"], part["edges"]
    require(len(nodes) == 10 and len(edges) == 9, "six leaves, two crosses and two elbows required")
    graph = [set() for _ in nodes]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    degree = [len(neighbors) for neighbors in graph]
    require(sorted(degree) == [1] * 6 + [2] * 2 + [4] * 2,
            "C2b must contain 6 terminals, 2 elbows and 2 genuine crosses")
    crosses = sorted((i for i, d in enumerate(degree) if d == 4),
                     key=lambda i: nodes[i][0])
    elbows = [i for i, d in enumerate(degree) if d == 2]
    left, right = crosses
    lx, ly = nodes[left]
    rx, ry = nodes[right]
    require(rx - lx >= 30 and ry - ly >= 28, "crosses must occupy distinct diagonal elevations")
    top = [i for i in elbows if nodes[i][1] == ly and lx + 12 <= nodes[i][0] <= rx - 12]
    bottom = [i for i in elbows if nodes[i][1] == ry and lx + 12 <= nodes[i][0] <= rx - 12]
    require(len(top) == len(bottom) == 1 and top[0] != bottom[0],
            "exactly one upper and one lower offset-backbone elbow required")
    a, b = top[0], bottom[0]
    require(nodes[a][0] == nodes[b][0] and
            {a} <= graph[left] and {b} <= graph[a] and
            {right} <= graph[b], "the connector must be horizontal/vertical/horizontal")
    require(right not in graph[left] and len(graph[a]) == len(graph[b]) == 2,
            "bypass/extra interior elbow connection rejected")
    leaves = [i for i, d in enumerate(degree) if d == 1]
    require(all(next(iter(graph[i])) in crosses for i in leaves),
            "all terminal stubs must attach directly to a genuine cross")
    for cross, direction in ((left, -1), (right, 1)):
        attached = [i for i in leaves if cross in graph[i]]
        require(len(attached) == 3, "each cross requires 3 outer terminals")
        vertical = [i for i in attached if nodes[i][0] == nodes[cross][0]]
        horizontal = [i for i in attached if nodes[i][1] == nodes[cross][1]]
        require(len(vertical) == 2 and len(horizontal) == 1 and
                any(nodes[i][1] < nodes[cross][1] for i in vertical) and
                any(nodes[i][1] > nodes[cross][1] for i in vertical) and
                direction * (nodes[horizontal[0]][0] - nodes[cross][0]) >= 12,
                "each cross needs upward/downward stubs and an outward horizontal terminal")
    return {"degree_multiset": sorted(degree),
            "crosses": [nodes[i] for i in crosses],
            "elbows": [nodes[a], nodes[b]],
            "terminal_count": len(leaves),
            "edge_count": len(edges),
            "cycle_count": len(edges) - len(nodes) + 1}


def screen(corpus: Path | None = None, path: Path = PROPOSAL) -> dict:
    doc = read_document(path)
    semantic = validate_stepped_structure(doc["program"])
    old = json.loads((HERE / "contrastive_proposals_v1.json").read_text(encoding="utf-8"))
    old_c2 = next(i["program"] for i in old["proposals"] if i["id"] == LEGACY_C2)
    require(legacy.signature(doc["program"]) != legacy.signature(old_c2),
            "C2b is an old C2 graph relabel")
    known = legacy.original_visible_splits(corpus) if corpus is not None else None
    if known is not None:
        require(all(legacy.signature(doc["program"]) not in known[part]
                    for part in ("TRAIN", "DEVELOPMENT")),
                "C2b aliases original TRAIN/DEVELOPMENT structural signature")
    receipts = []
    for seed in legacy.SEEDS:
        built = legacy.compiler.compile_program(
            doc["program"], seed=seed, focus=(0, 0, 96, 96),
            material=58, allowed_materials=(58,))
        grid = legacy.compiler.rasterize_operations(built["operations"], (0, 0, 96, 96))
        structure = legacy.compiler.verify_structure(built, grid)
        sem = built["semantic"]
        require(built["structural_preflight"]["passed"] and structure["passed"] and
                built["native_qualified"] is False and built["proposal_only"] is True,
                "compiler-only structural qualification rejected")
        require(len(sem["endpoints"]) == 6 and len(sem["junctions"]) == 2 and
                len(sem["diameter_probes"]) == 9 and sem["expected_cycles"] == 0,
                "compiled true-graph and complete-diameter probes must agree")
        receipts.append({"seed": seed, "program_sha256": built["program_sha256"],
                         "operations_sha256": built["operations_sha256"],
                         "operation_count": built["operation_count"],
                         "unique_cells": built["unique_cells"],
                         "native_qualified": False, "compiler_structure_pass": True})
    return {
        "status": "COMPILER_ONLY_REVISE_REVIEW_PENDING_NO_NATIVE_NO_TRAIN",
        "source_proposal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "legacy_original_c2_git_blob": LEGACY_BLOB,
        "original_native_source": legacy.SOURCE,
        "original_manifest_sha256": legacy.MANIFEST,
        "opened_splits": ["TRAIN", "DEVELOPMENT"] if known is not None else [],
        "final_targets_opened": False,
        "reviewer_disposition_old_c2": "REVISE",
        "witness": semantic, "seed_compilation": receipts,
        "model_calls": 0, "native_executions": 0,
        "admitted_train_groups": 0, "optimizer_updates": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = screen(args.corpus)
    except Exception as error:
        args.receipt.write_text(json.dumps({
            "status": "FAIL_NO_ADMISSION", "error": repr(error),
            "final_targets_opened": False, "model_calls": 0,
            "native_executions": 0, "admitted_train_groups": 0,
            "optimizer_updates": 0}, indent=2) + "\n", encoding="utf-8")
        raise
    args.receipt.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({"status": output["status"], "seeds": len(output["seed_compilation"])}))


if __name__ == "__main__":
    main()
