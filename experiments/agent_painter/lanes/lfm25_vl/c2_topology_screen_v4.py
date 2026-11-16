#!/usr/bin/env python3
"""C2d genuinely distinct Pipework-graph research, with visible original native comparisons.

No FINAL inspection, model inference, native Art, training admission, or numeric
auto-selection. Original C2/C2b/C2c, including failed runs, remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import c2_native_proximity_v3 as v3
import c2_stepped_research_v2 as v2
import contrastive_preflight as legacy

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "c2_topology_proposal_v4.json"
SCHEMA = "cybersand.lfm25_vl.c2-topology-research/4"
STATUS = "REVIEW_ONLY_SOURCE_AND_VISIBLE_SPLIT_NO_NATIVE_NO_TRAIN"
C2C_SHA = "49a4871a2456e1d0abc856dc49b11ac007e9a8664c9fda385e9c329ab8c3abcf"
ID = "lfm-c2d-vertical-mast-cross-two-tees-hanging-branches"
FOCUS = (0, 0, 96, 96)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_document(path: Path = PROPOSAL) -> dict:
    require(path.is_file() and not path.is_symlink(), "ordinary v4 proposal required")
    doc = json.loads(path.read_text(encoding="utf-8"))
    require(type(doc) is dict and set(doc) == {
        "schema", "status", "revision_of", "id", "original_c2_source_blob",
        "c2b_proposal_sha256", "c2c_proposal_sha256",
        "original_native_source", "original_manifest_sha256",
        "host_seeds", "request", "witness", "program",
    }, "v4 review document has unapproved shape")
    require(doc["schema"] == SCHEMA and doc["status"] == STATUS and doc["id"] == ID,
            "v4 schema/status/identity changed")
    require(doc["revision_of"] == "lfm-c2c-wide-asymmetric-bridge-two-cross-six-terminal-network"
            and doc["original_c2_source_blob"] == v2.LEGACY_BLOB
            and doc["c2b_proposal_sha256"] == v3.C2B_SHA
            and legacy.sha(v2.PROPOSAL) == v3.C2B_SHA
            and doc["c2c_proposal_sha256"] == C2C_SHA
            and legacy.sha(v3.PROPOSAL) == C2C_SHA,
            "original C2/C2b/C2c lineage/hash changed")
    require(doc["original_native_source"] == legacy.SOURCE
            and doc["original_manifest_sha256"] == legacy.MANIFEST
            and doc["host_seeds"] == list(legacy.SEEDS),
            "original native TRAIN/DEVELOPMENT identity changed")
    require(all(isinstance(doc[k], str) and len(doc[k]) >= 100
                for k in ("request", "witness")), "explicit authored goal and witness required")
    return doc


def topology(program: dict) -> dict:
    legacy.compiler.validate_program(program)
    require(program["family"] == "pipework" and program["version"] == 2
            and program["variation"] == 0
            and program["requirements"] == {"connected": True, "cycles": 0}
            and len(program["components"]) == 1,
            "v4 must be a connected zero-cycle pipework v2 program")
    c = program["components"][0]
    require(c["kind"] == "pipe_network" and c["diameter"] == 3
            and c["fittings"] is True, "v4 nominal diameter/fittings changed")
    nodes, edges = c["nodes"], c["edges"]
    require(len(nodes) == 10 and len(edges) == 9, "six terminals and four interior nodes required")
    graph = [set() for _ in nodes]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    degrees = [len(g) for g in graph]
    require(sorted(degrees) == [1]*6 + [2] + [3]*2 + [4],
            "v4 must have exactly one cross, two tees, one elbow, six terminals")
    cross = next(i for i, d in enumerate(degrees) if d == 4)
    tees = [i for i, d in enumerate(degrees) if d == 3]
    elbow = next(i for i, d in enumerate(degrees) if d == 2)
    require(nodes[cross] == [48, 42]
            and {tuple(nodes[i]) for i in tees} == {(19, 42), (76, 70)}
            and nodes[elbow] == [76, 42],
            "v4 vertical-mast branch positions changed")
    require(set(graph[cross]) == {1, 2, 3, 6}
            and set(graph[3]) == {0, 4, 5}
            and set(graph[6]) == {0, 7}
            and set(graph[7]) == {6, 8, 9},
            "v4 cross/tees/elbow adjacency changed")
    require(nodes[1][0] == nodes[2][0] == nodes[cross][0]
            and nodes[1][1] < nodes[cross][1] < nodes[2][1]
            and nodes[4][0] < nodes[3][0] < nodes[cross][0]
            and nodes[5][0] == nodes[3][0] and nodes[5][1] > nodes[3][1]
            and nodes[8][1] == nodes[9][1] == nodes[7][1]
            and nodes[8][0] < nodes[7][0] < nodes[9][0],
            "v4 upright mast / hanging tree geometry changed")
    current = legacy.signature(program)
    old_c2 = json.loads((HERE / "contrastive_proposals_v1.json").read_text(encoding="utf-8"))
    old_c2 = next(x["program"] for x in old_c2["proposals"]
                  if x["id"] == v2.LEGACY_C2)
    require(all(current != legacy.signature(previous)
                for previous in (old_c2, v2.read_document()["program"],
                                 v3.load_proposal()["program"])),
            "new C2d graph structurally aliases original C2/C2b/C2c")
    return {
        "degree_multiset": sorted(degrees),
        "cross": nodes[cross],
        "tees": sorted(nodes[i] for i in tees),
        "elbow": nodes[elbow],
        "terminal_count": degrees.count(1),
        "edge_count": len(edges),
        "cycles": len(edges) - len(nodes) + 1,
        "structural_signature_sha256": hashlib.sha256(current.encode()).hexdigest(),
    }


def screen(corpus: Path | None = None) -> dict:
    doc = read_document()
    shape = topology(doc["program"])
    require(shape["cycles"] == 0, "v4 unexpected cycle")
    known = legacy.original_visible_splits(corpus) if corpus is not None else None
    if known is not None:
        require(all(legacy.signature(doc["program"]) not in known[split]
                    for split in ("TRAIN", "DEVELOPMENT")),
                "v4 duplicates a frozen visible structural group")
    compilation = []
    for seed in legacy.SEEDS:
        built = legacy.compiler.compile_program(
            doc["program"], seed=seed, focus=FOCUS, material=58,
            allowed_materials=(58,))
        oracle = legacy.compiler.rasterize_operations(built["operations"], FOCUS)
        checked = legacy.compiler.verify_structure(built, oracle)
        sem = built["semantic"]
        require(built["structural_preflight"]["passed"] and checked["passed"]
                and built["proposal_only"] and not built["native_qualified"]
                and len(sem["endpoints"]) == 6
                and len(sem["junctions"]) == 3
                and len(sem["diameter_probes"]) == 9
                and sem["expected_cycles"] == 0,
                "v4 source/compiler/diameter check refused")
        compilation.append({
            "seed": seed, "program_sha256": built["program_sha256"],
            "operations_sha256": built["operations_sha256"],
            "operation_count": built["operation_count"],
            "compiler_unique_cells": built["unique_cells"],
            "native_qualified": False, "compiler_structure_pass": True,
        })
    comparisons = []
    if corpus is not None:
        points = v3.centreline(doc["program"])
        for split, group, seed, native, raw_sha in v3.visible_native_masks(corpus):
            comparisons.append({
                "split": split, "group_id": group, "seed": seed,
                "original_native_material_sha256": raw_sha,
                "c2d_centreline_to_original_native": v3.coverage(points, native),
            })
        require(len(comparisons) == 8, "8 original visible pipework masks required")
    return {
        "status": "C2D_NEW_GRAPH_SOURCE_AND_VISIBLE_RESEARCH_ONLY",
        "candidate_sha256": legacy.sha(PROPOSAL),
        "original_native_source": legacy.SOURCE,
        "original_manifest_sha256": legacy.MANIFEST,
        "prior_c2c_sha256": C2C_SHA,
        "topology": shape,
        "seed_compilation": compilation,
        "comparison_definition": "candidate centreline pixels within Chebyshev <=3 of original TRAIN/DEVELOPMENT native material-58 masks; no automatic selection",
        "original_visible_comparisons": comparisons,
        "opened_splits": ["TRAIN", "DEVELOPMENT"] if corpus is not None else [],
        "numeric_auto_accept_threshold": None,
        "review_disposition": "PENDING_INDEPENDENT_REVIEW_AND_SHEPHERD",
        "final_targets_opened": False,
        "model_calls": 0, "native_executions": 0,
        "admitted_train_groups": 0, "optimizer_updates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = screen(args.corpus)
    except Exception as exc:
        args.receipt.write_text(json.dumps({
            "status": "FAIL_OR_INCOMPLETE_NO_ADMISSION", "error": repr(exc),
            "final_targets_opened": False, "native_executions": 0,
            "admitted_train_groups": 0, "optimizer_updates": 0,
        }, indent=2) + "\n", encoding="utf-8")
        raise
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"status": result["status"], "visible_views": len(result["original_visible_comparisons"]),
                      "compiled_seed_cases": len(result["seed_compilation"])}))


if __name__ == "__main__":
    main()
