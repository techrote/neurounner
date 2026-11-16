#!/usr/bin/env python3
"""Versioned, non-admitted SPROCKET fitted Pipework studies.

These are *new topology classes*, not restyled copies or admitted TRAIN data.
The original P01-P06 native prototypes, their receipts and published hashes
remain immutable. No model output, optimizer, FINAL target or native Art claim.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

PAINTER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER_ROOT))
from compiler import canonical_json, compile_program, parse_program, structural_signature
from proposal_pipework import (
    GRAPHS as ORIGINAL_GRAPHS, HOST_SEEDS, ORIGINAL_COMPILER_SHA,
    file_sha, public_split_signatures,
)
from peer_topology import PEER_GRAPH_EDGES, isomorphic

# These two explicit graphs are not independent merely because their pixels
# are different: they have distinct adjacency, degree and cycle witnesses.
# All segments are orthogonal, with sufficiently spacious faces for collars.
RICH_GRAPHS = {
    "R09": {
        "description": "one wide ring; THREE unequal exterior service runs, one turning upward",
        "nodes": [[22,24],[68,24],[68,68],[22,68],
                  [82,24],[82,12],[10,68],[68,82]],
        "edges": [[0,1],[1,2],[2,3],[3,0],[1,4],[4,5],[3,6],[2,7]],
        "cycles": 1,
        "degrees": [1,1,1,2,2,3,3,3],
    },
    "R10": {
        "description": "three-bay rectangular ladder with terminal services on opposite corners",
        "nodes": [[18,24],[38,24],[58,24],[78,24],
                  [18,68],[38,68],[58,68],[78,68],[6,24],[90,68]],
        "edges": [[0,1],[1,2],[2,3],[4,5],[5,6],[6,7],
                  [0,4],[1,5],[2,6],[3,7],[0,8],[7,9]],
        "cycles": 3,
        "degrees": [1,1,2,2,3,3,3,3,3,3],
    },
}

def program_for(identifier: str) -> dict:
    graph = RICH_GRAPHS[identifier]
    return {
        "version": 2, "family": "pipework", "variation": 0,
        "components": [{
            "kind": "pipe_network",
            "nodes": deepcopy(graph["nodes"]),
            "edges": deepcopy(graph["edges"]),
            "diameter": 5,
            "fittings": True,
        }],
        "requirements": {"connected": True, "cycles": graph["cycles"]},
    }

def topology_witness(identifier: str) -> dict:
    g = RICH_GRAPHS[identifier]
    n, edges = g["nodes"], g["edges"]
    deg = Counter(i for a,b in edges for i in (a,b))
    if len(deg) != len(n) or sorted(deg.values()) != g["degrees"]:
        raise ValueError(f"{identifier}: incorrect degree witness")
    if len(edges) - len(n) + 1 != g["cycles"]:
        raise ValueError(f"{identifier}: incorrect cycle witness")
    for a,b in edges:
        x0,y0 = n[a]; x1,y1 = n[b]
        if (x0 == x1) == (y0 == y1):
            raise ValueError(f"{identifier}: not an orthogonal, nonzero segment")
    for peer, old in ORIGINAL_GRAPHS.items():
        if isomorphic(len(n), edges, len(old["nodes"]), old["edges"]):
            raise ValueError(f"{identifier}: duplicates original SPROCKET {peer}")
    for peer, (count, other_edges) in PEER_GRAPH_EDGES.items():
        if isomorphic(len(n), edges, count, other_edges):
            raise ValueError(f"{identifier}: duplicates pinned VOID {peer}")
    for other, graph in RICH_GRAPHS.items():
        if other != identifier and isomorphic(len(n), edges, len(graph["nodes"]), graph["edges"]):
            raise ValueError(f"{identifier}: duplicates another rich candidate {other}")
    return {
        "nodes": len(n), "edges": len(edges), "cycles": g["cycles"],
        "degree_histogram": dict(sorted(Counter(deg.values()).items())),
        "endpoints": sum(v == 1 for v in deg.values()),
        "junctions": sum(v >= 3 for v in deg.values()),
        "isomorphic_to_original_sprocket_or_pinned_void": False,
    }

def evaluate(corpus: Path | None = None) -> dict:
    if file_sha(PAINTER_ROOT / "compiler.py") != ORIGINAL_COMPILER_SHA:
        raise ValueError("frozen compiler source mismatch")
    known = public_split_signatures(corpus) if corpus is not None else {}
    rows = []
    seen = set()
    for identifier in RICH_GRAPHS:
        program = program_for(identifier)
        if parse_program(canonical_json(program)) != program:
            raise ValueError("strict schema round-trip changed the program")
        topology = topology_witness(identifier)
        signature = canonical_json(structural_signature(program))
        if signature in known or signature in seen:
            raise ValueError(f"{identifier}: public TRAIN/DEVELOPMENT or local signature collision")
        seen.add(signature)
        views = []
        for seed in HOST_SEEDS:
            compiled = compile_program(program, seed=seed)
            collars = compiled["semantic"]["fittings"]
            if not collars:
                raise ValueError(f"{identifier}: declared fittings produced no collars")
            if compiled["proposal_only"] is not True or compiled["native_qualified"] is not False:
                raise ValueError("compiler improperly asserted native execution")
            if compiled["operation_count"] > 512 or compiled["estimated_cell_work"] > 100000:
                raise ValueError("compiler operation budget exceeded")
            views.append({
                "host_seed": seed,
                "operation_count": compiled["operation_count"],
                "estimated_cell_work": compiled["estimated_cell_work"],
                "collars": len(collars),
                "operations_sha256": compiled["operations_sha256"],
            })
        rows.append({
            "id": identifier, "description": RICH_GRAPHS[identifier]["description"],
            "topology": topology,
            "structural_signature_sha256": sha256(signature.encode()).hexdigest(),
            "program_sha256": sha256(canonical_json(program).encode()).hexdigest(),
            "views": views, "native_positive": False, "train_admitted": False,
        })
    return {
        "schema": "cybersand.sprocket-fitted-pipework-proposals/1",
        "status": "source-compiler-proposals-only",
        "compiler_sha256": ORIGINAL_COMPILER_SHA,
        "original_public_split_checked": corpus is not None,
        "groups": len(rows), "compiler_seed_views": sum(len(r["views"]) for r in rows),
        "original_six_prototypes_unchanged": True,
        "native_positive_views": 0, "new_train_groups_admitted": 0,
        "model_generated_views": 0, "final_targets_opened": False,
        "requires": "independent spatial/held-out review, two-seed source-built Art receipts, original PNG/readback and shepherd admission",
        "candidates": rows,
    }

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    result = evaluate(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "groups": result["groups"],
                      "views": result["compiler_seed_views"], "native": 0,
                      "TRAIN": 0}, sort_keys=True))

if __name__ == "__main__":
    main()
