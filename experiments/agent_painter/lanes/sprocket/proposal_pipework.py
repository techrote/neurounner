#!/usr/bin/env python3
"""Source-compiled, non-admitted SPROCKET Pipework proposals.

No training, model output, native execution, FINAL reads or dataset mutation.
A successful check establishes only compiler feasibility and a conservative
TRAIN/DEVELOPMENT structural-signature screen; independent reviewer and two
source-built native seed readbacks remain mandatory before shared admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

PAINTER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER_ROOT))
from compiler import canonical_json, compile_program, parse_program, structural_signature

NATIVE_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
NATIVE_MANIFEST_SHA = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
ORIGINAL_TRAIN_SHA = "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1"
ORIGINAL_DEV_SHA = "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"
ORIGINAL_COMPILER_SHA = "4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6"
HOST_SEEDS = (104729, 130363)

# Coordinates are orthographic XY in v2's 96-cell design plane. Each graph
# differs by connectivity, not seed, translation, diameter or paraphrase.
# Never equate a proposal ID with an admitted corpus TRAIN ID.
GRAPHS = {
    "P01": {
        "description": "two nested degree-three branch nodes, four endpoints; acyclic",
        "nodes": [[18,45],[40,45],[65,45],[40,20],[65,20],[65,70]],
        "edges": [[0,1],[1,2],[1,3],[2,4],[2,5]],
        "cycles": 0, "degrees": [1,1,1,1,3,3],
    },
    "P02": {
        "description": "single degree-four cross hub and four terminal legs; NOT three-tooth comb",
        "nodes": [[48,48],[48,20],[76,48],[48,76],[20,48]],
        "edges": [[0,1],[0,2],[0,3],[0,4]],
        "cycles": 0, "degrees": [1,1,1,1,4],
    },
    "P03": {
        "description": "orthogonal perimeter ring with internal cross-chord, two cycles, zero endpoints",
        "nodes": [[24,24],[72,24],[72,48],[72,72],[24,72],[24,48]],
        "edges": [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[5,2]],
        "cycles": 2, "degrees": [2,2,2,2,3,3],
    },
    "P04": {
        "description": "one rectangular ring with TWO external spurs on distinct corners",
        "nodes": [[28,28],[64,28],[64,64],[28,64],[14,28],[78,64]],
        "edges": [[0,1],[1,2],[2,3],[3,0],[0,4],[2,5]],
        "cycles": 1, "degrees": [1,1,2,2,3,3],
    },
    "P05": {
        "description": "two loops sharing exactly one articulation vertex, plus one spur",
        "nodes": [[18,18],[48,18],[48,48],[18,48],[78,48],[78,78],[48,78],[18,72]],
        "edges": [[0,1],[1,2],[2,3],[3,0],[2,4],[4,5],[5,6],[6,2],[3,7]],
        "cycles": 2, "degrees": [1,2,2,2,2,2,3,4],
    },
    "P06": {
        "description": "three-by-three junction lattice, four independent closed faces",
        "nodes": [[20,20],[48,20],[76,20],[20,48],[48,48],[76,48],[20,76],[48,76],[76,76]],
        "edges": [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],
                  [0,3],[3,6],[1,4],[4,7],[2,5],[5,8]],
        "cycles": 4, "degrees": [2,2,2,2,3,3,3,3,4],
    },
}

def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def program_for(identifier: str) -> dict:
    item = GRAPHS[identifier]
    return {
        "version": 2, "family": "pipework", "variation": 0,
        "components": [{
            "kind": "pipe_network", "nodes": deepcopy(item["nodes"]), "edges": deepcopy(item["edges"]),
            "diameter": 3, "fittings": False,
        }],
        "requirements": {"connected": True, "cycles": item["cycles"]},
    }

def check_graph(identifier: str, program: dict) -> dict:
    item = GRAPHS[identifier]
    nodes = program["components"][0]["nodes"]
    edges = program["components"][0]["edges"]
    deg = Counter(i for edge in edges for i in edge)
    actual = sorted(deg.values())
    if actual != sorted(item["degrees"]) or len(edges)-len(nodes)+1 != item["cycles"]:
        raise ValueError(f"{identifier}: topology witness disagrees with explicit graph")
    # Strict parse first; compiler independently verifies connected graph,
    # undeclared crossings, degree <=4, diameter and bounded geometry.
    parsed = parse_program(canonical_json(program))
    if parsed != program:
        raise ValueError("strict JSON round-trip changed the program")
    return {"degree_histogram": dict(sorted(Counter(actual).items())),
            "endpoints": actual.count(1), "junctions": sum(d >= 3 for d in actual),
            "cycles": item["cycles"], "nodes": len(nodes), "edges": len(edges)}

def public_split_signatures(corpus: Path) -> dict[str, list[str]]:
    """Only original public TRAIN and DEVELOPMENT; FINAL content is never read."""
    manifest_file = corpus / "admitted-corpus.json"
    if file_sha(manifest_file) != NATIVE_MANIFEST_SHA:
        raise ValueError("original native corpus manifest digest mismatch")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("source_sha") != NATIVE_SOURCE or manifest.get("state") != "native-admitted":
        raise ValueError("original native admission source/state mismatch")
    if manifest.get("compiler_sha256") != ORIGINAL_COMPILER_SHA:
        raise ValueError("original compiler digest mismatch")
    found = {}
    for name, expected in (("TRAIN", ORIGINAL_TRAIN_SHA), ("DEVELOPMENT", ORIGINAL_DEV_SHA)):
        entry = manifest["split_files"][name]
        path = (corpus / entry["path"]).resolve()
        if not path.is_relative_to(corpus.resolve()) or not path.is_file():
            raise ValueError(f"{name}: invalid source file")
        if file_sha(path) != expected or entry["sha256"] != expected or entry["count"] != 12:
            raise ValueError(f"{name}: source split digest/count mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 12 or len({r["structural_group"] for r in rows}) != 6:
            raise ValueError(f"{name}: original structure count mismatch")
        if any(r.get("split") != name for r in rows):
            raise ValueError(f"{name}: cross-partition row")
        for row in rows:
            key = canonical_json(structural_signature(row["program"]))
            found.setdefault(key, []).append(f'{name}:{row["structural_group"]}')
    return found

def evaluate(corpus: Path | None = None) -> dict:
    if file_sha(PAINTER_ROOT / "compiler.py") != ORIGINAL_COMPILER_SHA:
        raise ValueError("strict frozen v2 compiler source changed")
    known = public_split_signatures(corpus) if corpus is not None else {}
    seen: set[str] = set()
    reports = []
    for identifier in GRAPHS:
        program = program_for(identifier)
        topology = check_graph(identifier, program)
        signature = canonical_json(structural_signature(program))
        if signature in seen:
            raise ValueError(f"{identifier}: duplicate prospective structural signature")
        seen.add(signature)
        if signature in known:
            raise ValueError(f"{identifier}: collides with original public split: {known[signature]}")
        seeds = []
        for seed in HOST_SEEDS:
            compiled = compile_program(program, seed=seed)
            if compiled["proposal_only"] is not True or compiled["native_qualified"] is not False:
                raise ValueError(f"{identifier}: compiler result mislabels native authority")
            if compiled["operation_count"] > 512 or compiled["estimated_cell_work"] > 100000:
                raise ValueError(f"{identifier}: plan exceeds budget")
            if compiled["semantic"]["expected_cycles"] != topology["cycles"]:
                raise ValueError(f"{identifier}: compiler and topology witness disagree")
            seeds.append({"seed":seed, "operations":compiled["operation_count"],
                          "cell_work":compiled["estimated_cell_work"],
                          "operations_sha256":compiled["operations_sha256"]})
        reports.append({
            "id": identifier, "description": GRAPHS[identifier]["description"],
            "program_sha256": sha256(canonical_json(program).encode()).hexdigest(),
            "structural_signature_sha256":sha256(signature.encode()).hexdigest(),
            "topology":topology, "views":seeds,
            "compiler_valid": True, "original_public_signature_collision":False,
            "native_admitted":False, "train_eligible":False,
        })
    return {
        "schema":"cybersand.sprocket-prospective-pipework-compiler-screen/1",
        "status":"compiler-only-not-native-admitted",
        "native_source":NATIVE_SOURCE,
        "original_corpus_manifest_sha256":NATIVE_MANIFEST_SHA if corpus else None,
        "compiler_sha256":ORIGINAL_COMPILER_SHA,
        "reviewable_structures":len(reports),
        "host_seed_views_compiled":len(reports)*len(HOST_SEEDS),
        "final_targets_opened":False,
        "requires":"reviewer composition screening, source-built native two-seed receipts and original PNGs",
        "candidates":reports,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, help="original actual native corpus root; reads TRAIN/DEVELOPMENT only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite existing screen result")
    report = evaluate(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"structures":report["reviewable_structures"],
                      "seed_views":report["host_seed_views_compiled"],
                      "native_admitted":False}, sort_keys=True))

if __name__ == "__main__":
    main()
