#!/usr/bin/env python3
"""Read-only visible-split topology comparison for prospective SMOL teacher data.

Uses only the positively admitted original TRAIN/DEVELOPMENT manifests. Never
opens FINAL. An exact *unlabeled graph* match is a caution, not dataset admission.
Compiler-signature noncollision alone is not sufficient split independence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from curriculum_preflight import MANIFEST, SEEDS, SOURCE, run as source_screen, sha


def canonical_tree(program: dict) -> str | None:
    """Canonical isomorphism code of the sole connected, cycle-free pipe graph.

    Node order, absolute coordinates, diameter, fittings, seed, and traversal
    order do not affect the result. Cyclic/multi-network cases are explicitly
    unsupported, never classified as structurally different.
    """
    if program.get("family") != "pipework":
        return None
    networks = [part for part in program.get("components", [])
                if part.get("kind") == "pipe_network"]
    if len(networks) != 1 or len(program.get("components", [])) != 1:
        return None
    network = networks[0]
    nodes = network.get("nodes", [])
    edges = network.get("edges", [])
    n = len(nodes)
    if n < 1 or n > 64:
        raise ValueError("unexpected pipe node count")
    adjacency = [set() for _ in range(n)]
    visited_edges = set()
    for pair in edges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("invalid pipe edge")
        a, b = pair
        if (type(a) is not int or type(b) is not int or
                not (0 <= a < n and 0 <= b < n) or a == b):
            raise ValueError("invalid pipe node endpoint")
        key = tuple(sorted((a, b)))
        if key in visited_edges:
            raise ValueError("duplicate pipe edge")
        visited_edges.add(key)
        adjacency[a].add(b)
        adjacency[b].add(a)
    if len(edges) != n - 1:
        return None  # cycles/multi-edge count are not certified tree shapes
    reached = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adjacency[v] - reached:
            reached.add(w)
            stack.append(w)
    if len(reached) != n:
        return None  # disconnected graph is not a tree

    remaining = n
    degree = [len(x) for x in adjacency]
    leaves = [v for v in range(n) if degree[v] <= 1]
    while remaining > 2:
        remaining -= len(leaves)
        next_leaves = []
        for v in leaves:
            for w in adjacency[v]:
                degree[w] -= 1
                if degree[w] == 1:
                    next_leaves.append(w)
        leaves = next_leaves
    if not leaves:
        raise ValueError("tree-center reduction failed")

    def encode(v: int, parent: int) -> str:
        return "(" + "".join(sorted(encode(w, v) for w in adjacency[v]
                                    if w != parent)) + ")"
    if len(leaves) == 1:
        return encode(leaves[0], -1)
    if len(leaves) == 2:
        a, b = leaves
        return "[" + "".join(sorted((encode(a, b), encode(b, a)))) + "]"
    raise ValueError("invalid tree-center cardinality")


def records_from_visible_splits(corpus: Path) -> dict:
    """Read ONLY already hash-verified TRAIN/DEVELOPMENT rows."""
    manifest = json.loads((corpus / "admitted-corpus.json").read_text())
    if sha(corpus / "admitted-corpus.json") != MANIFEST:
        raise ValueError("original native corpus manifest changed")
    if manifest.get("source_sha") != SOURCE:
        raise ValueError("original native corpus source changed")
    from curriculum_preflight import visible_splits
    visible_splits(corpus)  # checks both split file hashes, row counts, compiler
    result = {}
    for split in ("TRAIN", "DEVELOPMENT"):
        spec = manifest["split_files"][split]
        file = (corpus / spec["path"]).resolve()
        if not file.is_relative_to(corpus.resolve()) or sha(file) != spec["sha256"]:
            raise ValueError("visible split path/hash changed after custody check")
        result[split] = [json.loads(line) for line in file.read_text().splitlines()
                         if line.strip()]
    return result


def compare(candidates: list[dict], visible: dict) -> dict:
    """Group by graph shape; seed twins are never counted as novel structures."""
    result = []
    observed = {}
    for split in ("TRAIN", "DEVELOPMENT"):
        seen_shapes = set()
        for record in visible[split]:
            program = record["program"]
            if isinstance(program, str):
                from curriculum_preflight import compiler
                program = compiler.parse_program(program)
            tree = canonical_tree(program)
            if tree is not None:
                seen_shapes.add(tree)
        observed[split] = seen_shapes
    for item in candidates:
        tree = canonical_tree(item["program"])
        matches = {split: bool(tree is not None and tree in observed[split])
                   for split in ("TRAIN", "DEVELOPMENT")}
        result.append({
            "id": item["id"], "family": item["family"],
            "tree_graph_isomorphism_sha256":
                hashlib.sha256(tree.encode()).hexdigest() if tree else None,
            "tree_comparison_applicable": tree is not None,
            "same_unlabeled_tree_as_original_visible_split": matches,
            "disposition": ("QUARANTINE_POTENTIAL_VISIBLE_SPLIT_ALIAS" if
                            any(matches.values()) else
                            "REVIEW_ONLY_NO_AUTOMATIC_ADMISSION"),
            "limitation": ("Unlabeled graph topology ignores coordinates, material "
                           "thickness, fittings and artistic composition; different "
                           "graphs/unsupported geometry also need semantic review."),
        })
    return {
        "schema": "cybersand.smol-visible-split-neighbours/1",
        "status": "REVIEW_ONLY_NO_TRAIN_OR_NATIVE_ADMISSION",
        "original_native_source_sha": SOURCE,
        "original_manifest_sha256": MANIFEST,
        "checked_visible_splits": ["TRAIN", "DEVELOPMENT"],
        "final_target_or_image_opened": False,
        "admitted_train_structures": 0,
        "native_views": 0,
        "optimizer_steps": 0,
        "distinct_visible_pipe_tree_shapes":
            {split: len(observed[split]) for split in ("TRAIN", "DEVELOPMENT")},
        "candidates": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    screened = source_screen(args.corpus, args.proposals)
    if screened["native_execution"] or screened["admitted_teacher_rows"]:
        raise ValueError("source screen must remain research-only")
    proposal = json.loads(args.proposals.read_text())
    result = compare(proposal["proposals"], records_from_visible_splits(args.corpus))
    result["proposal_sha256"] = sha(args.proposals)
    result["source_screen_status"] = screened["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "flags": [r["id"] for r in result["candidates"]
                                if r["disposition"].startswith("QUARANTINE")]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
