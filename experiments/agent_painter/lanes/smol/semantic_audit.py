#!/usr/bin/env python3
"""Read-only SMOL DEVELOPMENT topology conformance, not a model/native evaluator.

Consume retained original first-pass outputs; do not train, retry, repair, run
native Art, or open FINAL. Only explicit pipework request constraints are scored.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

PIPE_REQUIREMENTS = {
    "pipework/three_branch_manifold/v1": {
        "endpoints": 5, "tees": 3, "crosses": 0, "cycles": 0,
        "upward_branches": 3, "diameter": 3,
    },
    "pipework/closed_loop_with_tap/v1": {
        "endpoints": 1, "tees": 1, "crosses": 0, "cycles": 1, "diameter": 3,
    },
}
ARMS = ("untuned-base", "trained-candidate")


def strict_json(value):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    def reject(value):
        raise ValueError("nonfinite JSON literal: " + value)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=reject)


def graph_metrics(program):
    if (type(program) is not dict or program.get("family") != "pipework"
            or type(program.get("components")) is not list
            or len(program["components"]) != 1):
        raise ValueError("not a single pipework program")
    pipe = program["components"][0]
    nodes, edges = pipe["nodes"], pipe["edges"]
    n = len(nodes)
    if n < 2 or len({tuple(p) for p in nodes}) != n:
        raise ValueError("duplicate or insufficient graph nodes")
    adjacency = [set() for _ in nodes]
    degree = [0] * n
    seen = set()
    horizontal = defaultdict(int)
    vertical = []
    for pair in edges:
        if (type(pair) is not list or len(pair) != 2
                or any(type(i) is not int or i < 0 or i >= n for i in pair)):
            raise ValueError("invalid edge")
        a, b = pair
        key = tuple(sorted((a, b)))
        if a == b or key in seen:
            raise ValueError("duplicate or zero-length edge")
        seen.add(key)
        degree[a] += 1
        degree[b] += 1
        adjacency[a].add(b)
        adjacency[b].add(a)
        p, q = nodes[a], nodes[b]
        if p[1] == q[1]:
            horizontal[p[1]] += abs(p[0] - q[0])
        elif p[0] == q[0]:
            vertical.append((p, q))
        else:
            raise ValueError("non-orthogonal edge")
    unseen = set(range(n))
    components = 0
    while unseen:
        components += 1
        visit = [unseen.pop()]
        while visit:
            for i in adjacency[visit.pop()] & unseen:
                unseen.remove(i)
                visit.append(i)
    trunk_y = (max(sorted(horizontal), key=lambda y: horizontal[y])
               if horizontal else None)
    upward = set()
    if trunk_y is not None:
        for p, q in vertical:
            if max(p[1], q[1]) == trunk_y and min(p[1], q[1]) < trunk_y:
                upward.add(p[0])
    return {
        "endpoints": degree.count(1), "tees": degree.count(3),
        "crosses": degree.count(4), "cycles": len(edges) - n + components,
        "connected_components": components, "upward_branches": len(upward),
        "diameter": pipe["diameter"], "nodes": n, "edges": len(edges),
    }


def audit_case(directory, record):
    identifier = record["id"]
    if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier) is None:
        raise ValueError("unsafe case identifier")
    if record["native_executed"] is not False:
        raise ValueError("native claims are forbidden for compile-only study")
    case_dir = directory / identifier
    completion = (case_dir / "completion.txt").read_bytes()
    if hashlib.sha256(completion).hexdigest() != record["completion_sha256"]:
        raise ValueError("raw completion digest mismatch")
    if strict_json((case_dir / "receipt.json").read_text()) != record:
        raise ValueError("original receipt mismatch")
    group = record["structural_group"]
    result = {
        "id": identifier, "group": group, "host_seed": record["host_seed"],
        "prompt_sha256": record["prompt_sha256"],
        "completion_sha256": record["completion_sha256"],
        "compiler_accepted": record["compiler_accepted"],
        "semantic_conforms": False, "native_executed": False,
    }
    if not record["compiler_accepted"]:
        result["status"] = "strict-first-pass-refusal"
    elif group not in PIPE_REQUIREMENTS:
        result["status"] = "compiles-but-nonpipe-semantics-not-scored"
    else:
        program = strict_json(completion.decode("utf-8"))
        observed = graph_metrics(program)
        required = PIPE_REQUIREMENTS[group]
        checks = {key: observed[key] == value for key, value in required.items()}
        checks["connected"] = observed["connected_components"] == 1
        result.update(required=required, observed=observed, checks=checks)
        result["semantic_conforms"] = all(checks.values())
        result["status"] = ("pipe-constraints-met" if result["semantic_conforms"]
                            else "compiles-but-request-topology-fails")
    return result


def audit(root, training_sha, evaluation_sha, corpus_sha, token_budget):
    summary = strict_json((root / "summary.json").read_text())
    expected = {
        "training_source_sha": training_sha, "evaluation_source_sha": evaluation_sha,
        "corpus_manifest_sha256": corpus_sha, "max_new_tokens": token_budget,
        "development_seed_cases": 12, "development_unique_prompt_count": 6,
        "development_structural_groups": 6, "native_execution": False,
        "final_targets_accessed": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError("original source/cohort mismatch: " + key)
    arms = {}
    for name in ARMS:
        directory = root / name
        if not directory.is_dir():
            raise ValueError("missing arm " + name)
        cases = []
        for path in sorted(directory.iterdir()):
            if not path.is_dir():
                raise ValueError("unexpected non-case file")
            record = strict_json((path / "receipt.json").read_text())
            if path.name != record["id"]:
                raise ValueError("case name differs from receipt")
            cases.append(audit_case(directory, record))
        if len(cases) != 12:
            raise ValueError("expected exactly 12 host-seed views")
        groups = defaultdict(list)
        for case in cases:
            groups[(case["group"], case["prompt_sha256"])].append(case)
        if len(groups) != 6 or not set(PIPE_REQUIREMENTS).issubset({g for g, _ in groups}):
            raise ValueError("wrong original held-out request groups")
        for pair in groups.values():
            if (len(pair) != 2 or len({p["host_seed"] for p in pair}) != 2
                    or len({p["completion_sha256"] for p in pair}) != 1):
                raise ValueError("host seed views did not reuse one identical generation")
        arms[name] = {
            "independent_requests": 6, "host_seed_views": 12,
            "compiler_accepted_views": sum(c["compiler_accepted"] for c in cases),
            "compiler_accepted_requests": sum(any(c["compiler_accepted"] for c in pair)
                                               for pair in groups.values()),
            "pipe_semantic_conforming_requests": sum(any(c["semantic_conforms"] for c in pair)
                                                      for pair in groups.values()),
            "all_family_semantic_score": None,
            "nonpipe_semantics_scored": False,
            "cases": cases,
        }
    return {
        "schema": "cybersand.agent-painter.smol-semantic-audit/1",
        "source_evaluation_run": 43535795817,
        "training_source_sha": training_sha, "evaluation_source_sha": evaluation_sha,
        "corpus_manifest_sha256": corpus_sha, "max_new_tokens": token_budget,
        "training": False, "native_execution": False, "final_opened": False,
        "repairs_or_retries": False,
        "limitation": "Scores only two frozen Pipework prompt constraints; no Art or visual success implied.",
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "output", "training-source", "evaluation-source", "corpus-manifest"):
        parser.add_argument("--" + key, required=True,
                            type=Path if key in ("input", "output") else str)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("do not overwrite evidence")
    result = audit(args.input, args.training_source, args.evaluation_source,
                   args.corpus_manifest, args.max_new_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arms": {key: {k: v for k, v in arm.items() if k != "cases"}
                                for key, arm in result["arms"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
