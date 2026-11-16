#!/usr/bin/env python3
"""LFM family-contrast research: frozen source + visible-split compiler preflight ONLY.

Never opens FINAL, trains a model, re-labels native teacher admission, mutates Art,
or repairs raw completions. Shared reviewer/shepherd own approval and admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PAINTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAINTER))
import compiler

SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
SEEDS = (104729, 130363)
SCHEMA = "cybersand.lfm25_vl.contrastive_research_proposals/1"
STATUS = "PROPOSED_NOT_TRAIN_NOT_NATIVE_ADMITTED"
FILE = Path(__file__).with_name("contrastive_proposals_v1.json")


def sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Untrusted or absent input: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(program: dict) -> str:
    return json.dumps(compiler.structural_signature(program), sort_keys=True, separators=(",", ":"))


def original_visible_splits(corpus: Path) -> dict[str, set[str]]:
    """Original TRAIN/DEVELOPMENT program signatures only: never inspect FINAL."""
    corpus = corpus.resolve()
    manifest_file = corpus / "admitted-corpus.json"
    if sha(manifest_file) != MANIFEST:
        raise ValueError("original source-633 native manifest hash changed")
    m = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (m.get("source_sha") != SOURCE or m.get("state") != "native-admitted"
            or m.get("compiler_sha256") != sha(PAINTER / "compiler.py")
            or set(m.get("split_files", {})) != {"TRAIN", "DEVELOPMENT", "FINAL"}):
        raise ValueError("native admission/source/compiler/split contract changed")
    result = {}
    for split in ("TRAIN", "DEVELOPMENT"):
        spec = m["split_files"][split]
        source = (corpus / spec["path"]).resolve()
        if not source.is_relative_to(corpus) or sha(source) != spec["sha256"]:
            raise ValueError(split + " split byte hash or path mismatch")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 12 or spec["count"] != 12 or len({row["group_id"] for row in rows}) != 6:
            raise ValueError(split + " split group or row count changed")
        if any(row["split"] != split for row in rows):
            raise ValueError(split + " mislabeled source split")
        sigs = set()
        for row in rows:
            p = row["program"]
            p = compiler.parse_program(p) if isinstance(p, str) else p
            sigs.add(signature(p))
        if len(sigs) != 6:
            raise ValueError(split + " original structural group uniqueness broken")
        result[split] = sigs
    return result


def preflight(item: dict) -> list[dict]:
    """Validate literal v2 program and its independent semantic witnesses."""
    p = item["program"]
    compiler.validate_program(p)
    if p["family"] != item["family"] or p["variation"] != 0:
        raise ValueError("proposal family or variation mismatch")
    components = p["components"]
    if p["family"] == "greebles":
        if (len(components) != 6 or len({c["detail"] for c in components}) != 5
                or sum(c["box"][0] < 40 for c in components) != 3
                or sum(c["box"][0] >= 56 for c in components) != 3
                or any(not (c["box"][0] + c["box"][2] <= 40 or c["box"][0] >= 56)
                       for c in components)):
            raise ValueError("missing six-panel two-bank/empty-central-chase witness")
        # Three *paired* vertical tiers, not merely three arbitrary panels on
        # each side. Preserve the full-height centre chase already required
        # above; x-bank counts alone do not prove a 3-by-2 composition.
        banks = (
            sorted((c["box"] for c in components if c["box"][0] + c["box"][2] <= 40),
                   key=lambda box: box[1]),
            sorted((c["box"] for c in components if c["box"][0] >= 56),
                   key=lambda box: box[1]),
        )
        if any(len(bank) != 3 or any(
                bank[i][1] + bank[i][3] + 4 > bank[i + 1][1]
                for i in range(2)) for bank in banks):
            raise ValueError("greebles must have three separated tiers per bank")
        if any(abs((2 * left[1] + left[3]) - (2 * right[1] + right[3])) > 16
               for left, right in zip(*banks)):
            raise ValueError("greebles require paired three-tier bank alignment")
    elif p["family"] == "pipework":
        c = components[0]
        degree = [0] * len(c["nodes"])
        graph = [[] for _ in degree]
        for a, b in c["edges"]:
            degree[a] += 1
            degree[b] += 1
            graph[a].append(b)
            graph[b].append(a)
        leaves = [i for i, d in enumerate(degree) if d == 1]
        rising = sum(c["nodes"][i][0] == c["nodes"][graph[i][0]][0]
                     and c["nodes"][i][1] < c["nodes"][graph[i][0]][1] for i in leaves)
        falling = sum(c["nodes"][i][0] == c["nodes"][graph[i][0]][0]
                      and c["nodes"][i][1] > c["nodes"][graph[i][0]][1] for i in leaves)
        if (sorted(degree) != [1, 1, 1, 1, 1, 1, 4, 4] or rising != 2 or falling != 2
                or c["diameter"] != 3 or c["fittings"] is not True
                or p["requirements"] != {"connected": True, "cycles": 0}):
            raise ValueError("not the independently specified six-terminal dual-cross network")
    else:
        kinds = sorted(c["kind"] for c in components)
        if (kinds != ["floor", "platform", "platform", "room", "stairs", "support", "support", "wall"]
                or len(p["requirements"]["route"]) != 2
                or p["requirements"]["clearance"] != 5
                or [o["side"] for o in components[0]["openings"]] != ["left", "right"]):
            raise ValueError("architecture hall/gantry/route composition witness absent")

    evidence = []
    for seed in SEEDS:
        compiled = compiler.compile_program(
            p, seed=seed, focus=(0, 0, 96, 96), material=58, allowed_materials=(58,))
        grid = compiler.rasterize_operations(compiled["operations"], (0, 0, 96, 96))
        structure = compiler.verify_structure(compiled, grid)
        if (not compiled["structural_preflight"]["passed"] or not structure["passed"]
                or compiled["native_qualified"] is not False or compiled["proposal_only"] is not True):
            raise ValueError("compiler-only proposal could not be independently verified")
        sem = compiled["semantic"]
        if p["family"] == "pipework" and (
                len(sem["endpoints"]) != 6 or len(sem["junctions"]) != 2
                or sem["expected_cycles"] != 0 or len(sem["diameter_probes"]) != 7):
            raise ValueError("host compiled graph topology/probe disagreement")
        if p["family"] == "architecture" and (
                len(sem["rooms"]) != 1 or len(sem["route"]) != 2 or sem["clearance"] != 5):
            raise ValueError("host compiled route/rooms disagreement")
        if p["family"] == "greebles" and len(sem["panels"]) != 6:
            raise ValueError("host compiled panels disagreement")
        evidence.append({
            "seed": seed, "program_sha256": compiled["program_sha256"],
            "operations_sha256": compiled["operations_sha256"],
            "operation_count": compiled["operation_count"],
            "unique_cells": compiled["unique_cells"],
            "compiler_structure_pass": True, "native_qualified": False,
        })
    return evidence


def screen(corpus: Path, path: Path) -> dict:
    if sha(path) == "":
        raise AssertionError("unreachable input hash")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("schema") != SCHEMA or document.get("status") != STATUS
            or document.get("original_native_source") != SOURCE
            or document.get("original_manifest_sha256") != MANIFEST
            or document.get("host_seeds") != list(SEEDS)):
        raise ValueError("invalid research-only proposal source contract")
    items = document.get("proposals")
    if (not isinstance(items, list) or len(items) != 3
            or {x["family"] for x in items} != {"greebles", "pipework", "architecture"}
            or len({x["id"] for x in items}) != 3):
        raise ValueError("exactly three independent family proposals required")
    known = original_visible_splits(corpus)
    seen = set()
    evidence = []
    for item in items:
        if not item.get("request") or not item.get("witness"):
            raise ValueError("literal design requirements/witness missing")
        key = signature(item["program"])
        if key in seen or any(key in known[s] for s in ("TRAIN", "DEVELOPMENT")):
            raise ValueError(item["id"] + " duplicates another proposal or visible held-out group")
        seen.add(key)
        witness = preflight(item)
        evidence.append({
            "id": item["id"], "family": item["family"],
            "signature_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "seed_compilation": witness,
        })
    return {
        "status": "COMPILER_SCREEN_ONLY_PENDING_SHEPHERD_REVIEW_NATIVE_AND_TRAIN_ADMISSION",
        "input_sha256": sha(path),
        "original_native_source_sha": SOURCE,
        "original_manifest_sha256": MANIFEST,
        "compiler_sha256": sha(PAINTER / "compiler.py"),
        "opened_splits": ["TRAIN", "DEVELOPMENT"],
        "final_targets_opened": False, "actual_model_calls": 0,
        "native_executions": 0, "admitted_train_groups": 0, "optimizer_updates": 0,
        "proposals": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--proposals", type=Path, default=FILE)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = screen(args.corpus, args.proposals)
    except Exception as exc:
        receipt = {"status": "FAIL_NO_ADMISSION", "error": repr(exc),
                   "final_targets_opened": False, "native_executions": 0,
                   "admitted_train_groups": 0, "optimizer_updates": 0}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "proposals": len(receipt["proposals"])}))


if __name__ == "__main__":
    main()
