#!/usr/bin/env python3
"""Model-free v2 proposal source/compiler/split screen. NO native admission or TRAIN rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import compiler
from curriculum_witness import inspect as inspect_composition

MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
SEEDS = (104729, 130363)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(program):
    return json.dumps(compiler.structural_signature(program), sort_keys=True, separators=(",", ":"))


def visible_splits(corpus):
    manifest_path = corpus / "admitted-corpus.json"
    if sha(manifest_path) != MANIFEST:
        raise ValueError("original manifest SHA256 changed")
    m = json.loads(manifest_path.read_text())
    if m.get("source_sha") != SOURCE or m.get("state") != "native-admitted" or m.get("compiler_sha256") != sha(ROOT / "compiler.py"):
        raise ValueError("native source/compiler identity changed")
    if set(m["split_files"]) != {"TRAIN", "DEVELOPMENT", "FINAL"}:
        raise ValueError("invalid original split partitions")
    found = {}
    for split in ("TRAIN", "DEVELOPMENT"):  # Deliberately never opens FINAL target files
        spec = m["split_files"][split]
        file = (corpus / spec["path"]).resolve()
        if not file.is_relative_to(corpus.resolve()) or sha(file) != spec["sha256"]:
            raise ValueError(split + ": source rows failed hash/path custody")
        rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if len(rows) != spec["count"] or len(rows) != 12:
            raise ValueError(split + ": original source cohort changed")
        seen = set()
        for row in rows:
            p = row["program"]
            if isinstance(p, str):
                p = compiler.parse_program(p)
            else:
                compiler.validate_program(p)
            seen.add(signature(p))
        found[split] = seen
    return found


def preflight(item):
    p = item["program"]
    compiler.validate_program(p)
    if p["variation"] != 0 or p["family"] != item["family"]:
        raise ValueError("proposal family or variation invalid")
    receipts = []
    for seed in SEEDS:
        result = compiler.compile_program(p, seed=seed, focus=(0, 0, 96, 96), material=58, allowed_materials=(58,))
        grid = compiler.rasterize_operations(result["operations"], (0, 0, 96, 96))
        if not compiler.verify_structure(result, grid)["passed"]:
            raise ValueError("independent proposal raster preflight failed")
        if not result["structural_preflight"]["passed"] or result["native_qualified"] or not result["proposal_only"]:
            raise ValueError("untrue prototype receipt")
        sem = result["semantic"]
        if item["family"] == "pipework":
            if (sorted(sem["degree"]) != [1, 1, 1, 1, 1, 3, 4] or len(sem["endpoints"]) != 5
                    or len(sem["junctions"]) != 2 or sem["expected_cycles"] != 0):
                raise ValueError("nested branch graph topology witness failed")
        elif item["family"] == "architecture":
            if len(sem["rooms"]) != 4 or len(sem["route"]) != 2 or sem["clearance"] != 5:
                raise ValueError("architecture route/alcove witness failed")
        else:
            if len(sem["panels"]) != 3 or len({x["detail"] for x in sem["panels"]}) != 3:
                raise ValueError("greeble asymmetric panel witness failed")
        receipts.append({"host_seed": seed, "program_sha256": result["program_sha256"],
                         "operations_sha256": result["operations_sha256"],
                         "operation_count": result["operation_count"], "unique_cells": result["unique_cells"],
                         "compiler_preflight": True, "native_qualified": False})
    return receipts


def run(corpus, proposals):
    record = json.loads(proposals.read_text())
    if (record.get("schema") != "cybersand.agent-painter.smol-curriculum-proposals/1"
            or record.get("status") != "PROPOSED_NOT_NATIVE_ADMITTED_NOT_TRAIN"
            or record.get("original_native_source") != SOURCE
            or record.get("original_manifest_sha256") != MANIFEST
            or record.get("fixed_seeds") != list(SEEDS)):
        raise ValueError("input is not the fixed source-bound proposal")
    items = record["proposals"]
    if len(items) != 3 or {p["family"] for p in items} != {"greebles", "pipework", "architecture"}:
        raise ValueError("expected one program per family")
    if len({p["id"] for p in items}) != 3:
        raise ValueError("duplicate proposal identity")
    known = visible_splits(corpus)
    seen = set()
    evidence = []
    for item in items:
        if not item["request"] or not item["witness"]:
            raise ValueError("missing explicit composition witness")
        key = signature(item["program"])
        if key in seen:
            raise ValueError("two candidate proposals share a structure")
        seen.add(key)
        for split in ("TRAIN", "DEVELOPMENT"):
            if key in known[split]:
                raise ValueError(item["id"] + ": structural collision with original " + split)
        composition = inspect_composition(item)
        details = preflight(item)
        evidence.append({"id": item["id"], "family": item["family"],
                         "structural_signature_sha256": hashlib.sha256(key.encode()).hexdigest(),
                         "literal_composition_witness": composition,
                         "compiled_seed_variants": details})
    return {"status": "COMPILER_PREFLIGHT_ONLY_PENDING_REVIEW_AND_POSITIVE_NATIVE_ADMISSION",
            "proposal_sha256": sha(proposals), "native_manifest_sha256": MANIFEST,
            "native_source_sha": SOURCE, "compiler_sha256": sha(ROOT / "compiler.py"),
            "comparison_splits": ["TRAIN", "DEVELOPMENT"], "final_targets_opened": False,
            "actual_training_steps": 0, "native_execution": False, "admitted_teacher_rows": 0,
            "candidates": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, default=Path(__file__).with_name("curriculum_proposals_v2.json"))
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = run(args.corpus, args.proposals)
    except Exception as exc:
        output = {"status": "FAIL_NO_ADMISSION", "error": type(exc).__name__ + ": " + str(exc),
                  "native_execution": False, "admitted_teacher_rows": 0,
                  "final_targets_opened": False}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
        raise
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "proposals": len(output["candidates"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
