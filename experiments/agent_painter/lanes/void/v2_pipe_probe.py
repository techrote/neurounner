#!/usr/bin/env python3
"""VOID lane-only structural preflight; not positive native Art admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compiler import compile_program, parse_program, structural_signature, digest

MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
NATIVE = "633920373a3d406a2ede61ce4137c324f76be23b"
FROZEN = {"TRAIN": "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1",
          "DEVELOPMENT": "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"}
SEEDS = (104729, 130363)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(corpus, proposals):
    manifest_file = corpus / "admitted-corpus.json"
    if manifest_file.is_symlink() or sha(manifest_file) != MANIFEST:
        raise ValueError("original native corpus manifest digest mismatch")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest["source_sha"] != NATIVE or manifest["state"] != "native-admitted":
        raise ValueError("original positive native identity differs")
    old, groups = {}, {}
    for split, expected in FROZEN.items():
        spec = manifest["split_files"][split]
        f = corpus / spec["path"]
        if (spec["path"] != split.lower()+".jsonl" or spec["sha256"] != expected
                or spec["count"] != 12 or f.is_symlink() or sha(f) != expected):
            raise ValueError("original split metadata/hash mismatch: "+split)
        # Only structural identity metadata are extracted. Do not use DEVELOPMENT
        # target programs; do not access FINAL case data or image files.
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] != split:
                raise ValueError("foreign split")
            rows.append((r["family"], r["group_id"], r["structural_signature_sha256"]))
        groups[split] = len(set(r[1] for r in rows))
        if len(rows) != 12 or groups[split] != 6:
            raise ValueError("original row/group counts differ")
        for family, group_id, signature in rows:
            old.setdefault(signature, []).append((split, family, group_id))
    document = json.loads(proposals.read_text(encoding="utf-8"))
    if (document["schema"] != "void-pipework-stage2-proposals/1"
            or document["state"] != "proposal-only"
            or document["no_final_targets_opened"] is not True
            or document["reference_native_source"] != NATIVE
            or document["reference_manifest_sha256"] != MANIFEST
            or len(document["candidates"]) != 5):
        raise ValueError("new stage source identity differs")
    ids, signatures, output = set(), set(), []
    for c in document["candidates"]:
        if c["id"] in ids or c["training_admitted"] is not False:
            raise ValueError("duplicate ID or false TRAIN admission claim")
        ids.add(c["id"])
        p = parse_program(json.dumps(c["program"], sort_keys=True, separators=(",", ":")))
        if p["family"] != "pipework" or p["components"][0]["diameter"] != 3:
            raise ValueError("unsupported family or pipe width")
        signature = digest(structural_signature(p))
        if signature in signatures or signature in old:
            raise ValueError("frozen original/candidate graph-signature collision: " + c["id"])
        signatures.add(signature)
        views = []
        for seed in SEEDS:
            v = compile_program(p, seed=seed, focus=(0, 0, 96, 96), material=58,
                                allowed_materials=(58,), max_operations=512, max_cell_work=100000)
            s = v["semantic"]
            witness = {"cycles": s["expected_cycles"], "endpoints": len(s["endpoints"]),
                       "junctions": len(s["junctions"]), "degree4": s["degree"].count(4)}
            if (witness != c["witness"] or v["structural_preflight"]["passed"] is not True
                    or len(s["diameter_probes"]) != len(p["components"][0]["edges"])):
                raise ValueError("semantic pipe-topology witness failure: " + c["id"])
            views.append({"seed": seed, "operations_sha256": v["operations_sha256"],
                          "rectangles": v["operation_count"], "cells": v["unique_cells"],
                          "compiler_preflight_pass": True, "actual_native_execution": False})
        output.append({"id": c["id"], "program_sha256": digest(p),
                       "signature_sha256": signature, "original_signature_collision": False,
                       "native_admitted": False, "views": views, "witness": c["witness"]})
    return {"status": "SOURCE_ONLY_PASS", "source_only": True, "proposals": output,
            "distinct_compositions": len(output), "seed_compilations": 2*len(output),
            "original_public_groups_screened": groups, "native_admitted_rows": 0,
            "real_native_replays": 0, "optimizer_updates": 0, "final_targets_opened": False,
            "next_gate": "independent structural reviewer and real source-built Godot/Art two-seed verification"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--proposals", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = verify(a.corpus, a.proposals)
    except Exception as exc:
        out = {"status": "FAIL", "error": type(exc).__name__+": "+str(exc),
               "source_only": True, "real_native_replays": 0,
               "optimizer_updates": 0, "final_targets_opened": False}
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({k: out.get(k) for k in ("status", "error", "distinct_compositions", "seed_compilations")}))
    return 0 if out["status"] == "SOURCE_ONLY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
