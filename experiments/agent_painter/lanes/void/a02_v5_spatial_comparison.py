#!/usr/bin/env python3
"""Compare immutable A02 v4/v5 native material masks against visible public Architecture.

Research-only: no FINAL extraction, candidate tuning, model, compiler, native rerun or TRAIN.
Original v4/v5 and source633 archives are immutable, SHA-256 pinned inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import zipfile

PUBLIC_ZIP = "50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39"
V4_ZIP = "deeabf0c0a79b924bdc6092b4f9514b376ae5e3bdbe013571bea46bbd472c38a"
V5_ZIP = "d8f1c1dbce65ef14d392a7bd557bd89e8af21563da14d4a5a61b84386285944b"
V5_CROP = "14738d7bd6747f2137c2604e1e7afa615622a7d31c2adf785bd64f95e3c67cf7"
V5_SOURCE = "64b5f1ca5b0f65ffbf9a7645a0d0828d35657f08"
V5_PROGRAM = "9df29e3d7f5d97d68ee69708882e3bd6b7b19a5ee3d019ca0a2a86145cc1aeba"
PUBLIC_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
SPLITS = {"TRAIN": "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1",
          "DEVELOPMENT": "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572"}
SEEDS = (104729, 130363)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_zip(path, expected, public=False):
    need(path.is_file() and not path.is_symlink(), "original artifact missing or symlink")
    need(sha(path.read_bytes()) == expected, "whole original ZIP hash mismatch")
    z = zipfile.ZipFile(path)
    need(len(z.namelist()) == len(set(z.namelist())), "duplicate ZIP members")
    # Public ZIP contains FINAL: never call testzip() or read arbitrary members.
    if not public:
        need(z.testzip() is None, "original native artifact CRC mismatch")
    return z


def mask(raw):
    need(len(raw) == 96 * 96 * 4, "original native material dimensions differ")
    cells = struct.unpack("<9216i", raw)
    need(set(cells) <= {0, 58}, "unexpected native material ID")
    return {index for index, material in enumerate(cells) if material == 58}


def iou(a, b):
    need(bool(a) and bool(b), "empty material mask")
    return len(a & b) / len(a | b)


def proximity_fraction(a, b, radius=3):
    """Fraction of A cells within Chebyshev radius of B at original coordinates."""
    need(bool(a) and bool(b) and type(radius) is int and 0 <= radius <= 8,
         "bad proximity arguments")
    near = set()
    for cell in b:
        x, y = cell % 96, cell // 96
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 96 and 0 <= ny < 96:
                    near.add(ny * 96 + nx)
    return len(a & near) / len(a)


def native_v5(z):
    reports = [n for n in z.namelist() if n == "native-verification.json"
               or n.endswith("/native-verification.json")]
    need(len(reports) == 1, "expected exactly one original native verifier")
    report = json.loads(z.read(reports[0]))
    need(report.get("passed") is True and report.get("source_sha") == V5_SOURCE
         and not report.get("failures"), "original native source/verification failure")
    gates = report.get("global_gates", {})
    need(len(gates) == 16 and all(v is True for v in gates.values()),
         "native safety gates incomplete")
    root = PurePosixPath(reports[0]).parent
    cases = report.get("cases", [])
    need(len(cases) == 2 and {r.get("id") for r in cases}
         == {f"void-v5-A02-floor-anchored-s{s}" for s in SEEDS},
         "missing original native host-seed cases")
    masks = []
    for case in cases:
        need(case.get("success") is True and case.get("program_sha256") == V5_PROGRAM
             and case.get("readback_sha256") == V5_CROP
             and case.get("filled_cells") == case.get("acknowledged_cells") == 662
             and case.get("split") == "UNADMITTED_TRAIN_PROPOSAL",
             "original native case/receipt identity differs")
        member = PurePosixPath(case["raw_materials"])
        need(not member.is_absolute() and ".." not in member.parts,
             "unsafe native material path")
        raw = z.read(str(root / member))
        need(sha(raw) == V5_CROP, "original native material bytes changed")
        masks.append(mask(raw))
    need(masks[0] == masks[1] and len(masks[0]) == 662,
         "A02 v5 host seeds cannot count as separate compositions")
    return masks[0]


def original_evidence(z, path, expected_hash):
    """Read only a selected visible native admission member and verify its SHA."""
    need(isinstance(path, str) and isinstance(expected_hash, str),
         "original visible native path/hash required")
    name = PurePosixPath(path)
    need(not name.is_absolute() and len(name.parts) >= 2
         and name.parts[0] == "evidence" and ".." not in name.parts,
         "unsafe original visible evidence path")
    data = z.read("corpus/" + str(name))
    need(sha(data) == expected_hash, "original visible native evidence SHA mismatch")
    return data


def visible_architecture(z, split, manifest):
    need(split in SPLITS, "sealed/unknown split refused")
    need(manifest.get("source_sha") == PUBLIC_SOURCE
         and manifest["split_files"][split]["sha256"] == SPLITS[split]
         and manifest["split_files"][split]["count"] == 12,
         "original corpus split/source identity differs")
    raw = z.read(f"corpus/{split.lower()}.jsonl")
    need(sha(raw) == SPLITS[split], "visible original split digest changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    need(len(rows) == 12 and len({r["group_id"] for r in rows}) == 6,
         "original structural grouping changed")
    selected = [r for r in rows if r["family"] == "architecture"]
    groups = {r["group_id"] for r in selected}
    need(len(selected) == 4 and len(groups) == 2
         and len({r["id"] for r in selected}) == 4
         and {(r["group_id"], r["seed"]) for r in selected}
             == {(group, seed) for group in groups for seed in SEEDS},
         "original Architecture requires both native seeds for each of two groups")
    need(groups <= set(manifest["structural_groups"][split]),
         "original Architecture group escaped its visible split")
    evidence = []
    for row in sorted(selected, key=lambda r: (r["group_id"], r["seed"])):
        need(row.get("split") == split
             and row.get("structural_group", row["group_id"]) == row["group_id"],
             "original visible row/structural group mismatch")
        admission = row.get("native_admission", {})
        need(admission.get("state") == "PASS"
             and admission.get("native_execution") is True
             and admission.get("verified") is True
             and admission.get("source_sha") == PUBLIC_SOURCE
             and admission.get("runtime_sha256") == manifest["runtime_sha256"],
             "original visible native admission not source-bound")
        paths = admission["paths"]
        raw_mask = original_evidence(z, paths["readback"], admission["readback_sha256"])
        receipt_raw = original_evidence(z, paths["receipt"], admission["receipt_sha256"])
        original_evidence(z, paths["png"], admission["png_sha256"])
        receipt = json.loads(receipt_raw)
        need(receipt.get("id") == row["id"] and receipt.get("seed") == row["seed"]
             and receipt.get("family") == "architecture"
             and receipt.get("passed") is True
             and receipt.get("program_sha256") == admission["program_sha256"]
             and receipt.get("operations_sha256") == admission["operations_sha256"]
             and receipt.get("crop_i32le_sha256") == admission["readback_sha256"]
             and receipt.get("outside_area_cells") == 0
             and receipt.get("unexpected_material_cells") == 0
             and receipt.get("deterministic_replay_checked") is True,
             "original visible native receipt not bound to seed/material")
        evidence.append((row["group_id"], row["seed"], mask(raw_mask),
                         admission["readback_sha256"], admission["receipt_sha256"],
                         admission["png_sha256"]))
    return evidence

def compare(v4, v5, peers):
    need(len(v4) == 524 and len(v5) == 662 and v4 < v5
         and len(v5 - v4) == 138, "historical v4-to-v5 138-cell-only addition differs")
    need(len(peers) == 8, "eight original Architecture seed views required")
    keys = [(split, group, seed) for split, group, seed, *_ in peers]
    groups = {(split, group) for split, group, _ in keys}
    need(len(groups) == 4 and {split for split, _ in groups} == set(SPLITS)
         and all(sum(part == split for part, _ in groups) == 2 for split in SPLITS)
         and set(keys) == {(split, group, seed)
                           for split, group in groups for seed in SEEDS},
         "duplicate or missing reference seed/structural group")
    values = []
    for split, name, seed, other, material_hash, receipt_hash, png_hash in peers:
        need(all(isinstance(h, str) and len(h) == 64 for h in
                 (material_hash, receipt_hash, png_hash)),
             "missing original reference evidence hashes")
        values.append({"split": split, "group": name, "reference_seed": seed,
                       "reference_readback_sha256": material_hash,
                       "reference_receipt_sha256": receipt_hash,
                       "reference_png_sha256": png_hash,
                       "peer_cells": len(other),
                       "v4_iou": round(iou(v4, other), 6),
                       "v5_iou": round(iou(v5, other), 6),
                       "v4_near3": round(proximity_fraction(v4, other), 6),
                       "v5_near3": round(proximity_fraction(v5, other), 6)})
    return values

def audit(v4_file, v5_file, public_file):
    with checked_zip(v4_file, V4_ZIP) as four, checked_zip(v5_file, V5_ZIP) as five, \
         checked_zip(public_file, PUBLIC_ZIP, public=True) as public:
        raw_manifest = public.read("corpus/admitted-corpus.json")
        manifest = json.loads(raw_manifest)
        need(sha(raw_manifest) == MANIFEST and
             manifest.get("source_sha") == PUBLIC_SOURCE,
             "original native corpus manifest/source mismatch")
        v4 = mask(four.read("void-v4-shaft-native/void-v4-A02-s104729.materials.i32le"))
        need(v4 == mask(four.read("void-v4-shaft-native/void-v4-A02-s130363.materials.i32le")),
             "v4 host seeds differ")
        v5 = native_v5(five)
        peers = [(split, group, seed, cells, readback_sha, receipt_sha, png_sha) for split in SPLITS
                 for group, seed, cells, readback_sha, receipt_sha, png_sha in visible_architecture(public, split, manifest)]
        return {"status": "READ_ONLY_A02_V5_SPATIAL_COMPARISON",
                "v4_zip_sha256": V4_ZIP, "v5_zip_sha256": V5_ZIP,
                "public_zip_sha256": PUBLIC_ZIP, "native_corpus_manifest_sha256": MANIFEST,
                "opened_splits": ["TRAIN", "DEVELOPMENT"], "final_targets_opened": False,
                "original_native_v4_cells": len(v4), "original_native_v5_cells": len(v5),
                "added_v5_cells": len(v5 - v4), "removed_v5_cells": len(v4 - v5),
                "distinct_architecture_compositions": 1, "v5_host_seed_views": 2,
                "reference_structural_groups": 4, "reference_native_seed_views": len(peers),
                "per_visible_original_architecture_group": compare(v4, v5, peers),
                "new_native_executions": 0, "new_train_admissions": 0,
                "new_optimizer_steps": 0, "model_generated_paintings": 0,
                "global_split_admission": "REVIEW_ONLY_HOLD",
                "limitations": ("Original-coordinate IoU and Chebyshev-3 occupied-cell proximity "
                                "are only spatial diagnostics, not translation invariance, "
                                "topological proof, global independence or model painter quality.")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v4-zip", type=Path, required=True)
    p.add_argument("--v5-zip", type=Path, required=True)
    p.add_argument("--public-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    need(not a.output.exists(), "do not overwrite an earlier evidence receipt")
    result = audit(a.v4_zip, a.v5_zip, a.public_zip)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
