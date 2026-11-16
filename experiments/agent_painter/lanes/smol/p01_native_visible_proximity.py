#!/usr/bin/env python3
"""P01 versus native-admitted *visible* Pipework: original-byte review only.

Read precisely TRAIN and DEVELOPMENT JSONL and their cited PNG/material/receipt
members. Never open FINAL records, targets or images. Same-coordinate overlap
and Chebyshev-3 proximity are diagnostics, not admission decisions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import struct
import zipfile

from check_p01_admission_review import check as check_p01
from visible_split_neighbours import canonical_tree

CORPUS_ZIP = "50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39"
P01_ZIP = "71ddc40478e06e3350f05eecd5e16ed77c082300f066be97bbccf8afb2370ff1"
MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
SPLITS = {
    "TRAIN": "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1",
    "DEVELOPMENT": "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572",
}
SEEDS = (104729, 130363)
CANDIDATE = "smol_p01_nested_branch_tree"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def check_archive(path: Path, expected: str) -> zipfile.ZipFile:
    require(path.is_file() and not path.is_symlink(), "missing/unsafe original archive")
    require(digest(path.read_bytes()) == expected, "original archive SHA256 mismatch")
    z = zipfile.ZipFile(path)
    names = z.namelist()
    require(len(names) == len(set(names)), "duplicate original archive members")
    return z


def safe_evidence(name: str) -> str:
    require(isinstance(name, str), "missing evidence path")
    parts = PurePosixPath(name).parts
    require(len(parts) == 2 and parts[0] == "evidence" and
            parts[1] not in (".", "..") and "\\" not in name,
            "unsafe or non-visible native evidence path")
    return "corpus/" + name


def cells_from_i32(raw: bytes) -> set[int]:
    require(len(raw) == 96 * 96 * 4, "unexpected source material crop")
    values = struct.unpack("<9216i", raw)
    require(set(values) <= {0, 58}, "unexpected source material ID")
    return {i for i, value in enumerate(values) if value == 58}


def chebyshev_coverage(a: set[int], b: set[int], radius: int = 3) -> tuple[int, int]:
    require(0 <= radius <= 4, "unbounded radius")
    require(bool(a), "empty candidate material")
    hits = 0
    for pixel in a:
        x, y = pixel % 96, pixel // 96
        if any(0 <= x + dx < 96 and 0 <= y + dy < 96 and
               ((y + dy) * 96 + x + dx) in b
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)):
            hits += 1
    return hits, len(a)


def compare(a: set[int], b: set[int]) -> dict:
    require(bool(a) and bool(b), "empty native material mask")
    forward = chebyshev_coverage(a, b)
    reverse = chebyshev_coverage(b, a)
    intersection = len(a & b)
    return {
        "candidate_cells": len(a),
        "visible_cells": len(b),
        "same_coordinate_intersection": intersection,
        "same_coordinate_union": len(a | b),
        "same_coordinate_iou": round(intersection / len(a | b), 6),
        "candidate_near_visible_radius3_hits": forward[0],
        "candidate_near_visible_radius3_denominator": forward[1],
        "candidate_near_visible_radius3_fraction": round(forward[0] / forward[1], 6),
        "visible_near_candidate_radius3_hits": reverse[0],
        "visible_near_candidate_radius3_denominator": reverse[1],
        "visible_near_candidate_radius3_fraction": round(reverse[0] / reverse[1], 6),
    }


def visible_pipework(z: zipfile.ZipFile) -> tuple[dict, list[dict]]:
    manifest_bytes = z.read("corpus/admitted-corpus.json")
    require(digest(manifest_bytes) == MANIFEST, "corpus manifest drift")
    manifest = json.loads(manifest_bytes)
    require(manifest.get("source_sha") == SOURCE and
            manifest.get("state") == "native-admitted" and
            set(manifest.get("split_files", {})) == {"TRAIN", "DEVELOPMENT", "FINAL"},
            "unapproved source or partitions")
    out = defaultdict(list)
    structures = []
    for split in ("TRAIN", "DEVELOPMENT"):
        spec = manifest["split_files"][split]
        require(spec["path"] == split.lower() + ".jsonl" and
                spec["sha256"] == SPLITS[split] and spec["count"] == 12,
                "visible split identity drift")
        raw = z.read("corpus/" + spec["path"])
        require(digest(raw) == SPLITS[split], "visible split bytes changed")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        require(len(rows) == 12 and len({r["id"] for r in rows}) == 12 and
                len({r["structural_group"] for r in rows}) == 6,
                "visible split row/group census mismatch")
        pipes = [r for r in rows if r["family"] == "pipework"]
        require(len(pipes) == 4 and len({r["structural_group"] for r in pipes}) == 2,
                "original visible Pipework cohort mismatch")
        groups = defaultdict(list)
        for r in pipes:
            groups[r["structural_group"]].append(r)
        for group, items in sorted(groups.items()):
            require(len(items) == 2 and sorted(r["seed"] for r in items) == list(SEEDS),
                    "seed variants must be one whole group")
            programs = [r["program"] for r in items]
            require(programs[0] == programs[1], "structural seed program disagreement")
            tree = canonical_tree(programs[0])
            structures.append({"split": split, "group": group,
                               "tree_sha256": digest(tree.encode()) if tree is not None else None,
                               "tree_supported": tree is not None})
            for r in items:
                admission = r["native_admission"]
                require(r.get("native_qualified") is True and
                        admission["verified"] is True and admission["native_execution"] is True and
                        admission["state"] == "PASS" and admission["source_sha"] == SOURCE,
                        "missing positive original Art admission")
                rawpath = safe_evidence(admission["paths"]["readback"])
                rawgrid = z.read(rawpath)
                require(digest(rawgrid) == admission["readback_sha256"],
                        "original material byte mismatch")
                for key in ("receipt", "png"):
                    member = safe_evidence(admission["paths"][key])
                    require(digest(z.read(member)) == admission[key + "_sha256"],
                            "original native " + key + " hash mismatch")
                receipt = json.loads(z.read(safe_evidence(admission["paths"]["receipt"])))
                require(receipt.get("passed") is True and receipt["seed"] == r["seed"] and
                        receipt["crop_i32le_sha256"] == admission["readback_sha256"] and
                        receipt["program_sha256"] == admission["program_sha256"] and
                        receipt["deterministic_replay_checked"] is True,
                        "original positive native receipt binding failed")
                material = cells_from_i32(rawgrid)
                require(len(material) == admission["affected_cells"] and
                        len(material) == admission["acknowledged_cells"],
                        "original material/acknowledgment count mismatch")
                out[(split, group, r["seed"])].append(material)
    require(len(structures) == 4 and len(out) == 8, "visible group/seed count")
    return out, structures


def evaluate(corpus_zip: Path, native_zip: Path, proposal_path: Path, offer_path: Path) -> dict:
    # The existing independently qualified P01 verifier binds the native PNG,
    # material grid, receipt, 596-cell acknowledgements and exact authored program.
    p01_proof = check_p01(offer_path, proposal_path, native_zip)
    require(p01_proof["status"] == "PASS_REVIEW_ONLY_NOT_SHARED_TRAIN_ADMISSION",
            "P01 native research qualification not established")
    proposals = json.loads(proposal_path.read_text())["proposals"]
    program = next(p["program"] for p in proposals if p["id"] == CANDIDATE)
    candidate_tree = canonical_tree(program)
    require(candidate_tree is not None, "P01 must be one valid connected tree")
    with check_archive(corpus_zip, CORPUS_ZIP) as corpus, check_archive(native_zip, P01_ZIP) as native:
        visible, structures = visible_pipework(corpus)
        cases = []
        for split in ("TRAIN", "DEVELOPMENT"):
            groups = sorted({group for sp, group, _seed in visible if sp == split})
            for group in groups:
                s = next(x for x in structures if x["split"] == split and x["group"] == group)
                s["same_unlabeled_tree_as_P01"] = (
                    s["tree_sha256"] == digest(candidate_tree.encode())
                    if s["tree_supported"] else None)
                for seed in SEEDS:
                    p01_name = "smol-v2-native/" + CANDIDATE + "-s" + str(seed) + ".materials.i32le"
                    original = native.read(p01_name)
                    expected = json.loads(offer_path.read_text())["cases"][str(seed)]["material_sha256"]
                    require(digest(original) == expected, "P01 native material digest drift")
                    p01_cells = cells_from_i32(original)
                    require(len(p01_cells) == 596, "P01 physical cell count drift")
                    native_cells = visible[(split, group, seed)]
                    require(len(native_cells) == 1, "duplicate visible seed")
                    cases.append({"split": split, "visible_group": group, "seed": seed,
                                  **compare(p01_cells, native_cells[0])})
    return {
        "schema": "cybersand.smol-p01-native-visible-proximity/1",
        "status": "READ_ONLY_SPLIT_AND_NATIVE_DIAGNOSTIC_HOLD",
        "original_corpus_zip_sha256": CORPUS_ZIP,
        "original_native_source_sha": SOURCE,
        "original_manifest_sha256": MANIFEST,
        "visible_split_sha256": SPLITS,
        "original_P01_zip_sha256": P01_ZIP,
        "original_P01_native_source_sha": p01_proof["source_sha"],
        "P01_program_tree_sha256": digest(candidate_tree.encode()),
        "P01_original_native_seed_views": 2,
        "visible_original_Pipework_structural_groups": structures,
        "original_visible_native_seed_views_read": 8,
        "same_seed_comparisons": cases,
        "comparisons": len(cases),
        "new_native_executions": 0,
        "new_shared_train_admissions": 0,
        "new_optimizer_steps": 0,
        "model_generated_paintings": 0,
        "final_targets_or_images_opened": False,
        "split_safe_or_artistic_independence_approved": False,
        "next_gate": "Independent reviewer whole-structure/visual decision; shepherd shared native TRAIN admission and common frozen DEVELOPMENT suite",
        "limitation": "Coordinates/proximity are descriptive, not a hard novelty threshold or full layout/graph certificate; two seeds are one composition.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-zip", required=True, type=Path)
    p.add_argument("--native-zip", required=True, type=Path)
    p.add_argument("--proposals", required=True, type=Path)
    p.add_argument("--offer", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    require(not a.output.exists(), "refuse evidence overwrite")
    result = evaluate(a.corpus_zip, a.native_zip, a.proposals, a.offer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"],
                      "new_shared_train_admissions": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
