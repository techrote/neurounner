#!/usr/bin/env python3
"""P01 vs R10: read-only, original native Art material peer diagnostic.

One SMOL P01 tree and one SPROCKET R10 ladder, at the same two host-owned
seeds. Different graph classes do not imply split safety; proximity is only
a descriptive measurement. Never open FINAL or treat native research as TRAIN.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image
from check_p01_admission_review import check as check_p01
from p01_native_visible_proximity import (
    SEEDS, CANDIDATE, P01_ZIP, cells_from_i32, compare, digest, require
)

R10_ZIP = "f7e673c414487668598eb26dc618daa019b2fc83bc70ed373cd574049f20e53c"
R10_SOURCE = "ab101910921a7cf7f30b3dc83d150d214aaa060d"
R10_PROGRAM = "0f52d4f6b252d17f9f335e8c8a65a44f78626e79cd26298881a78446f3f7b745"
PREFIX = "sprocket-rich-v2-native/"


def archive(path: Path, expected: str) -> zipfile.ZipFile:
    require(path.is_file() and not path.is_symlink(), "missing or symlink original native ZIP")
    require(digest(path.read_bytes()) == expected, "original native archive SHA mismatch")
    z = zipfile.ZipFile(path)
    names = z.namelist()
    require(len(names) == len(set(names)), "duplicate original ZIP member")
    require(all(not n.startswith("/") and "\\" not in n and
                ".." not in Path(n).parts for n in names), "unsafe original ZIP member")
    require(z.testzip() is None, "original ZIP CRC mismatch")
    return z


def r10_native(z: zipfile.ZipFile) -> tuple[dict[int, set[int]], list[dict]]:
    summary = json.loads(z.read(PREFIX + "qualification-summary.json"))
    verified = json.loads(z.read(PREFIX + "native-verification.json"))
    produced = json.loads(z.read(PREFIX + "candidates.json"))
    require(summary.get("source_sha") == verified.get("source_sha") == R10_SOURCE,
            "R10 source mismatch")
    require(summary.get("native_verified") is True and verified.get("passed") is True
            and summary.get("shared_train_groups_admitted") == 0
            and summary.get("model_training") is False
            and summary.get("final_targets_opened") is False,
            "R10 native research authority drift")
    check = {r["id"]: r for r in verified["cases"]}
    examples = {r["id"]: r for r in produced["examples"]}
    require(len(check) == len(verified["cases"]) and
            len(examples) == len(produced["examples"]), "R10 duplicate case identity")
    masks, evidence = {}, []
    for seed in SEEDS:
        case_id = f"sprocket-R10-s{seed}"
        require(case_id in check and case_id in examples, "missing R10 seed case")
        b, a = check[case_id], examples[case_id]
        require(a["seed"] == b["seed"] == seed and
                a["split"] == "PROPOSAL_ONLY" and
                b["program_sha256"] == R10_PROGRAM and
                b["structure"]["passed"] is True and
                b["acknowledged_cells"] == 1724,
                "R10 original proposal/verification mismatch")
        native_name = PREFIX + case_id
        raw = z.read(native_name + ".materials.i32le")
        png = z.read(native_name + ".png")
        matpng = z.read(native_name + ".materials.png")
        rec = z.read(native_name + ".receipt.json")
        require(digest(raw) == b["readback_sha256"] and
                digest(png) == b["image_sha256"] and
                digest(rec) == b["receipt_sha256"],
                "R10 original receipt/hash mismatch")
        receipt = json.loads(rec)
        require(receipt.get("passed") is True and
                receipt["crop_i32le_sha256"] == digest(raw),
                "R10 original native receipt not positive")
        cells = cells_from_i32(raw)
        require(len(cells) == 1724, "R10 material cell count drift")
        material = Image.open(BytesIO(matpng))
        display = Image.open(BytesIO(png))
        require(material.mode == "L" and material.size == (96, 96) and
                display.mode == "RGB" and display.size == (96, 96),
                "R10 native image size/mode mismatch")
        expected = bytes(58 if i in cells else 0 for i in range(9216))
        require(material.tobytes() == expected, "R10 native mask PNG/readback mismatch")
        require(all((display.getpixel((i % 96, i // 96)) == (232, 226, 205))
                    == (i in cells) for i in range(9216)),
                "R10 display PNG/readback mismatch")
        masks[seed] = cells
        evidence.append({"seed": seed, "readback_sha256": digest(raw),
                         "display_png_sha256": digest(png),
                         "receipt_sha256": digest(rec),
                         "acknowledged_material_cells": len(cells)})
    require(len(masks) == 2, "R10 must have two native seed views")
    return masks, evidence


def audit(p01_path: Path, r10_path: Path, proposals: Path, offer: Path) -> dict:
    old = check_p01(offer, proposals, p01_path)
    require(old["status"] == "PASS_REVIEW_ONLY_NOT_SHARED_TRAIN_ADMISSION",
            "P01 original native admission review not qualified")
    with archive(p01_path, P01_ZIP) as p, archive(r10_path, R10_ZIP) as r:
        r10_masks, receipts = r10_native(r)
        p01_masks = {}
        original_offer = json.loads(offer.read_text())
        for seed in SEEDS:
            name = "smol-v2-native/" + CANDIDATE + f"-s{seed}.materials.i32le"
            raw = p.read(name)
            require(digest(raw) == original_offer["cases"][str(seed)]["material_sha256"],
                    "P01 material hash mismatch")
            cells = cells_from_i32(raw)
            require(len(cells) == 596, "P01 native material count changed")
            p01_masks[seed] = cells
        rows = [{"seed": seed, **compare(p01_masks[seed], r10_masks[seed])}
                for seed in SEEDS]
    require(len(rows) == 2 and {r["seed"] for r in rows} == set(SEEDS),
            "original two-seed peer comparison incomplete")
    return {
        "schema": "cybersand.smol-p01-r10-original-native-peer/1",
        "status": "ORIGINAL_NATIVE_PEER_DIAGNOSTIC_REVIEW_ONLY",
        "p01_original_native_source": old["source_sha"],
        "p01_original_zip_sha256": P01_ZIP,
        "r10_original_native_source": R10_SOURCE,
        "r10_original_zip_sha256": R10_ZIP,
        "r10_program_sha256": R10_PROGRAM,
        "original_R10_receipts": receipts,
        "host_seed_views": list(SEEDS),
        "comparisons": rows,
        "independent_compositions": 2,
        "seed_comparison_pairs": 2,
        "new_native_executions": 0,
        "optimizer_updates": 0,
        "model_generated_views": 0,
        "new_shared_train_admissions": 0,
        "final_access": False,
        "admission": "HOLD_REVIEWER_AND_SHEPHERD",
        "limitation": "IoU/proximity cannot prove graph/silhouette novelty; actual research-native drawings, not model paintings; check visual originals separately.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p01", required=True, type=Path)
    ap.add_argument("--r10", required=True, type=Path)
    ap.add_argument("--proposals", required=True, type=Path)
    ap.add_argument("--offer", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    require(not a.output.exists(), "immutable report already exists")
    result = audit(a.p01, a.r10, a.proposals, a.offer)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "pairs": len(result["comparisons"]),
                      "new_train": result["new_shared_train_admissions"]}))


if __name__ == "__main__":
    main()
