#!/usr/bin/env python3
"""SPROCKET A07/A08: original source-built native Architecture vs visible original splits.

Research only. Only named TRAIN/DEVELOPMENT ZIP entries are read. FINAL remains
unopened. Both pinned host seeds are views, never new structural groups.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import zipfile

from PIL import Image
from threefamily_v3 import program_for
from r10_native_mask_proximity import metrics, cells, must, SIDE

CORPUS_ZIP = "50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39"
NATIVE_ZIP = "3a7397883b146a28da990f0a37b891897285c2c7c81cd9dd2aaa3895d0d196dd"
CORPUS_MANIFEST = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
CORPUS_SOURCE = "633920373a3d406a2ede61ce4137c324f76be23b"
NATIVE_SOURCE = "63ee8b2b561068a72e0652e54b6e4193cb47d5e6"
NATIVE_PREFIX = "sprocket-v3-native/"
SEEDS = (104729, 130363)
CANDIDATES = {"A07": 781, "A08": 924}
GROUPS = {
    "train": {"architecture/stair_and_landing_hall/v1", "architecture/paired_doorway_rooms/v1"},
    "development": {"architecture/stacked_rooms_and_shaft/v1", "architecture/bridge_between_towers/v1"},
}

def digest(data: bytes) -> str:
    return sha256(data).hexdigest()

def original_pixels(raw: bytes, material_png: bytes, display_png: bytes):
    occupied, values = cells(raw)
    mat = Image.open(BytesIO(material_png))
    must(mat.mode == "L" and mat.size == (SIDE, SIDE)
         and mat.tobytes() == bytes(values), "native material PNG disagrees with i32le")
    rgb = Image.open(BytesIO(display_png))
    must(rgb.mode == "RGB" and rgb.size == (SIDE, SIDE)
         and all((rgb.getpixel((i % SIDE, i // SIDE)) == (232,226,205)) == (v == 58)
                 for i,v in enumerate(values)),
         "native RGB PNG disagrees with material readback")
    return occupied

def require_exact_cases(rows, expected):
    seen = [(row["structural_group"], row["seed"]) for row in rows]
    must(len(seen) == len(set(seen)), "duplicate visible split structure/seed")
    must(set(seen) == {(group, seed) for group in expected for seed in SEEDS},
         "missing or unexpected visible Architecture group/seed")

def audit(corpus_path: Path, native_path: Path):
    must(digest(corpus_path.read_bytes()) == CORPUS_ZIP, "original corpus ZIP hash drift")
    must(digest(native_path.read_bytes()) == NATIVE_ZIP, "original native ZIP hash drift")
    masks = {}
    cases = []
    with zipfile.ZipFile(native_path) as native:
        # Do not testzip() the corpus: sealed FINAL entries must not be decompressed.
        summary = json.loads(native.read(NATIVE_PREFIX + "qualification-summary.json"))
        verifier = json.loads(native.read(NATIVE_PREFIX + "native-verification.json"))
        produced = json.loads(native.read(NATIVE_PREFIX + "candidates.json"))["examples"]
        must(summary["source_sha"] == verifier["source_sha"] == NATIVE_SOURCE
             and summary["native_verified"] is True
             and summary["actual_native_positive_views"] == 8
             and summary["shared_train_groups_admitted"] == 0
             and summary["model_training"] is False
             and summary["model_generated_views"] == 0
             and summary["final_targets_opened"] is False
             and verifier["passed"] is True,
             "original four-design native Art research not positive")
        examples = {r["id"]: r for r in produced}
        checks = {r["id"]: r for r in verifier["cases"]}
        must(len(examples) == len(produced) == 8 and len(checks) == len(verifier["cases"]) == 8,
             "duplicate or missing original research case ID")
        for candidate, count in CANDIDATES.items():
            original_program = program_for(candidate)
            for seed in SEEDS:
                name = f"sprocket-{candidate}-s{seed}"
                must(name in examples and name in checks,
                     "original Architecture native case missing")
                ex, ck = examples[name], checks[name]
                must(ex["family"] == "architecture" and ex["seed"] == seed
                     and ex["split"] == "PROPOSAL_ONLY" and ex["program"] == original_program
                     and ck["seed"] == seed and ck["success"] is True
                     and ck["deterministic_replay"] is True
                     and ck["acknowledged_cells"] == count
                     and ck["affected_cells"] == count,
                     "candidate program or native acknowledgement mismatch")
                raw = native.read(NATIVE_PREFIX + name + ".materials.i32le")
                material = native.read(NATIVE_PREFIX + name + ".materials.png")
                display = native.read(NATIVE_PREFIX + name + ".png")
                receipt_bytes = native.read(NATIVE_PREFIX + name + ".receipt.json")
                receipt = json.loads(receipt_bytes)
                mask = original_pixels(raw, material, display)
                must(len(mask) == count and digest(raw) == ck["readback_sha256"]
                     and digest(display) == ck["image_sha256"]
                     and digest(receipt_bytes) == ck["receipt_sha256"]
                     and receipt["passed"] is True
                     and receipt["terminal"]["status"] == "completed"
                     and receipt["id"] == name and int(receipt["seed"]) == seed
                     and receipt["terminal"]["changed"] == count,
                     "original Architecture native bytes/receipt not positive")
                masks[(candidate, seed)] = mask
                cases.append({"candidate": candidate, "seed": seed,
                              "acknowledged_cells": count,
                              "readback_sha256": digest(raw),
                              "display_png_sha256": digest(display),
                              "receipt_sha256": digest(receipt_bytes)})
    rows_out = []
    with zipfile.ZipFile(corpus_path) as corpus:
        manifest_bytes = corpus.read("corpus/admitted-corpus.json")
        must(digest(manifest_bytes) == CORPUS_MANIFEST, "original corpus manifest drift")
        manifest = json.loads(manifest_bytes)
        must(manifest["source_sha"] == CORPUS_SOURCE, "original corpus producer drift")
        for split, expected in GROUPS.items():
            blob = corpus.read("corpus/" + split + ".jsonl")
            must(digest(blob) == manifest["split_files"][split.upper()]["sha256"],
                 "original visible split hash mismatch")
            allrows = [json.loads(line) for line in blob.splitlines() if line.strip()]
            must(len(allrows) == 12, "original visible partition denominator changed")
            selected = [r for r in allrows if r["family"] == "architecture"]
            require_exact_cases(selected, expected)
            for r in sorted(selected, key=lambda x: (x["structural_group"], x["seed"])):
                name = r["native_case_id"]
                must(name == r["id"] and name.startswith("architecture-"),
                     "invalid original Architecture native ID")
                admission = r["native_admission"]
                must(r["native_qualified"] is True and admission["native_execution"] is True
                     and admission["verified"] is True and admission["state"] == "PASS"
                     and admission["source_sha"] == CORPUS_SOURCE,
                     "visible original native admission not positive")
                raw = corpus.read(name + ".materials.i32le")
                mat = corpus.read(name + ".materials.png")
                rgb = corpus.read(name + ".png")
                receipt_bytes = corpus.read(name + ".receipt.json")
                receipt = json.loads(receipt_bytes)
                reference = original_pixels(raw, mat, rgb)
                must(digest(raw) == admission["readback_sha256"]
                     and digest(rgb) == admission["png_sha256"]
                     and digest(receipt_bytes) == admission["receipt_sha256"]
                     and receipt["passed"] is True
                     and receipt["id"] == name
                     and receipt["crop_i32le_sha256"] == digest(raw)
                     and receipt["terminal"]["status"] == "completed"
                     and len(reference) == receipt["filled_cells"],
                     "original visible native receipt/readback mismatch")
                for candidate in sorted(CANDIDATES):
                    rows_out.append({"candidate": candidate,
                                     "split": split.upper(),
                                     "reference_group": r["structural_group"],
                                     "reference_seed": r["seed"],
                                     "reference_native_readback_sha256": digest(raw),
                                     "reference_png_sha256": digest(rgb),
                                     **metrics(masks[(candidate, r["seed"])], reference)})
    must(len(rows_out) == 16
         and len({(r["candidate"], r["split"], r["reference_group"], r["reference_seed"])
                  for r in rows_out}) == 16,
         "Architecture comparison coverage is incomplete or duplicate")
    return {
        "schema": "cybersand.sprocket-a07-a08-visible-native-spatial/1",
        "status": "ORIGINAL_NATIVE_RESEARCH_SPLIT_REVIEW_ONLY",
        "original_corpus_source_sha": CORPUS_SOURCE,
        "original_native_research_producer_sha": NATIVE_SOURCE,
        "original_corpus_zip_sha256": CORPUS_ZIP,
        "original_native_research_zip_sha256": NATIVE_ZIP,
        "original_corpus_manifest_sha256": CORPUS_MANIFEST,
        "image_size": [SIDE, SIDE],
        "metric": "exact-mask IoU and directional Chebyshev radius-3 coverage",
        "candidate_structures": 2,
        "candidate_seed_views": 4,
        "reference_structures": 4,
        "reference_seed_views": 8,
        "pairwise_comparisons": len(rows_out),
        "original_native_cases": cases,
        "results": rows_out,
        "new_native_executions": 0,
        "new_model_outputs": 0,
        "optimizer_updates": 0,
        "new_shared_train_admissions": 0,
        "final_targets_opened": False,
        "reviewer_authorized_train": False,
        "disposition": "HOLD_FOR_WHOLE_PEER_AND_SPLIT_REVIEW",
        "limitation": "Only original material morphology/orthographic source research. "
                      "No gameplay traversal, independent split approval, model painting, "
                      "or positive shared TRAIN admission.",
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-zip", type=Path, required=True)
    p.add_argument("--native-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    must(not a.output.exists(), "refuse overwrite")
    result = audit(a.corpus_zip, a.native_zip)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "candidate_structures": 2,
                      "reference_seed_views": 8, "pairs": result["pairwise_comparisons"],
                      "TRAIN": 0}, sort_keys=True))

if __name__ == "__main__":
    main()
