#!/usr/bin/env python3
"""Fail-closed original Art-byte verification for one SMOL P01 review offer.

This script cannot admit TRAIN, train a model, open FINAL, or generate Art.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

from PIL import Image

ARCHIVE_SHA = "71ddc40478e06e3350f05eecd5e16ed77c082300f066be97bbccf8afb2370ff1"
PROPOSAL_SHA = "4abb6c020434977418822f669f3355f70cb2499852ff8f5e4f822cbce89c3772"
PRODUCER = "6e412178c9b71cb0196d8e35408470077b50f884"
GROUP = "smol_p01_nested_branch_tree"
SEEDS = (104729, 130363)


def need(ok, why):
    if not ok:
        raise ValueError("P01 admission-review refused: " + why)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def load(raw):
    def pairs(seq):
        result = {}
        for key, value in seq:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def check(offer_file, proposal_file, archive_file):
    offer = load(offer_file.read_bytes())
    raw_proposal = proposal_file.read_bytes()
    proposal = load(raw_proposal)
    need(offer["schema"] == "cybersand.agent-painter.smol-p01-admission-review/1"
         and offer["status"] == "REVIEW_ONLY_AWAIT_GLOBAL_SELECTION_NOT_TRAIN"
         and offer["candidate_id"] == GROUP and offer["proposed_split"] == "UNADMITTED_RESEARCH_ONLY"
         and offer["authoritative_group_id"] is None, "unapproved curriculum identity")
    need(offer["native_source_sha"] == PRODUCER
         and offer["original_corpus_source_sha"] == "633920373a3d406a2ede61ce4137c324f76be23b"
         and offer["original_corpus_sha256"] == "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"
         and offer["original_native_run"] == 43601281142
         and offer["original_native_artifact"] == 14117110667, "native/corpus provenance")
    need(offer["proposal_file_sha256"] == PROPOSAL_SHA == sha(raw_proposal), "original authored source digest")
    need(offer["original_archive_sha256"] == ARCHIVE_SHA == sha(archive_file.read_bytes()), "unaltered original native ZIP")
    need(offer["fixed_seeds"] == list(SEEDS) and offer["structural_groups"] == 1
         and set(offer["cases"]) == {str(s) for s in SEEDS}, "seed/group census")
    need(offer["training_steps"] == offer["admitted_train_groups"] == offer["model_generated_views"] == 0
         and offer["final_targets_opened"] is False, "false training/admission/model/FINAL claim")
    candidates = [v for v in proposal["proposals"] if v["id"] == GROUP]
    need(len(candidates) == 1 and candidates[0]["family"] == "pipework", "authored candidate")
    authored = candidates[0]
    need(authored["request"] == offer["authored_request"], "original authored request")
    program = authored["program"]
    program_hash = sha(json.dumps(program, sort_keys=True, separators=(",", ":")).encode())
    need(offer["program_sha256"] == program_hash == "1e5905db9e4f71e67119433f89eea9ad304eae812bcfd14f1b0afd96e06ab344", "program digest")
    graph = program["components"][0]
    degrees = [0] * len(graph["nodes"])
    for left, right in graph["edges"]:
        degrees[left] += 1
        degrees[right] += 1
    need(len(graph["nodes"]) == 7 and len(graph["edges"]) == 6
         and sorted(degrees) == [1, 1, 1, 1, 1, 3, 4]
         and graph["diameter"] == 3
         and program["requirements"] == {"connected": True, "cycles": 0}, "nested fan structure")
    need(offer["observed_topology"] == {
         "endpoints": 5, "tees": 1, "crosses": 1,
         "upward_trunk_risers": 1, "cycles": 0, "diameter": 3
         } and offer["heldout_three_riser_topology"] == {
         "endpoints": 5, "tees": 3, "crosses": 0,
         "upward_trunk_risers": 3, "cycles": 0, "diameter": 3
         }, "held-out label separation")
    checks = []
    with zipfile.ZipFile(archive_file) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) == 88 and z.testzip() is None, "ZIP CRC/unique members")
        summary = load(z.read("smol-v2-native/smol-v2-native-summary.json"))
        need(summary["native_passed"] is True and summary["native_source_sha"] == PRODUCER
             and summary["program_proposals_sha256"] == PROPOSAL_SHA
             and summary["verified_native_views"] == 6 and summary["distinct_structural_groups"] == 3
             and summary["admitted_train_groups"] == summary["optimizer_steps"] == summary["model_outputs"] == 0
             and summary["final_targets_opened"] is False, "original native outcome")
        native_candidates = load(z.read("smol-v2-native/candidates.json"))["examples"]
        for seed in SEEDS:
            name = f"{GROUP}-s{seed}"
            item = next((x for x in native_candidates if x["id"] == name), None)
            need(item is not None and item["seed"] == seed and item["family"] == "pipework"
                 and item["program"] == program and item["prompt"] == authored["request"]
                 and item["split"] == "PROPOSAL_ONLY_UNADMITTED" and item["training_approved"] is False
                 and item["compiled"]["program_sha256"] == program_hash, "unchanged research input")
            base = "smol-v2-native/" + name
            grid_bytes = z.read(base + ".materials.i32le")
            png_bytes = z.read(base + ".png")
            id_png_bytes = z.read(base + ".materials.png")
            receipt_bytes = z.read(base + ".receipt.json")
            original = next(x for x in summary["cases"] if x["id"] == name)
            recorded = offer["cases"][str(seed)]
            need(len(grid_bytes) == 96 * 96 * 4
                 and sha(grid_bytes) == recorded["material_sha256"] == original["readback_sha256"]
                 and sha(png_bytes) == recorded["png_sha256"] == original["png_sha256"]
                 and sha(id_png_bytes) == recorded["materials_png_sha256"]
                 and sha(receipt_bytes) == recorded["receipt_sha256"] == original["receipt_sha256"],
                 "original bytes/manifest hash")
            cells = struct.unpack("<9216i", grid_bytes)
            need(set(cells) == {0, 58} and cells.count(58) == recorded["material_cells"] == 596,
                 "native material IDs")
            receipt = load(receipt_bytes)
            need(receipt["passed"] is True and int(receipt["seed"]) == seed
                 and receipt["program_sha256"] == program_hash
                 and receipt["filled_cells"] == 596
                 and receipt["outside_area_cells"] == receipt["unexpected_material_cells"] == 0
                 and receipt["deterministic_replay_checked"] is True, "native case/safety")
            for key in ("terminal", "replay_terminal"):
                terminal = receipt[key]
                need(terminal["status"] == "completed"
                     and terminal["changed"] == terminal["checked"] == 596
                     and all(terminal[k] == 0 for k in
                             ("refused", "excluded", "unchanged", "human_pending", "human_refused")),
                     "native Art acknowledgements/replay")
            for blob, filled, empty in (
                (png_bytes, (232, 226, 205), (13, 21, 28)),
                (id_png_bytes, (58, 58, 58), (0, 0, 0)),
            ):
                with Image.open(io.BytesIO(blob)) as image:
                    need(image.size == (96, 96), "physical PNG size")
                    pixels = list(image.convert("RGB").getdata())
                need(all(rgb == (filled if material == 58 else empty)
                         for rgb, material in zip(pixels, cells)), "physical PNG pixel mismatch")
            checks.append({"seed": seed, "native_material_cells": 596,
                           "display_and_material_pixels_verified": 18432,
                           "native_acknowledged": 596})
    return {
        "status": "PASS_REVIEW_ONLY_NOT_SHARED_TRAIN_ADMISSION",
        "source_sha": PRODUCER, "original_native_run": 43601281142,
        "original_artifact": 14117110667, "original_zip_sha256": ARCHIVE_SHA,
        "independent_structural_groups": 1, "original_seed_views_verified": 2,
        "raster_pixels_verified": 36864, "admitted_train_groups": 0,
        "training_steps": 0, "model_generated_views": 0,
        "final_targets_opened": False, "cases": checks,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offer", type=Path, required=True)
    p.add_argument("--proposals", type=Path, required=True)
    p.add_argument("--original-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        result = check(args.offer, args.proposals, args.original_zip)
    except Exception as error:
        result = {"status": "FAIL_REVIEW_ONLY", "error": str(error),
                  "admitted_train_groups": 0, "training_steps": 0}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
