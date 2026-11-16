#!/usr/bin/env python3
"""Fail-closed independent custody check of the LFM C1 review offer.

Original failed native research remains FAILED. This validates archived original
bytes and the lane's conditional review packet, NOT new native TRAIN admission,
new model inference, training, or final split access. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

HERE = Path(__file__).resolve().parent
NATIVE_PREFIX = "lfm-contrastive-native/"
C1 = "lfm-c1-six-bay-service-chase"
SOURCE_SHA = "abef683ceaa0fb2782e2d4b65db2c4c3c475e52d"
ORIGINAL_ZIP_SHA256 = "e90014c8883bb89e1caaebfdb00cf28f2078604553a6b8de803e6caf9305aebf"
PROPOSAL_SHA256 = "9485c71ecfec5df5fae3cb68acc78dcc1b725208fdd9b30d5a21bf7662f8349f"
CORPUS_SHA256 = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(*, offer_path: Path, proposal_path: Path, original_zip: Path) -> dict:
    checks = 0

    def ok(condition: bool, message: str) -> None:
        nonlocal checks
        require(condition, message)
        checks += 1

    offer = json.loads(offer_path.read_text(encoding="utf-8"))
    proposals_bytes = proposal_path.read_bytes()
    proposals = json.loads(proposals_bytes)
    ok(offer["schema"] == "cybersand.lfm25_vl.c1-shared-admission-review-offer/1",
       "offer schema")
    ok(offer["status"] == "CONDITIONAL_YES_FOR_SHARED_ADMISSION_REVIEW_ONLY",
       "review status cannot imply admission")
    selection = offer["selection"]
    ok(selection["review_not_admission"] is True and
       selection["seed_variants_are_not_independent"] is True and
       selection["independent_structural_groups"] == 1, "review-only group")
    fail = offer["fail_closed"]
    for key in ("original_teacher_hand_authored_not_model_generated",
                "no_auto_merge_no_provider_promotion"):
        ok(fail[key] is True, "required true field " + key)
    ok(fail["final_target_or_image_opened"] is False, "FINAL must remain sealed")
    for key in ("new_native_executions_for_this_offer", "new_common_train_groups",
                "optimizer_updates", "useful_model_vN_freezes"):
        ok(type(fail[key]) is int and fail[key] == 0,
           "review packet cannot claim new model/native results: " + key)
    for key in ("candidate_c2_not_in_this_offer", "candidate_c3_quarantined",
                "requires_reviewer_global_yes_and_shepherd_lease_before_native_common_admission",
                "must_keep_original_failed_run_and_retrospective_audit_separate"):
        ok(fail[key] is True, "protected disposition " + key)

    originals = offer["originals"]
    ok(originals["original_native_job_outcome"] == "FAILED_POST_RUN_INPUT_CUSTODY_KEY"
       and originals["original_failed_job_reclassified"] is False, "retain failed original")
    ok(originals["original_failure_key"] == "inputs_unchanged" and
       originals["expected_verifier_key"] == "input_unchanged", "custody spelling")
    ok(originals["native_pilot_source_sha"] ==
       "633920373a3d406a2ede61ce4137c324f76be23b", "native original pilot")
    ok(originals["original_corpus_manifest_sha256"] == CORPUS_SHA256, "original corpus")
    ok(originals["original_proposal_file_sha256"] == digest(proposals_bytes) ==
       PROPOSAL_SHA256, "original proposal bytes")
    ok(originals["original_native_research_producer_sha"] == SOURCE_SHA, "research source")
    ok(originals["original_native_artifact"]["zip_sha256"] ==
       ORIGINAL_ZIP_SHA256, "archive declared digest")
    ok(offer["authored_program"]["proposal_id"] == C1, "proposal id")
    ok(proposals["schema"] == "cybersand.lfm25_vl.contrastive_research_proposals/1",
       "research source schema")
    source_c1 = next(x for x in proposals["proposals"] if x["id"] == C1)
    program = source_c1["program"]
    ok(program["family"] == "greebles" and program["version"] == 2, "source family/version")
    panels = program["components"]
    ok(len(panels) == 6 and all(x["kind"] == "panel" for x in panels), "six panels")
    ok(offer["authored_program"]["panel_boxes"] == [x["box"] for x in panels],
       "exact original panel boxes")
    ok(offer["authored_program"]["details"] == [x["detail"] for x in panels],
       "exact original details")
    ok(len(set(offer["authored_program"]["details"])) == 5, "five distinct details")
    ok(all(box[0] + box[2] <= 44 or box[0] >= 56 for box in
           offer["authored_program"]["panel_boxes"]), "full-height empty chase")
    ok(len([x for x in panels if x["box"][0] < 44]) == 3 and
       len([x for x in panels if x["box"][0] >= 56]) == 3, "three panels per bank")

    zip_bytes = original_zip.read_bytes()
    ok(digest(zip_bytes) == ORIGINAL_ZIP_SHA256, "original ZIP complete SHA256")
    with zipfile.ZipFile(original_zip) as archive:
        ok(len(archive.namelist()) == 62 and archive.testzip() is None, "ZIP 62/62 CRC")
        old_process = json.loads(archive.read(NATIVE_PREFIX + "process-result.json"))
        ok(old_process["exit_code"] == 0 and old_process["inputs_unchanged"] is True
           and "input_unchanged" not in old_process, "original proven verification typo")
        old_verification = json.loads(
            archive.read(NATIVE_PREFIX + "native-verification.json"))
        ok(old_verification["passed"] is False and
           "input_unchanged" in old_verification["failures"], "failed verifier preserved")
        old_summary = json.loads(
            archive.read(NATIVE_PREFIX + "lfm-contrastive-native-summary.json"))
        ok(old_summary["native_verifier_passed"] is False and
           old_summary["new_common_train_admissions"] == 0, "old overall failure")
        screen = json.loads(archive.read(NATIVE_PREFIX + "pre-native-source-screen.json"))
        ok(screen["input_sha256"] == PROPOSAL_SHA256 and
           screen["original_manifest_sha256"] == CORPUS_SHA256, "screen custody")
        cases = offer["authored_program"]["original_seed_cases"]
        ok([c["seed"] for c in cases] == [104729, 130363], "two original seeds only")
        reviewed = []
        for case in cases:
            name = case["id"]
            prefix = NATIVE_PREFIX + name
            receipt_bytes = archive.read(prefix + ".receipt.json")
            receipt = json.loads(receipt_bytes)
            png = archive.read(prefix + ".png")
            raw = archive.read(prefix + ".materials.i32le")
            ok(digest(receipt_bytes) == case["original_receipt_sha256"], name + " receipt")
            ok(digest(png) == case["native_png_sha256"], name + " original PNG")
            ok(digest(raw) == case["native_raw_i32le_sha256"] ==
               receipt["crop_i32le_sha256"], name + " material bytes")
            ok(len(raw) == 96 * 96 * 4, name + " 96x96 crop")
            cells = list(struct.iter_unpack("<i", raw))
            ids = {value[0] for value in cells}
            filled = sum(1 for value, in cells if value == 58)
            ok(ids.issubset({0, 58}) and filled == case["filled_and_acknowledged_cells"] ==
               receipt["filled_cells"], name + " material id/count")
            ok(receipt["passed"] is True and
               receipt["terminal"]["status"] == "completed" and
               receipt["terminal"]["changed"] == filled and
               receipt["terminal"]["refused"] == 0 and
               receipt["outside_area_cells"] == 0, name + " native acknowledgement")
            ok(receipt["program_sha256"] == offer["authored_program"]["program_sha256"],
               name + " same author program")
            ok(receipt["operations_sha256"] == case["operation_sha256"] and
               receipt["operation_count"] == case["operation_count"],
               name + " actual compiler operations")
            ok(receipt["history_and_reopen_checked"] is
               case["history_undo_redo_reopen_checked"], name + " history flag")
            reviewed.append({"case": name, "filled_cells": filled,
                             "original_png_sha256": digest(png)})
    return {
        "schema": "cybersand.lfm25_vl.c1-review-offer-custody-result/1",
        "status": "PASS_REVIEW_ONLY_NOT_NATIVE_TRAIN_ADMISSION",
        "checks": checks, "original_zip_sha256": digest(zip_bytes),
        "original_failed_run_retained": True,
        "reviewed_original_seed_cases": reviewed,
        "new_native_executions": 0, "train_admissions": 0,
        "optimizer_updates": 0, "useful_model_freezes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offer", type=Path,
                        default=HERE / "C1_SHARED_ADMISSION_REVIEW_OFFER.json")
    parser.add_argument("--proposals", type=Path,
                        default=HERE / "contrastive_proposals_v1.json")
    parser.add_argument("--original-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(offer_path=args.offer, proposal_path=args.proposals,
                    original_zip=args.original_zip)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
