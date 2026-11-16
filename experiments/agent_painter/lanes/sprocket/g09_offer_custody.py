#!/usr/bin/env python3
"""Fail-closed G09 teacher packet/native-byte custody; review only, never TRAIN."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from compiler import canonical_json
from threefamily_v3 import program_for

EXPECTED_PROGRAM_SHA = "62fc07e8e27a67c020acd023a2285ddf21fc4f5f8701fbf8e52ae0943893ee51"
EXPECTED_SOURCE = "63ee8b2b561068a72e0652e54b6e4193cb47d5e6"
EXPECTED_ZIP_SHA = "3a7397883b146a28da990f0a37b891897285c2c7c81cd9dd2aaa3895d0d196dd"
EXPECTED_SEEDS = (104729, 130363)
EXPECTED_CELLS = (847, 900)

def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def sha_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())

def verify_offer(packet: dict, original: Path) -> dict:
    """Read only original source-built prospective native evidence, never FINAL."""
    require(packet["schema"] == "cybersand.sprocket-g09-global-admission-review-offer/1"
            and packet["status"] == "ACCEPT_FOR_SHARED_ADMISSION_REVIEW_ONLY",
            "not a version-1 review-only packet")
    src = packet["authored_program"]
    require(src["source_producer_sha"] == "af673d4497f199d2ce10a6aa7ff1b89dffc23c37"
            and src["source_program_key"] == "STUDIES['G09']['program']",
            "source locator/provenance mismatch")
    program = program_for("G09")
    digest = sha_bytes(canonical_json(program).encode("utf-8"))
    require(digest == EXPECTED_PROGRAM_SHA and src.get("program_sha256") == digest,
            "canonical source program SHA-256 mismatch")
    require(packet["immutable_inputs"]["original_native_research_producer_sha"] == EXPECTED_SOURCE
            and packet["immutable_inputs"]["original_research_artifact"]["zip_sha256"] == EXPECTED_ZIP_SHA,
            "original native producer/archive identity drift")
    label = packet.get("proposed_training_label_gate")
    require(type(label) is dict and label.get("status") == "NOT_AUTHORED_NOT_ADMITTED"
            and label.get("request") is None and label.get("complete_target_json") is None
            and label.get("target_eos_token_id") == 5
            and label.get("eos_appended_and_verified") is False
            and label.get("root_instruct_jinja_verified_for_new_row") is False
            and label.get("positive_shared_native_train_receipt") is None
            and label.get("global_structural_group_id") is None
            and label.get("ready_for_optimizer") is False,
            "attempted TRAIN/target/EOS promotion without positive admission")
    safety = packet["safety_and_next_gate"]
    require(safety["newly_admitted_train_groups"] == 0
            and safety["optimizer_updates_in_this_packet"] == 0
            and safety["research_teacher_only"] is True
            and safety["final_targets_or_images_opened"] is False
            and safety["draft_hold"] is True and safety["merge_authorized"] is False,
            "research safety gate changed")
    summary = read_json(original / "qualification-summary.json")
    verify = read_json(original / "native-verification.json")
    candidates = read_json(original / "candidates.json")["examples"]
    require(summary["source_sha"] == EXPECTED_SOURCE
            and summary["native_verified"] is True
            and summary["actual_native_positive_views"] == 8
            and summary["shared_train_groups_admitted"] == 0
            and summary["final_targets_opened"] is False
            and verify["source_sha"] == EXPECTED_SOURCE
            and verify["passed"] is True,
            "original native research did not qualify")
    review_cases = src["exact_native_cases"]
    require(len(review_cases) == 2
            and [c["seed"] for c in review_cases] == list(EXPECTED_SEEDS)
            and len({c["id"] for c in review_cases}) == 2,
            "two original distinct seed views required")
    count = 0
    for i, packet_case in enumerate(review_cases):
        seed = EXPECTED_SEEDS[i]
        cid = f"sprocket-G09-s{seed}"
        require(packet_case["id"] == cid
                and packet_case["family"] == "greebles"
                and packet_case["acknowledged_material58_cells"] == EXPECTED_CELLS[i],
                "original native case identity/cell count differs")
        def unique(rows):
            matches = [x for x in rows if x["id"] == cid]
            require(len(matches) == 1, "original native case missing/duplicated")
            return matches[0]
        candidate = unique(candidates)
        native = unique(verify["cases"])
        summary_case = unique(summary["cases"])
        receipt_file = original / f"{cid}.receipt.json"
        receipt = read_json(receipt_file)
        require(candidate["split"] == "PROPOSAL_ONLY"
                and candidate["seed"] == seed
                and candidate["program"] == program
                and candidate["compiled"]["program_sha256"] == digest
                and native["split"] == "PROPOSAL_ONLY"
                and native["seed"] == seed
                and native["success"] is True
                and native["deterministic_replay"] is True
                and native["program_sha256"] == digest
                and native["affected_cells"] == EXPECTED_CELLS[i]
                and native["acknowledged_cells"] == EXPECTED_CELLS[i],
                "original native candidate/verifier program or case mismatch")
        require(receipt["passed"] is True
                and receipt["program_sha256"] == digest
                and receipt["id"] == cid
                and int(receipt["seed"]) == seed
                and receipt["deterministic_replay_checked"] is True
                and receipt["unexpected_material_cells"] == 0
                and receipt["terminal"]["status"] == "completed"
                and receipt["terminal"]["changed"] == EXPECTED_CELLS[i],
                "original native receipt is not positive/matched")
        for field, filename in (
            ("original_png_sha256", cid + ".png"),
            ("native_i32le_sha256", cid + ".materials.i32le"),
            ("native_receipt_sha256", cid + ".receipt.json"),
        ):
            actual = sha_file(original / filename)
            require(actual == packet_case[field], "original native file checksum mismatch: " + filename)
        require(summary_case["png_sha256"] == packet_case["original_png_sha256"]
                and summary_case["readback_sha256"] == packet_case["native_i32le_sha256"]
                and summary_case["receipt_sha256"] == packet_case["native_receipt_sha256"]
                and native["image_sha256"] == packet_case["original_png_sha256"]
                and native["readback_sha256"] == packet_case["native_i32le_sha256"]
                and native["receipt_sha256"] == packet_case["native_receipt_sha256"],
                "original native verifier/summary disagree with packet")
        count += 1
    return {
        "schema": "cybersand.sprocket-g09-offer-custody-witness/1",
        "status": "REVIEW_ONLY_NOT_TRAIN_ADMITTED",
        "program_sha256": digest,
        "original_native_producer_sha": EXPECTED_SOURCE,
        "original_native_archive_sha256": EXPECTED_ZIP_SHA,
        "native_seed_views_verified": count,
        "new_native_views": 0,
        "new_optimizer_steps": 0,
        "new_train_groups": 0,
        "model_generated_views": 0,
        "final_opened": False,
        "exact_new_request_and_target": False,
        "eos_new_label_verified": False,
        "next_gate": "Shepherd and reviewer global split identity, exact framed request/target/EOS, and separately positive source-built shared TRAIN native admission."
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--offer", type=Path, default=HERE / "G09_SHARED_ADMISSION_REVIEW_OFFER.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "immutable receipt already exists")
    report = verify_offer(read_json(args.offer), args.native)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verified_original_seed_views": 2, "train_admitted": 0, "status": report["status"]}))

if __name__ == "__main__":
    main()
