#!/usr/bin/env python3
"""Source-only reconciliation of original Sprocket NCA-only diagnostic evidence.

This does not run a model, optimizer, evaluation fixture or training. Original
PR #159 receipts belong to a later and DIFFERENT source cohort from historical
formal pretraining adjudications #122/#124. Never turn their narrow observation
into scientific #72 PASS, cross-variant competence or training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ncr_001_sprocket_learning_artifact_readback as original
import ncr_001_sprocket_learning_git_custody as custody

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/sprocket-learning-20261028"
OUT = ROOT / "runtime/ncr-001/sprocket-postwarmup-diagnostic-reconciliation.json"
SOURCE_PR = 159
CUSTODY_PR = 160
SOURCE_HEAD = "df8479aae34550cba014d910a73e30e59cd493e2"
CUSTODY_HEAD = "d168fa41ae35536e893a5bc548a7b21965c45999"
RUN = 40710171622
ORIGINAL_BLOBS = {
    "manifest.json": "afd9fec5a60702ed6a99274470125e52f2c9c38c",
    "original-receipt-bundle.part001.b64": "7a25b40a3e6c6ee4b1fab5d42015154d6f959333",
    "original-receipt-bundle.part002.b64": "ca847ad6796576010d081d8013b5b67d97411a18",
    "original-receipt-bundle.part003.b64": "5c7d57b18997fe05ea5e3f1f3e85c88b49a8c33b",
    "actions-decoded/source-original-job-122892989282.log": "9c968cbed65ceaef92127d12dec8aaba74b7f181",
    "actions-decoded/chat-original-job-122893504080.log": "20dd75820ed8fe7bb491a229321bc8b9d18fe9cf",
    "actions-decoded/instruct-original-job-122893504180.log": "a1bb8e45dcdeaf9f2fc8d984feed5de679231ef7",
    "actions-decoded/initial-readback-formatter-hash-failure-job-122910071921.log": "8da1661f51dc2b682db2d012bb63983c13776965",
}
DIAGNOSTIC_OBSERVED = "OBSERVED_NCA_ONLY_ONE_STEP_SYNTHETIC__SCIENTIFIC_REVIEW_PENDING"
PARTIAL_LOSS = "PARTIAL_FINITE_SYNTHETIC_LOSS_AND_THREE_DEPTH_FIELDS_ONLY"
FINAL = "BLOCKED_INDEPENDENT_FULL_72_SCIENTIFIC_ADJUDICATION__NO_TRAIN"


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("NCR001_SPROCKET_POSTWARMUP_REFUSED: " + why)


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii")
                        + b"\x00" + raw).hexdigest()


def check_committed_bytes(evidence: Path = EVIDENCE) -> None:
    """Reject byte edits even when stripped Base64 or parsed JSON is identical."""
    for name, expected in ORIGINAL_BLOBS.items():
        require(gitblob((evidence / name).read_bytes()) == expected,
                "changed original committed Git blob: " + name)


def reconcile() -> dict:
    check_committed_bytes()
    manifest, members = custody.load_archive()
    require(manifest["original_source_head"] == SOURCE_HEAD
            and manifest["original_run_id"] == RUN
            and original.HEAD == SOURCE_HEAD and original.RUN_ID == str(RUN),
            "original physical source/run mismatch")
    require(manifest["train_authorized"] is False
            and manifest["full_72_conformance_pass"] is False
            and manifest["independent_scientific_review"] is False
            and manifest["comparative_training_performed"] is False,
            "archive invents scientific acceptance or comparative training")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) ==
            {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "four-lane preregistered protocol or training gate changed")
    lanes = {}
    fingerprints = []
    for variant, expected_job in (("chat", 122893504080),
                                  ("instruct", 122893504180)):
        receipt = json.loads(members[variant]["receipt.json"])
        independent = original.receipt_gate(receipt, variant, source_tree=True)
        require(receipt["runner"]["run_id"] == str(RUN)
                and receipt["diagnostic_optimizer_steps"] == 1
                and receipt["synthetic_sample"]["heldout"] is False
                and receipt["full_72_conformance_pass"] is False
                and receipt["tool_memory_semantics_qualified"] is False,
                "synthetic diagnostic changed scope")
        lane_key, weight_file, weight_sha = original.WEIGHTS[variant]
        require(receipt["source_identity"]["weight_path"] == weight_file
                and independent["source_weight_sha256"] == weight_sha
                and independent["diagnostic_optimizer_steps"] == 1,
                "wrong variant or missing original one-step evidence")
        fingerprints.append(independent["backbone_fingerprint"])
        lanes[lane_key] = {
            "variant": variant,
            "physical_job_id": expected_job,
            "original_checkpoint_file": weight_file,
            "original_checkpoint_sha256": weight_sha,
            "backbone_fingerprint_unchanged": True,
            "one_in_memory_nca_only_diagnostic_step": True,
            "warmup_synthetic_supervised_loss": independent["warmup_loss"],
            "postwarmup_synthetic_supervised_loss": independent["post_warmup_loss"],
            "postwarmup_adapter_gradient_abs_sums": independent[
                "post_warmup_gradient_abs_sums"],
            "base_weight_unchanged_after_step": DIAGNOSTIC_OBSERVED,
            "nonzero_adapter_gradient_after_warmup": DIAGNOSTIC_OBSERVED,
            "finite_state_and_loss": PARTIAL_LOSS,
            "full_72_per_model_gate": False,
            "tool_memory_semantics_qualified": False,
        }
    require(len(set(fingerprints)) == 2,
            "distinct Chat/root Instruct backbones conflated")
    return {
        "schema": "ncr001-sprocket-postwarmup-diagnostic-reconciliation-v1",
        "status": FINAL,
        "physical_source_pr": SOURCE_PR,
        "physical_source_head": SOURCE_HEAD,
        "physical_source_run": RUN,
        "custody_pr": CUSTODY_PR,
        "custody_head": CUSTODY_HEAD,
        "evidence_class": "POSTLEASE_ORIGINAL_PHYSICAL_NCA_ONLY_DIAGNOSTIC",
        "historical_formal_pretraining_adjudications": {
            "sprocket_chat": {"pr": 124,
                              "head": "b2461a66241bc2fd0fbc389d7b3b2d10f729d2ab",
                              "scope": "OLDER_PRE_DIAGNOSTIC_COHORT_UNCHANGED"},
            "sprocket_instruct": {"pr": 122,
                                  "head": "90f3006dd59b2a731db19064472d9f2215ac62d9",
                                  "scope": "OLDER_PRE_DIAGNOSTIC_COHORT_UNCHANGED"},
        },
        "lanes": lanes,
        "new_physical_model_execution": False,
        "new_optimizer_steps": 0,
        "original_diagnostic_steps_per_lane": 1,
        "comparative_nca_mlp_training": False,
        "trained_model_checkpoint_created": False,
        "heldouts_opened": False,
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "independent_scientific_admission": False,
        "owner_comment_review_exception_scientific_pass": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = reconcile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
