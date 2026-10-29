#!/usr/bin/env python3
"""NCR-001E: conservative original-source Sprocket dual-lane #72 delta adjudication.

This script re-verifies Git-custodied ORIGINAL PR #202 2048-token Chat/root
Instruct physical receipts and combines them with a separately re-audited,
exact-head #194 original NCA-only one-step diagnostic. No physical model load,
no optimizer, no heldout, no training, no revision of old #122/#124 reports.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import ncr_001e_masked2048_original_custody as custody

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/ncr-001/evidence-review/pr202-masked2048-original"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
OUT = ROOT / "runtime/ncr001-sprocket-integrated-72-delta.json"
EXPECTED_ORIGINAL_MASKED_HEAD = "b52af229dbfd0220d8d89f0d18e58733a2c3b56b"
EXPECTED_MASKED_SCRIPT_GIT_BLOB = "c92a410dc6605fa1878ded44265906e0a9769d08"
EXPECTED_DIAGNOSTIC_HEAD = "e5ac7fb0659bd2e640a3007e8cf798820069fc2d"
EXPECTED_DIAGNOSTIC_SOURCE = "df8479aae34550cba014d910a73e30e59cd493e2"
EXPECTED_DIAGNOSTIC_RUN = 40710171622
PUBLISHED_REPORT = ROOT / "experiments/ncr-001/adjudication/sprocket-source-matched-2048-delta-20261029.json"
PUBLISHED_REPORT_GIT_BLOB = "2aef9415f1cd1c6fcc324aaf25e0f5ebbfc7b805"
CACHE_ATOL = 0.0005
CACHE_RTOL = 0.0005
EVIDENCE_ONLY = "BLOCKED_FULL_72_SCIENTIFIC_CONFORMANCE__BOUNDED_TWO_SPROCKET_DELTA"
FOUR_LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
REQUIRED = {
    "same_base_logits", "base_weight_unchanged_after_step",
    "causal_suffix_invariance", "full_prefix_vs_incremental_cache",
    "per_token_isolation", "per_request_reset",
    "nonzero_adapter_gradient_after_warmup",
    "finite_state_and_loss", "cold_reload_state_dict",
}
SOURCE_MODELS = {
    "chat": ("sprocket_chat", "chat/model.safetensors",
             "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "instruct": ("sprocket_instruct", "model.safetensors",
                 "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
SYNTHETIC_DIAGNOSTIC_OBSERVED = "OBSERVED_NCA_ONLY_ONE_STEP_SYNTHETIC__SCIENTIFIC_REVIEW_PENDING"
SYNTHETIC_LOSS_PARTIAL = "PARTIAL_FINITE_SYNTHETIC_LOSS_AND_THREE_DEPTH_FIELDS_ONLY"


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("NCR001_SPROCKET_INTEGRATED_72_REFUSED: " + reason)


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii")
                        + b"\0" + raw).hexdigest()


def authoritative_source_gate() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    refuse(gitblob(PROTOCOL.read_bytes()) == custody.PROTOCOL_BLOB
           and gitblob(PINS.read_bytes()) == custody.PINS_BLOB,
           "immutable four-model protocol or source-pin Git blob changed")
    refuse(protocol["experiment"] == "NCR-001"
           and protocol["train_authorized"] is False
           and set(protocol["models"]) == set(pins["models"]) == FOUR_LANES
           and set(protocol["conformance"]["required_checks"]) == REQUIRED
           and protocol["conformance"]["any_failure_blocks_model_qualification"] is True
           and protocol["conformance"]["full_vs_cached_prefill_atol"] ==
               protocol["conformance"]["full_vs_cached_prefill_rtol"] == CACHE_ATOL,
           "no-train four-model scientific protocol changed")
    script = ROOT / "scripts/ncr_001e_sprocket_masked_cold_2048.py"
    refuse(gitblob(script.read_bytes()) == EXPECTED_MASKED_SCRIPT_GIT_BLOB,
           "original pinned 2048 executable bytes changed")
    for variant, (lane, path, digest) in SOURCE_MODELS.items():
        p = pins["models"][lane]
        m = protocol["models"][lane]
        refuse(p["hf_weight_file"] == m["weight_file"] == path
               and p["weight_declared_lfs_sha256_not_downloaded"] == digest
               and p["weight_bytes"] == 1002207984
               and p["revision"] == m["revision"] ==
                   "eb157d4ddddc4619e661d4a2426bb2dd000fc668",
               "original Chat/root variant source identity changed: " + variant)
    return protocol


def original_masked_receipts(evidence: Path = ARCHIVE) -> dict:
    """Use original producer/consumer JSON ONLY after primary byte-level custody PASS."""
    result = custody.audit(evidence)
    refuse(result["status"] ==
           "PASS_ORIGINAL_PR202_3ZIP_3LOG_GIT_BLOB_SOURCE_AND_NUMERIC_RECEIPT"
           and result["source_sha"] == EXPECTED_ORIGINAL_MASKED_HEAD
           and result["source_run"] == 40939386209
           and result["original_zip_count"] == 3
           and result["original_decoded_log_count"] == 3
           and result["train_authorized"] is False
           and result["full_72_conformance_pass"] is False,
           "original full ZIP/log/source/physical witness custody failed")
    records = {}
    for variant in SOURCE_MODELS:
        path = evidence / ("original-" + variant + "-" +
                           str(custody.ARTIFACT_IDS[variant]) + ".zip")
        with zipfile.ZipFile(path) as z:
            records[variant] = {
                "producer": json.loads(z.read("producer.json")),
                "consumer": json.loads(z.read("consumer.json")),
            }
    return records


def validate_diagnostic(diagnostic: dict) -> None:
    refuse(type(diagnostic) is dict and
           diagnostic.get("schema") ==
               "ncr001-sprocket-postwarmup-diagnostic-reconciliation-v1"
           and diagnostic.get("status") ==
               "BLOCKED_INDEPENDENT_FULL_72_SCIENTIFIC_ADJUDICATION__NO_TRAIN"
           and diagnostic.get("physical_source_head") == EXPECTED_DIAGNOSTIC_SOURCE
           and diagnostic.get("physical_source_run") == EXPECTED_DIAGNOSTIC_RUN
           and diagnostic.get("original_diagnostic_steps_per_lane") == 1
           and diagnostic.get("new_optimizer_steps") == 0
           and diagnostic.get("comparative_nca_mlp_training") is False
           and diagnostic.get("heldouts_opened") is False
           and diagnostic.get("train_authorized") is False
           and diagnostic.get("full_72_conformance_pass") is False
           and diagnostic.get("independent_scientific_admission") is False,
           "original diagnostic is not the independently verified no-train #194 report")
    earlier = diagnostic.get("historical_formal_pretraining_adjudications", {})
    refuse(earlier.get("sprocket_chat", {}).get("head") ==
               "b2461a66241bc2fd0fbc389d7b3b2d10f729d2ab"
           and earlier.get("sprocket_instruct", {}).get("head") ==
               "90f3006dd59b2a731db19064472d9f2215ac62d9",
           "older original #122/#124 adjudication heads changed")
    refuse(set(diagnostic.get("lanes", {})) ==
               {"sprocket_chat", "sprocket_instruct"},
           "missing or conflated original Chat and root Instruct learning diagnostics")


def compose(diagnostic: dict, originals: dict) -> dict:
    """Fail-closed scoped scientific *delta*; never marks full model #72 PASS."""
    validate_diagnostic(diagnostic)
    refuse(set(originals) == set(SOURCE_MODELS),
           "original 2048 physical receipts missing or variant-substituted")
    lanes = {}
    for variant, (lane, weight_file, weight_sha) in SOURCE_MODELS.items():
        producer = originals[variant]["producer"]
        consumer = originals[variant]["consumer"]
        diag = diagnostic["lanes"][lane]
        required_diagnostic_keys = {
            "original_checkpoint_file", "original_checkpoint_sha256",
            "one_in_memory_nca_only_diagnostic_step",
            "backbone_fingerprint_unchanged",
            "base_weight_unchanged_after_step",
            "nonzero_adapter_gradient_after_warmup",
            "finite_state_and_loss", "full_72_per_model_gate",
            "tool_memory_semantics_qualified",
            "warmup_synthetic_supervised_loss",
            "postwarmup_synthetic_supervised_loss",
        }
        refuse(type(diag) is dict and required_diagnostic_keys <= set(diag),
               "missing mandatory original source-bound diagnostic lane fields")
        refuse(producer["variant"] == consumer["variant"] == lane
               and producer["status"] == "PASS_MASKED2048_PRODUCER"
               and consumer["status"] == "PASS_MASKED2048_SEPARATE_PYTHON_COLD_CONSUMER"
               and producer["runner"]["head_sha"] ==
                   consumer["runner"]["head_sha"] == EXPECTED_ORIGINAL_MASKED_HEAD
               and str(producer["runner"]["run_id"]) ==
                   str(consumer["runner"]["run_id"]) == "40939386209"
               and producer["runner"]["pid"] != consumer["runner"]["pid"]
               and producer["runner"]["host"] == consumer["runner"]["host"],
               "original physical source, variant, PID or successful phase changed")
        for rep in (producer, consumer):
            refuse(rep["train_authorized"] is False
                   and rep["full_72_conformance_pass"] is False
                   and rep["optimizer_steps"] == 0
                   and rep["heldouts_opened"] is False
                   and rep["generated_tool_memory_semantics_qualified"] is False
                   and rep["right_pad_tail_is_synthetic_gap_not_publisher_wire"] is True
                   and rep["original_frozen_model_executed"] is True
                   and rep["verified_frozen_source_files"][weight_file]["sha256"] ==
                       weight_sha
                   and rep["frozen_before_sha256"] == rep["frozen_after_sha256"],
                   "original frozen physical source, negative or training gate changed")
        refuse(producer["source_sha256"] == consumer["source_sha256"] and
               producer["synthetic_input_sha256"] ==
                   consumer["synthetic_input_sha256"] and
               producer["verified_frozen_source_files"] ==
                   consumer["verified_frozen_source_files"],
               "cold process did not reload identical sources and token IDs")
        refuse(diag["original_checkpoint_file"] == weight_file and
               diag["original_checkpoint_sha256"] == weight_sha and
               diag["one_in_memory_nca_only_diagnostic_step"] is True and
               diag["backbone_fingerprint_unchanged"] is True and
               diag["base_weight_unchanged_after_step"] ==
                   SYNTHETIC_DIAGNOSTIC_OBSERVED and
               diag["nonzero_adapter_gradient_after_warmup"] ==
                   SYNTHETIC_DIAGNOSTIC_OBSERVED and
               diag["finite_state_and_loss"] == SYNTHETIC_LOSS_PARTIAL and
               diag["full_72_per_model_gate"] is False and
               diag["tool_memory_semantics_qualified"] is False,
               "original bounded NCA-only learning evidence misrepresented")
        before = diag["warmup_synthetic_supervised_loss"]
        after = diag["postwarmup_synthetic_supervised_loss"]
        refuse(all(type(x) in (int, float) and math.isfinite(x) and x > 0
                   for x in (before, after)),
               "original diagnostic synthetic loss absent or nonfinite")
        errors = {}
        for side in ("left", "right"):
            cache = producer["checks"][side + "_2048_full_vs_four_real_cached"]
            cold = consumer["checks"][side + "_cold_new_python_same_model_2048"]
            refuse(cache["pass"] is True and cache["finite"] is True
                   and cache["shape"] == [2, 4, 32000]
                   and cache["atol"] == cache["rtol"] == CACHE_ATOL
                   and type(cache["max_abs"]) in (int, float)
                   and 0 <= cache["max_abs"] <= CACHE_ATOL
                   and producer["checks"][side + "_mask_corruption_sentinel"] is True
                   and cold["pass"] is True and cold["exact"] is True
                   and cold["max_abs"] == 0 and
                   cold["shape"] == [2, 4, 32000],
                   "2048 batched genuine KV/cold result or pad sentinel changed")
            errors[side] = cache["max_abs"]
        lanes[lane] = {
            "frozen_weight_file": weight_file,
            "frozen_weight_sha256": weight_sha,
            "source_physical_2048_run": 40939386209,
            "source_diagnostic_19_unit_run": EXPECTED_DIAGNOSTIC_RUN,
            "observed_bounded": {
                "masked_batch_size": 2,
                "masked_sequence_physical_positions": 2048,
                "short_row_true_tokens": 1536,
                "actual_one_token_kv_increments": 4,
                "masked_left_right_cache_max_abs": errors,
                "cold_new_python_same_runner_last_four_logits_exact": True,
                "source_frozen_backbone_unchanged": True,
                "one_synthetic_NCA_only_optimizer_diagnostic_step": True,
                "postwarmup_adapter_gradient_nonzero": True,
                "synthetic_loss_before_step": before,
                "synthetic_loss_after_step": after,
                "invalid_pad_mask_sentinel_detected_both_sides": True,
            },
            "partial_or_unobserved": {
                "full_prefix_by_prefix_each_token_across_2048": "NOT_OBSERVED",
                "all_context_numeric_and_resource_stability": "NOT_FULLY_OBSERVED",
                "generated_publisher_tool_result_memory_semantics": "NOT_ADMITTED",
                "right_pad_gap_publisher_valid": False,
                "matched_NCA_MLP_training_or_quality": "NOT_PERFORMED",
                "full_72_final_admission": False,
            },
            "full_72_per_model_gate": False,
            "train_authorized": False,
        }
    refuse(lanes["sprocket_chat"]["frozen_weight_sha256"] !=
           lanes["sprocket_instruct"]["frozen_weight_sha256"],
           "two different model identities were collapsed")
    return {
        "schema": "ncr001-sprocket-source-matched-two-lane-72-delta-v1",
        "status": EVIDENCE_ONLY,
        "masked_source_pr": 202, "masked_original_head": EXPECTED_ORIGINAL_MASKED_HEAD,
        "masked_original_run": 40939386209,
        "permanent_masked_custody_pr": 203,
        "one_step_diagnostic_source_pr": 159,
        "one_step_diagnostic_run": EXPECTED_DIAGNOSTIC_RUN,
        "independent_diagnostic_reconciliation_head": EXPECTED_DIAGNOSTIC_HEAD,
        "historical_formal_reports_preserved": {
            k: {"pr": v["pr"], "head": v["head"]}
            for k, v in diagnostic["historical_formal_pretraining_adjudications"].items()
        },
        "lanes": lanes, "new_frozen_model_execution": False,
        "new_optimizer_steps": 0, "comparative_NCA_MLP_training": False,
        "heldouts_opened": False, "train_authorized": False,
        "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False,
        "generated_tool_memory_semantics_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-checkout", type=Path, required=True,
                        help="immutable original #194 checkout, re-audited with its own script")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    source = args.diagnostic_checkout.resolve()
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                  text=True).strip()
    refuse(sha == EXPECTED_DIAGNOSTIC_HEAD,
           "independent physical diagnostic checkout exact SHA changed")
    script = source / "scripts/ncr_001_sprocket_postwarmup_reconcile.py"
    report_path = args.output.resolve().with_suffix(".diagnostic.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(script), "--output", str(report_path)],
                   check=True, capture_output=True, text=True)
    diagnostic = json.loads(report_path.read_text(encoding="utf-8"))
    authoritative_source_gate()
    result = compose(diagnostic, original_masked_receipts())
    raw_published = PUBLISHED_REPORT.read_bytes()
    refuse(gitblob(raw_published) == PUBLISHED_REPORT_GIT_BLOB,
           "permanent independently derived scientific-delta report Git blob changed")
    refuse(json.loads(raw_published) == result,
           "regenerated original-source scientific-delta differs from committed report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
