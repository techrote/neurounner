#!/usr/bin/env python3
"""Read-only, fail-closed #72 source-bound Sprocket evidence scope reconciliation.

PR #204 retains original B2 four cached steps and original NCA-only learning;
PR #207 retains independent B1 2048 cached steps. Their different cohorts are
NEVER combined into a claim that one run exercised all conditions at once.
Only immutable original Git evidence and source-only Python stdlib are read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PR204 = "16f8610d856cccba27a7b35e5a99b96526652b50"
PR204_REPORT_BLOB = "09d8bc81b4568931c964a3c1b070997ae9cba00d"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
PR202_RUN = 40939386209
PR205_RUN = 40987371706
NCA_DIAGNOSTIC_RUN = 40710171622
STATUS = "BLOCKED_FULL_72_SCIENTIFIC_CONFORMANCE__THREE_DISTINCT_SOURCE_COHORTS"
PUBLISHED = ROOT / "experiments/ncr-001/adjudication/sprocket-three-cohort-source-delta-20261030.json"
MODEL = {
    "sprocket_chat": ("chat", "chat/model.safetensors",
        "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97",
        0.00026702880859375, 2.2232532501220703e-05, 2.2232532501220703e-05),
    "sprocket_instruct": ("instruct", "model.safetensors",
        "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        0.000110626220703125, 1.5735626220703125e-05, 2.2649765014648438e-05),
}


def refuse(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("NCR001_THREE_COHORT_REFUSED: " + reason)


def gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def compose(prior: dict, masked: dict, streaming: dict) -> dict:
    """Validate independent original-source decisions before composing scoped fields."""
    refuse(prior.get("schema") == "ncr001-sprocket-source-matched-two-lane-72-delta-v1"
           and prior.get("status") ==
                "BLOCKED_FULL_72_SCIENTIFIC_CONFORMANCE__BOUNDED_TWO_SPROCKET_DELTA"
           and prior.get("masked_original_head") ==
                "b52af229dbfd0220d8d89f0d18e58733a2c3b56b"
           and prior.get("masked_original_run") == PR202_RUN
           and prior.get("one_step_diagnostic_run") == NCA_DIAGNOSTIC_RUN
           and prior.get("new_frozen_model_execution") is False
           and prior.get("new_optimizer_steps") == 0
           and prior.get("train_authorized") is False
           and prior.get("full_72_conformance_pass") is False
           and prior.get("actual_compute_budget_qualified") is False,
           "historical reconciled B2/diagnostic scope changed")
    refuse(masked.get("status") ==
               "PASS_ORIGINAL_PR202_3ZIP_3LOG_GIT_BLOB_SOURCE_AND_NUMERIC_RECEIPT"
           and masked.get("source_sha") ==
               "b52af229dbfd0220d8d89f0d18e58733a2c3b56b"
           and masked.get("source_run") == PR202_RUN
           and masked.get("original_zip_count") == 3
           and masked.get("original_decoded_log_count") == 3
           and masked.get("train_authorized") is False
           and masked.get("full_72_conformance_pass") is False,
           "immutable original masked B2 3ZIP/3log audit absent")
    refuse(streaming.get("status") ==
               "PASS_ORIGINAL_PR205_3ZIP_3LOG_GIT_BLOBS_2048_STREAM"
           and streaming.get("source_sha") ==
               "0c866f5b65380ef6629a5bf084d017e886b91e52"
           and streaming.get("original_run") == PR205_RUN
           and streaming.get("original_zip_count") == 3
           and streaming.get("original_decoded_log_count") == 3
           and streaming.get("train_authorized") is False
           and streaming.get("full_72_conformance_pass") is False,
           "immutable original B1 all-token 3ZIP/3log audit absent")
    refuse(set(prior.get("lanes", {})) == set(MODEL)
           and set(streaming.get("lanes", {})) == {"source", "chat", "instruct"}
           and streaming["lanes"]["source"] ==
               {"original_physical_model": False, "source_tests": 12},
           "two independently frozen Sprocket variants required")
    lanes = {}
    for name, (variant, path, sha, b1_max, left, right) in MODEL.items():
        old = prior["lanes"][name]
        new = streaming["lanes"][variant]
        m = old["observed_bounded"]
        unknown = old["partial_or_unobserved"]
        refuse(old["frozen_weight_file"] == path and
               old["frozen_weight_sha256"] == sha and
               old["source_physical_2048_run"] == PR202_RUN and
               old["source_diagnostic_19_unit_run"] == NCA_DIAGNOSTIC_RUN and
               old["train_authorized"] is False and
               old["full_72_per_model_gate"] is False,
               "original B2/NCA lane swapped or promoted: " + name)
        refuse(m["masked_batch_size"] == 2
               and m["masked_sequence_physical_positions"] == 2048
               and m["short_row_true_tokens"] == 1536
               and m["actual_one_token_kv_increments"] == 4
               and m["cold_new_python_same_runner_last_four_logits_exact"] is True
               and m["masked_left_right_cache_max_abs"] ==
                    {"left": left, "right": right}
               and m["one_synthetic_NCA_only_optimizer_diagnostic_step"] is True
               and m["postwarmup_adapter_gradient_nonzero"] is True
               and unknown["right_pad_gap_publisher_valid"] is False
               and unknown["full_prefix_by_prefix_each_token_across_2048"] ==
                   "NOT_OBSERVED"
               and unknown["generated_publisher_tool_result_memory_semantics"] ==
                   "NOT_ADMITTED"
               and unknown["matched_NCA_MLP_training_or_quality"] ==
                   "NOT_PERFORMED",
               "masked/cold/diagnostic historical scope changed: " + name)
        refuse(new["model"] == path and new["max_abs"] == b1_max
               and new["tokens"] == 2048
               and new["uncached_prefixes"] == 10
               and new["optimizer_steps"] == 0,
               "all-token streaming original-source numerical scope changed: " + name)
        lanes[name] = {
            "frozen_weight_file": path,
            "frozen_weight_sha256": sha,
            "b2_masked_physical_positions": 2048,
            "b2_short_row_valid_tokens": 1536,
            "b2_genuine_incremental_kv_positions": 4,
            "b2_masked_left_cache_max_abs": left,
            "b2_masked_right_cache_max_abs": right,
            "b2_separate_python_same_runner_cold_last_four_exact": True,
            "b2_right_synthetic_gap_is_publisher_wire": False,
            "b1_unpadded_valid_token_positions": 2048,
            "b1_genuine_incremental_kv_positions": 2048,
            "b1_full_vs_incremental_max_abs": b1_max,
            "b1_independent_uncached_prefixes": 10,
            "b1_all_2048_uncached_prefix_forwards_executed": False,
            "separate_19_unit_synthetic_nca_only_update_observed": True,
            "separate_postwarmup_adapter_gradient_nonzero": True,
            "combined_b1_b2_cold_and_learning_same_physical_cohort": False,
            "per_model_72_scientific_pass": False,
        }
    return {
        "schema": "ncr001e-sprocket-source-matched-three-cohort-v1",
        "status": STATUS,
        "source_only": True,
        "original_physical_runs": [NCA_DIAGNOSTIC_RUN, PR202_RUN, PR205_RUN],
        "masked_prior_pr204_head": PR204,
        "masked_prior_report_git_blob": PR204_REPORT_BLOB,
        "streaming_pr205_head": "0c866f5b65380ef6629a5bf084d017e886b91e52",
        "streaming_original_custody_pr207_head": "cb016f6dceb79e8c7c5fb2c032479c4be5f461e7",
        "protocol_blob": PROTOCOL_BLOB,
        "pins_blob": PINS_BLOB,
        "lanes": lanes,
        "new_physical_execution": False,
        "new_optimizer_steps": 0,
        "comparative_nca_mlp_training": False,
        "heldouts_opened": False,
        "generated_tool_memory_semantics_admitted": False,
        "historic_root_instruct_six_token_tool_prefix_complete": False,
        "void1_historical_eos_training_impact": "UNMEASURED",
        "train_authorized": False,
        "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False,
    }


def live_originals(prior_root: Path) -> tuple[dict, dict, dict]:
    from ncr_001e_alltoken2048_original_custody import audit as stream_audit
    refuse(gitblob((ROOT / "experiments/ncr-001/protocol.json").read_bytes()) ==
           PROTOCOL_BLOB and
           gitblob((ROOT / "experiments/ncr-001/source-pins.json").read_bytes()) ==
           PINS_BLOB, "four-model protocol/source pins changed")
    protocol = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
    refuse(protocol["train_authorized"] is False and
           set(protocol["models"]) ==
             {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
           "training policy or four-model lane identity changed")
    sha = subprocess.check_output(
        ["git", "-C", str(prior_root), "rev-parse", "HEAD"], text=True).strip()
    refuse(sha == PR204, "not immutable PR204 original source")
    raw_report = (prior_root / "experiments/ncr-001/adjudication/"
                  "sprocket-source-matched-2048-delta-20261029.json").read_bytes()
    refuse(gitblob(raw_report) == PR204_REPORT_BLOB,
           "immutable published PR204 report bytes modified")
    sys.path.insert(0, str(prior_root / "scripts"))
    try:
        import ncr_001e_masked2048_original_custody as masked_custody
        masked = masked_custody.audit()
    finally:
        sys.path.pop(0)
    return json.loads(raw_report), masked, stream_audit()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    data = compose(*live_originals(a.prior.resolve()))
    refuse(json.loads(PUBLISHED.read_text(encoding="utf-8")) == data,
           "published derived source decision differs from independently audited originals")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": data["status"], "original_runs": data["original_physical_runs"],
                      "scoped_lanes": list(data["lanes"]), "source_only": True}, sort_keys=True))


if __name__ == "__main__":
    main()
