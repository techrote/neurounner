#!/usr/bin/env python3
"""Deterministic NCR-001E Sprocket Instruct #72 lane adjudication.

Reads only durable, already-reviewed evidence from the stacked draft chain.
No network, model execution, optimizer, held-out fixture access, or training.

The result is deliberately fail-closed: strong source/runtime evidence may pass
individual pre-training subgates, but the lane cannot be promoted to final
#72 PASS while protocol.required_checks include learning/update checks and
protocol.train_authorized is false.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review"
DATASET_WIRE = ROOT / "datasets/ncr-001-sprocket-wire-diagnostic/provenance.json"
OUT = ROOT / "runtime/ncr-001/sprocket-instruct-adjudication.json"

MODEL_KEY = "sprocket_instruct"
WEIGHT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
CHAT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
MODEL_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
HOOK_SHA = "b95c010f9b2ba3cbd9f81317981316bb9c8acd19"
PR102_HEAD = "3592841802d85c442ca2521479b5d8795bee7233"
PR113_HEAD = "7dec2f91b6542854762542807978e5bddead4f64"
CACHE_ATOL = 0.0005
CACHE_RTOL = 0.0005
FOUR_MODELS = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}

REQUIRED_CHECKS = {
    "same_base_logits",
    "base_weight_unchanged_after_step",
    "causal_suffix_invariance",
    "full_prefix_vs_incremental_cache",
    "per_token_isolation",
    "per_request_reset",
    "nonzero_adapter_gradient_after_warmup",
    "finite_state_and_loss",
    "cold_reload_state_dict",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("NCR Sprocket Instruct adjudication REFUSED: " + reason)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_path(name: str) -> Path:
    return EVIDENCE / name / "provenance.json"


def check_common_source(record: dict, *, weight_path: tuple[str, ...],
                        hook_path: tuple[str, ...]) -> None:
    value = record
    for key in weight_path:
        value = value[key]
    require(value == WEIGHT_SHA, "evidence checkpoint SHA differs: " + "/".join(weight_path))
    value = record
    for key in hook_path:
        value = value[key]
    require(value == HOOK_SHA, "evidence hook SHA differs: " + "/".join(hook_path))


def build_report() -> dict:
    protocol = load(PROTOCOL)
    pins = load(PINS)
    require(protocol["experiment"] == "NCR-001" and protocol["train_authorized"] is False,
            "training policy or experiment identity changed")
    require(set(protocol["models"]) == set(pins["models"]) == FOUR_MODELS,
            "four independent model keys changed")
    require(set(protocol["conformance"]["required_checks"]) == REQUIRED_CHECKS,
            "required #72 conformance checklist changed")
    require(protocol["conformance"]["any_failure_blocks_model_qualification"] is True,
            "fail-closed per-model qualification policy changed")
    require(protocol["conformance"]["full_vs_cached_prefill_atol"] == CACHE_ATOL
            and protocol["conformance"]["full_vs_cached_prefill_rtol"] == CACHE_RTOL,
            "cache tolerance changed")
    pin = pins["models"][MODEL_KEY]
    chat = pins["models"]["sprocket_chat"]
    require(pin["repo_id"] == "kandivault/sprocket-500m"
            and pin["revision"] == MODEL_REV and pin["variant"] == "instruct"
            and pin["hf_weight_file"] == "model.safetensors"
            and pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT_SHA
            and pin["weight_bytes"] == 1002207984,
            "Sprocket Instruct pin changed")
    require(chat["variant"] == "chat"
            and chat["hf_weight_file"] == "chat/model.safetensors"
            and chat["weight_declared_lfs_sha256_not_downloaded"] == CHAT_SHA
            and CHAT_SHA != WEIGHT_SHA,
            "Sprocket Chat/Instruct source separation changed")

    fresh = load(evidence_path("sprocket-fresh"))
    cross = load(evidence_path("sprocket-cross-process"))
    extended = load(evidence_path("sprocket-extended-context"))
    wire = load(evidence_path("sprocket-publisher-wire"))
    batch = load(evidence_path("pr102-masked-batch"))
    right = load(evidence_path("pr113-rightpad-generation"))
    diagnostic = load(DATASET_WIRE)

    # Exact immutable source identity across independent runs.
    check_common_source(fresh,
        weight_path=("source_identity", "instruct_actual_weights_sha256"),
        hook_path=("source_identity", "instruct_original_implementation_sha"))
    check_common_source(cross,
        weight_path=("variants", "instruct", "weight_sha256"),
        hook_path=("variants", "instruct", "source_hook_sha"))
    check_common_source(extended,
        weight_path=("variants", "instruct", "checkpoint_sha256"),
        hook_path=("variants", "instruct", "original_hook_commit"))
    check_common_source(wire,
        weight_path=("model", "weight_sha256"),
        hook_path=("model", "original_hook_commit"))
    check_common_source(batch,
        weight_path=("source", "model_weight_sha256"),
        hook_path=("source", "original_frozen_instruct_hook_commit"))
    check_common_source(right,
        weight_path=("original_model", "weight_sha256"),
        hook_path=("original_model", "hook_source_commit"))

    # Source/runtime evidence must remain original, no-training and within fixed gates.
    require(fresh["source_run"]["original_actions_result"] == "success"
            and fresh["source_run"]["actual_model_optimizer_steps"] == 0
            and fresh["observed_results"]["zero_init_logits_exact_up_to_128_units_both"] is True
            and fresh["observed_results"]["left_right_padded_batch_noop_pass_both"] is True
            and fresh["observed_results"]["manually_activated_nca_instruct_cache_max_abs"] <= CACHE_ATOL
            and fresh["observed_results"]["future_suffix_max_abs_both"] == 0
            and fresh["observed_results"]["A_B_A_request_isolation_equal_both"] is True
            and fresh["observed_results"]["same_process_new_adapter_cold_reload_equal_both"] is True
            and fresh["observed_results"]["training_performed"] is False,
            "fresh source/padding/cache/cold evidence changed")

    ci = cross["variants"]["instruct"]
    require(cross["original_run_id"] == 40515083104
            and ci["job_outcome"] == "success"
            and ci["context_length_units"] == 256
            and ci["cached_prefill"] == 240
            and ci["incremental_decode_steps"] == 16
            and ci["cached_max_abs"] <= CACHE_ATOL
            and ci["strict_future_max_abs"] == ci["request_reset_max_abs"] ==
            ci["cold_replay_max_abs"] == 0
            and cross["qualification"]["separate_python_process"] is True
            and cross["qualification"]["actual_training_performed"] is False
            and cross["qualification"]["optimizer_steps"] == 0,
            "fresh-interpreter/cold source evidence changed")

    ei = extended["variants"]["instruct"]
    require(extended["original_run_id"] == 40544384293
            and ei["cache_2048_max_abs"] <= CACHE_ATOL
            and ei["raw_control_ids_cache_max_abs"] <= CACHE_ATOL
            and extended["evidence_limit"]["optimizer_steps"] == 0
            and extended["evidence_limit"]["train_authorized"] is False
            and extended["evidence_limit"]["heldout_evaluation"] is False,
            "2048 source/cache evidence changed")

    wc = wire["conformance"]
    require(wire["original_run"]["id"] == 40567832272
            and wire["publisher"]["commit"] ==
            "a003767e70d766bc14b40c89792543ee8641c8bc"
            and wire["publisher"]["is_crypto_training_build_attestation"] is False
            and wire["model"]["tap_indices"] == [7, 16, 25]
            and wc["status"] == "PASS_SOURCE_PUBLISHER_WIRE_CACHE_ONLY"
            and max(wc["tool_call_cached_max_abs"], wc["tool_result_cached_max_abs"],
                    wc["memory_write_cached_max_abs"]) <= CACHE_ATOL
            and wc["strict_future_suffix_max_abs"] == wc["request_reset_max_abs"] == 0
            and wire["limits"]["train_authorized"] is False
            and wire["limits"]["optimizer_steps"] == 0
            and wire["limits"]["model_generated_tool_or_memory_semantics_qualified"] is False,
            "publisher structural wire evidence/limits changed")

    bnum = batch["numerical"]
    blim = batch["evidence_limit"]
    require(batch["original_actions"]["run_id"] == 40590487839
            and batch["status"] ==
            "PASS_SOURCE_BOUND_REVIEWER_MASKED_BATCH_PREPASS__FINAL_72_BLOCKED"
            and bnum["left_padded_batch_cached_four_max_abs"] <= CACHE_ATOL
            and bnum["left_valid_padded_vs_single_max_abs"] ==
            bnum["right_valid_padded_vs_single_max_abs"] ==
            bnum["mutated_masked_pad_tool_result_vs_valid_max_abs"] ==
            bnum["cross_batch_request_isolation_max_abs"] ==
            bnum["fresh_request_A_B_A_max_abs"] == 0
            and bnum["positive_bad_unmasked_pad_sentinel_max_abs"] > 0
            and bnum["frozen_tensors_before_sha256"] ==
            bnum["frozen_tensors_after_sha256"]
            and blim["train_authorized"] is False and blim["optimizer_steps"] == 0
            and blim["heldout_evaluation"] is False
            and blim["model_generated_tool_semantics_qualified"] is False,
            "masked batch source evidence/limits changed")

    rr = right["results"]
    rs = right["scope"]
    require(right["original_actions"]["run_id"] == 40598034739
            and right["status"] ==
            "PASS_RIGHTPAD_GAPPED_KV_AND_ORIGINAL_HF_GENERATE_ONLY__FINAL_72_BLOCKED"
            and max(rr["rightpad_control_full_vs_cache_max_abs"],
                    rr["rightpad_unrelated_full_vs_cache_max_abs"],
                    rr["max_HF_generate_full_vs_cache_abs"]) <= CACHE_ATOL
            and rr["rightpad_masked_tool_result_abs"] == 0
            and rr["negative_unmasked_gap_max_abs"] > 0
            and rr["source_frozen_zero_exact"] is True
            and rr["request_reset_exact"] is True
            and rr["frozen_base_tensor_sha256_before"] ==
            rr["frozen_base_tensor_sha256_after"]
            and rr["model_generated_tool_call_competence_qualified"] is False
            and rr["general_memory_behavior_qualified"] is False
            and rs["train_authorized"] is False and rs["optimizer_steps"] == 0
            and rs["actual_training"] is False and rs["heldout_evaluation"] is False,
            "right-gap/HF.generate source evidence/limits changed")

    # Preserve negative model-output evidence; source serialization != competence.
    require(diagnostic["limits"]["train_authorized"] is False
            and diagnostic["limits"]["optimizer_steps"] == 0
            and diagnostic["limits"]["model_generated_tool_memory_semantics_qualified"] is False
            and diagnostic["original_results"]["strict_tool_envelopes_seen"] == 0
            and diagnostic["original_results"]["assistant_memory_write_markers_seen"] == 0,
            "diagnostic negative evidence falsely promoted")

    required = protocol["conformance"]["required_checks"]
    matrix = {
        "same_base_logits": {
            "status": "PASS",
            "evidence": ["40469703065", "40590487839", "40598034739"],
            "note": "Exact frozen-vs-zero logits observed at multiple lengths/padding/generation paths."
        },
        "base_weight_unchanged_after_step": {
            "status": "BLOCKED_BY_TRAINING_POLICY",
            "evidence": [],
            "note": "No optimizer step is permitted while protocol.train_authorized=false."
        },
        "causal_suffix_invariance": {
            "status": "PASS",
            "evidence": ["40469703065", "40515083104", "40567832272"],
            "note": "Future suffix influence and A→B→A request leakage were exactly zero."
        },
        "full_prefix_vs_incremental_cache": {
            "status": "PASS",
            "evidence": ["40515083104", "40544384293", "40573188117",
                         "40590487839", "40598034739"],
            "note": "Short, 256, 2048-tail, all-position structural, padded-gap, and actual HF.generate cache paths remain within 5e-4."
        },
        "per_token_isolation": {
            "status": "PASS_SOURCE_RUNTIME",
            "evidence": ["40573188117", "40590487839", "40598034739"],
            "note": "All-position/padded/cached hook traces and masked-control sentinels demonstrate per-token source isolation."
        },
        "per_request_reset": {
            "status": "PASS",
            "evidence": ["40469703065", "40515083104", "40590487839", "40598034739"],
            "note": "Independent A→B→A/request/batch/generate isolation was exact."
        },
        "nonzero_adapter_gradient_after_warmup": {
            "status": "BLOCKED_BY_TRAINING_POLICY",
            "evidence": [],
            "note": "No backward/training warmup is admitted while train_authorized=false."
        },
        "finite_state_and_loss": {
            "status": "PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY",
            "evidence": ["40590487839", "40598034739"],
            "note": "Finite states/logits are demonstrated; supervised training loss is intentionally not executed."
        },
        "cold_reload_state_dict": {
            "status": "PASS_PRETRAIN",
            "evidence": ["40469703065", "40515083104"],
            "note": "Same-process cold adapter reconstruction plus separate-Python-process frozen-base/adapter replay are exact."
        },
    }
    require(set(matrix) == set(required), "adjudication matrix does not cover every required protocol check")

    blocking = [name for name, row in matrix.items()
                if not row["status"].startswith("PASS")]
    require(set(blocking) == {
        "base_weight_unchanged_after_step",
        "nonzero_adapter_gradient_after_warmup",
        "finite_state_and_loss",
    }, "unexpected final blockers; do not silently reinterpret source evidence")

    report = {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "issue": 72,
        "model_key": MODEL_KEY,
        "model_variant": "instruct",
        "adjudication": "BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS",
        "per_model_gate": False,
        "source_runtime_pretraining_subgate": True,
        "train_authorized": False,
        "optimizer_steps_observed": 0,
        "heldout_evaluation": False,
        "reviewed_pr_heads": {
            "pr102": PR102_HEAD,
            "pr113": PR113_HEAD,
        },
        "source": {
            "repo_id": pin["repo_id"],
            "revision": MODEL_REV,
            "weight_file": "model.safetensors",
            "weight_sha256": WEIGHT_SHA,
            "weight_bytes": 1002207984,
            "original_hook_commit": HOOK_SHA,
            "taps_after_block_zero_based": [7, 16, 25],
            "distinct_chat_weight_sha256": CHAT_SHA,
        },
        "protocol": {
            "sha256": sha256(PROTOCOL),
            "source_pins_sha256": sha256(PINS),
            "cache_atol": CACHE_ATOL,
            "cache_rtol": CACHE_RTOL,
            "any_failure_blocks_model_qualification": True,
            "required_checks": required,
        },
        "matrix": matrix,
        "blocking_required_checks": blocking,
        "nonblocking_scope_limits": {
            "publisher_source_is_crypto_training_build_attestation": False,
            "model_generated_tool_memory_competence_qualified": False,
            "comparative_NCA_MLP_training_performed": False,
            "sealed_holdouts_opened": False,
            "all_parent_prs_merged": False,
        },
        "evidence_runs": [
            40469703065, 40515083104, 40544384293, 40567832272,
            40573188117, 40590487839, 40598034739,
        ],
        "decision_note": (
            "Do not set sprocket_instruct_conformance_pass=true and do not authorize "
            "training. The source/runtime pre-training portion is independently "
            "well-supported; final #72 remains blocked only by protocol-required "
            "learning/update/loss checks that are prohibited while train_authorized=false."
        ),
    }
    return report


def main() -> None:
    report = build_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
