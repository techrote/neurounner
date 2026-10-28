#!/usr/bin/env python3
"""Deterministic NCR-001E Sprocket Chat #72 lane adjudication.

No model execution, network, optimizer, heldout fixture access or training.
The report re-reads durable original source/runtime evidence and deliberately
keeps the final per-model gate false while required learning/update/loss checks
remain prohibited by protocol.train_authorized=false.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
EVIDENCE = ROOT / "experiments/ncr-001/evidence-review"
OUT = ROOT / "runtime/ncr-001/sprocket-chat-adjudication.json"

MODEL_KEY = "sprocket_chat"
WEIGHT_SHA = "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"
INSTRUCT_SHA = "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"
MODEL_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
HOOK_SHA = "83a56835a7476f17a17980d28dd1a7c14bb6f2f6"
FORMAT_BLOB = "39cf015735e74a1d1e08682b623404bc1e600b9b"
PR113_HEAD = "7dec2f91b6542854762542807978e5bddead4f64"
PR123_HEAD = "57bf036f1a8728b35545473b475e3306eee440d7"
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


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("NCR Sprocket Chat adjudication REFUSED: " + why)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(name: str) -> dict:
    return load(EVIDENCE / name / "provenance.json")


def verify_original_shared_sprocket_custody() -> None:
    """Require byte-exact original Sprocket Chat/Instruct archives, not summaries.

    Original run/runner/model SHA, fail-closed numeric controls and unchanged
    receipt bytes are revalidated using the existing independent auditors.
    The Chat-only HF.generate original-byte audit also runs below. No model,
    tokenizer, sealed fixtures, optimizer or network are invoked here.
    """
    import ncr_001e_sprocket_receipt_audit as fresh_audit
    import ncr_001e_sprocket_process_receipt_audit as process_audit
    import ncr_001e_sprocket_extended_receipt_audit as extended_audit

    fresh = fresh_audit.audit()
    require(fresh["status"] ==
            "PASS_ORIGINAL_TRIPLE_SPROCKET_RECEIPT_BYTE_HASH_AND_NUMERIC_READBACK"
            and fresh["source_run_id"] == 40469703065
            and fresh["training_authorized"] is False,
            "source-fresh original receipt custody failed")
    process = process_audit.audit()
    require(process["status"] == process_audit.READBACK_STATUS
            and process["source_integrity_readback_pass"] is True
            and process["source_run_id"] == 40515083104
            and process["train_authorized"] is False,
            "source cross-process original receipt custody failed")
    extended = extended_audit.audit()
    require(extended["status"] == extended_audit.STATUS
            and extended["original_real_model_run_id"] == 40544384293
            and extended["training_authorized"] is False,
            "2048-context original receipt custody failed")


def build_report() -> dict:
    protocol = load(PROTOCOL)
    pins = load(PINS)
    require(protocol["experiment"] == "NCR-001"
            and protocol["train_authorized"] is False,
            "protocol experiment/training flag changed")
    require(set(protocol["models"]) == set(pins["models"]) == FOUR_MODELS,
            "four independent model identities changed")
    require(set(protocol["conformance"]["required_checks"]) == REQUIRED_CHECKS,
            "required #72 checklist changed")
    require(protocol["conformance"]["any_failure_blocks_model_qualification"] is True,
            "fail-closed qualification policy changed")
    require(protocol["conformance"]["full_vs_cached_prefill_atol"] == CACHE_ATOL
            and protocol["conformance"]["full_vs_cached_prefill_rtol"] == CACHE_RTOL,
            "cache tolerances changed")

    pin = pins["models"][MODEL_KEY]
    instruct = pins["models"]["sprocket_instruct"]
    require(pin["repo_id"] == "kandivault/sprocket-500m"
            and pin["revision"] == MODEL_REV and pin["variant"] == "chat"
            and pin["hf_weight_file"] == "chat/model.safetensors"
            and pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT_SHA
            and pin["weight_bytes"] == 1002207984,
            "Sprocket Chat source pin changed")
    require(instruct["variant"] == "instruct"
            and instruct["hf_weight_file"] == "model.safetensors"
            and instruct["weight_declared_lfs_sha256_not_downloaded"] == INSTRUCT_SHA
            and WEIGHT_SHA != INSTRUCT_SHA,
            "Chat/Instruct checkpoint distinction changed")

    verify_original_shared_sprocket_custody()

    fresh = evidence("sprocket-fresh")
    cross = evidence("sprocket-cross-process")
    extended = evidence("sprocket-extended-context")
    generated = evidence("sprocket-chat-generate")

    require(fresh["source_identity"]["chat_actual_weights_sha256"] == WEIGHT_SHA
            and fresh["source_identity"]["chat_original_implementation_sha"] == HOOK_SHA
            and fresh["source_identity"]["instruct_actual_weights_sha256"] == INSTRUCT_SHA
            and fresh["source_identity"]["tool_and_memory_behavior_independently_qualified"]
            is False,
            "fresh source role/variant evidence changed")
    fr = fresh["observed_results"]
    require(fresh["source_run"]["run_id"] == 40469703065
            and fresh["source_run"]["original_actions_result"] == "success"
            and fresh["source_run"]["actual_model_optimizer_steps"] == 0
            and fr["zero_init_logits_exact_up_to_128_units_both"] is True
            and fr["left_right_padded_batch_noop_pass_both"] is True
            and fr["manually_activated_nca_chat_cache_max_abs"] <= CACHE_ATOL
            and fr["future_suffix_max_abs_both"] == 0
            and fr["same_process_new_adapter_cold_reload_equal_both"] is True
            and fr["A_B_A_request_isolation_equal_both"] is True
            and fr["wrong_variant_weight_sha_revision_refused_both"] is True
            and fr["training_performed"] is False
            and fr["tool_and_memory_semantics_pass"] is False,
            "fresh Chat source/padding/cache/cold evidence changed")

    cv = cross["variants"]["chat"]
    cq = cross["qualification"]
    require(cross["original_run_id"] == 40515083104
            and cv["weight_file"] == "chat/model.safetensors"
            and cv["weight_sha256"] == WEIGHT_SHA
            and cv["source_hook_sha"] == HOOK_SHA
            and cv["job_outcome"] == "success"
            and cv["context_length_units"] == 256
            and cv["cached_prefill"] == 240
            and cv["incremental_decode_steps"] == 16
            and cv["cached_max_abs"] <= CACHE_ATOL
            and cv["strict_future_max_abs"] == cv["request_reset_max_abs"] ==
            cv["cold_replay_max_abs"] == 0
            and cq["separate_python_process"] is True
            and cq["source_verified_tool_memory_argument_result_grammar"] is False
            and cq["train_authorized"] is False
            and cq["actual_training_performed"] is False
            and cq["optimizer_steps"] == 0,
            "separate-process Chat source/cache/cold evidence changed")

    ev = extended["variants"]["chat"]
    el = extended["evidence_limit"]
    require(extended["original_run_id"] == 40544384293
            and ev["checkpoint_file"] == "chat/model.safetensors"
            and ev["checkpoint_sha256"] == WEIGHT_SHA
            and ev["original_hook_commit"] == HOOK_SHA
            and ev["cache_2048_max_abs"] <= CACHE_ATOL
            and ev["raw_control_ids_cache_max_abs"] <= CACHE_ATOL
            and el["optimizer_steps"] == 0 and el["train_authorized"] is False
            and el["tool_memory_semantics_qualified"] is False
            and el["heldout_evaluation"] is False,
            "2048 Chat source/control/cache evidence changed")

    gs, gr, scope = generated["source"], generated["results"], generated["scope"]
    require(generated["original_actions"]["run_id"] == 40603226200
            and generated["status"] ==
            "PASS_CHAT_RIGHTGAP_AND_HF_GENERATE_PRETRAIN_ONLY__FINAL_72_BLOCKED"
            and gs["variant"] == "chat"
            and gs["checkpoint_file"] == "chat/model.safetensors"
            and gs["checkpoint_sha256"] == WEIGHT_SHA
            and gs["distinct_instruct_checkpoint_sha256"] == INSTRUCT_SHA
            and gs["original_hook_commit"] == HOOK_SHA
            and gs["formatter_git_blob"] == FORMAT_BLOB
            and gs["ordinary_role_format_only"] is True
            and gs["instruct_tool_memory_semantics_inherited"] is False,
            "new Chat source identity/ordinary-role scope changed")
    require(max(gr["rightgap_long_max_abs"],
                gr["rightgap_short_max_abs"],
                gr["max_generate_cache_on_off_abs"]) <= CACHE_ATOL
            and gr["masked_padding_control_id7_max_abs"] == 0
            and gr["invalid_unmasked_padding_max_abs"] > 0
            and gr["frozen_zero_exact"] is True
            and gr["frozen_tensor_sha_before"] == gr["frozen_tensor_sha_after"]
            and gr["cache_atol"] == CACHE_ATOL and gr["cache_rtol"] == CACHE_RTOL
            and scope["train_authorized"] is False and scope["optimizer_steps"] == 0
            and scope["actual_training"] is False and scope["heldout_evaluation"] is False
            and scope["tool_memory_semantics_qualified"] is False
            and scope["final_four_model_72_conformance"] is False
            and scope["model_generated_tool_memory_competence"] is False,
            "Chat right-gap/generate results or no-training limits changed")

    # Re-run the independent immutable original receipt verifier, still no model.
    sys.path.insert(0, str(ROOT / "scripts"))
    from ncr_001e_sprocket_chat_generate_audit import load_originals, verify_originals
    originals, provenance = load_originals()
    verify_originals(originals, provenance)

    matrix = {
        "same_base_logits": {
            "status": "PASS",
            "evidence": ["40469703065", "40603226200"],
            "note": "Frozen Chat logits are exact with zero-initialized NCA over source roles, padding, and generation."
        },
        "base_weight_unchanged_after_step": {
            "status": "BLOCKED_BY_TRAINING_POLICY",
            "evidence": [],
            "note": "No optimizer step is authorized while protocol.train_authorized=false."
        },
        "causal_suffix_invariance": {
            "status": "PASS",
            "evidence": ["40469703065", "40515083104"],
            "note": "Future suffix and independent request influence were exactly zero."
        },
        "full_prefix_vs_incremental_cache": {
            "status": "PASS",
            "evidence": ["40469703065", "40515083104", "40544384293", "40603226200"],
            "note": "48/256/2048-tail, right-gap, and real HF generate cache-on/off paths stay within 5e-4."
        },
        "per_token_isolation": {
            "status": "PASS_SOURCE_RUNTIME",
            "evidence": ["40515083104", "40603226200"],
            "note": "Per-forward three-tap traces, true one-token cache calls, and masked padding/control sentinels remain isolated."
        },
        "per_request_reset": {
            "status": "PASS",
            "evidence": ["40469703065", "40515083104", "40603226200"],
            "note": "A→B→A, separate-process replay, batched padding, and post-generation request reset are exact."
        },
        "nonzero_adapter_gradient_after_warmup": {
            "status": "BLOCKED_BY_TRAINING_POLICY",
            "evidence": [],
            "note": "No backward/training warmup is admitted while train_authorized=false."
        },
        "finite_state_and_loss": {
            "status": "PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY",
            "evidence": ["40603226200"],
            "note": "Finite NCA state/logits are runtime-verified; supervised training loss is intentionally not executed."
        },
        "cold_reload_state_dict": {
            "status": "PASS_PRETRAIN",
            "evidence": ["40469703065", "40515083104"],
            "note": "Fresh adapter construction and separate-Python-process state/replay are exact."
        },
    }
    require(set(matrix) == set(protocol["conformance"]["required_checks"]),
            "matrix no longer covers every protocol-required check")
    blockers = [k for k, v in matrix.items() if not v["status"].startswith("PASS")]
    require(set(blockers) == {
        "base_weight_unchanged_after_step",
        "nonzero_adapter_gradient_after_warmup",
        "finite_state_and_loss",
    }, "unexpected blockers or false PASS")

    return {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "issue": 72,
        "model_key": MODEL_KEY,
        "model_variant": "chat",
        "adjudication": "BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS",
        "per_model_gate": False,
        "source_runtime_pretraining_subgate": True,
        "train_authorized": False,
        "optimizer_steps_observed": 0,
        "heldout_evaluation": False,
        "reviewed_pr_heads": {
            "pr113": PR113_HEAD,
            "pr123": PR123_HEAD,
        },
        "source": {
            "repo_id": pin["repo_id"],
            "revision": MODEL_REV,
            "weight_file": "chat/model.safetensors",
            "weight_sha256": WEIGHT_SHA,
            "weight_bytes": 1002207984,
            "original_hook_commit": HOOK_SHA,
            "taps_after_block_zero_based": [7, 16, 25],
            "ordinary_formatter_git_blob": FORMAT_BLOB,
            "distinct_instruct_weight_sha256": INSTRUCT_SHA,
        },
        "protocol": {
            "sha256": sha(PROTOCOL),
            "source_pins_sha256": sha(PINS),
            "cache_atol": CACHE_ATOL,
            "cache_rtol": CACHE_RTOL,
            "any_failure_blocks_model_qualification": True,
            "required_checks": protocol["conformance"]["required_checks"],
        },
        "matrix": matrix,
        "blocking_required_checks": blockers,
        "source_semantic_scope": {
            "ordinary_user_assistant_role_format_verified": True,
            "instruct_tool_memory_grammar_inherited": False,
            "chat_tool_memory_semantics_qualified": False,
            "publisher_chat_excludes_tool_memory_finetuning": True,
        },
        "nonblocking_scope_limits": {
            "comparative_NCA_MLP_training_performed": False,
            "sealed_holdouts_opened": False,
            "model_generated_tool_memory_competence_qualified": False,
            "all_parent_prs_merged": False,
        },
        "evidence_runs": [40469703065, 40515083104, 40544384293, 40603226200],
        "preserved_negative_runs": [40603167288],
        "decision_note": (
            "Do not set sprocket_chat_conformance_pass=true and do not authorize "
            "training. Ordinary Chat source/runtime pre-training conformance is "
            "well-supported and distinct from Instruct tool/memory semantics; final "
            "#72 remains blocked by required learning/update/loss checks that cannot "
            "run while train_authorized=false."
        ),
    }


def main() -> None:
    report = build_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
