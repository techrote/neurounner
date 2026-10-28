#!/usr/bin/env python3
"""Deterministic post-lease void.1 NCR-001E/#72 adjudication.

Consumes only durable original evidence already committed in the stacked draft:
- corrected causal-EOS + post-lease 32/128/cold evidence (#167/#168)
- post-lease 512/1024/2048 + GenerationMixin evidence (#169/#170)

No network, model execution, optimizer, backward pass, held-out fixture access,
comparative training, or merge. Strong source/runtime evidence may pass the
pretraining subgate, but final #72 must remain blocked while protocol-required
learning/update checks are absent and train_authorized=false.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
POST = ROOT / "experiments/ncr-001/controls-evidence/void-postlease-20261028"
MAX = ROOT / "experiments/ncr-001/controls-evidence/void-postlease-maxctx-20261028"
OUT = ROOT / "runtime/ncr-001/void1-postlease-adjudication.json"

MODEL_KEY = "void1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHTS = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
SERIALIZER = "9fbdee880697903aaf5b91cc29442c9e53479e50"
FROZEN = "7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904"
TOL = 0.0005
FOUR_MODELS = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
REQUIRED = {
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
        raise RuntimeError("void.1 post-lease #72 adjudication REFUSED: " + why)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_original_evidence() -> tuple[dict, dict, dict]:
    # Import established byte/semantic auditors only after protocol checks.
    import ncr_001_void_postlease_original_audit as short_audit
    import ncr_001_void_postlease_maxctx_original_audit as max_audit

    producer, consumer = short_audit.audit(
        (POST / "source-original.zip").read_bytes(),
        (POST / "qwen-original.zip").read_bytes(),
    )
    max_receipt = max_audit.audit(
        max_audit.decode_original("source"),
        max_audit.decode_original("qwen"),
    )
    return producer, consumer, max_receipt


def build_report() -> dict:
    protocol = load(PROTOCOL)
    pins = load(PINS)
    require(protocol["experiment"] == "NCR-001" and protocol["train_authorized"] is False,
            "experiment/training policy changed")
    require(set(protocol["models"]) == set(pins["models"]) == FOUR_MODELS,
            "four frozen lane identities changed")
    conf = protocol["conformance"]
    require(set(conf["required_checks"]) == REQUIRED
            and conf["any_failure_blocks_model_qualification"] is True,
            "required #72 checklist/fail-closed rule changed")
    require(conf["full_vs_cached_prefill_atol"] == TOL
            and conf["full_vs_cached_prefill_rtol"] == TOL,
            "cache tolerance changed")

    lane = protocol["models"][MODEL_KEY]
    pin = pins["models"][MODEL_KEY]
    require(lane["repo_id"] == pin["repo_id"] == "dotlabs/void.1"
            and lane["revision"] == pin["revision"] == REV
            and lane["weight_file"] == pin["hf_weight_file"] == "model.safetensors"
            and lane["taps_after_block_zero_based"] == [7, 15, 23]
            and lane["expected_config"]["hidden_size"] == 640
            and lane["expected_config"]["num_hidden_layers"] == 24
            and lane["expected_config"]["max_position_embeddings"] == 2048
            and pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHTS
            and pin["weight_bytes"] == 360622744
            and pin["tokenization_byte_py_sha256"] == TOKENIZER,
            "void.1 source/tokenizer/geometry pin changed")

    post_manifest = load(POST / "manifest.json")
    max_manifest = load(MAX / "manifest.json")
    require(post_manifest["original_run"] == 40747035876
            and post_manifest["original_source_head"] ==
            "9aed5ca0574a6885c5d60c07d16561e0f73d5043"
            and post_manifest["training_authorized"] is False
            and post_manifest["comparative_training_performed"] is False
            and post_manifest["scientific_status"] ==
            "SOURCE_BOUND_FORWARD_ONLY__72_FINAL_BLOCKED",
            "post-lease short/cold manifest changed")
    require(max_manifest["original_run"] == 40755894558
            and max_manifest["original_source_head"] ==
            "1438f45efc0dab8ed669a814b4897fc7a839302f"
            and max_manifest["training_authorized"] is False
            and max_manifest["optimizer_steps"] == 0
            and max_manifest["backward_executed"] is False
            and max_manifest["heldout_opened"] is False
            and max_manifest["comparative_training_performed"] is False
            and max_manifest["full_72_conformance_pass"] is False
            and max_manifest["scientific_status"] ==
            "SOURCE_BOUND_POSTLEASE_MAXCTX_PRETRAIN_PASS__FINAL_72_BLOCKED",
            "post-lease max-context manifest changed")
    for source in (post_manifest["pinned_source_hashes"],
                   max_manifest["source_identity"]):
        require(source["corrected_serializer_sha256"] ==
                "a2f575fa536792aaf8cc283a329b23eb8e98c5a07b7a18e4b1c622d5a1435118"
                and (source.get("serializer_commit") or
                     source.get("corrected_serializer_commit")) == SERIALIZER
                and source["postlease_nca_source_sha256"] ==
                "b85516da0b2020985d03b12a498e51d883433c335858c6d93cce8fda7f63748a"
                and source["postlease_mlp_source_sha256"] ==
                "f68d72092658319147d571e70dff682bdb501333e94275f3b3099b71c11a92d5",
                "post-lease wrapper/EOS source identity changed")
    require(max_manifest["source_identity"]["model_safetensors_sha256"] == WEIGHTS
            and max_manifest["source_identity"]["byte_tokenizer_source_sha256"] == TOKENIZER,
            "max-context actual source identity changed")
    require(max_manifest["physical_result"]["frozen_backbone_sha256"] == FROZEN
            and max_manifest["physical_result"]["actual_eos_id_emitted"] is False,
            "frozen/EOS negative result changed")
    require(all(0 <= value <= TOL
                for value in max_manifest["physical_result"]["cache_max_abs"].values()),
            "max-context cache evidence exceeds frozen tolerance")

    producer, consumer, max_receipt = _audit_original_evidence()
    require(producer["runner"]["pid"] != consumer["runner"]["pid"],
            "separate-Python cold replay identity missing")
    require(producer["frozen_backbone_sha256_before"] ==
            producer["frozen_backbone_sha256_after"] ==
            consumer["frozen_backbone_sha256_before"] ==
            consumer["frozen_backbone_sha256_after"] == FROZEN,
            "post-lease frozen tensor identity changed")
    require(max_receipt["frozen_backbone_sha256_before"] ==
            max_receipt["frozen_backbone_sha256_after"] == FROZEN,
            "max-context frozen tensor identity changed")
    require(all(math.isfinite(case["native_frozen_loss"])
                and math.isfinite(case["manual_shifted_ce"])
                for case in producer["cases"]),
            "corrected causal-EOS forward losses are nonfinite")
    require(all(g["actual_eos_id_emitted"] is False
                and g["learned_eos_claim"] is False
                for g in max_receipt["generations"]),
            "four-step non-EOS result falsely promoted")

    matrix = {
        "same_base_logits": {
            "status": "PASS",
            "evidence_runs": [40747035876, 40755894558],
            "note": "Post-lease NCA disabled/zero frozen logits are exact at corrected-EOS cases and 512/1024/2048 contexts."
        },
        "base_weight_unchanged_after_step": {
            "status": "BLOCKED_BY_TRAINING_POLICY",
            "evidence_runs": [],
            "note": "Forward-only frozen fingerprints are unchanged, but no optimizer step is permitted or observed; the protocol explicitly requires unchanged base weights after a diagnostic update."
        },
        "causal_suffix_invariance": {
            "status": "PASS",
            "evidence_runs": [40747035876, 40755894558],
            "note": "Post-lease future-suffix invariance is exact, including the full 2048-unit NCA witness."
        },
        "full_prefix_vs_incremental_cache": {
            "status": "PASS",
            "evidence_runs": [40747035876, 40755894558],
            "note": "Original frozen Qwen3 passes 32/128 and 512/1024/2048 incremental-cache checks plus real GenerationMixin cache on/off under unchanged 5e-4 tolerances."
        },
        "per_token_isolation": {
            "status": "PASS_SOURCE_RUNTIME",
            "evidence_runs": [40747035876, 40755894558],
            "note": "Shared decoder-root lease, explicit per-forward field creation, causal sentinels and cached-token source traces reject cross-adapter/token contamination."
        },
        "per_request_reset": {
            "status": "PASS",
            "evidence_runs": [40747035876],
            "note": "Original post-lease A→B→A request isolation is exact for the NCA and matched control."
        },
        "nonzero_adapter_gradient_after_warmup": {
            "status": "BLOCKED_BY_TRAINING_POLICY",
            "evidence_runs": [],
            "note": "No post-lease backward/warmup is admitted in this stack; manually nonzero readouts are diagnostic, not learned gradients."
        },
        "finite_state_and_loss": {
            "status": "PARTIAL_BLOCKED_WARMUP_LOSS",
            "evidence_runs": [40747035876, 40755894558],
            "note": "States/logits are finite and corrected causal-EOS frozen losses match independent shifted CE, but the required supervised adapter loss after warmup has not been executed."
        },
        "cold_reload_state_dict": {
            "status": "PASS_PRETRAIN",
            "evidence_runs": [40747035876],
            "note": "NCA and matched-MLP adapter binaries replay exactly in a separate Python interpreter against the same pinned frozen Qwen3 source."
        },
    }
    require(set(matrix) == set(conf["required_checks"]),
            "adjudication matrix does not cover every protocol-required check")
    blockers = [k for k, v in matrix.items() if not v["status"].startswith("PASS")]
    require(set(blockers) == {
        "base_weight_unchanged_after_step",
        "nonzero_adapter_gradient_after_warmup",
        "finite_state_and_loss",
    }, "unexpected final blocker set")

    report = {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "issue": 72,
        "model_key": MODEL_KEY,
        "model_variant": "base_byte",
        "adjudication":
            "BLOCKED_REQUIRED_LEARNING_CHECKS__POSTLEASE_SOURCE_RUNTIME_PRETRAIN_PASS",
        "per_model_gate": False,
        "source_runtime_pretraining_subgate": True,
        "train_authorized": False,
        "optimizer_steps_observed": 0,
        "backward_executed": False,
        "heldout_evaluation": False,
        "comparative_training_performed": False,
        "independent_scientific_review": False,
        "source": {
            "repo_id": "dotlabs/void.1",
            "revision": REV,
            "weight_file": "model.safetensors",
            "weight_sha256": WEIGHTS,
            "weight_bytes": 360622744,
            "tokenizer_source_sha256": TOKENIZER,
            "corrected_serializer_commit": SERIALIZER,
            "taps_after_block_zero_based": [7, 15, 23],
            "max_context_units": 2048,
            "eos_id": 258,
        },
        "protocol": {
            "sha256": sha256(PROTOCOL),
            "source_pins_sha256": sha256(PINS),
            "cache_atol": TOL,
            "cache_rtol": TOL,
            "required_checks": conf["required_checks"],
            "any_failure_blocks_model_qualification": True,
        },
        "evidence": {
            "corrected_eos_postlease_run": 40747035876,
            "corrected_eos_postlease_source_head":
                "9aed5ca0574a6885c5d60c07d16561e0f73d5043",
            "postlease_max_context_run": 40755894558,
            "postlease_max_context_source_head":
                "1438f45efc0dab8ed669a814b4897fc7a839302f",
            "max_context_cache_max_abs":
                max_manifest["physical_result"]["cache_max_abs"],
            "max_context_generation_records": 12,
            "max_context_generation_ids": [101, 110, 32, 109],
            "actual_eos_258_emitted_in_four_step_probe": False,
            "frozen_backbone_sha256": FROZEN,
        },
        "byte_eos_dependency": {
            "future_serializer_source_runtime_status":
                "PASS_FORWARD_ONLY__NO_LEARNED_STOPPING_CLAIM",
            "corrected_terminal_eos_id": 258,
            "historical_train002b_effect_measured": False,
            "four_step_generation_emitted_eos": False,
            "note": "Correct target construction and original-host causal forward loss are supported; learned EOS stopping and historical effect size remain unmeasured."
        },
        "matrix": matrix,
        "blocking_required_checks": blockers,
        "nonblocking_scope_limits": {
            "matched_mlp_postlease_source_runtime_prepass": True,
            "matched_nca_mlp_comparative_training_performed": False,
            "sealed_holdouts_opened": False,
            "learned_eos_stopping_qualified": False,
            "all_parent_prs_merged": False,
        },
        "decision_note": (
            "Do not set void1_conformance_pass=true and do not authorize comparative "
            "training. Post-lease exact-source/cache/cold/EOS-serialization runtime "
            "evidence is strong through 2048 units, but the protocol-required "
            "learning/update/supervised-warmup checks remain absent while "
            "train_authorized=false."
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
