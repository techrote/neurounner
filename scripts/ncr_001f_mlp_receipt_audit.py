"""NCR-001F: independent byte-exact audit of original real SmolLM2 MLP receipt.

No HF weights, PyTorch, network, optimizer or sealed evaluation materials.
The original source Actions ZIP was independently retrieved and SHA-verified
before its exact 5,505-byte JSON entry was committed. This script rechecks the
committed raw bytes, source pins, numerical gates and negative admission flags.
It does NOT independently execute a real model or qualify issue #72.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/ncr-001/controls-evidence/real-smollm2-mlp"
ORIGINAL_RECEIPT_SHA256 = "27aef63e0855a6cd93f6dd721d88a7ef2ec17978a5c15b8163437b8d57e2b86d"
ORIGINAL_ARTIFACT_ZIP_SHA256 = "d143c244d5e4ae9de5ddd0c186589caae5c2694eb67219af131563c51ee8447a"
ORIGINAL_SOURCE_BRANCH_COMMIT = "eddfaa4f6704c58fe336180031a26c9212490f25"
SOURCE_WEIGHT_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
FROZEN_FULL_TENSOR_DIGEST = "f2bf4fb1ff77f0fb74d49020170bc0ce7f0a141c4d039653030d57059fdbf086"
FIXED_ATOL = 0.0005


def audit_payload(receipt: dict, provenance: dict) -> dict:
    """Source/semantics-only review; archive byte SHA checked by audit() first."""
    source = provenance["original_source"]
    observed = provenance["observed"]
    if (
        source["original_raw_sha256"] != ORIGINAL_RECEIPT_SHA256
        or source["original_raw_size_bytes"] != 5505
        or source["original_zip_sha256"] != ORIGINAL_ARTIFACT_ZIP_SHA256
        or source["source_head_commit"] != ORIGINAL_SOURCE_BRANCH_COMMIT
        or source["actions_run_id"] != 40491265310
        or source["job_id"] != 122145756840
        or source["job_conclusion"] != "success"
        or source["original_artifact_id"] != 12727648261
    ):
        raise RuntimeError("Original source/run/archive identity drifted")
    if receipt["status"] != "PASS_SOURCE_BOUND_REAL_SMOLLM2_MLP_HOOK_PREPASS__TRAINING_BLOCKED":
        raise RuntimeError("Original receipt was not source-qualified preliminary PASS")
    if (receipt["source_identity"]["model_id"] != "HuggingFaceTB/SmolLM2-135M-Instruct"
            or receipt["source_identity"]["revision"] != "12fd25f77366fa6b3b4b768ec3050bf629380bac"
            or receipt["source_identity"]["weight_path"] != "model.safetensors"
            or receipt["source_identity"]["weight_sha256"] != SOURCE_WEIGHT_SHA256
            or receipt["source_identity"]["taps"] != [9, 19, 29]
            or receipt["source_identity"]["width"] != 576
            or receipt["source_identity"]["model_type"] != "llama"):
        raise RuntimeError("Frozen SmolLM2 identity or tap layout changed")
    checkpoint = receipt["source_files"]["model.safetensors"]
    if (checkpoint["sha256"] != SOURCE_WEIGHT_SHA256
            or checkpoint["expected_sha256"] != SOURCE_WEIGHT_SHA256
            or checkpoint["bytes"] != 269060552):
        raise RuntimeError("Actual downloaded checkpoint bytes have wrong identity")
    if (receipt["base_frozen_parameters"] != 134515008
            or receipt["gated_mlp_trainable_parameters"] != 296576
            or receipt["nca_trainable_parameters_reference"] != 297048):
        raise RuntimeError("Parameter identity or matched control was altered")
    for key in ("train_authorized", "model_training_performed", "nca_training_performed",
                "comparative_training_performed", "independent_ncr_001e_conformance_pass",
                "heldout_data_opened", "four_model_actual_host_mlp_qualified"):
        if receipt[key] is not False:
            raise RuntimeError("False training/heldout/full-conformance gate became true: " + key)
    if receipt["optimizer_steps"] != 0:
        raise RuntimeError("Unadmitted optimizer step detected")
    if (receipt["frozen_base_in_memory_sha256_before"] != FROZEN_FULL_TENSOR_DIGEST
            or receipt["frozen_base_in_memory_sha256_after"] != FROZEN_FULL_TENSOR_DIGEST):
        raise RuntimeError("Full frozen model in-memory tensor digest changed")
    checks = receipt["checks"]
    for name in (
        "zero_init_16_tokens", "zero_init_64_tokens", "zero_init_128_tokens",
        "zero_init_masked_batch2_left", "zero_init_masked_batch2_right",
        "nontrained_48_token_full_vs_real_incremental_KV",
        "future_suffix_causal_invariance", "request_A_B_A_reset",
        "new_adapter_same_base_cold_reload",
    ):
        row = checks[name]
        if row["close"] is not True or not (0 <= row["max_abs"] <= row["atol"]):
            raise RuntimeError("Original causal/cache/noop numerical gate failed: " + name)
        if name.startswith("zero_init") and row["max_abs"] != 0.0:
            raise RuntimeError("Original initial no-op had nonzero logit error: " + name)
    if (checks["nontrained_48_token_full_vs_real_incremental_KV"]["atol"] != FIXED_ATOL
            or checks["nontrained_48_token_full_vs_real_incremental_KV"]["max_abs"]
            != 8.869171142578125e-05
            or checks["future_suffix_causal_invariance"]["max_abs"] != 0.0
            or checks["wrong_checkpoint_variant_refusal"] is not True
            or checks["zero_init_greedy_generate"]["equal"] is not True
            or checks["readout_gradient_only_no_optimizer"]["frozen_base_grads_none"] is not True
            or checks["readout_gradient_only_no_optimizer"]["grad_l1"] <= 0
            or checks["readout_gradient_only_no_optimizer"]["optimizer_steps"] != 0
            or checks["full_base_tensor_bytes_unchanged"] is not True):
        raise RuntimeError("Actual-source reviewer negative or gradient requirement failed")
    if (
        receipt["runner"]["github_run_id"] != str(source["actions_run_id"])
        or receipt["runner"]["github_sha"] != source["github_job_effective_sha"]
        or receipt["runner"]["cpu_threads"] != 4
        or receipt["runner"]["torch"] != "2.14.1+cpu"
        or receipt["runner"]["transformers"] != "4.57.6"
        or receipt["source_sha256"]["scripts/ncr_001f_frozen_hooks.py"] != "a11c9ffa9853b300bb24ad827e4d550539007d057c03e6370b11cbab5a5f15c6"
        or receipt["source_sha256"]["experiments/ncr-001/protocol.json"] != "a76d38f80a57012ca8e62fba183d1cf2e91fd0986032f1adcdfb0aef843bf0b4"
    ):
        raise RuntimeError("Runner identity / source code SHA not preserved")
    if observed["optimizer_steps"] != 0 or observed["train_authorized"] is not False:
        raise RuntimeError("Promoted source provenance falsely reports training admission")
    return {
        "status": "PASS_ORIGINAL_SMOLLM2_MLP_RAW_RECEIPT_SOURCE_READBACK__NO_FINAL_CONFORMANCE",
        "original_raw_sha256": ORIGINAL_RECEIPT_SHA256,
        "original_zip_sha256": ORIGINAL_ARTIFACT_ZIP_SHA256,
        "source_commit": ORIGINAL_SOURCE_BRANCH_COMMIT,
        "source_run": source["actions_run_id"],
        "actual_real_model_inference_performed_by_source_job": True,
        "independent_real_model_execution_by_this_audit": False,
        "actual_model_qualification_scope": "smollm2_mlp_only",
        "four_model_ncr_001e_pass": False,
        "tool_and_memory_semantics_qualified": False,
        "model_training_performed": False,
        "train_authorized": False,
    }


def audit(evidence_dir: Path = EVIDENCE) -> dict:
    receipt_bytes = (evidence_dir / "receipt.json").read_bytes()
    if len(receipt_bytes) != 5505 or hashlib.sha256(receipt_bytes).hexdigest() != ORIGINAL_RECEIPT_SHA256:
        raise RuntimeError("Original raw JSON receipt byte hash/size mismatch")
    receipt = json.loads(receipt_bytes)
    provenance = json.loads((evidence_dir / "provenance.json").read_text(encoding="utf-8"))
    return audit_payload(receipt, provenance)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
