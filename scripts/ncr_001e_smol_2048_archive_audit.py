#!/usr/bin/env python3
"""No-model, fail-closed audit of original SmolLM2 #150 Actions 40651696185.

This runs only against committed read-only receipts, NOT against HF weights.
It cannot promote a source pretraining result into a final #72 PASS.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/ncr-001/evidence-review/smolllm2-2048"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
MODEL_SHA = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
HOOK_SHA = "ec0d4a80e15f203a06c26ee9c23858f78976370b"
ORIGINAL_HEAD = "b452d3e5b112507b39be57bbfcac7f0ab721b609"
CACHE_BOUND = 0.0005
BLOCKERS = {
    "base_weight_unchanged_after_step",
    "nonzero_adapter_gradient_after_warmup",
    "finite_state_and_loss",
}


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(why)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def validate_reports(producer: dict, consumer: dict, manifest: dict, protocol: dict, pins: dict) -> None:
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) == {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "Protocol training gate or four model identities changed")
    require(protocol["conformance"]["full_vs_cached_prefill_atol"] == CACHE_BOUND
            and protocol["conformance"]["full_vs_cached_prefill_rtol"] == CACHE_BOUND,
            "Protocol numerical tolerance altered")
    source = pins["models"]["smollm2"]
    require(source["repo_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
            and source["revision"] == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
            and source["hf_weight_file"] == "model.safetensors"
            and source["weight_declared_lfs_sha256_not_downloaded"] == MODEL_SHA
            and source["weight_bytes"] == 269060552,
            "Original frozen SmolLM2 variant/source pin changed")
    sr = manifest["source_run"]
    require(sr["run_id"] == 40651696185 and sr["job_id"] == 122693027277
            and sr["tested_exact_head"] == ORIGINAL_HEAD
            and sr["source_hook_commit"] == HOOK_SHA
            and sr["full_actual_weight_sha256"] == MODEL_SHA
            and sr["job_conclusion"] == "success",
            "Recorded original Actions/head/hook/weight run identity changed")
    require(set(manifest["required_final_gate"]["learning_checks_blocked"]) == BLOCKERS
            and manifest["required_final_gate"]["train_authorized"] is False
            and manifest["required_final_gate"]["per_model_gate"] is False
            and manifest["required_final_gate"]["optimizer_steps"] == 0,
            "Manifest false final training admission or learning check omission")
    require(manifest["qualification"] == "SOURCE_RUNTIME_PRETRAIN_PASS__FINAL_72_BLOCKED",
            "Pretraining archive spuriously promoted to final PASS")
    require(manifest["original_actions_artifact"]["id"] == 12800615190
            and manifest["original_actions_artifact"]["zip_sha256"] ==
            "ae0b26e835f451a513527b7bad3a52e783248300a7a677b1f075a19d658b0f09"
            and manifest["original_actions_artifact"]["original_zip_committed_in_git"] is False,
            "Missing or falsely relabelled original transient Actions ZIP")
    require(producer["phase"] == "producer" and consumer["phase"] == "consumer"
            and producer["status"] == consumer["status"] == "PASS_PRETRAIN_ONLY",
            "Original source/runtime phase failed")
    for entry in (producer, consumer):
        require(entry["lane"] == "smollm2" and entry["experiment"] == "NCR-001E"
                and entry["training_authorized"] is False
                and entry["training_performed"] is False
                and entry["optimizer_steps"] == 0
                and entry["per_model_gate"] is False
                and entry["heldout_evaluation"] is False
                and entry["comparison_training"] is False
                and entry["full_four_model_conformance_pass"] is False,
                "Original phase spuriously claims training, qualification or evaluation")
        require(entry["process"]["github_run_id"] == "40651696185"
                and entry["process"]["github_head_sha"] == ORIGINAL_HEAD
                and entry["process"]["runner_name"] == sr["runner_name"]
                and entry["source"]["actual_full_weight_sha256"] == MODEL_SHA
                and entry["source"]["hook_commit"] == HOOK_SHA,
                "Original actual host/source identity not maintained")
        require(entry["source"]["protocol_sha256"] == sr["protocol_sha256"]
                and entry["source"]["source_pins_sha256"] == sr["pins_sha256"],
                "Original source/protocol hashes changed")
    require(producer["source"] == consumer["source"], "Two processes used different source")
    require(producer["process"]["pid"] != consumer["process"]["pid"]
            and producer["process"]["pid"] == consumer["producer_pid"]
            and producer["process"]["github_run_id"] == consumer["producer_run_id"]
            and producer["process"]["cpu_model"] == consumer["process"]["cpu_model"],
            "Same Python process or mismatched cold replay")
    checks = producer["checks"]
    for size in (512, 1024, 2048):
        row = checks[f"zero_noop_{size}"]
        require(row["exact"] is True and row["max_abs"] == 0
                and row["within"] is True and row["atol"] == 0.0001
                and row["rtol"] == 0.0001,
                f"{size} frozen source zero-init parity failed")
        row = checks[f"active_tail_{size}_prefill_{size - 16}_plus_16"]
        require(row["within"] is True and row["max_abs"] <= CACHE_BOUND
                and row["atol"] == row["rtol"] == CACHE_BOUND,
                f"{size} true 16-unit cached decode failed")
        require(row["max_abs"] == manifest["original_results"]["cache_16_tail_max_abs"][str(size)],
                f"{size} original archived numeric summary mismatched")
    require(checks["future_suffix_256"]["max_abs"] == 0
            and checks["request_A_B_A_exact"] is True,
            "Original causality or request reset broke")
    require(checks["backbone_before_after_tensor_sha256"]["before"] ==
            checks["backbone_before_after_tensor_sha256"]["after"],
            "Frozen backbone bytes changed")
    require(checks["manually_activated_untrained_effect_max_abs"] > 0,
            "Manual untrained NCA readout never affected logits")
    for mode in ("active", "disabled", "zero"):
        row = checks[f"hf_generate_{mode}_cache_on_off_scores"]
        require(row["within"] is True and row["max_abs"] <= CACHE_BOUND
                and row["atol"] == row["rtol"] == CACHE_BOUND,
                f"Original HF.generate {mode} cache comparison failed")
    for mode in (False, True):
        row = checks[f"hf_generate_frozen_vs_zero_scores_cache_{mode}"]
        require(row["within"] is True and row["max_abs"] == 0
                and row["atol"] == row["rtol"] == 0.0001,
                "Original zero-adapter generation scores changed")
    all_ids = list(checks["hf_generate_raw_ids"].values())
    require(len(all_ids) == 6 and all(len(v) == 4 for v in all_ids)
            and all(x == all_ids[0] for x in all_ids),
            "Original HF.generate output IDs changed across modes")
    cold = consumer["checks"]["separate_process_reloaded_active_adapter_exact"]
    require(cold["within"] is True and cold["max_abs"] == 0
            and cold["atol"] == cold["rtol"] == 0
            and consumer["checks"]["wrong_revision_cold_load_refused"] is True,
            "Original independently reloaded model/adapter comparison failed")
    require(producer["cold_reference"]["sha256"] ==
            manifest["original_members"]["cold-reference.pt"]["sha256"]
            and producer["cold_reference"]["adapter_state_sha256"] ==
            manifest["original_members"]["adapter/adapter.pt"]["sha256"]
            and producer["cold_reference"]["adapter_metadata_sha256"] ==
            manifest["original_members"]["adapter/metadata.json"]["sha256"],
            "Original transient state handoff digests do not match manifest")


def audit(archive: Path = ARCHIVE) -> dict:
    meta = json.loads((archive / "provenance.json").read_bytes())
    receipts = {}
    for name in ("producer.json", "consumer.json"):
        raw = (archive / name).read_bytes()
        wanted = meta["original_members"][name]
        require(len(raw) == wanted["bytes"]
                and digest(raw) == wanted["sha256"]
                and git_blob(raw) == wanted["git_blob_sha"],
                "Tampered/reformatted original Actions JSON " + name)
        receipts[name] = json.loads(raw)
    protocol_raw = PROTOCOL.read_bytes()
    pins_raw = PINS.read_bytes()
    require(digest(protocol_raw) == meta["source_run"]["protocol_sha256"]
            and digest(pins_raw) == meta["source_run"]["pins_sha256"],
            "Frozen original protocol or pinned source bytes changed")
    validate_reports(receipts["producer.json"], receipts["consumer.json"], meta,
                     json.loads(protocol_raw), json.loads(pins_raw))
    return {
        "status": "PASS_ORIGINAL_SMOL_2048_SOURCE_RUNTIME_RECEIPT_AUDIT__FINAL_72_BLOCKED",
        "original_actions_run": 40651696185,
        "original_actions_artifact_id": 12800615190,
        "original_physical_model_producer_pid": receipts["producer.json"]["process"]["pid"],
        "original_physical_model_consumer_pid": receipts["consumer.json"]["process"]["pid"],
        "full_model_72_gate": False,
        "train_authorized": False,
        "original_model_invocations_added_by_this_audit": 0,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
