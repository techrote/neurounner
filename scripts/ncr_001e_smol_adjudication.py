#!/usr/bin/env python3
"""Formal NCR-001E SmolLM2 PRETRAIN-ONLY adjudication of original physical receipts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from ncr_001e_smol_2048_archive_audit import audit as audit_2048

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"experiments/ncr-001/evidence-review/smolllm2-fresh"
LONG=ROOT/"experiments/ncr-001/evidence-review/smolllm2-2048"
PROTOCOL=ROOT/"experiments/ncr-001/protocol.json"
REPORT=ROOT/"experiments/ncr-001/adjudication/smolllm2-pretrain.json"
OLD_SHA="fb337b69d5533585dd93a1788c4236a6e08b14912ef8121b6ecd0c4a0bcab81f"
MODEL_SHA="5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
HOOK_SHA="ec0d4a80e15f203a06c26ee9c23858f78976370b"
BLOCKERS=("base_weight_unchanged_after_step","nonzero_adapter_gradient_after_warmup","finite_state_and_loss")
RUNTIME=("same_base_logits","causal_suffix_invariance","full_prefix_vs_incremental_cache","per_token_isolation","per_request_reset","cold_reload_state_dict")

def require(value, reason):
    if not value: raise ValueError("NCR-001E Smol adjudication REFUSED: "+reason)

def verify_old(old,custody,raw):
    src=custody["source_run"]
    require(hashlib.sha256(raw).hexdigest()==OLD_SHA and src["original_report_sha256"]==OLD_SHA and src["original_report_bytes"]==len(raw)
        and src["run_id"]==40436942220 and src["tested_review_commit"]=="3fa59c86ac61483ee8fcdd452e917e205ed17bb7"
        and src["pinned_implementation_commit"]==HOOK_SHA and src["result"]=="success","old source receipt or custodial identity")
    require(old["source_model"]=="HuggingFaceTB/SmolLM2-135M-Instruct"
        and old["source_revision"]=="12fd25f77366fa6b3b4b768ec3050bf629380bac"
        and old["integration_commit"]==HOOK_SHA and old["actual_full_model_safetensors"]=={"bytes":269060552,"sha256":MODEL_SHA}
        and old["optimizer_steps"]==0 and old["training_performed"] is False and old["comparative_training_admitted"] is False
        and old["model_quality_test_performed"] is False and old["full_four_model_conformance_pass"] is False
        and old["status"]=="PASS_REVIEWER_FRESH_SMOLLM2_FROZEN_CACHE_COLD_ADAPTER_PREPASS","old variant/training baseline")
    c=old["checks"]
    for n in (16,64,128):
        row=c[f"zero_init_longer_{n}"]
        require(row["within"] is True and row["max_abs"]==0 and row["atol"]==row["rtol"]==.0001,"old zero-adapter length")
    for side in ("left","right"):
        row=c[f"zero_init_{side}_padding_batch2"]
        require(row["within"] is True and row["max_abs"]==0 and row["atol"]==row["rtol"]==.0001,"old padded batch "+side)
    for name in ("nonzero_48_full_vs_incremental_KV","nonzero_future_suffix_isolation"):
        row=c[name]
        require(row["within"] is True and row["max_abs"]<=.0005 and row["atol"]==row["rtol"]==.0005,"old cache/causal bound")
    require(c["request_A_B_A_isolation"]["exact"] is True
        and c["cold_adapter_reinstantiation_same_frozen_base"]["max_abs"]==0
        and c["cold_checkpoint_variant_refusal"]["rejected"] is True
        and c["zero_init_generate_5_tokens"]["identical"] is True
        and c["manual_untrained_effect_max_abs"]>0,"old reset/cold/manual diagnostics")

def build(protocol,old,producer,consumer,old_custody,new_custody):
    require(protocol["train_authorized"] is False
        and set(protocol["models"])=={"smollm2","sprocket_chat","sprocket_instruct","void1"}
        and set(protocol["conformance"]["required_checks"])==set(RUNTIME)|set(BLOCKERS),"closed admission/check set")
    require(producer["lane"]==consumer["lane"]=="smollm2" and producer["source"]["actual_full_weight_sha256"]==MODEL_SHA
        and producer["source"]["hook_commit"]==HOOK_SHA
        and producer["optimizer_steps"]==consumer["optimizer_steps"]==0
        and producer["per_model_gate"] is consumer["per_model_gate"] is False
        and producer["training_authorized"] is consumer["training_authorized"] is False,"new variant/training gate")
    require(new_custody["source_run"]["run_id"]==40651696185
        and producer["process"]["pid"]!=consumer["process"]["pid"]
        and producer["process"]["github_run_id"]==consumer["process"]["github_run_id"]
        and old_custody["source_run"]["run_id"]!=new_custody["source_run"]["run_id"],"fake separate interpreter")
    return {
        "schema_version":1,"experiment":"NCR-001E","issue":72,"model_lane":"smollm2",
        "model_repo":"HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_revision":"12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "full_checkpoint_sha256":MODEL_SHA,"source_hook_commit":HOOK_SHA,
        "status":"BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS",
        "source_runtime_pretrain_pass":True,"final_72_pass":False,"per_model_gate":False,
        "train_authorized":False,"optimizer_steps":0,"comparative_training_performed":False,
        "original_physical_actions_runs":[40436942220,40651696185],
        "original_receipt_sha256":[OLD_SHA,new_custody["original_members"]["producer.json"]["sha256"],new_custody["original_members"]["consumer.json"]["sha256"]],
        "measured_zero_noop_lengths":[16,64,128,512,1024,2048],
        "measured_true_cached_lengths":[48,512,1024,2048],
        "measured_masked_batch_modes":["left","right"],
        "verified_hf_generate_cache_modes":[False,True],
        "verified_separate_interpreters_same_host":True,"cross_host_reboot_verified":False,
        "preregistered_check_status":{**{x:"PASS_SOURCE_RUNTIME_PRETRAIN" for x in RUNTIME},**{x:"BLOCKED_UNAUTHORIZED_LEARNING" for x in BLOCKERS}},
        "not_claimed":["optimizer-step or trained-adapter frozen-backbone proof","post-warmup gradients and supervised finite loss","cross-host independent restarts","held-out predictive quality or matched NCA/MLP comparison","acceptance of other three model lanes"],
        "qualification_scope":"Archived original physical runs; no new model invocation in adjudication",
    }

def adjudicate():
    prior=audit_2048()
    require(prior["full_model_72_gate"] is False and prior["original_model_invocations_added_by_this_audit"]==0,"archive false promotion")
    raw=(OLD/"receipt.json").read_bytes()
    old=json.loads(raw); oldmeta=json.loads((OLD/"provenance.json").read_bytes())
    verify_old(old,oldmeta,raw)
    return build(json.loads(PROTOCOL.read_bytes()),old,json.loads((LONG/"producer.json").read_bytes()),
        json.loads((LONG/"consumer.json").read_bytes()),oldmeta,json.loads((LONG/"provenance.json").read_bytes()))

def canonical(report):
    return (json.dumps(report,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()

def check_archive():
    report=adjudicate()
    require(REPORT.read_bytes()==canonical(report),"committed adjudication differs from source evidence")
    return report

if __name__=="__main__":
    print(json.dumps(check_archive(),sort_keys=True,indent=2))
