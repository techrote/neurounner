#!/usr/bin/env python3
"""Deterministic NCR-001E void.1 #72 lane adjudication.

No model execution, dynamic tokenizer import, optimizer, backward pass or
heldout access. Re-reads durable original frozen-model evidence only and keeps
the final per-model gate false while learning/update/loss checks remain blocked
by protocol.train_authorized=false.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"experiments/ncr-001/protocol.json"
PINS=ROOT/"experiments/ncr-001/source-pins.json"
EVIDENCE=ROOT/"experiments/ncr-001/evidence-review"
OUT=ROOT/"runtime/ncr-001/void1-adjudication.json"

MODEL_KEY="void1"
REPO="dotlabs/void.1"
REV="9c2be5087772867d45961abd11f6b35110617563"
WEIGHT="ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
TOKENIZER="279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
HOOK="5fa3dc8c0de153098d1800f7b0c809819620aef4"
ATOL=RTOL=0.0005
FOUR={"smollm2","sprocket_chat","sprocket_instruct","void1"}
REQUIRED={
 "same_base_logits","base_weight_unchanged_after_step",
 "causal_suffix_invariance","full_prefix_vs_incremental_cache",
 "per_token_isolation","per_request_reset",
 "nonzero_adapter_gradient_after_warmup","finite_state_and_loss",
 "cold_reload_state_dict",
}


def require(ok,why):
    if not ok:
        raise RuntimeError("NCR void1 adjudication REFUSED: "+why)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recursive_values(obj,key):
    out=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k==key: out.append(v)
            out.extend(recursive_values(v,key))
    elif isinstance(obj,list):
        for v in obj: out.extend(recursive_values(v,key))
    return out


def require_any_value(obj,key,expected,why):
    require(expected in recursive_values(obj,key),why)


def require_all_bool_if_present(obj,key,expected,why):
    values=recursive_values(obj,key)
    require(values and all(v is expected for v in values),why)


def evidence(name,file="provenance.json"):
    return load(EVIDENCE/name/file)


def build_report():
    protocol=load(PROTOCOL); pins=load(PINS)
    require(protocol["train_authorized"] is False,"training policy changed")
    require(set(protocol["models"])==set(pins["models"])==FOUR,
            "four model identities changed")
    require(set(protocol["conformance"]["required_checks"])==REQUIRED,
            "#72 required checklist changed")
    require(protocol["conformance"]["any_failure_blocks_model_qualification"] is True,
            "fail-closed qualification policy changed")
    require(protocol["conformance"]["full_vs_cached_prefill_atol"]==
            protocol["conformance"]["full_vs_cached_prefill_rtol"]==ATOL==RTOL,
            "cache tolerance changed")
    lane=protocol["models"][MODEL_KEY]; pin=pins["models"][MODEL_KEY]
    require(lane["repo_id"]==pin["repo_id"]==REPO
            and lane["revision"]==pin["revision"]==REV
            and lane["weight_file"]==pin["hf_weight_file"]=="model.safetensors"
            and pin["weight_declared_lfs_sha256_not_downloaded"]==WEIGHT
            and pin["weight_bytes"]==360622744
            and pin["tokenization_byte_py_sha256"]==
            lane["tokenizer_reference"]["train002b_source_sha256"]==TOKENIZER
            and lane["tokenizer_reference"]["ids"]==
            {"pad":256,"bos":257,"eos":258}
            and lane["expected_config"]["max_position_embeddings"]==2048
            and lane["taps_after_block_zero_based"]==[7,15,23],
            "exact void model/tokenizer/tap/context source changed")

    fresh_path=EVIDENCE/"void1-utf8-fresh"/"receipt.json"
    fresh=load(fresh_path)
    fresh_prov=evidence("void1-utf8-fresh")
    cross=evidence("void1-cross-process")
    maxctx=evidence("void1-2048")

    require(sha(fresh_path)==
            "a595e5bc61762f35085fbaeb24c1a5a8d4e654ef3a344a1c56232046a51e7e51",
            "fresh UTF8 original receipt bytes changed")
    # Fail-closed source identities across independently written evidence.
    for obj,label in ((fresh,"fresh"),(fresh_prov,"fresh provenance"),
                      (cross,"cross process"),(maxctx,"2048")):
        require(WEIGHT in json.dumps(obj,sort_keys=True),
                label+" missing exact full checkpoint SHA")
        require(TOKENIZER in json.dumps(obj,sort_keys=True),
                label+" missing exact ByteTokenizer source SHA")
        require(HOOK in json.dumps(obj,sort_keys=True),
                label+" missing exact original hook commit")

    # The original archive records this ID under source_run.run_id, not
    # workflow_run_id. Bind the actual nested field rather than inventing an
    # absent provenance key or relying on a text occurrence of the run number.
    require(fresh_prov["source_run"]["run_id"]==40464956343
            and fresh_prov["source_run"]["run_status"]=="success",
            "fresh actual-model Actions run/status changed")
    require_any_value(cross,"original_run_id",40549549886,
                      "cross-process actual-model run changed")
    require(maxctx["original_actions"]["run_id"]==40605678381
            and maxctx["original_actions"]["model_result"]=="success",
            "2048 actual-model run/result changed")

    # Fresh UTF8 original evidence: exact no-op, padding, 48-cache, causal/reset,
    # cold reload, finite manually activated adapter and zero training.
    require(fresh["status"]==
            "PASS_REVIEWED_VOID_UTF8_REAL_MODEL_CACHE_COLD_ADAPTER_PREPASS",
            "fresh receipt is not an actual source-bound prepass PASS")
    require(fresh["checks"]["nonzero_48_UTF8_full_vs_KV_cache"]["max_abs"]
            ==2.86102294921875e-05
            and fresh["checks"]["nonzero_48_UTF8_full_vs_KV_cache"]["within"] is True,
            "fresh 48-unit cache metric changed")
    require(fresh["checks"]["manual_nontrained_readout_effect"]["max_abs"]
            ==0.062297821044921875
            and fresh["checks"]["manual_nontrained_readout_effect"]["trained"] is False,
            "fresh manual-untrained effect changed")
    require(fresh["optimizer_steps"]==0
            and fresh["model_training_performed"] is False
            and fresh["comparative_training_admitted"] is False
            and fresh["full_four_model_conformance_pass"] is False,
            "fresh receipt falsely claims training/qualification")
    # Archived provenance itself explicitly preserves scope.
    require_any_value(fresh_prov,"optimizer_steps",0,
                      "fresh provenance optimizer boundary changed")

    # Require byte-exact original Actions producer/consumer receipts, not just
    # their editable source provenance summary. No Qwen3/tokenizer execution.
    sys.path.insert(0,str(ROOT/"scripts"))
    import ncr_001e_void_process_receipt_audit as process_audit
    original_process = process_audit.audit()
    require(original_process["status"] == process_audit.STATUS
            and original_process["original_actual_model_run_id"] == 40549549886
            and original_process["full_original_receipts_verified"] == 5
            and original_process["original_actual_frozen_processes"] == 2
            and original_process["actual_cached_tail_error"] == 9.5367431640625e-06
            and original_process["train_authorized"] is False
            and original_process["optimizer_steps"] == 0
            and original_process["original_model_reexecuted"] is False,
            "original process archive/runner source audit failed")

    # Separate Python process: exact 256 units / 240+16 cached replay, no leakage.
    results=cross["actual_results"]
    require(cross["original_run_id"]==40549549886
            and results["producer_max_abs_cached"]==9.5367431640625e-06
            and results["consumer_max_abs_cached"]==9.5367431640625e-06,
            "cross-process cache original metric/run changed")
    require(results["tokenizer_units"]==256
            and results["prefill_units"]==240
            and results["cached_incremental_units"]==16
            and results["producer_pid"]!=results["consumer_pid"],
            "cross-process source units or independent PIDs changed")
    require_any_value(cross,"optimizer_steps",0,
                      "cross-process optimizer boundary changed")
    require_any_value(cross,"train_authorized",False,
                      "cross-process training authorization changed")
    require_any_value(cross,"full_2048_context_proven",False,
                      "historical pre-2048 evidence limit removed")

    # New 2048 review must close exactly that inference gap.
    require(maxctx["status"]==
            "PASS_VOID1_512_1024_2048_AND_HF_GENERATE_PRETRAIN_ONLY__FINAL_72_BLOCKED"
            and maxctx["results"]["full_2048_context_proven"] is True,
            "2048 source/runtime subgate is not passed")
    require(maxctx["source"]["checkpoint_sha256"]==WEIGHT
            and maxctx["source"]["tokenizer_source_sha256"]==TOKENIZER
            and maxctx["source"]["original_hook_commit"]==HOOK
            and maxctx["source"]["max_context_units"]==2048,
            "2048 source identity changed")
    require(max(maxctx["results"]["cache_max_abs"].values())<=ATOL
            and maxctx["results"]["strict_future_suffix_2048_max_abs"]==0
            and maxctx["results"]["request_reset_max_abs"]==0
            and maxctx["results"]["max_generate_cache_on_off_abs"]<=ATOL
            and maxctx["results"]["frozen_tensor_sha_before"]==
            maxctx["results"]["frozen_tensor_sha_after"],
            "2048/cache/generate/frozen tensor evidence changed")
    require(maxctx["scope"]["train_authorized"] is False
            and maxctx["scope"]["optimizer_steps"]==0
            and maxctx["scope"]["historical_byte_eos_impact_measured"] is False
            and maxctx["scope"]["future_byte_eos_serializer_integrated"] is False
            and maxctx["scope"]["eos_non_emission_in_four_units_is_not_propensity_estimate"]
            is True,
            "2048 no-training/EOS boundary changed")

    # Re-run exact original 2048 archive verifier (still no model).
    sys.path.insert(0,str(ROOT/"scripts"))
    from ncr_001e_void_2048_archive_audit import load_originals,verify_originals
    originals,provenance=load_originals()
    verify_originals(originals,provenance)

    matrix={
      "same_base_logits":{
        "status":"PASS",
        "evidence":["40464956343","40605678381"],
        "note":"Exact frozen-vs-zero logits verified for UTF8 source, padding, 512/1024/2048 and generation."
      },
      "base_weight_unchanged_after_step":{
        "status":"BLOCKED_BY_TRAINING_POLICY","evidence":[],
        "note":"No optimizer step is authorized while train_authorized=false."
      },
      "causal_suffix_invariance":{
        "status":"PASS",
        "evidence":["40464956343","40549549886","40605678381"],
        "note":"Future suffix influence and independent request replay are exact zero."
      },
      "full_prefix_vs_incremental_cache":{
        "status":"PASS",
        "evidence":["40464956343","40549549886","40605678381"],
        "note":"48, 256/240+16, and 512/1024/2048 N-16 tails plus HF generate cache on/off remain below 5e-4."
      },
      "per_token_isolation":{
        "status":"PASS_SOURCE_RUNTIME",
        "evidence":["40549549886","40605678381"],
        "note":"Sequential one-byte cached calls reset finite three-tap field state; no global carryover observed."
      },
      "per_request_reset":{
        "status":"PASS",
        "evidence":["40464956343","40549549886","40605678381"],
        "note":"A→B→A and fresh-process replay are exact."
      },
      "nonzero_adapter_gradient_after_warmup":{
        "status":"BLOCKED_BY_TRAINING_POLICY","evidence":[],
        "note":"Manual readout interventions are deliberately untrained; no backward warmup is authorized."
      },
      "finite_state_and_loss":{
        "status":"PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY",
        "evidence":["40464956343","40549549886","40605678381"],
        "note":"Finite states/logits are proven through max context; supervised training loss is not executed."
      },
      "cold_reload_state_dict":{
        "status":"PASS_PRETRAIN",
        "evidence":["40464956343","40549549886"],
        "note":"Fresh adapter reconstruction and separate-Python-process adapter/frozen-base replay are exact."
      },
    }
    require(set(matrix)==set(protocol["conformance"]["required_checks"]),
            "adjudication matrix incomplete")
    blockers=[k for k,v in matrix.items() if not v["status"].startswith("PASS")]
    require(set(blockers)=={
        "base_weight_unchanged_after_step",
        "nonzero_adapter_gradient_after_warmup",
        "finite_state_and_loss"},
        "unexpected blocker or false learning PASS")

    return {
      "schema_version":1,
      "experiment":"NCR-001E",
      "issue":72,
      "model_key":"void1",
      "model_variant":"qwen3_byte",
      "adjudication":"BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS",
      "per_model_gate":False,
      "source_runtime_pretraining_subgate":True,
      "train_authorized":False,
      "optimizer_steps_observed":0,
      "heldout_evaluation":False,
      "source":{
        "repo_id":REPO,"revision":REV,
        "weight_file":"model.safetensors","weight_sha256":WEIGHT,
        "weight_bytes":360622744,
        "tokenizer_source_sha256":TOKENIZER,
        "ids":{"pad":256,"bos":257,"eos":258,"vocab":259},
        "original_hook_commit":HOOK,
        "taps_after_block_zero_based":[7,15,23],
        "max_context_units":2048,
      },
      "protocol":{
        "sha256":sha(PROTOCOL),"source_pins_sha256":sha(PINS),
        "cache_atol":ATOL,"cache_rtol":RTOL,
        "any_failure_blocks_model_qualification":True,
        "required_checks":protocol["conformance"]["required_checks"],
      },
      "matrix":matrix,
      "blocking_required_checks":blockers,
      "byte_tokenizer_scope":{
        "actual_reviewed_tokenizer_runtime_qualified":True,
        "mixed_utf8_roundtrips_and_padding_source_bound":True,
        "full_2048_context_proven":True,
        "future_byte_eos_serializer_integration_in_this_lane":False,
        "historical_byte_eos_effect_measured":False,
        "four_unit_eos_non_emission_is_propensity_evidence":False,
      },
      "nonblocking_scope_limits":{
        "comparative_NCA_MLP_training_performed":False,
        "sealed_holdouts_opened":False,
        "trained_adapter_created":False,
        "all_parent_prs_merged":False,
      },
      "evidence_runs":[40464956343,40549549886,40605678381],
      "tokenizer_source_runtime_runs":[40462933022,40463929003],
      "decision_note":(
        "Do not set void1_conformance_pass=true and do not authorize training. "
        "Exact ByteTokenizer/Qwen3 source/runtime including full 2048 context is "
        "well-supported; final #72 remains blocked by required learning/update/loss "
        "checks that cannot run while train_authorized=false. BYTE-EOS future "
        "serializer integration is a separate future-training dependency."
      ),
    }


def main():
    report=build_report()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                   encoding="utf-8")
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
