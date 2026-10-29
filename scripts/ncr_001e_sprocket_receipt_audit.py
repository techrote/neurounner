#!/usr/bin/env python3
"""Verify three exact original NCR-001E Sprocket source/cache receipts archived in Git.

The .json.gz.b64 files decode losslessly to the original Actions JSON bytes.
No model, tokenizer or network is used in this independent custody check.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"experiments/ncr-001/evidence-review/sprocket-fresh"
PROVENANCE=EVIDENCE/"provenance.json"
PROTOCOL=ROOT/"experiments/ncr-001/protocol.json"
PINS=ROOT/"experiments/ncr-001/source-pins.json"


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def require(ok:bool,label:str)->None:
    if not ok:raise RuntimeError(label)


def original_receipt(meta:dict)->dict:
    path=ROOT/meta["archive"]
    data=path.read_text(encoding="utf-8").strip()
    gz=base64.b64decode(data,validate=True)
    require(sha(gz)==meta["gzip_sha256"],"Original lossless gzip hash mismatch")
    raw=gzip.decompress(gz)
    require(len(raw)==meta["original_raw_size_bytes"],"Original raw file byte length mismatch")
    require(sha(raw)==meta["original_raw_sha256"],"Original raw file SHA256 mismatch")
    return json.loads(raw)


def audit()->dict:
    prov=json.loads(PROVENANCE.read_text())
    p=json.loads(PROTOCOL.read_text())
    pins=json.loads(PINS.read_text())
    require(p["train_authorized"] is False,"Source protocol must forbid training")
    require(set(prov["original_sources"])==
            {"pinned_role_inventory","chat_fresh","instruct_fresh"},
            "Original triple source custody incomplete")
    original={k:original_receipt(v) for k,v in prov["original_sources"].items()}
    inventory=original["pinned_role_inventory"]
    require(inventory["status"]==
            "PASS_PINNED_SPROCKET_ROLE_FORMAT_AND_TOKEN_INVENTORY__BEHAVIOR_BLOCKED",
            "Source role inventory status mismatch")
    require(inventory["model_card"]["sha256"]==
            prov["source_identity"]["model_card_sha256"],
            "Pinned README SHA mismatch")
    require(inventory["training_performed"] is False and
            inventory["model_weights_downloaded"] is False,
            "Source-only role auditor improperly loaded/trained model")
    require(inventory["tool_and_memory_behavior_qualified"] is False,
            "Source-only control-token presence falsely reported as behavior")
    for variant,src in (("chat","chat_fresh"),("instruct","instruct_fresh")):
        model=original[src]
        cfg=p["models"]["sprocket_"+variant]
        pin=pins["models"]["sprocket_"+variant]
        expectedsha=prov["source_identity"][variant+"_actual_weights_sha256"]
        require(model["status"]==
                "PASS_FRESH_SPROCKET_"+variant.upper()+"_SOURCE_ROLES_CACHE_COLD_PREPASS",
                f"{variant} true frozen-host source status mismatch")
        require(model["variant"]==variant and
                model["model_revision"]==cfg["revision"]==pin["revision"] and
                pin["variant"]==variant,
                f"{variant} source checkpoint variant/revision identity mismatch")
        require(model["source_files"][cfg["weight_file"]]["sha256"]==
                expectedsha==pin["weight_declared_lfs_sha256_not_downloaded"],
                f"{variant} independent full weight SHA mismatch")
        require(model["source_files"][cfg["weight_file"]]["bytes"]==
                pin["weight_bytes"],
                f"{variant} actual model byte length mismatch")
        require(model["source_files"]["README.md"]["sha256"]==
                inventory["model_card"]["sha256"],
                "Role model card not same exact upstream README")
        require(model["format_source"]["model_card_sha256"]==
                inventory["model_card"]["sha256"] and
                model["format_source"]["source_verified_dialogue"] is True,
                "Pinned ordinary-role formatting authority mismatch")
        require(model["format_source"]["tool_and_memory_behavior_qualified"] is False,
                "Source-only control tokens falsely qualified as a tool protocol")
        require(model["actual_model_training"] is False and
                model["optimizer_steps"]==0 and
                model["full_four_model_conformance_pass"] is False,
                "Untrained frozen source falsely labelled a learner result")
        require(model["nca_parameters"]==658904 and
                model["frozen_base_parameters"]==501090560,
                f"{variant} parameters changed")
        for k in ("zero_init_16_source_role_prefix",
                  "zero_init_64_source_role_prefix",
                  "zero_init_128_source_role_prefix",
                  "zero_init_left_padded_source_roles",
                  "zero_init_right_padded_source_roles"):
            require(model["checks"][k]["close"] is True and
                    model["checks"][k]["max_abs"]==0,
                    f"{variant} initial no-op parity failed: {k}")
        require(model["checks"]["zero_init_greedy_generation"]["identical"] is True,
                f"{variant} zero-init generation changed")
        for k in ("nonzero_source_role_future_suffix",
                  "cold_adapter_new_instance_same_frozen_base"):
            require(model["checks"][k]["close"] is True and
                    model["checks"][k]["max_abs"]==0,
                    f"{variant} causal/cold adapter identity violation: {k}")
        require(model["checks"]["request_A_B_A_no_field_leak"]["exact"] is True,
                f"{variant} cross-request cellular state leak")
        for k in ("model_variant","checkpoint_sha256","model_revision"):
            require(model["checks"]["cold_"+k+"_source_refusal"]["rejected"] is True,
                    f"{variant} model source refusal not proved: {k}")
        cache=model["checks"]["nonzero_role_48_full_vs_incremental_cache"]
        require(cache["close"] is True and
                cache["max_abs"]<=p["conformance"]["full_vs_cached_prefill_atol"],
                f"{variant} dynamic cached forward out of preregistered tolerance")
        require(model["checks"]["manual_nontrained_readout_effect"]["max_abs"]>0
                and model["checks"]["manual_nontrained_readout_effect"]["trained"] is False,
                f"{variant} manual nontrained perturbation not demonstrated")
    require(prov["source_identity"]["chat_actual_weights_sha256"] !=
            prov["source_identity"]["instruct_actual_weights_sha256"],
            "Root Instruct and Chat are not interchangeable weights")
    return {
        "status":"PASS_ORIGINAL_TRIPLE_SPROCKET_RECEIPT_BYTE_HASH_AND_NUMERIC_READBACK",
        "role_inventory_sha256":prov["original_sources"]["pinned_role_inventory"]["original_raw_sha256"],
        "variant_raw_sha256":{
            variant:prov["original_sources"][variant+"_fresh"]["original_raw_sha256"]
            for variant in ("chat","instruct")},
        "source_run_id":prov["source_run"]["run_id"],
        "tool_and_memory_behavior_qualified":False,
        "fresh_separate_process_cold_model_reload_proved":False,
        "full_four_model_conformance_pass":False,
        "training_authorized":False,
    }


if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
