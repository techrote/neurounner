#!/usr/bin/env python3
"""Independent standard-library replay for DETECTIVE-002D evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.detective_002b_runtime import marker_observations  # noqa: E402

CONFIG=ROOT/"experiments/detective-002/lfm-v5-api-runtime.json"
SOURCE=ROOT/"experiments/detective-002/source-evidence/source-evidence.json"
BATTERY=ROOT/"experiments/detective-002/runtime-smoke.json"
PRIOR_CONFIG=ROOT/"experiments/detective-002/lfm-v5-runtime.json"
PRIOR_RESULT=ROOT/"datasets/detective-002c/result.json"


class VerifyError(ValueError): pass


def need(v:bool,msg:str)->None:
    if not v: raise VerifyError(msg)


def read(path:Path)->Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()


def blob(path:Path)->str:
    data=path.read_bytes();h=hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"));h.update(data)
    return h.hexdigest()


def verify(root:Path)->dict:
    cfg=read(CONFIG);source=read(SOURCE);battery=read(BATTERY)
    prior=read(PRIOR_RESULT)
    expected={
        "README.md","manifest.json","result.json",
        "provenance/source-artifacts.json","provenance/hashes.sha256"
    }
    present={p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()}
    need(present==expected,"002D archive file set mismatch")
    recorded=(root/"provenance/hashes.sha256").read_text().splitlines()
    actual=[
        f"{sha(root/rel)}  {rel}"
        for rel in sorted(expected-{"provenance/hashes.sha256"})
    ]
    need(recorded==actual,"002D archive hash inventory mismatch")

    auth=cfg["source_authority"]
    need(blob(SOURCE)==auth["stage_a_git_blob_sha1"],"002D Stage-A blob drift")
    need(blob(BATTERY)==auth["battery_git_blob_sha1"],"002D battery blob drift")
    need(blob(PRIOR_CONFIG)==auth["prior_config_git_blob_sha1"],
         "002D prior-config blob drift")
    need(blob(PRIOR_RESULT)==auth["prior_result_git_blob_sha1"],
         "002D prior-result blob drift")

    candidate=next(x for x in source["candidates"]
                   if x["candidate_id"]=="lfm25-350m")
    model=cfg["model"]
    need(
        candidate["repo"]==model["repo"]
        and candidate["resolved_revision"]==model["revision"]
        and candidate["weight"]["lfs_sha256"]==model["weight_sha256"]
        and candidate["weight"]["size"]==model["weight_bytes"]
        and candidate["tokenizer_config"]["sha256"]
            ==model["tokenizer_config_sha256"]
        and candidate["chat_template_file"]["sha256"]
            ==model["chat_template_sha256"]
        and candidate["config"]["sha256"]==model["config_sha256"]
        and candidate["production_license_gate"] is True,
        "002D model source authority mismatch")

    token=prior["tokenizer_stage"]
    need(prior["experiment"]=="DETECTIVE-002C"
         and prior["status"]=="tokenizer_incompatible"
         and prior["resolved_packages"]["transformers"]=="5.20.1"
         and token["tokenizer_class"]=="TokenizersBackend"
         and token["tokenizer_backend"]=="tokenizers"
         and token["tokenizer_error"]["type"]=="AttributeError"
         and token["tokenizer_error"]["message"]==""
         and token["source"]["weight_downloaded"] is False
         and prior["model_stage"] is None,
         "002D frozen 002C control mismatch")

    result=read(root/"result.json")
    manifest=read(root/"manifest.json")
    prov=read(root/"provenance/source-artifacts.json")
    need(result.get("schema")=="detective-002d-lfm-api-result-v1"
         and result.get("experiment")=="DETECTIVE-002D"
         and result.get("issue")==135,
         "002D result identity mismatch")
    need(result.get("status") in cfg["decisions"]["allowed"],
         "002D unapproved outcome")
    need(result.get("config_sha256")==sha(CONFIG)
         and result.get("source_evidence_sha256")==sha(SOURCE)
         and result.get("battery_sha256")==sha(BATTERY)
         and result.get("prior_config_sha256")==sha(PRIOR_CONFIG)
         and result.get("prior_result_sha256")==sha(PRIOR_RESULT),
         "002D source digest mismatch")
    need(result.get("model")==model
         and result.get("runtime_contract")==cfg["runtime"],
         "002D model/runtime contract mismatch")
    need(result.get("optimizer_steps")==0
         and result.get("paid_api_calls")==0
         and result.get("relevance_mapping_accessed") is False
         and result.get("production_license_cleared") is False
         and result.get("training_selection_allowed") is False
         and result.get("population_expansion_admitted") is False,
         "002D authority widened")
    packages=result.get("resolved_packages") or {}
    need(packages.get("transformers")=="5.20.1"
         and packages.get("torch")=="2.14.1+cpu"
         and packages.get("python","").startswith("3.12"),
         "002D resolved package mismatch")

    d=result.get("diagnostic_stage") or {}
    need(d.get("status") in {
        "tokenizer_api_pass",
        "template_render_incompatible",
        "tokenization_api_incompatible"
    },"002D diagnostic status invalid")
    source_stage=d.get("source") or {}
    need(source_stage.get("config_sha256")==model["config_sha256"]
         and source_stage.get("tokenizer_config_sha256")
            ==model["tokenizer_config_sha256"]
         and source_stage.get("chat_template_sha256")
            ==model["chat_template_sha256"],
         "002D small-source identity mismatch")

    if result["status"] in {
        "template_render_incompatible","tokenization_api_incompatible"
    }:
        need(result["status"]==d["status"],
             "002D top-level/diagnostic mismatch")
        need(source_stage.get("weight_downloaded") is False,
             "002D tokenizer diagnostic failure downloaded weight")
        need("weight_stage" not in result and result.get("model_stage") is None,
             "002D diagnostic failure entered model stage")
        need(isinstance(d.get("error"),dict),
             "002D diagnostic failure missing error record")
    else:
        need(d["status"]=="tokenizer_api_pass",
             "002D model stage lacks diagnostic PASS")
        need(d.get("tokenizer_class")=="TokenizersBackend"
             and d.get("tokenizer_backend")=="tokenizers",
             "002D tokenizer class/backend mismatch")
        prompts=d.get("prompts");encoded=d.get("tokenized_prompts")
        need(isinstance(prompts,list) and len(prompts)==4
             and isinstance(encoded,list) and len(encoded)==4,
             "002D diagnostic prompt sets incomplete")
        for p,e,task_cfg in zip(prompts,encoded,battery["battery"]):
            need(p["task_id"]==task_cfg["id"]==e["task_id"],
                 "002D prompt task order mismatch")
            need(isinstance(p["formatted_text"],str) and p["formatted_text"],
                 "002D formatted text missing")
            need(hashlib.sha256(
                p["formatted_text"].encode("utf-8")).hexdigest()
                ==p["formatted_text_sha256"]
                ==e["formatted_text_sha256"],
                "002D formatted text digest mismatch")
            need("input_ids" in e["mapping_keys"]
                 and 0<e["input_tokens"]<=512
                 and re.fullmatch(r"[0-9a-f]{64},?",e["input_ids_sha256"]+",")
                    is not None,
                 "002D tokenized mapping/input identity invalid")

        w=result.get("weight_stage") or {}
        need(source_stage.get("weight_downloaded") is True
             and w.get("selected_weight_path")==model["weight_path"]
             and w.get("selected_weight_sha256")==model["weight_sha256"]
             and w.get("selected_weight_bytes")==model["weight_bytes"]
             and w.get("staged_weight_sha256")==model["weight_sha256"],
             "002D model-stage weight identity mismatch")
        ms=result.get("model_stage") or {}
        if result["status"]=="model_runtime_incompatible":
            need(ms.get("status")=="model_runtime_incompatible"
                 and isinstance(ms.get("error"),dict),
                 "002D model incompatibility missing error")
        else:
            need(result["status"]=="runtime_compatibility_pass"
                 and ms.get("status")=="runtime_compatibility_pass",
                 "002D PASS classification mismatch")
            env=ms.get("environment") or {}
            need(env.get("transformers")=="5.20.1"
                 and env.get("torch_threads")==4
                 and env.get("cuda_available") is False,
                 "002D PASS environment mismatch")
            tasks=ms.get("tasks")
            need(isinstance(tasks,list) and len(tasks)==4,
                 "002D PASS task count mismatch")
            total=0
            for task_cfg,p,e,t in zip(
                battery["battery"],prompts,encoded,tasks):
                need(t["task_id"]==task_cfg["id"]
                     and t["messages"]==p["messages"]
                     and t["formatted_text_sha256"]
                        ==p["formatted_text_sha256"]
                     and t["prompt_token_count"]==e["input_tokens"]
                     and t["prompt_token_sha256"]==e["input_ids_sha256"],
                     "002D model prompt differs from diagnostic gate")
                need(t["generated_tokens"]==len(t["generated_token_ids"])
                     and 0<=t["generated_tokens"]<=64,
                     "002D generated token record invalid")
                need(bool(t["hit_max_new_tokens"])
                     ==(t["generated_tokens"]==64),
                     "002D truncation marker mismatch")
                need(t["observations"]
                     ==marker_observations(task_cfg,t["response"]),
                     "002D mechanical observations do not replay")
                total+=t["generated_tokens"]
            need(total==ms.get("total_generated_tokens") and total<=256,
                 "002D decode budget mismatch")

    need(manifest.get("schema")=="detective-002d-runtime-archive-v1"
         and manifest.get("status")==result["status"]
         and manifest.get("result_sha256")==sha(root/"result.json")
         and manifest.get("config_sha256")==sha(CONFIG)
         and manifest.get("prior_result_sha256")==sha(PRIOR_RESULT),
         "002D manifest mismatch")
    need(re.fullmatch(r"[0-9a-f]{40}",manifest.get("source_head",""))
         and str(manifest.get("workflow_run_id","")).isdigit(),
         "002D run/source identity missing")
    need(prov.get("source_head")==manifest["source_head"]
         and prov.get("workflow_run_id")==manifest["workflow_run_id"]
         and prov.get("stage_a_git_blob_sha1")==auth["stage_a_git_blob_sha1"]
         and prov.get("battery_git_blob_sha1")==auth["battery_git_blob_sha1"]
         and prov.get("prior_config_git_blob_sha1")
            ==auth["prior_config_git_blob_sha1"]
         and prov.get("prior_result_git_blob_sha1")
            ==auth["prior_result_git_blob_sha1"]
         and prov.get("transformers")=="5.20.1"
         and prov.get("transformers_wheel_sha256")
            ==cfg["runtime"]["transformers_wheel_sha256"]
         and prov.get("training_steps")==0
         and prov.get("external_provider_calls")==0,
         "002D provenance mismatch")
    return {
        "verdict":"DETECTIVE_002D_SOURCE_READBACK_PASS",
        "status":result["status"],
        "prior_002c_status":"tokenizer_incompatible",
        "diagnostic_status":d["status"],
        "weight_downloaded":source_stage.get("weight_downloaded"),
        "optimizer_steps":0,
        "production_license_cleared":False,
        "training_selection_allowed":False,
        "population_expansion_admitted":False
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(verify(args.evidence),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
