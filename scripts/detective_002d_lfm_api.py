#!/usr/bin/env python3
"""DETECTIVE-002D explicit Transformers-v5 chat-template API requalification."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.detective_002b_runtime import (  # noqa: E402
    cpu_model,current_rss_kib,marker_observations
)
from scripts.detective_002c_lfm_runtime import (  # noqa: E402
    SourceIntegrityError,ContractError,small_snapshot,download_weight
)

CONFIG=ROOT/"experiments/detective-002/lfm-v5-api-runtime.json"
SOURCE=ROOT/"experiments/detective-002/source-evidence/source-evidence.json"
BATTERY=ROOT/"experiments/detective-002/runtime-smoke.json"
PRIOR_CONFIG=ROOT/"experiments/detective-002/lfm-v5-runtime.json"
PRIOR_RESULT=ROOT/"datasets/detective-002c/result.json"


def read_json(path:Path)->Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,
                               ensure_ascii=False)+"\n",encoding="utf-8")


def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path:Path)->str:
    data=path.read_bytes();h=hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"));h.update(data)
    return h.hexdigest()


def load_contract()->tuple[dict,dict,dict]:
    cfg=read_json(CONFIG)
    source=read_json(SOURCE)
    battery=read_json(BATTERY)
    prior_cfg=read_json(PRIOR_CONFIG)
    prior=read_json(PRIOR_RESULT)

    if (cfg.get("schema"),cfg.get("experiment"),cfg.get("issue")) != (
        "detective-002d-lfm-v5-api-v1","DETECTIVE-002D",135):
        raise ContractError("002D identity drift")

    auth=cfg["source_authority"]
    for path,key in (
        (SOURCE,"stage_a_git_blob_sha1"),
        (BATTERY,"battery_git_blob_sha1"),
        (PRIOR_CONFIG,"prior_config_git_blob_sha1"),
        (PRIOR_RESULT,"prior_result_git_blob_sha1"),
    ):
        if git_blob_sha1(path)!=auth[key]:
            raise SourceIntegrityError(
                f"002D frozen source blob drift: {path.relative_to(ROOT)}")
    if auth["prior_merge_commit"]!="13bf4ce95772dfc5399b8767cdb8643962a71a66":
        raise ContractError("002C merge authority drift")

    candidate=next(
        (x for x in source["candidates"]
         if x.get("candidate_id")=="lfm25-350m"),None)
    if candidate is None:
        raise SourceIntegrityError("LFM source record missing")

    model=cfg["model"]
    if (
        candidate["repo"]!=model["repo"]
        or candidate["resolved_revision"]!=model["revision"]
        or candidate["weight"]["path"]!=model["weight_path"]
        or candidate["weight"]["lfs_sha256"]!=model["weight_sha256"]
        or candidate["weight"]["size"]!=model["weight_bytes"]
        or candidate["tokenizer_config"]["sha256"]
            !=model["tokenizer_config_sha256"]
        or candidate["chat_template_file"]["sha256"]
            !=model["chat_template_sha256"]
        or candidate["config"]["sha256"]!=model["config_sha256"]
        or candidate["license"]!=model["license"]
        or candidate["production_license_gate"] is not True
    ):
        raise SourceIntegrityError("002D Stage-A LFM identity drift")

    # 002D intentionally keeps the exact 002C runtime and changes API handling.
    if (
        prior_cfg["runtime"]["transformers"]!=cfg["runtime"]["transformers"]
        or prior_cfg["runtime"]["transformers_wheel_sha256"]
            !=cfg["runtime"]["transformers_wheel_sha256"]
        or prior_cfg["runtime"]["torch_cpu"]!=cfg["runtime"]["torch_cpu"]
        or prior_cfg["runtime"]["threads"]!=cfg["runtime"]["threads"]
        or prior_cfg["runtime"]["seed"]!=cfg["runtime"]["seed"]
        or prior_cfg["model"]["weight_sha256"]!=model["weight_sha256"]
    ):
        raise SourceIntegrityError("002D changed more than chat-template API handling")

    expected_prior=cfg["prior_result"]
    token=prior.get("tokenizer_stage") or {}
    error=token.get("tokenizer_error") or {}
    if (
        prior.get("experiment")!=expected_prior["experiment"]
        or prior.get("status")!=expected_prior["status"]
        or prior.get("resolved_packages",{}).get("transformers")
            !=expected_prior["transformers"]
        or token.get("tokenizer_class")!=expected_prior["tokenizer_class"]
        or token.get("tokenizer_backend")!=expected_prior["tokenizer_backend"]
        or error.get("type")!=expected_prior["error_type"]
        or error.get("message")!=expected_prior["error_message"]
        or token.get("source",{}).get("weight_downloaded")
            is not expected_prior["weight_downloaded"]
        or prior.get("model_stage") is not None
    ):
        raise SourceIntegrityError("002D prior 002C failure identity drift")

    if battery.get("experiment")!="DETECTIVE-002B" or len(battery["battery"])!=4:
        raise SourceIntegrityError("002D battery authority drift")
    if cfg["runtime"]!={
        "python":"3.12","torch_cpu":"2.14.1+cpu",
        "transformers":"5.20.1",
        "transformers_wheel_sha256":
            "5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19",
        "threads":4,"seed":260213,"do_sample":False,
        "max_prompt_tokens":512,"max_new_tokens_per_prompt":64,
        "max_generated_tokens_total":256,"max_minutes":25,
        "optimizer_steps":0,"paid_api_calls":0,
        "gpu_required":False,"dtype":"float32"
    }:
        raise ContractError("002D runtime drift")
    if cfg["diagnostic"]!={
        "text_render":{
            "tokenize":False,"add_generation_prompt":True,
            "return_dict":False
        },
        "tokenized_render":{
            "tokenize":True,"add_generation_prompt":True,
            "return_dict":True,"return_tensors":"pt"
        },
        "require_mapping_keys":["input_ids"],
        "compare_text_sha256":True,
        "compare_input_id_sha256":True,
        "download_weight_only_after_both_render_paths_pass":True
    }:
        raise ContractError("002D diagnostic contract drift")
    if cfg["decisions"]!={
        "allowed":[
            "runtime_compatibility_pass",
            "template_render_incompatible",
            "tokenization_api_incompatible",
            "model_runtime_incompatible"
        ],
        "source_integrity_fail_is_hard_error":True,
        "production_license_cleared":False,
        "training_selection_allowed":False,
        "population_expansion_admitted":False,
        "relevance_mapping_accessed":False
    }:
        raise ContractError("002D authority drift")
    return cfg,candidate,battery


def tokenizer_diagnostic(cfg:dict,candidate:dict,battery:dict,stage:Path)->dict:
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__!="5.20.1":
        raise SourceIntegrityError("002D Transformers runtime is not 5.20.1")
    source=small_snapshot(candidate,stage)
    started=time.monotonic()
    try:
        tok=AutoTokenizer.from_pretrained(
            stage,local_files_only=True,trust_remote_code=False)
    except Exception as exc:
        # Tokenizer construction already passed in frozen 002C. A new failure
        # is preserved as the first diagnostic stage, not hidden.
        return {
            "status":"template_render_incompatible",
            "source":source,
            "transformers":transformers.__version__,
            "tokenizer_load_seconds":time.monotonic()-started,
            "error":{
                "phase":"tokenizer_construction",
                "type":type(exc).__name__,
                "message":str(exc),
                "repr":repr(exc)
            },
            "prompts":[]
        }
    load_seconds=time.monotonic()-started
    class_name=type(tok).__name__
    tc=read_json(stage/"tokenizer_config.json")
    if class_name!="TokenizersBackend":
        raise SourceIntegrityError(
            f"002D expected TokenizersBackend, got {class_name}")
    if tc.get("backend")!="tokenizers" or tc.get("tokenizer_class")!="TokenizersBackend":
        raise SourceIntegrityError("002D frozen tokenizer declaration drift")

    prompt_records=[]
    tensor_records=[]
    for task in battery["battery"]:
        messages=[
            {"role":"system","content":battery["system"]},
            {"role":"user","content":task["user"]}
        ]
        try:
            formatted=tok.apply_chat_template(
                messages,tokenize=False,add_generation_prompt=True,
                return_dict=False)
            if not isinstance(formatted,str) or not formatted:
                raise TypeError(
                    f"expected nonempty str, got {type(formatted).__name__}")
        except Exception as exc:
            return {
                "status":"template_render_incompatible",
                "source":source,
                "transformers":transformers.__version__,
                "tokenizer_class":class_name,
                "tokenizer_backend":"tokenizers",
                "tokenizer_load_seconds":load_seconds,
                "error":{
                    "phase":"text_render",
                    "task_id":task["id"],
                    "type":type(exc).__name__,
                    "message":str(exc),
                    "repr":repr(exc)
                },
                "prompts":prompt_records
            }
        text_sha=hashlib.sha256(formatted.encode("utf-8")).hexdigest()
        prompt_records.append({
            "task_id":task["id"],"family":task["family"],
            "messages":messages,
            "formatted_text":formatted,
            "formatted_text_sha256":text_sha
        })

        try:
            encoded=tok.apply_chat_template(
                messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors="pt")
            if not isinstance(encoded,Mapping):
                raise TypeError(
                    f"expected mapping, got {type(encoded).__name__}")
            if "input_ids" not in encoded:
                raise KeyError("input_ids")
            ids=encoded["input_ids"]
            count=int(ids.shape[-1])
            if count<=0 or count>cfg["runtime"]["max_prompt_tokens"]:
                raise ContractError("002D prompt token budget violated")
        except (SourceIntegrityError,ContractError):
            raise
        except Exception as exc:
            return {
                "status":"tokenization_api_incompatible",
                "source":source,
                "transformers":transformers.__version__,
                "tokenizer_class":class_name,
                "tokenizer_backend":"tokenizers",
                "tokenizer_load_seconds":load_seconds,
                "error":{
                    "phase":"tokenized_render",
                    "task_id":task["id"],
                    "type":type(exc).__name__,
                    "message":str(exc),
                    "repr":repr(exc)
                },
                "prompts":prompt_records,
                "tokenized_prompts":[
                    {k:v for k,v in x.items() if k!="_encoded"}
                    for x in tensor_records
                ]
            }
        serial={
            "task_id":task["id"],"family":task["family"],
            "mapping_type":type(encoded).__name__,
            "mapping_keys":sorted(str(k) for k in encoded.keys()),
            "input_tokens":count,
            "input_ids_sha256":hashlib.sha256(
                ids.numpy().tobytes()).hexdigest(),
            "formatted_text_sha256":text_sha,
            "_encoded":encoded
        }
        tensor_records.append(serial)

    return {
        "status":"tokenizer_api_pass",
        "source":source,
        "transformers":transformers.__version__,
        "tokenizer_class":class_name,
        "tokenizer_backend":"tokenizers",
        "tokenizer_load_seconds":load_seconds,
        "prompts":prompt_records,
        "tokenized_prompts":[
            {k:v for k,v in x.items() if k!="_encoded"}
            for x in tensor_records
        ],
        "_tokenizer":tok,
        "_tensors":tensor_records
    }


def run(out:Path)->dict:
    cfg,candidate,battery=load_contract()
    import torch,transformers,huggingface_hub,tokenizers,safetensors
    from transformers import AutoModelForCausalLM

    if transformers.__version__!="5.20.1":
        raise SourceIntegrityError("002D exact Transformers version mismatch")
    if torch.cuda.is_available():
        raise ContractError("002D is CPU-only")
    torch.set_num_threads(4)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.manual_seed(cfg["runtime"]["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

    result={
        "schema":"detective-002d-lfm-api-result-v1",
        "experiment":"DETECTIVE-002D","issue":135,
        "status":"RUNNING",
        "config_sha256":file_sha(CONFIG),
        "source_evidence_sha256":file_sha(SOURCE),
        "battery_sha256":file_sha(BATTERY),
        "prior_config_sha256":file_sha(PRIOR_CONFIG),
        "prior_result_sha256":file_sha(PRIOR_RESULT),
        "model":cfg["model"],"runtime_contract":cfg["runtime"],
        "resolved_packages":{
            "python":platform.python_version(),
            "torch":torch.__version__,
            "transformers":transformers.__version__,
            "huggingface_hub":huggingface_hub.__version__,
            "tokenizers":tokenizers.__version__,
            "safetensors":safetensors.__version__
        },
        "optimizer_steps":0,"paid_api_calls":0,
        "relevance_mapping_accessed":False,
        "production_license_cleared":False,
        "training_selection_allowed":False,
        "population_expansion_admitted":False,
        "diagnostic_stage":None,"model_stage":None
    }
    write_json(out,result)
    wall=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="detective002d-lfm-") as td:
        stage=Path(td)/"model"
        diag=tokenizer_diagnostic(cfg,candidate,battery,stage)
        result["diagnostic_stage"]={
            k:v for k,v in diag.items() if not k.startswith("_")}
        write_json(out,result)
        if diag["status"]!="tokenizer_api_pass":
            result["status"]=diag["status"]
            result["wall_seconds"]=time.monotonic()-wall
            result["peak_rss_kib"]=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write_json(out,result)
            return result

        # Both text and dict render paths passed. Only now is model weight eligible.
        weight=download_weight(candidate,stage)
        result["diagnostic_stage"]["source"]["weight_downloaded"]=True
        result["weight_stage"]=weight
        write_json(out,result)

        tok=diag["_tokenizer"]
        tensors=diag["_tensors"]
        try:
            start=time.monotonic()
            model=AutoModelForCausalLM.from_pretrained(
                stage,local_files_only=True,trust_remote_code=False,
                dtype=torch.float32)
            model.config.use_cache=True;model.eval()
            env={
                "platform":platform.platform(),
                "python":platform.python_version(),
                "cpu":cpu_model(),"cpu_count":os.cpu_count(),
                "torch":torch.__version__,
                "transformers":transformers.__version__,
                "torch_threads":torch.get_num_threads(),
                "torch_interop_threads":torch.get_num_interop_threads(),
                "cuda_available":torch.cuda.is_available(),
                "model_load_seconds":time.monotonic()-start,
                "rss_after_load_kib":current_rss_kib(),
                "peak_rss_after_load_kib":int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            }
            result["model_stage"]={
                "status":"RUNNING","environment":env,"tasks":[]}
            write_json(out,result)
            total=0
            for task_cfg,text_rec,tensor_rec in zip(
                battery["battery"],diag["prompts"],tensors):
                encoded=tensor_rec["_encoded"]
                input_ids=encoded["input_ids"]
                prefill_start=time.monotonic()
                with torch.inference_mode():
                    pre=model(**encoded,use_cache=False)
                prefill=time.monotonic()-prefill_start
                del pre
                gen_start=time.monotonic()
                with torch.inference_mode():
                    gen=model.generate(
                        **encoded,
                        max_new_tokens=cfg["runtime"]["max_new_tokens_per_prompt"],
                        do_sample=False,use_cache=True,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                        return_dict_in_generate=True)
                elapsed=time.monotonic()-gen_start
                tail=gen.sequences[0,input_ids.shape[-1]:].tolist()
                total+=len(tail)
                if total>cfg["runtime"]["max_generated_tokens_total"]:
                    raise ContractError("002D decode budget exceeded")
                response=tok.decode(tail,skip_special_tokens=True)
                raw=tok.decode(tail,skip_special_tokens=False)
                result["model_stage"]["tasks"].append({
                    "task_id":task_cfg["id"],"family":task_cfg["family"],
                    "messages":text_rec["messages"],
                    "formatted_text_sha256":
                        text_rec["formatted_text_sha256"],
                    "prompt_token_count":tensor_rec["input_tokens"],
                    "prompt_token_sha256":tensor_rec["input_ids_sha256"],
                    "mapping_keys":tensor_rec["mapping_keys"],
                    "prefill_seconds":prefill,
                    "prompt_tokens_per_second":
                        tensor_rec["input_tokens"]/prefill if prefill else None,
                    "generation_seconds":elapsed,
                    "generated_token_ids":tail,
                    "generated_tokens":len(tail),
                    "completion_tokens_per_second":
                        len(tail)/elapsed if elapsed else None,
                    "hit_max_new_tokens":
                        len(tail)==cfg["runtime"]["max_new_tokens_per_prompt"],
                    "response":response,
                    "response_with_special_tokens":raw,
                    "observations":marker_observations(task_cfg,response)
                })
                write_json(out,result)
            if len(result["model_stage"]["tasks"])!=4:
                raise ContractError("002D incomplete model battery")
            result["model_stage"]["status"]="runtime_compatibility_pass"
            result["model_stage"]["total_generated_tokens"]=total
            result["status"]="runtime_compatibility_pass"
            result["wall_seconds"]=time.monotonic()-wall
            result["peak_rss_kib"]=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write_json(out,result)
            del model;gc.collect()
            return result
        except (SourceIntegrityError,ContractError):
            raise
        except Exception as exc:
            result["model_stage"]=result["model_stage"] or {"tasks":[]}
            result["model_stage"]["status"]="model_runtime_incompatible"
            result["model_stage"]["error"]={
                "type":type(exc).__name__,"message":str(exc),"repr":repr(exc)}
            result["status"]="model_runtime_incompatible"
            result["wall_seconds"]=time.monotonic()-wall
            result["peak_rss_kib"]=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write_json(out,result)
            return result


def archive(result_path:Path,out:Path,source_head:str,
            run_id:str,run_url:str)->dict:
    cfg,_,_=load_contract()
    result=read_json(result_path)
    if result.get("schema")!="detective-002d-lfm-api-result-v1":
        raise ContractError("002D result schema mismatch")
    if result.get("status") not in cfg["decisions"]["allowed"]:
        raise ContractError("002D result status not admitted")
    if out.exists(): shutil.rmtree(out)
    (out/"provenance").mkdir(parents=True)
    shutil.copy2(result_path,out/"result.json")
    manifest={
        "schema":"detective-002d-runtime-archive-v1",
        "experiment":"DETECTIVE-002D","issue":135,
        "source_head":source_head,
        "workflow_run_id":str(run_id),"workflow_run_url":run_url,
        "status":result["status"],
        "config_sha256":file_sha(CONFIG),
        "source_evidence_sha256":file_sha(SOURCE),
        "battery_sha256":file_sha(BATTERY),
        "prior_config_sha256":file_sha(PRIOR_CONFIG),
        "prior_result_sha256":file_sha(PRIOR_RESULT),
        "result_sha256":file_sha(out/"result.json"),
        "optimizer_steps":0,"paid_api_calls":0,
        "relevance_mapping_accessed":False,
        "production_license_cleared":False,
        "training_selection_allowed":False,
        "population_expansion_admitted":False
    }
    write_json(out/"manifest.json",manifest)
    write_json(out/"provenance/source-artifacts.json",{
        "source_head":source_head,"workflow_run_id":str(run_id),
        "workflow_run_url":run_url,
        "stage_a_git_blob_sha1":
            cfg["source_authority"]["stage_a_git_blob_sha1"],
        "battery_git_blob_sha1":
            cfg["source_authority"]["battery_git_blob_sha1"],
        "prior_config_git_blob_sha1":
            cfg["source_authority"]["prior_config_git_blob_sha1"],
        "prior_result_git_blob_sha1":
            cfg["source_authority"]["prior_result_git_blob_sha1"],
        "transformers":"5.20.1",
        "transformers_wheel_sha256":
            cfg["runtime"]["transformers_wheel_sha256"],
        "model_weight_sha256":cfg["model"]["weight_sha256"],
        "training_steps":0,"external_provider_calls":0,
        "relevance_mapping_accessed":False
    })
    (out/"README.md").write_text(
        "# DETECTIVE-002D LFM v5 chat-template API requalification\n\n"
        "Focused follow-up to DETECTIVE-002C. Same exact LFM2.5-350M source, "
        "same exact Transformers 5.20.1 wheel and same B01-B04 prompts. The "
        "only intended change is explicit v5 chat-template return handling: "
        "text render with return_dict=False, then tokenized mapping with "
        "return_dict=True and input_ids access. Model weight is downloaded "
        "only after both paths pass. No training/selection/license/admission.\n",
        encoding="utf-8")
    inv=[]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        rel=p.relative_to(out).as_posix()
        if rel!="provenance/hashes.sha256":
            inv.append(f"{file_sha(p)}  {rel}")
    (out/"provenance/hashes.sha256").write_text(
        "\n".join(inv)+"\n",encoding="utf-8")
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path)
    ap.add_argument("--archive-from",type=Path)
    ap.add_argument("--archive-out",type=Path)
    ap.add_argument("--source-head");ap.add_argument("--run-id");ap.add_argument("--run-url")
    args=ap.parse_args()
    if args.archive_from:
        if not all((args.archive_out,args.source_head,args.run_id,args.run_url)):
            raise SystemExit("archive mode requires archive-out/source-head/run-id/run-url")
        print(json.dumps(archive(
            args.archive_from,args.archive_out,args.source_head,
            args.run_id,args.run_url),indent=2,sort_keys=True))
    else:
        if args.out is None: raise SystemExit("--out required")
        res=run(args.out)
        print(json.dumps({
            "status":res["status"],
            "diagnostic_status":res["diagnostic_stage"]["status"],
            "model_status":(res.get("model_stage") or {}).get("status"),
            "weight_downloaded":
                res["diagnostic_stage"]["source"]["weight_downloaded"],
            "optimizer_steps":0,
            "population_expansion_admitted":False
        },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
