"""MEMPLANT Garden v2: creative Sprocket + LFM, with audited rule compilation.

Unlike the v1 experiment, this uses a fail-closed source-conditioned mechanical
compiler *after* LFM's raw design and separately reports raw vs repaired success.
The compiled decisions are deterministic, NOT evidence LFM learned the rules.
Only ordinary TEXT postcards are passed to the current pinned GGUF models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiments.memplant.memorygarden.garden import Invention,evaluate,INSTRUCTIONS
from experiments.memplant.memorygarden.subjects import (
    load_qwen_state,train_only_readout,select_worlds)
from experiments.memplant.memorygarden.garden_compiler import (
    parse_authenticated_laws,compile_design,constraints)

PROMPT_HEADER = """You are a mechanical inventor and specification drafter.
In the design below, the source facts are immutable and any contradicted claims
must be rejected. Preserve the imaginative idea but obey every REQUIRED field.
Output exactly one valid JSON object with name, frame, collector, sequence, explanation.
Do not treat the creative seed as a physics authority.
"""


def prompt_for(seed,evidence,requirement):
    p=PROMPT_HEADER+"Creative seed (may be physically wrong):\n"+seed+"\n\n"
    p+="Source evidence:\n"+evidence+"\n\n"
    if requirement is None:
        p+="No authenticated law information: use your best guess.\n"
    else:
        p+=("Mechanically REQUIRED fields from the authenticated observation:\n"+
            json.dumps(requirement,sort_keys=True)+"\n")
    return p+"\n"+INSTRUCTIONS


def run(source,out,*,max_tokens=170):
    from scripts.experiment import Server,load_kv
    repo=Path(__file__).resolve().parents[3]
    config=load_kv(repo/".cache/neurounner/paths.env")
    executable=Path(config["LLAMA_SERVER"])
    sprocket_file=Path(config["SPROCKET_MODEL"])
    lfm_file=Path(config["LFM_MODEL"])
    if not all(p.is_file() for p in (executable,sprocket_file,lfm_file)):
        raise FileNotFoundError("Missing pinned model; invoke scripts/bootstrap.sh all")
    v,source_manifest=load_qwen_state(source)
    worlds=select_worlds(source_manifest["cases"],
                         train_only_readout(v,source_manifest["cases"]))
    sprocket=Server("memplant-garden2-sprocket",executable,sprocket_file,
                    18961,threads=2,context=2048)
    lfm=Server("memplant-garden2-lfm",executable,lfm_file,
               18962,threads=2,context=2048)
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    report={
        "schema":"memplant-garden-compiler-v2",
        "source_sha256":source_manifest["channels"]["v"]["sha256"],
        "subjects":[sprocket_file.name,lfm_file.name],
        "source_channel":"real Qwen pooled V -> fixed supervised ridge -> TEXT only",
        "model_generated_mechanical_choices_scored_separately":True,
        "deterministic_compiler_choices_not_ascribed_to_LFM":True,
        "conditions":["no_memory","state_memory","opposite_memory","full_source"],
        "records":[],"status":"STARTED",
    }
    t0=time.monotonic()
    try:
        sprocket.start()
        lfm.start()
        for index,(actual,estimate,source_case) in enumerate(worlds):
            seed_prompt=(
                "Invent a small peculiar machine with metal, heat, vibrating dust "
                "and a dust collector. Give one unusual mechanism and a failure "
                "mode in two sentences, without giving JSON or conventional choices."
            )
            seed=sprocket.chat(seed_prompt,max_tokens=110,temperature=0.75,seed=1457+index)
            for cond in report["conditions"]:
                if cond=="no_memory":
                    evidence="No authenticated world observations."
                    kind=None
                elif cond=="state_memory":
                    evidence=estimate.postcard()
                    kind="card"
                elif cond=="opposite_memory":
                    evidence=estimate.inverted().postcard()
                    kind="card"
                else:
                    evidence=source_case["source"]
                    kind="source"
                law=None
                requirement=None
                if kind is not None:
                    # Parsing only given evidence, never hidden evaluator labels.
                    law=parse_authenticated_laws(evidence,kind=kind,name=actual.name+"-"+cond)
                    requirement=constraints(law)
                prompt=prompt_for(seed["content"],evidence,requirement)
                row={
                    "case_id":actual.name,"condition":cond,
                    "true_laws_for_evaluation":[actual.heat,actual.vibration],
                    "ridge_estimate":[estimate.heat,estimate.vibration],
                    "evidence_sha256":hashlib.sha256(evidence.encode()).hexdigest(),
                    "compiler_law_source":kind,
                    "sprocket_raw_creative_seed":seed["content"],
                    "lfm_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                }
                try:
                    response=lfm.chat(prompt,max_tokens=max_tokens,temperature=0.05,
                                      seed=807+index)
                    row["lfm_response_raw"]=response
                    try:
                        raw=Invention.from_text(response["content"])
                        row["raw_invention"]=raw.__dict__
                        row["raw_score"]=evaluate(actual,raw)
                        compiled,trace=compile_design(
                            raw,evidence=evidence if kind is not None else None,
                            kind=kind,name=actual.name+"-"+cond)
                        row["compiled_invention"]=compiled.__dict__
                        row["compiled_score"]=evaluate(actual,compiled)
                        if law is not None:
                            row["compiled_score_presented_world"]=evaluate(law,compiled)
                        row["compiler_trace"]=trace
                    except (ValueError,json.JSONDecodeError) as e:
                        row["parse_error"]=str(e)[:300]
                        row["raw_score"]={"passed":False,"component_accuracy":0.0}
                        row["compiled_score"]={"passed":False,"component_accuracy":0.0}
                except Exception as e:
                    row["request_error"]=type(e).__name__+": "+str(e)[:300]
                    row["raw_score"]={"passed":False,"component_accuracy":0.0}
                    row["compiled_score"]={"passed":False,"component_accuracy":0.0}
                report["records"].append(row)
                dest.write_text(json.dumps(report,indent=2)+"\n")
                print(actual.name,cond,"raw",row["raw_score"].get("passed"),
                      "compiled",row["compiled_score"].get("passed"),flush=True)
        report["summary"]={}
        for cond in report["conditions"]:
            rr=[r for r in report["records"] if r["condition"]==cond]
            report["summary"][cond]={
                "n":len(rr),
                "lfm_valid_json":sum("raw_invention" in r for r in rr),
                "raw_lfm_functional":sum(bool(r["raw_score"]["passed"]) for r in rr),
                "compiled_functional":sum(bool(r["compiled_score"]["passed"]) for r in rr),
                "component_changes_from_lfm":sum(len(r.get("compiler_trace",{}).get("changed_fields",[]))
                                                 for r in rr),
                "correct_for_presented_counterfactual":sum(
                    bool(r.get("compiled_score_presented_world",{}).get("passed"))
                    for r in rr)
            }
        source_by_id={r["case_id"]:r for r in report["records"] if r["condition"]=="state_memory"}
        opposite_by_id={r["case_id"]:r for r in report["records"] if r["condition"]=="opposite_memory"}
        comparisons=[]
        for name,x in source_by_id.items():
            y=opposite_by_id[name]
            if "compiled_invention" not in x or "compiled_invention" not in y:
                continue
            fields=("frame","collector","sequence")
            changes=[key for key in fields if x["compiled_invention"][key]!=y["compiled_invention"][key]]
            comparisons.append(dict(case=name,changed_fields=changes,
                                    all_three_reversed=len(changes)==3))
        report["counterfactual_pair_tests"]=comparisons
        report["elapsed_seconds"]=round(time.monotonic()-t0,3)
        report["status"]="COMPLETE"
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]=type(e).__name__+": "+str(e)
        raise
    finally:
        sprocket.stop()
        lfm.stop()
        dest.write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--source",default="runtime/memplant/garden/source")
    a.add_argument("--out",default="runtime/memplant/garden/compiler2/report.json")
    p=a.parse_args()
    run(p.source,p.out)
