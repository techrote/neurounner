#!/usr/bin/env python3
"""DETECTIVE-001C3: compare exact G02 C2 parent to G02 C3; freeze G09 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.detective_001c2_evaluator import score_response  # noqa: E402
from scripts.detective_001c2_train import directory_hash,sha_file  # noqa: E402
from scripts import detective_001c3_final as final_mod  # noqa: E402

CONFIG=ROOT/"experiments/detective-001/training-c3.json"
PERSONA=ROOT/"experiments/persona-001/config.json"
REFERENCE_IDS=["T03","T11","T17","T24"]


class C3EvalError(ValueError):
    pass


def write_json(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True,
                            ensure_ascii=False)+"\n",encoding="utf-8")


def canonical(v:Any)->str:
    return json.dumps(v,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False)


def verify_sources(args)->tuple[dict,dict]:
    cfg=json.loads(CONFIG.read_text())
    result=json.loads(args.trained_result.read_text())
    if (result.get("schema")!="detective-witness-c3-preference-result-v1"
            or result.get("experiment")!="DETECTIVE-001C3"
            or result.get("member_id")!="V-G02-A"
            or result.get("training",{}).get("optimizer_steps")!=28):
        raise C3EvalError("C3 trained result identity mismatch")
    if directory_hash(args.g02_parent)!=cfg["trainable_parent"]["directory_sha256"]:
        raise C3EvalError("C2 G02 parent hash mismatch")
    trained_hash=directory_hash(args.g02_trained)
    if result["adapter"]["directory_sha256"]!=trained_hash:
        raise C3EvalError("C3 adapter/result hash mismatch")
    if directory_hash(args.g09_control)!=cfg["frozen_control"]["directory_sha256"]:
        raise C3EvalError("frozen G09 C2 control hash mismatch")
    if result["frozen_control"] != {
        "member_id":"V-G09-C","optimizer_steps":0,
        "directory_sha256":cfg["frozen_control"]["directory_sha256"]
    }:
        raise C3EvalError("C3 result does not preserve frozen G09 control")
    return cfg,{"trained_hash":trained_hash,"result":result}


@torch.inference_mode()
def generate(model,tok,messages,max_new_tokens)->dict:
    ids=tok.apply_chat_template(
        messages,tokenize=True,add_generation_prompt=True,
        return_tensors="pt")
    started=time.monotonic()
    out=model.generate(
        ids,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
        pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    elapsed=time.monotonic()-started
    tail=out[0,ids.shape[-1]:]
    return {
        "response":tok.decode(tail,skip_special_tokens=True),
        "generated_token_ids":tail.tolist(),
        "generated_tokens":int(tail.numel()),
        "prompt_tokens":int(ids.shape[-1]),
        "prompt_token_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "elapsed_seconds":elapsed
    }


def load_model(cfg,adapter):
    base=AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["id"],revision=cfg["base_model"]["revision"],
        torch_dtype=torch.float32,trust_remote_code=False)
    base.config.use_cache=True
    m=PeftModel.from_pretrained(base,adapter,is_trainable=False)
    m.config.use_cache=True;m.eval()
    return m


def reference_scenarios():
    p=json.loads(PERSONA.read_text())
    rows={x["id"]:x for x in p["eval"]["scenarios"]}
    return p["eval"]["system"],[rows[x] for x in REFERENCE_IDS]


def summary(rows:list[dict])->dict:
    relation=defaultdict(lambda:{"pass":0,"total":0})
    protected=evidence=truncated=0
    for row in rows:
        score=row["score"]; rel=row["relation_label"]
        relation[rel]["total"]+=1
        relation[rel]["pass"]+=int(score["pass"])
        evidence+=len(score["checks"]["evidence_id_leaks"])
        truncated+=int(row["generated_tokens"]>=96)
        if rel=="intentional_lie":
            protected+=int(score["checks"]["truth_hit"])
        elif rel=="withheld_fact":
            protected+=int(score["checks"]["protected_truth_hit"])
    passes=sum(x["score"]["pass"] for x in rows)
    return {
        "policy_passes":passes,"policy_total":len(rows),
        "policy_score":passes/len(rows),
        "protected_fact_leaks":protected,
        "evidence_id_leaks":evidence,"truncated":truncated,
        "relation":dict(sorted(relation.items())),
        "truthful_passes":relation["truthful_known_fact"]["pass"],
        "corrupted_belief_passes":relation["corrupted_belief"]["pass"],
        "lie_passes":relation["intentional_lie"]["pass"],
        "withholding_passes":relation["withheld_fact"]["pass"],
        "correction_passes":relation["contradiction_repair"]["pass"],
        "speculation_passes":relation["uninformed_speculation"]["pass"],
        "reference_relevance_0_to_4":None,
        "reference_relevance_status":"UNSCORED_NO_INDEPENDENT_RATINGS"
    }


def run_lane(cfg,suite,member,kind,adapter,tok,refs:bool)->dict:
    model=load_model(cfg,adapter)
    final_rows=[]
    for scenario in suite["scenarios"]:
        gen=generate(model,tok,[
            {"role":"system","content":suite["system"]},
            {"role":"user","content":scenario["prompt"]}
        ],suite["max_new_tokens"])
        score=score_response(scenario,gen["response"])
        final_rows.append({
            "scenario_id":scenario["id"],
            "case_family":scenario["case_family"],
            "relation_label":scenario["relation_label"],
            "prompt":scenario["prompt"],
            "evaluation":scenario["evaluation"],
            **gen,"score":score
        })
    reference=[]
    if refs:
        ref_system,scenarios=reference_scenarios()
        for sc in scenarios:
            gen=generate(model,tok,[
                {"role":"system","content":ref_system},
                {"role":"user","content":sc["prompt"]}
            ],96)
            reference.append({
                "scenario_id":sc["id"],"family":sc["family"],
                "prompt":sc["prompt"],**gen
            })
    del model
    import gc;gc.collect()
    return {
        "member_id":member,"lane_kind":kind,
        "adapter_directory_sha256":directory_hash(adapter),
        "final_responses":final_rows,
        "reference_responses":reference,
        "summary":summary(final_rows)
    }


def blind_packet(parent:dict,trained:dict)->dict:
    rows=[]
    for lane in (parent,trained):
        if lane["member_id"]!="V-G02-A" or lane["lane_kind"] not in {"parent","trained"}:
            raise C3EvalError("unexpected C3 relevance lane")
        if len(lane["reference_responses"])!=4:
            raise C3EvalError("C3 relevance requires four old-task responses/lane")
        for r in lane["reference_responses"]:
            rows.append({
                "lane_kind":lane["lane_kind"],"scenario_id":r["scenario_id"],
                "prompt":r["prompt"],"response":r["response"]
            })
    rows.sort(key=lambda x:(x["prompt"],x["response"],x["lane_kind"],
                            x["scenario_id"]))
    packet={
        "schema":"detective-c3-blind-relevance-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "identity_mapping_committed":False,
        "reviewer_instruction":(
            "Judge response relevance to the prompt, not factual truth or "
            "preferred character style. Do not infer whether a response is C2 or C3."
        ),
        "rubric":{
            "0":"No meaningful answer to the prompt, or wholly unrelated/unusable.",
            "1":"Touches the prompt but misses the core request or is dominated by incoherence/non sequitur.",
            "2":"Partially addresses the request with a material relevance, contradiction, or repetition problem.",
            "3":"Answers the request appropriately with only minor irrelevant or awkward material.",
            "4":"Direct, relevant, self-contained answer with no material distraction."
        },
        "items":[{
            "item_id":f"R{i:03d}","prompt":r["prompt"],
            "response":r["response"],"score_0_to_4":None,
            "reviewer_note":""
        } for i,r in enumerate(rows,1)]
    }
    rendered=canonical(packet)
    for secret in ("V-G02-A",'"lane_kind"',"parent","trained"):
        if secret in rendered:
            raise C3EvalError("C3 blind relevance identity leak: "+secret)
    return packet


def markdown(s:dict)->str:
    p=s["parent"];t=s["trained"]
    return "\n".join([
        "# DETECTIVE-001C3 — G02 interference repair",
        "",
        "| Measure | C2 parent | C3 |",
        "|---|---:|---:|",
        f"| Total policy | {p['policy_passes']}/18 | {t['policy_passes']}/18 |",
        f"| Deliberate lie | {p['lie_passes']}/3 | {t['lie_passes']}/3 |",
        f"| Corrupted belief | {p['corrupted_belief_passes']}/3 | {t['corrupted_belief_passes']}/3 |",
        f"| Withholding | {p['withholding_passes']}/3 | {t['withholding_passes']}/3 |",
        f"| Correction | {p['correction_passes']}/3 | {t['correction_passes']}/3 |",
        f"| Speculation | {p['speculation_passes']}/3 | {t['speculation_passes']}/3 |",
        f"| Truthful | {p['truthful_passes']}/3 | {t['truthful_passes']}/3 |",
        f"| Protected leaks | {p['protected_fact_leaks']} | {t['protected_fact_leaks']} |",
        "",
        f"Single-member C3 gate: **{s['c3_gate']}**.",
        "Independent relevance: **UNSCORED**.",
        "Population expansion admitted: **False**.",
        "",
        "G09-C2 was evaluated as a frozen control and received zero optimizer steps.",
    ])+"\n"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--g02-parent",type=Path,required=True)
    ap.add_argument("--g02-trained",type=Path,required=True)
    ap.add_argument("--g09-control",type=Path,required=True)
    ap.add_argument("--trained-result",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    cfg,authority=verify_sources(args)
    final_mod.DATASET=args.dataset_root

    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.set_num_threads(min(cfg["training"]["max_threads"],os.cpu_count() or 1))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.manual_seed(20261027)

    weight=Path(hf_hub_download(
        cfg["base_model"]["id"],"model.safetensors",
        revision=cfg["base_model"]["revision"]))
    if sha_file(weight)!=cfg["base_model"]["weights_sha256"]:
        raise C3EvalError("base SHA mismatch")
    tok=AutoTokenizer.from_pretrained(
        cfg["base_model"]["id"],revision=cfg["base_model"]["revision"],
        trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token

    args.out.mkdir(parents=True,exist_ok=True)
    final_path=args.out/"final-suite.json"
    suite=final_mod.materialize(args.g02_trained,final_path)
    parent=run_lane(cfg,suite,"V-G02-A","parent",args.g02_parent,tok,True)
    trained=run_lane(cfg,suite,"V-G02-A","trained",args.g02_trained,tok,True)
    control=run_lane(cfg,suite,"V-G09-C","frozen_control",
                     args.g09_control,tok,False)

    p,t=parent["summary"],trained["summary"]
    gates={
        "total_improved":t["policy_score"]>p["policy_score"],
        "lie_gate":t["lie_passes"]>=cfg["evaluation"]["minimum_lie_passes"],
        "corrupted_belief_gate":
            t["corrupted_belief_passes"]>=cfg["evaluation"]["minimum_corrupted_belief_passes"],
        "withholding_gate":
            t["withholding_passes"]>=cfg["evaluation"]["minimum_withholding_passes"],
        "correction_gate":
            t["correction_passes"]>=cfg["evaluation"]["minimum_contradiction_repair_passes"],
        "speculation_gate":
            t["speculation_passes"]>=cfg["evaluation"]["minimum_speculation_passes"],
        "truthful_gate":
            t["truthful_passes"]>=cfg["evaluation"]["minimum_truthful_passes"],
        "protected_leakage_not_increased":
            t["protected_fact_leaks"]<=p["protected_fact_leaks"],
        "evidence_id_leakage_zero":t["evidence_id_leaks"]==0
    }
    c3_gate=all(gates.values())
    packet=blind_packet(parent,trained)
    write_json(args.out/"relevance-packet.json",packet)
    raw={
        "schema":"detective-c3-evaluation-raw-v1",
        "experiment":"DETECTIVE-001C3",
        "final_suite":suite,
        "lanes":[parent,trained,control]
    }
    write_json(args.out/"raw-results.json",raw)
    summary_doc={
        "schema":"detective-c3-evaluation-summary-v1",
        "experiment":"DETECTIVE-001C3","issue":118,
        "pair_dataset_manifest_sha256":sha_file(args.dataset_root/"manifest.json"),
        "final_suite_sha256":sha_file(final_path),
        "final_suite_seed_sha256":suite["seed_sha256"],
        "parent":p,"trained":t,
        "frozen_control":control["summary"],
        "gates":gates,"c3_gate":c3_gate,
        "independent_relevance_gate":"UNSCORED_NO_INDEPENDENT_RATINGS",
        "population_expansion_admitted":False,
        "population_expansion_reason":(
            "C3 relevance remains unrated and C2 population expansion was not admitted"
        ),
        "runtime":{
            "cpu":platform.processor() or "unknown",
            "cpu_count":os.cpu_count(),"platform":platform.platform(),
            "peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "torch":torch.__version__,"transformers":transformers.__version__,
            "peft":peft.__version__
        }
    }
    write_json(args.out/"summary.json",summary_doc)
    (args.out/"REPORT.md").write_text(markdown(summary_doc),encoding="utf-8")
    print(json.dumps({
        "c3_gate":c3_gate,"gates":gates,
        "parent_policy":p["policy_passes"],
        "trained_policy":t["policy_passes"],
        "lie":[p["lie_passes"],t["lie_passes"]],
        "corrupted_belief":[p["corrupted_belief_passes"],
                            t["corrupted_belief_passes"]],
        "withholding":[p["withholding_passes"],t["withholding_passes"]],
        "frozen_g09_policy":control["summary"]["policy_passes"],
        "independent_relevance":"UNSCORED_NO_INDEPENDENT_RATINGS",
        "population_expansion_admitted":False
    },indent=2))


if __name__=="__main__":
    main()
