"""G1C frozen cross-model postcard audit on entirely new counterfactual worlds.

Uses IMMUTABLE G1B checkpoint, no adapter training, no tuning, and explicit
continuous-blend donor swaps. Each recipient input is source-fenced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file

from experiments.memplant.dreamgraft001.bridge import answer_loss,format_prompt,generation_from_prefix
from experiments.memplant.dreamgraft001.contract import write_json
from experiments.memplant.dreamgraft001.g0_live import load_pinned,measure_rss,target_ids
from experiments.memplant.g1.features import QUESTION_SET,TRAIT,RECIPIENT,RECIPIENT_REV
from experiments.memplant.g1.imprint import binary_log_odds,inferred_word

PINNED_CHECKPOINT_SHA256="e1099d611ba577cd3e56208a7c9777d49cbf6a65380c15b62ef65ed3c9bc0f81"
PINNED_G1B_RUN=43693630530
PINNED_G1B_ARTIFACT=14154107117

def load_sources(directory):
    directory=Path(directory)
    m=json.loads((directory/"manifest.json").read_text())
    if m["schema"]!="memplant-g1c-confirm-v1":raise RuntimeError("wrong source schema")
    p=directory/"confirm_v.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest()!=m["npz_sha256"]:
        raise RuntimeError("confirm feature archive digest mismatch")
    with np.load(p,allow_pickle=False) as a:
        xs=np.asarray(a["v"],dtype="float32")
    if xs.shape!=(16,2048):
        raise RuntimeError(f"unexpected confirm source tensor {xs.shape}")
    if hashlib.sha256(xs.tobytes()).hexdigest()!=m["feature_sha256"]:
        raise RuntimeError("confirm feature tensor hash mismatch")
    return torch.from_numpy(xs).clone(),m

def get_frozen_adapter(directory):
    path=Path(directory)/"source_router_and_prefixes.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PINNED_CHECKPOINT_SHA256:
        raise RuntimeError("G1B pretrained model has wrong hash; do not silently refit")
    state=load_file(str(path),device="cpu")
    expected={"prefix_hardens","prefix_softens","ridge_alpha","ridge_train_features"}
    if set(state)!=expected:raise RuntimeError("G1B checkpoint schema changed")
    if state["ridge_alpha"].shape!=(32,) or state["ridge_train_features"].shape!=(32,2048):
        raise RuntimeError("wrong frozen supervised ridge geometry")
    if state["prefix_hardens"].shape!=(1,8,2048) or state["prefix_softens"].shape!=(1,8,2048):
        raise RuntimeError("wrong frozen SmolLM2 pseudo-embedding dimensions")
    return state

def source_scores(v,state):
    v=v.float()
    norm=(v-v.mean(1,keepdim=True))/(v.std(1,keepdim=True,unbiased=False)+1e-6)
    return ((norm@state["ridge_train_features"].T)/2048)@state["ridge_alpha"]

def conditioned_prefix(state,scores,records,idx,condition):
    p0=state["prefix_hardens"]
    p1=state["prefix_softens"]
    twin_map={r["case_id"]:j for j,r in enumerate(records)}
    twin=twin_map[records[idx]["twin_id"]]
    other=((idx//2+1)%8)*2+(1-idx%2)
    if condition=="genuine_blend":source=idx
    elif condition=="swapped_blend":source=twin
    elif condition=="wrong_family_blend":source=other
    else:source=idx
    score=float(scores[source])
    if condition in ("genuine_blend","swapped_blend","wrong_family_blend"):
        weight=torch.sigmoid(torch.tensor(score*25.0)).item()
        return (weight*p0+(1.0-weight)*p1).contiguous(),source
    if condition=="hard_gate":
        return (p0 if score>=0 else p1),source
    if condition=="equal_blend":
        return (0.5*p0+0.5*p1).contiguous(),source
    if condition=="zero":
        return torch.zeros_like(p0),source
    if condition=="random":
        gen=torch.Generator().manual_seed(11883+idx)
        noise=torch.randn(p0.shape,generator=gen)
        return noise*(p0.norm()/(noise.norm()+1e-12)),source
    raise ValueError("bad audit condition")

def audit(receiver,tokenizer,state,v,manifest):
    records=manifest["records"]
    scores=source_scores(v,state)
    labels={name:target_ids(tokenizer," "+name) for name in TRAIT}
    prompts=[format_prompt(tokenizer,q) for q in QUESTION_SET]
    byid={r["case_id"]:i for i,r in enumerate(records)}
    conditions=("genuine_blend","swapped_blend","wrong_family_blend",
                "hard_gate","equal_blend","zero","random","native_text")
    results=[]
    t0=time.monotonic()
    for idx,record in enumerate(records):
        for order,question in enumerate(QUESTION_SET):
            for condition in conditions:
                if condition=="native_text":
                    input_ids=format_prompt(tokenizer,"Source document:\n"+record["source_text"]+"\n\n"+question)
                    prefix=None
                    presented=idx
                else:
                    input_ids=prompts[order]
                    prefix,presented=conditioned_prefix(state,scores,records,idx,condition)
                margin=binary_log_odds(receiver,input_ids,labels,prefix)
                selected=inferred_word(margin)
                expected=records[presented]["answer"]
                results.append({
                    "case_id":record["case_id"],"family":record["family"],
                    "order":order,"condition":condition,
                    "predicted":selected,
                    "source_original_truth":record["answer"],
                    "source_presented_truth":expected,
                    "correct_presented":selected==expected,
                    "correct_original":selected==record["answer"],
                    "numeric_source_id":records[presented]["case_id"] if condition!="native_text" else None,
                    "log_odds_hardens_minus_softens":round(margin,6),
                    "receiver_saw_source_text":condition=="native_text"})
    summary={}
    for cond in conditions:
        rows=[r for r in results if r["condition"]==cond]
        summary[cond]={
            "n":len(rows),
            "presented_accuracy":round(sum(r["correct_presented"] for r in rows)/len(rows),4),
            "original_accuracy":round(sum(r["correct_original"] for r in rows)/len(rows),4),
            "mean_signed_presented_margin":round(sum(
                (1 if r["source_presented_truth"]=="hardens" else -1)*r["log_odds_hardens_minus_softens"]
                for r in rows)/len(rows),6)}
    lookup={(x["case_id"],x["order"],x["condition"]):x for x in results}
    swaps=[]
    for case in records:
        for order in (0,1):
            true=lookup[case["case_id"],order,"genuine_blend"]
            swapped=lookup[case["case_id"],order,"swapped_blend"]
            swaps.append({"case":case["case_id"],"order":order,
                          "flip":true["predicted"]!=swapped["predicted"],
                          "true_follows_donor":true["correct_presented"],
                          "swap_follows_replacement":swapped["correct_presented"]})
    score_cases=[{"case_id":record["case_id"],"family":record["family"],
                  "expected":record["answer"],"ridge_score":round(float(score),7),
                  "ridge_sign_correct":bool((score>=0)==(1 if record["answer"]=="hardens" else 0))}
                 for record,score in zip(records,scores,strict=True)]
    return {"source_readout":{"ridge_correct":sum(x["ridge_sign_correct"] for x in score_cases),
                              "ridge_total":len(score_cases),
                              "source_scores":score_cases},
            "n_families":8,"n_cases":16,"n_order_cases":32,
            "conditions":summary,
            "counterfactual_swap":{"n":len(swaps),
                                   "flips":sum(x["flip"] for x in swaps),
                                   "both_correct":sum(x["true_follows_donor"] and x["swap_follows_replacement"] for x in swaps),
                                   "cases":swaps},
            "all_raw_scores":results,"evaluation_seconds":round(time.monotonic()-t0,3)}

def creative(receiver,tokenizer,state,v,manifest):
    records=manifest["records"]
    score=source_scores(v,state)
    samples=[]
    prompt=format_prompt(tokenizer,"Describe an impossible material and an imaginative invention that uses its behavior under heat. Be concrete.")
    for idx in (0,1):
        for cond in ("genuine_blend","swapped_blend","zero"):
            prefix,_=conditioned_prefix(state,score,records,idx,cond)
            s,ids=generation_from_prefix(receiver,tokenizer,prompt,prefix,max_tokens=36)
            samples.append({"case_id":records[idx]["case_id"],"condition":cond,
                            "raw_text":s,"token_ids":ids})
    return {"samples":samples,"n":len(samples),"unscored_and_unpolished":True}

def run(source,checkpoint,out_dir):
    root=Path(out_dir)
    root.mkdir(parents=True,exist_ok=True)
    report={"schema":"memplant-g1c-confirm-v1","status":"STARTED",
            "provenance":{"g1b_run":PINNED_G1B_RUN,"g1b_artifact":PINNED_G1B_ARTIFACT,
                          "checkpoint_sha256":PINNED_CHECKPOINT_SHA256},
            "no_training_or_adjustment":True,
            "claims":{"only_supervised_low_bandwidth_numerical_communication_test":True,
                      "no_native_KV_translation":True,"no_causal_allele_or_descendant":True}}
    try:
        torch.set_num_threads(min(torch.get_num_threads(),4))
        v,manifest=load_sources(source)
        state=get_frozen_adapter(checkpoint)
        report["source"]={"donor":manifest["donor"],
                          "source_sha256":manifest["feature_sha256"],
                          "fresh_test_families":8,"sources":16}
        model,tok,identity=load_pinned(RECIPIENT,RECIPIENT_REV)
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
        report["receiver"]=identity
        report["audit"]=audit(model,tok,state,v,manifest)
        report["creative_postcards"]=creative(model,tok,state,v,manifest)
        report["rss_end_mib"]=measure_rss()
        report["status"]="COMPLETED_FROZEN_CONFIRMATION"
        write_json(root/"report.json",report)
        print(json.dumps({"status":report["status"],
                          "accuracy":report["audit"]["conditions"],
                          "source_swap":{k:v for k,v in report["audit"]["counterfactual_swap"].items() if k!="cases"},
                          "ridge":report["audit"]["source_readout"]["ridge_correct"],
                          "creative":report["creative_postcards"]["n"]},indent=2))
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]={"type":type(e).__name__,"message":str(e),
                         "trace":traceback.format_exc()[-4500:]}
        write_json(root/"report.json",report)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="runtime/memplant/g1c/source")
    p.add_argument("--checkpoint",default="runtime/memplant/g1c/checkpoint")
    p.add_argument("--out",default="runtime/memplant/g1c/results")
    a=p.parse_args()
    run(a.source,a.checkpoint,a.out)
