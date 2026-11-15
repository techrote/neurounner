"""G1B two-stage numerical Dream Postcard (frozen-model exploratory control).

A's REAL pooled V state -> supervised ridge 1-bit readout -> B-native
trained *continuous* prefix. This is deliberately NOT direct KV transplantation.
A hard-gated prefix is a one-bit classifier+lookup control; the continuous
interpolated prefix is a separate, more difficult experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch
from safetensors.torch import save_file

from experiments.memplant.g1.features import QUESTION_SET, TRAIT, RECIPIENT, RECIPIENT_REV
from experiments.memplant.g1.imprint import binary_log_odds, inferred_word
from experiments.memplant.dreamgraft001.bridge import answer_loss,format_prompt,generation_from_prefix
from experiments.memplant.dreamgraft001.contract import write_json
from experiments.memplant.dreamgraft001.g0_live import load_pinned,measure_rss,target_ids


def source_readout(raw,records):
    """Predeclared dual ridge classifier, lambda=1.0; only 32 train sources."""
    xs=torch.from_numpy(raw).float()
    xs=(xs-xs.mean(dim=1,keepdim=True))/(xs.std(dim=1,keepdim=True,unbiased=False)+1e-6)
    train=[i for i,r in enumerate(records) if r["split"]=="train"]
    y=torch.tensor([1.0 if r["answer"]=="hardens" else -1.0 for r in records])
    xt=xs[train]
    K=(xt@xt.T)/xs.shape[1]
    alpha=torch.linalg.solve(K+torch.eye(len(train))*1.0,y[train])
    score=(xs@xt.T)/xs.shape[1]@alpha
    stats={}
    for split in ("train","validation","fresh_test"):
        chosen=[i for i,r in enumerate(records) if r["split"]==split]
        stats[split]={"correct":sum(bool((score[i]>=0)==(y[i]>0)) for i in chosen),
                      "total":len(chosen),
                      "cases":[{"case":records[i]["case_id"],"label":records[i]["answer"],
                                "ridge_score":round(float(score[i]),7),
                                "correct":bool((score[i]>=0)==(y[i]>0))}
                               for i in chosen]}
    return score,alpha,xt,stats


def build_recipient_prefixes(model,tok,*,updates=80):
    torch.manual_seed(221)
    sigma=max(float(model.get_input_embeddings().weight.detach().std()),0.01)
    width=int(model.config.hidden_size)
    prototypes=torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn(1,8,width)*sigma) for _ in range(2)])
    optim=torch.optim.Adam(prototypes.parameters(),lr=0.006)
    prompts=[format_prompt(tok,q) for q in QUESTION_SET]
    targets=[target_ids(tok," "+x) for x in TRAIT]
    history=[]
    for step in range(updates):
        clazz=step%2
        order=(step//2)%2
        optim.zero_grad(set_to_none=True)
        loss=answer_loss(model,prompts[order],targets[clazz],prototypes[clazz])
        if not torch.isfinite(loss):raise RuntimeError("nonfinite prototype CE loss")
        loss.backward()
        if prototypes[clazz].grad is None or not torch.isfinite(prototypes[clazz].grad).all():
            raise RuntimeError("missing/invalid prefix gradient")
        norm=float(torch.nn.utils.clip_grad_norm_([prototypes[clazz]],5.0))
        if not math.isfinite(norm) or norm<=0:raise RuntimeError("nonpositive prefix gradient")
        optim.step()
        history.append({"step":step+1,"label":TRAIT[clazz],"order":order,
                        "loss_before":round(float(loss.item()),6),
                        "gradient_norm_before_clip":round(norm,5),
                        "rss_mib":measure_rss()})
    result=[p.detach().clone() for p in prototypes]
    for p in model.parameters():
        if p.grad is not None:raise RuntimeError("receiver model accumulated gradients")
    return result,{"updates":updates,"lr":0.006,"history":history,
                   "last_losses":{t:[x["loss_before"] for x in history if x["label"]==t][-4:] for t in TRAIT},
                   "peak_rss_mib":max(x["rss_mib"] for x in history)}


def choose_prefix(protos,scores,idx,records,condition,*,seed):
    ref={r["case_id"]:i for i,r in enumerate(records)}
    twin=ref[records[idx]["twin_id"]]
    score=float(scores[twin] if condition=="swapped_gate" else scores[idx])
    if condition in ("genuine_gate","swapped_gate"):
        return protos[0] if score>=0 else protos[1]
    if condition=="blend":
        p=torch.sigmoid(torch.tensor(score*25.0)).item()
        return (p*protos[0]+(1-p)*protos[1]).contiguous()
    if condition=="zero":
        return torch.zeros_like(protos[0])
    if condition=="random":
        gen=torch.Generator().manual_seed(seed)
        x=torch.randn(protos[0].shape,generator=gen)
        return x* (protos[0].norm()/(x.norm()+1e-12))
    if condition=="hard_only":return protos[0]
    if condition=="soft_only":return protos[1]
    raise ValueError(f"unexpected condition: {condition}")


def evaluate(model,tok,protos,scores,records):
    answer_ids={x:target_ids(tok," "+x) for x in TRAIT}
    generic=[format_prompt(tok,q) for q in QUESTION_SET]
    conditions=("genuine_gate","blend","swapped_gate","zero","random",
                "hard_only","soft_only","native_text_reference")
    idx={r["case_id"]:i for i,r in enumerate(records)}
    tested=[i for i,r in enumerate(records) if r["split"]=="fresh_test"]
    rows=[]
    for i in tested:
        r=records[i]
        for order,q in enumerate(QUESTION_SET):
            for c in conditions:
                if c=="native_text_reference":
                    # This *separately marked* comparison intentionally supplies text.
                    prompt=format_prompt(tok,"Source document:\n"+r["source_text"]+"\n\n"+q)
                    pref=None
                else:
                    prompt=generic[order]
                    pref=choose_prefix(protos,scores,i,records,c,seed=736+i+1000*order)
                margin=binary_log_odds(model,prompt,answer_ids,pref)
                predicted=inferred_word(margin)
                target=records[idx[r["twin_id"]]]["answer"] if c=="swapped_gate" else r["answer"]
                rows.append({"case_id":r["case_id"],"family":r["family"],
                             "condition":c,"order":order,
                             "predicted":predicted,"presented_truth":target,
                             "original_truth":r["answer"],
                             "correct_for_presented":predicted==target,
                             "correct_for_original":predicted==r["answer"],
                             "log_odds_hardens_minus_softens":round(margin,6),
                             "source_text_visible_to_B":c=="native_text_reference"})
    overall={}
    for c in conditions:
        block=[x for x in rows if x["condition"]==c]
        overall[c]={
            "n":len(block),
            "accuracy_for_presented_truth":round(sum(x["correct_for_presented"] for x in block)/len(block),4),
            "accuracy_for_original_truth":round(sum(x["correct_for_original"] for x in block)/len(block),4),
            "mean_source_signed_margin":round(sum(
                (1 if x["presented_truth"]=="hardens" else -1)*x["log_odds_hardens_minus_softens"]
                for x in block)/len(block),6)}
    k={(x["case_id"],x["order"],x["condition"]):x for x in rows}
    flips=[]
    for i in tested:
        for order in (0,1):
            a=k[records[i]["case_id"],order,"genuine_gate"]
            b=k[records[i]["case_id"],order,"swapped_gate"]
            flips.append(a["predicted"] != b["predicted"])
    return {"fresh_families":len({records[i]["family"] for i in tested}),
            "n_cases":len(tested),"conditions":overall,
            "swapped_donor_prediction_flip_rate":round(sum(flips)/len(flips),4),
            "paired_evaluations":len(flips),"raw":rows}


def raw_creative(model,tok,protos,scores,records):
    indices=[i for i,r in enumerate(records) if r["split"]=="fresh_test"][:2]
    tasks=("Invent a peculiar fictional metal and explain how heating changes its behavior.",
           "Describe a machine with one peculiar failure caused by warming its metal parts.")
    results=[]
    for i in indices:
        for task_id,task in enumerate(tasks):
            prompt=format_prompt(tok,task)
            for cond in ("genuine_gate","blend","swapped_gate","zero"):
                prefix=choose_prefix(protos,scores,i,records,cond,seed=1480+i+task_id*10)
                content,ids=generation_from_prefix(model,tok,prompt,prefix,max_tokens=48)
                results.append({"case_id":records[i]["case_id"],"task":task,
                                "condition":cond,"raw":content,"token_ids":ids})
    return {"n":len(results),"raw":results,
            "not_transfer_evidence":True,"unpolished_and_unselected":True}


def run(source_directory,output_directory):
    root=Path(source_directory)
    out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["schema"]!="memplant-g1b-postcards-v1":
        raise RuntimeError("wrong source manifest")
    payload=root/"source_v.npz"
    if hashlib.sha256(payload.read_bytes()).hexdigest()!=manifest["source_array_sha256"]:
        raise RuntimeError("source payload checksum mismatch")
    with np.load(payload,allow_pickle=False) as npz:
        xs=np.asarray(npz["v"],dtype="float32")
    if xs.shape!=(48,2048) or hashlib.sha256(xs.tobytes()).hexdigest()!=manifest["feature_sha256"]:
        raise RuntimeError("source V shape/hash mismatch")
    records=manifest["records"]
    assert [sum(x["split"]==s for x in records) for s in ("train","validation","fresh_test")]==[32,8,8]
    report={"schema":"memplant-g1b-postcards-v1","status":"STARTED",
            "source":manifest["donor"],"recipient_name":RECIPIENT,
            "claims":{"direct_KV_transplant":False,"JIT_context_state_compilation":False,
                      "low_bandwidth_source_feature_compilation_test":True,
                      "not_general_causal_inheritance":True}}
    try:
        torch.set_num_threads(min(4,torch.get_num_threads()))
        scores,alpha,train_x,stats=source_readout(xs,records)
        report["ridge"]={"lambda":1.0,"calibration_multiplier_for_blend":25.0,
                         "simple_linear_supervised_semantic_gate":True,
                         "split_accuracy":stats}
        model,tok,identity=load_pinned(RECIPIENT,RECIPIENT_REV)
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
        report["receiver"]=identity
        prototypes,train_data=build_recipient_prefixes(model,tok,updates=80)
        report["native_prefix_training"]=train_data
        report["fresh_test"]=evaluate(model,tok,prototypes,scores,records)
        report["raw_postcards"]=raw_creative(model,tok,prototypes,scores,records)
        save_file({"prefix_hardens":prototypes[0].contiguous(),
                   "prefix_softens":prototypes[1].contiguous(),
                   "ridge_alpha":alpha.contiguous(),
                   "ridge_train_features":train_x.contiguous()},
                  str(out/"source_router_and_prefixes.safetensors"))
        report["rss_end_mib"]=measure_rss()
        report["status"]="COMPLETED_EXPLORATORY"
        write_json(out/"report.json",report)
        print(json.dumps({"status":report["status"],
                          "ridge_fresh":stats["fresh_test"]["correct"],
                          "prototype_last_losses":train_data["last_losses"],
                          "receiver":report["fresh_test"]["conditions"],
                          "source_swap_flips":report["fresh_test"]["swapped_donor_prediction_flip_rate"],
                          "raw_creative":report["raw_postcards"]["n"]},indent=2))
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]={"type":type(e).__name__,"message":str(e),
                         "trace":traceback.format_exc()[-3500:]}
        write_json(out/"report.json",report)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="runtime/memplant/g1b/source")
    p.add_argument("--out",default="runtime/memplant/g1b/experiment")
    a=p.parse_args()
    run(a.source,a.out)


if __name__=="__main__":
    main()
