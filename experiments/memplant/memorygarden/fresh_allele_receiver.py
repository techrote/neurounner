"""Fresh-world recipient confirmation for pre-existing additive Counterfactual Alleles.

Frozen Qwen source encoder; unchanged original-DG002 training-only ridge and
allele deltas; unchanged four frozen SmolLM2-1.7B native soft prefixes.
No fresh-world labels or texts reach B. No fitting in this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from experiments.memplant.memorygarden.fresh_allele_source import cases as fresh_cases
from experiments.memplant.allele_receiver.experiment import (
    fit_train_only_ridge,learned_deltas,make_prefix)
from experiments.memplant.dreamgraft002.evaluate import (
    load_inputs,batch_four_choices,predicted,label_accuracy)
from experiments.memplant.dreamgraft002.prefixes import QUESTIONS
from experiments.memplant.dreamgraft001.bridge import format_prompt
from experiments.memplant.dreamgraft001.g0_live import load_pinned,measure_rss
from experiments.memplant.dreamgraft002.source import (
    RECEIVER,RECEIVER_REV,save_json)

PREVIOUS_SOURCE_HASH="473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75"
PREVIOUS_PREFIX_HASH="89fdb9c2edf8f2d3fffba6a124e6bcb62177229a4ee5d70522a17ed02eba8cc4"
CONDITIONS=("original","heat_delta","heat_reencode","vib_delta",
            "vib_reencode","random_heat","wrong_axis_heat")

def load_fresh(root):
    root=Path(root)
    meta=json.loads((root/"manifest.json").read_text())
    if meta["schema"]!="memplant-fresh-allele-v1" or meta["records"]!=fresh_cases():
        raise RuntimeError("Fresh source code and manifest are inconsistent")
    path=root/"fresh_v.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["feature_file_sha256"]:
        raise RuntimeError("Fresh source ZIP content checksum changed")
    with np.load(path,allow_pickle=False) as z:
        v=torch.tensor(np.array(z["v"],dtype="float32"))
    if tuple(v.shape)!=(32,2048) or hashlib.sha256(v.numpy().tobytes()).hexdigest()!=meta["feature_array_sha256"]:
        raise RuntimeError("Fresh true donor feature array checksum changed")
    return v,meta

def edited(idx,condition,rows,v,heat,vib):
    row=rows[idx]
    lookup={r["case_id"]:i for i,r in enumerate(rows)}
    h=lookup[row["heat_twin"]]
    k=lookup[row["vibe_twin"]]
    if condition=="original":return v[idx],idx
    if condition=="heat_reencode":return v[h],h
    if condition=="vib_reencode":return v[k],k
    if condition=="heat_delta":
        sign=1 if row["heat"]=="hardens" else -1
        return v[idx]+sign*heat,h
    if condition=="vib_delta":
        sign=1 if row["vibration"]=="floats" else -1
        return v[idx]+sign*vib,k
    if condition=="random_heat":
        sign=1 if row["heat"]=="hardens" else -1
        rng=torch.Generator(device="cpu").manual_seed(20261116+idx)
        noise=torch.randn(heat.shape,generator=rng)
        noise*=heat.norm()/(noise.norm()+1e-10)
        return v[idx]+sign*noise,h
    if condition=="wrong_axis_heat":
        sign=1 if row["heat"]=="hardens" else -1
        wrong=vib*(heat.norm()/(vib.norm()+1e-10))
        return v[idx]+sign*wrong,h
    raise ValueError(condition)

def run(fresh_dir,old_dir,prefix_dir,out,shard):
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    report={"schema":"memplant-fresh-allele-recipient-v1","shard":shard,
            "status":"STARTED","frozen_adapter":True,
            "previously_exposed_sealed_worlds_excluded":True,
            "fresh_worlds_used_for_fitting":False,
            "new_training_updates":0,"records":[]}
    try:
        fresh,manifest=load_fresh(fresh_dir)
        old,features,pmeta,prototypes=load_inputs(old_dir,prefix_dir)
        if old["channels"]["v"]["sha256"]!=PREVIOUS_SOURCE_HASH or pmeta["prototypes_sha256"]!=PREVIOUS_PREFIX_HASH:
            raise RuntimeError("Original DG002 data or B-native carriers do not match source custody")
        train_x=features["v"]
        decode=fit_train_only_ridge(train_x,old["cases"])
        delta_h,delta_v=learned_deltas(train_x,old["cases"])
        rows=manifest["records"]
        group_ids=set(range(shard*2,shard*2+2))
        indices=[i for i,r in enumerate(rows) if int(r["family"].split("-")[-1]) in group_ids]
        if len(indices)!=8 or not (0<=shard<4):
            raise RuntimeError("Shard must have two full unseen counterfactual families")
        report["families"]=sorted({rows[i]["family"] for i in indices})
        report["provenance"]={
            "donor_revision":manifest["donor"]["revision"],
            "old_train_v_sha256":PREVIOUS_SOURCE_HASH,
            "new_fresh_v_sha256":manifest["feature_array_sha256"],
            "recipient_prefix_sha256":PREVIOUS_PREFIX_HASH,
            "training_pair_count_each_edit":40,
            "ridge_regularization":1.0,
            "no_source_reencoding_during_recipient_readback":True,
        }
        receiver,tok,recipient_identity=load_pinned(RECEIVER,RECEIVER_REV)
        receiver.eval()
        for p in receiver.parameters():p.requires_grad_(False)
        prompt=format_prompt(tok,QUESTIONS[0])
        report["recipient_revision"]=recipient_identity["revision"]
        start=time.monotonic()
        for completed,index in enumerate(indices,1):
            source=rows[index]
            for cond in CONDITIONS:
                tensor,target_index=edited(index,cond,rows,fresh,delta_h,delta_v)
                logits=decode(tensor.unsqueeze(0))[0]
                prefix=make_prefix(prototypes,logits)
                scores=batch_four_choices(receiver,tok,prompt,prefix)
                pred=predicted(scores)
                truth=rows[target_index]["answer_class"]
                flags=label_accuracy(pred,truth)
                report["records"].append({
                    "case_id":source["case_id"],"family":source["family"],
                    "condition":cond,"presented_source":rows[target_index]["case_id"],
                    "predicted_class":pred,"expected_class":truth,
                    "correct_target":flags,
                    "correct_original":label_accuracy(pred,source["answer_class"]),
                    "ridge_heat":round(float(logits[0]),6),
                    "ridge_vibration":round(float(logits[1]),6),
                    "raw_candidate_log_probs":[round(float(x),5) for x in scores]})
            report["completed_fresh_cases"]=completed
            save_json(dest,report)
            if completed%2==0:print("Fresh allele B readback",completed,"/8",flush=True)
        summary={}
        for cond in CONDITIONS:
            rr=[r for r in report["records"] if r["condition"]==cond]
            summary[cond]={
                "n":len(rr),
                "heat_correct":sum(r["correct_target"]["heat"] for r in rr),
                "vibration_correct":sum(r["correct_target"]["vibration"] for r in rr),
                "both_correct":sum(r["correct_target"]["joint"] for r in rr),
            }
        lookup={(r["case_id"],r["condition"]):r for r in report["records"]}
        for condition,axis in (("heat_delta","heat"),("vib_delta","vibration")):
            kept=changed=selective=0
            for i in indices:
                case=rows[i]["case_id"]
                old_cls=lookup[case,"original"]["predicted_class"]
                new_cls=lookup[case,condition]["predicted_class"]
                if axis=="heat":
                    flipped=(old_cls//2)!=(new_cls//2)
                    preserved=(old_cls%2)==(new_cls%2)
                else:
                    flipped=(old_cls%2)!=(new_cls%2)
                    preserved=(old_cls//2)==(new_cls//2)
                kept+=preserved
                changed+=flipped
                selective+=flipped and preserved
            summary[condition].update(
                actual_target_flips=changed,other_rule_preserved=kept,
                selective_flips=selective)
        report["summary"]=summary
        report["rss_mib"]=measure_rss()
        report["elapsed_seconds"]=round(time.monotonic()-start,3)
        report["status"]="COMPLETE_FROZEN_FRESH_WORLD_EXPLORATORY"
        save_json(dest,report)
        print(json.dumps({"shard":shard,"summary":summary},indent=2),flush=True)
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]={"type":type(e).__name__,"message":str(e),
                         "traceback":traceback.format_exc()[-4500:]}
        save_json(dest,report)
        raise

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--fresh",required=True)
    a.add_argument("--original",required=True)
    a.add_argument("--prefixes",required=True)
    a.add_argument("--out",required=True)
    a.add_argument("--shard",type=int,required=True)
    p=a.parse_args()
    run(p.fresh,p.original,p.prefixes,p.out,p.shard)
