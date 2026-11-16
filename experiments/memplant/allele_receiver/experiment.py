"""MEMPLANT source-latent Counterfactual Allele -> recipient-native prefix readback.

No model training. Learn mean additive heat/vibration deltas only from DG002
training families, modify already-computed frozen Qwen V feature tensors from
held-out worlds (no text re-encoding), and recompile through the original
trained frozen 2-bit ridge + four fixed frozen SmolLM2-1.7B soft prefixes.

This is source-FEATURE surgery, not in-place recipient native KV surgery.
Previously inspected DG002 sealed worlds: exploratory, not confirmatory.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import torch

from experiments.memplant.dreamgraft002.source import RECEIVER, RECEIVER_REV, save_json
from experiments.memplant.dreamgraft002.evaluate import (
    load_inputs, batch_four_choices, predicted, label_accuracy, SCORE_SLOPE)
from experiments.memplant.dreamgraft002.prefixes import QUESTIONS
from experiments.memplant.dreamgraft001.bridge import format_prompt
from experiments.memplant.dreamgraft001.g0_live import load_pinned, measure_rss

REGULARISATION=1.0
EXPECTED_DONOR_V_SHA256="473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75"
EXPECTED_B_PREFIX_SHA256="89fdb9c2edf8f2d3fffba6a124e6bcb62177229a4ee5d70522a17ed02eba8cc4"


def fit_train_only_ridge(v,rows):
    train=[i for i,r in enumerate(rows) if r["split"]=="train"]
    xx=(v-v.mean(dim=1,keepdim=True))/(v.std(dim=1,keepdim=True,unbiased=False)+1e-6)
    bank=xx[train]
    truth=torch.tensor([
        [1 if r["heat"]=="hardens" else -1,
         1 if r["vibration"]=="floats" else -1] for r in rows],dtype=torch.float32)
    alpha=torch.linalg.solve(bank@bank.T/2048+
                             REGULARISATION*torch.eye(len(train)),truth[train])
    def decode(feature):
        x=(feature-feature.mean(dim=1,keepdim=True))/(feature.std(dim=1,keepdim=True,unbiased=False)+1e-6)
        return (x@bank.T)/2048@alpha
    return decode


def learned_deltas(v,rows):
    byid={r["case_id"]:i for i,r in enumerate(rows)}
    heat=[];vibration=[]
    for i,r in enumerate(rows):
        if r["split"]!="train":
            continue
        if r["heat"]=="hardens":
            heat.append(v[byid[r["heat_twin"]]]-v[i])
        if r["vibration"]=="floats":
            vibration.append(v[byid[r["vibe_twin"]]]-v[i])
    if len(heat)!=40 or len(vibration)!=40:
        raise RuntimeError("training-only mutation pairs incomplete")
    return torch.stack(heat).mean(0),torch.stack(vibration).mean(0)


def make_prefix(prototypes,score):
    a=float(torch.sigmoid(score[0]*SCORE_SLOPE))
    b=float(torch.sigmoid(score[1]*SCORE_SLOPE))
    weights=(a*b,a*(1-b),(1-a)*b,(1-a)*(1-b))
    return sum(prototypes[f"class_{i}"]*weights[i] for i in range(4)).contiguous()


def plan(rows,features,heat_delta,vib_delta,idx,condition):
    row=rows[idx]
    byid={r["case_id"]:i for i,r in enumerate(rows)}
    if condition=="original":
        return features[idx],idx
    if condition=="heat_allele":
        sign=1 if row["heat"]=="hardens" else -1
        return features[idx]+sign*heat_delta,byid[row["heat_twin"]]
    if condition=="vibration_allele":
        sign=1 if row["vibration"]=="floats" else -1
        return features[idx]+sign*vib_delta,byid[row["vibe_twin"]]
    if condition=="reencode_heat":
        ref=byid[row["heat_twin"]]
        return features[ref],ref
    if condition=="reencode_vibration":
        ref=byid[row["vibe_twin"]]
        return features[ref],ref
    if condition=="random_heat":
        sign=1 if row["heat"]=="hardens" else -1
        g=torch.Generator().manual_seed(3700+idx)
        x=torch.randn(heat_delta.shape,generator=g)
        x=x*(heat_delta.norm()/(x.norm()+1e-8))
        return features[idx]+sign*x,byid[row["heat_twin"]]
    raise ValueError("unknown plan condition")


def run(source,checkpoint,out):
    root=Path(out);root.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    report={"schema":"memplant-recipient-allele-v1",
            "status":"STARTED","new_training_updates":0,
            "interpretation":"source-state arithmetic recomputed as B-native prefix; not live recipient KV editing",
            "prior_source_test_exposed":True,
            "rows":[]}
    try:
        source_manifest,features,pm,prototypes=load_inputs(source,checkpoint)
        if source_manifest["channels"]["v"]["sha256"]!=EXPECTED_DONOR_V_SHA256:
            raise RuntimeError("source V hashes changed")
        if pm["prototypes_sha256"]!=EXPECTED_B_PREFIX_SHA256:
            raise RuntimeError("frozen B-native carriers changed")
        rows=source_manifest["cases"]
        v=features["v"]
        decode=fit_train_only_ridge(v,rows)
        delta_heat,delta_vib=learned_deltas(v,rows)
        model,tokenizer,identity=load_pinned(RECEIVER,RECEIVER_REV)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        prompt=format_prompt(tokenizer,QUESTIONS[0])
        # Three entire previously sealed source families; each has all four laws.
        sealed=[i for i,r in enumerate(rows) if r["split"]=="sealed"]
        families=sorted({rows[i]["family"] for i in sealed})[:3]
        ids=[i for i in sealed if rows[i]["family"] in families]
        report["sample"]={"families":families,"cases":len(ids),
                          "previously_inspected":True}
        report["provenance"]={"source_hash":EXPECTED_DONOR_V_SHA256,
                              "prefix_hash":EXPECTED_B_PREFIX_SHA256,
                              "receiver_revision":identity["revision"]}
        conditions=("original","heat_allele","reencode_heat",
                    "vibration_allele","reencode_vibration","random_heat")
        start=time.monotonic()
        for n,i in enumerate(ids):
            source_case=rows[i]
            for cond in conditions:
                state,target_idx=plan(rows,v,delta_heat,delta_vib,i,cond)
                score=decode(state.unsqueeze(0))[0]
                prefix=make_prefix(prototypes,score)
                raw=batch_four_choices(model,tokenizer,prompt,prefix)
                choice=predicted(raw)
                flags=label_accuracy(choice,rows[target_idx]["answer_class"])
                original_flags=label_accuracy(choice,source_case["answer_class"])
                report["rows"].append({
                    "case":source_case["case_id"],"condition":cond,
                    "presented_source_id":rows[target_idx]["case_id"],
                    "predicted_class":choice,"correct_target":flags,
                    "correct_original":original_flags,
                    "signed_ridge_outputs":[round(float(x),6) for x in score],
                    "answer_log_probs":[round(float(x),5) for x in raw]})
            # Emit partial evidence for bounded CI and failure salvage.
            report["completed_cases"]=n+1
            save_json(root,report)
            if (n+1)%4==0:
                print("recipient allele readback:",n+1,"/",len(ids),flush=True)
        by={}
        for cond in conditions:
            part=[r for r in report["rows"] if r["condition"]==cond]
            by[cond]={
                "n":len(part),
                "heat_correct":sum(r["correct_target"]["heat"] for r in part),
                "vibration_correct":sum(r["correct_target"]["vibration"] for r in part),
                "both_correct":sum(r["correct_target"]["joint"] for r in part),
            }
        look={(r["case"],r["condition"]):r for r in report["rows"]}
        for edited,axis in (("heat_allele","heat"),("vibration_allele","vibration")):
            selected=[]
            for i in ids:
                x=look[(rows[i]["case_id"],"original")]
                y=look[(rows[i]["case_id"],edited)]
                changed=x["predicted_class"]//2!=y["predicted_class"]//2 if axis=="heat" else x["predicted_class"]%2!=y["predicted_class"]%2
                preserved=x["predicted_class"]%2==y["predicted_class"]%2 if axis=="heat" else x["predicted_class"]//2==y["predicted_class"]//2
                selected.append(changed and preserved)
            by[edited]["selective_prediction_flips"]=sum(selected)
        report["summary"]=by
        report["elapsed_seconds"]=round(time.monotonic()-start,3)
        report["rss_mib"]=measure_rss()
        report["status"]="COMPLETED_EXPLORATORY"
        save_json(root,report)
        print(json.dumps(by,indent=2),flush=True)
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]=type(e).__name__+": "+str(e)
        save_json(root,report)
        raise


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="runtime/memplant/allele-b/source")
    ap.add_argument("--prefixes",default="runtime/memplant/allele-b/prefixes")
    ap.add_argument("--out",default="runtime/memplant/allele-b/report.json")
    a=ap.parse_args()
    run(a.source,a.prefixes,a.out)
