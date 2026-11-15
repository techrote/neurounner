"""DREAMGRAFT G1: real frozen 1.7B receiver and matched donor-feature arms.

Each method gets the exact same adapter architecture, train case order, answer
objective, 48-update cap and receiver. No test-set tuning or G2/G3 claims.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from experiments.memplant.dreamgraft001.bridge import (
    DreamPrefix, answer_loss, format_prompt, generation_from_prefix)
from experiments.memplant.dreamgraft001.contract import write_json
from experiments.memplant.dreamgraft001.g0_live import load_pinned, measure_rss, target_ids
from .features import (
    QUESTION_SET, RECIPIENT, RECIPIENT_REV, SCHEMA, TRAIT, cases)


def binary_log_odds(model, prompts, answers, prefix):
    """Full conditional answer-token log-likelihood difference, H minus S.

    Token counts may differ. Mean NLL alone is unsuitable for comparing
    different-length verbal answers, so multiply by answer length.
    """
    with torch.no_grad():
        log_probs=[]
        for label in TRAIT:
            answer=answers[label]
            loss=answer_loss(model,prompts,answer,prefix=prefix)
            log_probs.append(-float(loss.item()) * int(answer.shape[1]))
    return log_probs[0]-log_probs[1]


def inferred_word(log_odds):
    return "hardens" if log_odds>=0 else "softens"


def compute_condition(adapter, feats, idx, control, *, twin_index, seed):
    with torch.no_grad():
        prefix=adapter(feats[idx:idx+1])
        if control=="genuine":
            return prefix
        if control=="swapped":
            return adapter(feats[twin_index:twin_index+1])
        if control=="zero":
            return torch.zeros_like(prefix)
        if control=="random":
            generator=torch.Generator(device="cpu").manual_seed(seed)
            noise=torch.randn(prefix.shape,generator=generator,dtype=prefix.dtype)
            return noise * (prefix.norm()/(noise.norm()+1e-12))
    raise ValueError(f"bad condition {control}")


def eval_split(receiver,tokenizer,adapter,feats,manifest,split,*,conditions):
    records=manifest["records"]
    indices=[i for i,r in enumerate(records) if r["split"]==split]
    idx_by_name={r["case_id"]:i for i,r in enumerate(records)}
    prompts=[format_prompt(tokenizer,q) for q in QUESTION_SET]
    answers={t:target_ids(tokenizer," "+t) for t in TRAIT}
    details=[]
    for idx in indices:
        item=records[idx]
        twin=idx_by_name[item["twin_id"]]
        for order in range(2):
            for c in conditions:
                target=records[twin]["answer"] if c=="swapped" else item["answer"]
                prefix=compute_condition(adapter,feats,idx,c,twin_index=twin,
                                         seed=5151+order*1000+idx)
                score=binary_log_odds(receiver,prompts[order],answers,prefix)
                pred=inferred_word(score)
                details.append({
                    "case_id":item["case_id"],"family":item["family"],
                    "order":order,"method":manifest["_method"],"condition":c,
                    "true_source":item["answer"],"target_source":target,
                    "original_truth_correct":pred==item["answer"],
                    "followed_presented_source":pred==target,
                    "predicted":pred,"log_odds_hardens_minus_softens":round(score,6)})
    by={}
    for c in conditions:
        rows=[x for x in details if x["condition"]==c]
        by[c]={
            "n":len(rows),
            "accuracy_vs_presented_source":round(sum(x["followed_presented_source"] for x in rows)/len(rows),4),
            "accuracy_vs_original_source":round(sum(x["original_truth_correct"] for x in rows)/len(rows),4),
            "mean_signed_target_margin":round(sum(
                (1 if x["target_source"]=="hardens" else -1)*x["log_odds_hardens_minus_softens"]
                for x in rows)/len(rows),6),
        }
    paired={}
    if "genuine" in conditions and "swapped" in conditions:
        pairs={(x["case_id"],x["order"],x["condition"]):x for x in details}
        flips=[]
        for idx in indices:
            for order in (0,1):
                a=pairs[records[idx]["case_id"],order,"genuine"]
                b=pairs[records[idx]["case_id"],order,"swapped"]
                flips.append(a["predicted"]!=b["predicted"])
        paired={"source_swap_prediction_flip_rate":round(sum(flips)/len(flips),4),
                "n_source_order_pairs":len(flips)}
    return {"split":split,"n_cases":len(indices),"n_families":len(set(records[i]["family"] for i in indices)),
            "conditions":by,"paired":paired,"cases":details}


def creative_postcards(receiver,tokenizer,adapter,feats,manifest):
    records=manifest["records"]
    test=[i for i,r in enumerate(records) if r["split"]=="sealed"][:2]
    idx_by_name={r["case_id"]:i for i,r in enumerate(records)}
    tasks=(
        "Invent an unusual fictional material governed by the world you inherited. Name it and describe how heating affects it.",
        "Imagine a small machine built from the inherited material. Explain what goes wrong when it is warmed.",
    )
    raw=[]
    for idx in test:
        twin=idx_by_name[records[idx]["twin_id"]]
        for task_no,task in enumerate(tasks):
            p=format_prompt(tokenizer,task)
            for c in ("genuine","swapped","zero"):
                pref=compute_condition(adapter,feats,idx,c,twin_index=twin,
                                       seed=5151+idx+1000*task_no)
                text,ids=generation_from_prefix(receiver,tokenizer,p,pref,max_tokens=48)
                raw.append({"case_id":records[idx]["case_id"],"task":task,
                            "task_index":task_no,"condition":c,
                            "raw_text":text,"token_ids":ids})
    return {"data_status":"unscored exploratory, not transplant evidence",
            "no_editor":True,"count":len(raw),"raw":raw}


def run(method, input_dir, output_dir, *, steps=48,max_fit_seconds=720):
    from safetensors.torch import save_file
    random.seed(615)
    np.random.seed(615)
    torch.manual_seed(615)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    if method not in ("v","hidden","embedding"):
        raise ValueError("unknown method")
    root=Path(input_dir)
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"manifest.json").read_text())
    feature_bytes=(root/"features.npz").read_bytes()
    if hashlib.sha256(feature_bytes).hexdigest()!=manifest["feature_file_sha256"]:
        raise RuntimeError("feature archive checksum mismatch")
    xs=cases()
    if ([c.record() for c in xs]!=manifest["records"] or manifest["schema"]!=SCHEMA):
        raise RuntimeError("wrong synthetic experiment or changed data manifest")
    with np.load(root/"features.npz",allow_pickle=False) as arrays:
        features={k:np.asarray(arrays[k],dtype=np.float32) for k in ("v","hidden","embedding")}
    for key,arr in features.items():
        if arr.shape!=(48,2048) or hashlib.sha256(arr.tobytes()).hexdigest()!=manifest["channels"][key]["features_sha256"]:
            raise RuntimeError("feature array metadata failed verification")
    feats=torch.from_numpy(features[method]).clone().contiguous()
    report={
        "schema":SCHEMA,"method":method,"status":"STARTED",
        "utc":datetime.now(timezone.utc).isoformat(),
        "source_provenance":{"donor_id":manifest["donor"]["id"],
                             "donor_revision":manifest["donor"]["revision"],
                             "donor_model_sha256":manifest["donor"]["weight_sha256"],
                             "donor_tokenizer_sha256":manifest["donor"]["tokenizer_sha256"],
                             "source_feature_hash":manifest["channels"][method]["features_sha256"]},
        "split_counts":{split:sum(x["split"]==split for x in manifest["records"])
                        for split in ("train","validation","sealed")},
        "fit_steps_requested":steps,
        "g1_exploratory_only":True,"g2_not_executed":True,
        "receiver_does_not_receive_source_text":True,
    }
    try:
        receiver,tokenizer,provenance=load_pinned(RECIPIENT,RECIPIENT_REV)
        report["recipient"]=provenance
        for p in receiver.parameters():p.requires_grad_(False)
        receiver.eval()
        assert int(receiver.config.hidden_size)==2048
        sigma=max(float(receiver.get_input_embeddings().weight.detach().std()),.01)
        adapter=DreamPrefix(2048,2048,slots=8,bottleneck=64,
                            scale=sigma*2.0).eval()
        report["adapter_parameter_count"]=sum(p.numel() for p in adapter.parameters())
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.003,weight_decay=0.005)
        prompts=[format_prompt(tokenizer,q) for q in QUESTION_SET]
        answers={label:target_ids(tokenizer," "+label) for label in TRAIT}
        report["answer_token_lengths"]={k:int(v.shape[1]) for k,v in answers.items()}
        training=[i for i,r in enumerate(manifest["records"]) if r["split"]=="train"]
        # Deterministic, identical example/option order and optimiser for all arms.
        indices=[]
        for epoch in range(math.ceil(steps/len(training))):
            batch=training.copy()
            random.Random(801+epoch).shuffle(batch)
            indices+=batch
        indices=indices[:steps]
        trace=[];fit_start=time.monotonic()
        for step,index in enumerate(indices):
            if time.monotonic()-fit_start>=max_fit_seconds:
                break
            opt_order=step%2
            optimizer.zero_grad(set_to_none=True)
            prefix=adapter(feats[index:index+1])
            target=manifest["records"][index]["answer"]
            loss=answer_loss(receiver,prompts[opt_order],answers[target],prefix)
            if not torch.isfinite(loss):raise RuntimeError("nonfinite training loss")
            loss.backward()
            grads=[p.grad for p in adapter.parameters()]
            if any(g is None or not torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError("missing/nonfinite adapter gradient")
            grad_norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.0))
            if not math.isfinite(grad_norm) or grad_norm<=0:
                raise RuntimeError("nonpositive/invalid gradient")
            optimizer.step()
            trace.append({"step":step+1,"case_id":manifest["records"][index]["case_id"],
                          "option_order":opt_order,"target":target,
                          "loss":round(float(loss.item()),6),
                          "gradient_norm_before_clip":round(grad_norm,6),
                          "elapsed_seconds":round(time.monotonic()-fit_start,2),
                          "rss_mib":measure_rss()})
        if not trace:raise RuntimeError("no updates executed")
        report["training"]={
            "updates":len(trace),"fit_seconds":round(time.monotonic()-fit_start,3),
            "max_fitting_seconds":max_fit_seconds,
            "training_target":"masked teacher-forced word CE; no KL distillation",
            "optimizer_lr":0.003,"adapter_scale":sigma*2.0,
            "peak_reported_rss_mib":max(x["rss_mib"] for x in trace),
            "trace":trace}
        for p in receiver.parameters():
            if p.grad is not None:raise RuntimeError("frozen B received gradient")
        # Tiny model-only checkpoint; test set not used for selection.
        save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
                  str(output/("adapter-"+method+".safetensors")))
        manifest["_method"]=method
        report["validation"]=eval_split(receiver,tokenizer,adapter,feats,manifest,
                                         "validation",conditions=("genuine","swapped","zero","random"))
        # Sealed set evaluated once, after fixed step recipe and checkpoint.
        report["sealed"]=eval_split(receiver,tokenizer,adapter,feats,manifest,
                                    "sealed",conditions=("genuine","swapped","zero","random"))
        if method=="v":
            report["creative_postcards"]=creative_postcards(receiver,tokenizer,adapter,feats,manifest)
        report["runtime"]={"end_rss_mib":measure_rss(),"torch":torch.__version__,
                           "total_fit_and_eval_seconds":round(time.monotonic()-fit_start,3)}
        report["status"]="COMPLETED_EXPLORATORY"
        report["claims"]={"g1_smoke_completed":True,
                          "memory_transplant_proven":False,
                          "kv_structural_advantage_proven":False,
                          "generalization_or_composition_proven":False}
        write_json(output/"report.json",report)
        print(json.dumps({"method":method,"updates":len(trace),
                          "validation":report["validation"]["conditions"]["genuine"],
                          "sealed":report["sealed"]["conditions"],
                          "swap":report["sealed"]["paired"],
                          "seconds":report["runtime"]["total_fit_and_eval_seconds"]},indent=2))
    except Exception as exc:
        report["status"]="BLOCKED"
        report["error"]={"type":type(exc).__name__,"message":str(exc),"trace":traceback.format_exc()[-5000:]}
        write_json(output/"report.json",report)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=("v","hidden","embedding"),required=True)
    p.add_argument("--features",default="runtime/memplant/g1/features")
    p.add_argument("--out",default="runtime/memplant/g1/fit")
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--max-fit-seconds",type=int,default=720)
    args=p.parse_args()
    run(args.method,args.features,args.out,steps=args.steps,max_fit_seconds=args.max_fit_seconds)


if __name__=="__main__":
    main()
