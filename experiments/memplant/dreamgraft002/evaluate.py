"""DREAMGRAFT-002: frozen two-rule source readout and recipient test.

A's real V / hidden / lexical features -> independent supervised two-bit ridge
-> four B-native learned prefixes combined continuously. Model weights and
prototypes are frozen during tests. This is not native KV transplantation.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file
from experiments.memplant.dreamgraft001.bridge import format_prompt,generation_from_prefix
from experiments.memplant.dreamgraft001.g0_live import load_pinned,measure_rss
from experiments.memplant.dreamgraft002.source import (
    VERSION,RECEIVER,RECEIVER_REV,HEAT,VIBRATION,PAIR_ANSWERS,
    make_cases,hash_bytes,save_json)
from experiments.memplant.dreamgraft002.prefixes import QUESTIONS,CREATIVE,INTERACTION_QUESTION

REGULARISATION=1.0
SCORE_SLOPE=15.0
CHANNELS=("v","hidden","embedding")

def load_inputs(src,pref):
    src=Path(src)
    m=json.loads((src/"manifest.json").read_text())
    if m["schema"]!=VERSION or m["cases"]!=make_cases():
        raise RuntimeError("source corpus/schema does not match immutable test")
    p=src/"features.npz"
    if sha256(p.read_bytes()).hexdigest()!=m["feature_container_hash"]:
        raise RuntimeError("donor source feature file hash mismatch")
    with np.load(p,allow_pickle=False) as z:
        data={k:torch.from_numpy(np.array(z[k],dtype="float32")).contiguous()
              for k in CHANNELS}
    for k,x in data.items():
        if x.shape!=(128,2048) or hash_bytes(x.numpy().tobytes())!=m["channels"][k]["sha256"]:
            raise RuntimeError("invalid donor feature tensor: "+k)
    pp=Path(pref)
    pm=json.loads((pp/"report.json").read_text())
    if pm.get("status")!="COMPLETED" or pm.get("model_revision")!=RECEIVER_REV:
        raise RuntimeError("prefix carrier not qualified for frozen recipient")
    pb=pp/"prototypes.safetensors"
    if sha256(pb.read_bytes()).hexdigest()!=pm["prototypes_sha256"]:
        raise RuntimeError("prefix payload digest mismatch")
    prefixes=load_file(str(pb),device="cpu")
    if set(prefixes)!={f"class_{i}" for i in range(4)}:
        raise RuntimeError("wrong prefix checkpoint keys")
    assert all(x.shape==(1,8,2048) for x in prefixes.values())
    return m,data,pm,prefixes

def two_bit_ridge(features,cases):
    if features.shape!=(128,2048):raise ValueError("unexpected feature geometry")
    xs=(features-features.mean(dim=1,keepdim=True))/(features.std(dim=1,keepdim=True,unbiased=False)+1e-6)
    train=[i for i,x in enumerate(cases) if x["split"]=="train"]
    # Heat: +1 for hardens. Vibration: +1 for floats.
    Y=torch.tensor([[1 if x["heat"]=="hardens" else -1,
                     1 if x["vibration"]=="floats" else -1]
                    for x in cases],dtype=torch.float32)
    tr=xs[train]
    gram=tr@tr.T/2048
    alpha=torch.linalg.solve(gram+REGULARISATION*torch.eye(len(train)),Y[train])
    raw=(xs@tr.T)/2048@alpha
    report={}
    for split in ("train","validation","sealed"):
        ids=[i for i,x in enumerate(cases) if x["split"]==split]
        report[split]={
            "n":len(ids),
            "heat_correct":sum(bool((raw[i,0]>=0)==(Y[i,0]>0)) for i in ids),
            "vibration_correct":sum(bool((raw[i,1]>=0)==(Y[i,1]>0)) for i in ids),
            "both_correct":sum(bool(((raw[i]>=0)==(Y[i]>0)).all()) for i in ids),
            "distinct_families":len({cases[i]["family"] for i in ids})}
    return raw,report

def continuous_prefix(prefixes,scores,i,cases,condition,seed):
    index={x["case_id"]:j for j,x in enumerate(cases)}
    src=i
    if condition=="heat_swap":src=index[cases[i]["heat_twin"]]
    elif condition=="vibration_swap":src=index[cases[i]["vibe_twin"]]
    elif condition=="both_swap":src=index[cases[i]["both_twin"]]
    elif condition in ("genuine","hard_gate","zero","random","equal_blend"):pass
    else:raise ValueError("bad numeric condition "+condition)
    z=scores[src]
    p0,p1=(float(torch.sigmoid(z[k]*SCORE_SLOPE)) for k in (0,1))
    if condition=="hard_gate":
        p0=float(z[0]>=0);p1=float(z[1]>=0)
    if condition=="equal_blend":p0=p1=0.5
    # Four weights sum to 1; bilinear composition is explicitly supervised.
    weights=((p0*p1),(p0*(1-p1)),((1-p0)*p1),((1-p0)*(1-p1)))
    combo=sum(prefixes[f"class_{k}"]*weights[k] for k in range(4))
    if condition=="zero":combo=torch.zeros_like(combo)
    elif condition=="random":
        gen=torch.Generator(device="cpu").manual_seed(seed)
        noise=torch.randn(combo.shape,generator=gen)
        combo=noise*(combo.norm()/(noise.norm()+1e-8))
    return combo.contiguous(),src,{"p_heat":p0,"p_vibration":p1}

def batch_four_choices(receiver,tok,prompt,prototypes_prefix):
    """One batched forward for four different-length candidate continuations.

    Uses TOTAL log-likelihood of each complete phrase; no mean-per-token
    shortcut that would bias toward different-length phrase candidates.
    """
    K=len(PAIR_ANSWERS)
    assert K==4 and prompt.shape[0]==1
    pieces=[tok(txt,add_special_tokens=False,return_tensors="pt")["input_ids"][0] for txt in PAIR_ANSWERS]
    if any(len(ids)<1 for ids in pieces):raise RuntimeError("empty candidate answer")
    maxlen=max(len(ids) for ids in pieces)
    pad=int(tok.eos_token_id)
    tokens=torch.full((K,maxlen),pad,dtype=torch.long)
    mask=torch.zeros(K,maxlen,dtype=torch.long)
    for k,ids in enumerate(pieces):
        tokens[k,:len(ids)]=ids
        mask[k,:len(ids)]=1
    with torch.no_grad():
        emb=receiver.get_input_embeddings()
        prompt_emb=emb(prompt).expand(K,-1,-1)
        answer_emb=emb(tokens)
        slot=0
        if prototypes_prefix is None:
            inputs=torch.cat([prompt_emb,answer_emb],dim=1)
            prefix_mask=torch.ones(K,prompt.shape[1],dtype=torch.long)
        else:
            slot=prototypes_prefix.shape[1]
            assert prototypes_prefix.shape==(1,8,2048)
            inputs=torch.cat([prototypes_prefix.expand(K,-1,-1),prompt_emb,answer_emb],dim=1)
            prefix_mask=torch.ones(K,slot+prompt.shape[1],dtype=torch.long)
        attn=torch.cat([prefix_mask,mask],dim=1)
        logits=receiver(inputs_embeds=inputs,attention_mask=attn,
                        use_cache=False,return_dict=True).logits
        base=slot+prompt.shape[1]
        score=[]
        for k,ids in enumerate(pieces):
            # answer token t is predicted at sequence logit index base+t-1.
            chosen=logits[k,base-1:base-1+len(ids)].float()
            logp=torch.log_softmax(chosen,dim=-1)
            score.append(float(logp.gather(-1,ids.view(-1,1)).sum().item()))
    return score

def predicted(scores):
    return int(max(range(4),key=lambda k:scores[k]))

def label_accuracy(cls,target):
    return dict(
        heat=HEAT[cls//2]==HEAT[target//2],
        vibration=VIBRATION[cls%2]==VIBRATION[target%2],
        joint=cls==target)

def score_method(receiver,tok,features,cases,prefixes,method,with_creative=False):
    scores,ridge=two_bit_ridge(features,cases)
    seals=[i for i,x in enumerate(cases) if x["split"]=="sealed"]
    idx={x["case_id"]:i for i,x in enumerate(cases)}
    prompt=format_prompt(tok,QUESTIONS[0])
    conditions=("genuine","heat_swap","vibration_swap","both_swap","hard_gate")
    if method=="v":conditions+=("zero","random","equal_blend","no_prefix","native_text")
    cases_results=[];started=time.monotonic()
    for n,i in enumerate(seals):
        row=cases[i]
        for condition in conditions:
            if condition=="no_prefix":
                pp=None;src=i;meta={}
                prompt_i=prompt
            elif condition=="native_text":
                pp=None;src=i;meta={}
                prompt_i=format_prompt(tok,"Source document:\n"+row["source"]+"\n\n"+QUESTIONS[0])
            else:
                pp,src,meta=continuous_prefix(prefixes,scores,i,cases,condition,seed=2201+i)
                prompt_i=prompt
            result=batch_four_choices(receiver,tok,prompt_i,pp)
            pred=predicted(result)
            actual=cases[src]["answer_class"]
            flags=label_accuracy(pred,actual)
            orig=label_accuracy(pred,row["answer_class"])
            cases_results.append(dict(
                case_id=row["case_id"],family=row["family"],method=method,
                condition=condition,source_from=cases[src]["case_id"],
                target_class=actual,original_class=row["answer_class"],
                predicted_class=pred,candidate_log_likelihoods=[round(x,5) for x in result],
                correct=flags,correct_original=orig,
                source_text_in_receiver=(condition=="native_text"),
                interp_metadata=meta))
        if (n+1)%6==0:
            print(f"DG002 {method} recipient scoring {n+1}/{len(seals)}",flush=True)
    by={}
    for cond in conditions:
        group=[x for x in cases_results if x["condition"]==cond]
        by[cond]=dict(
            n=len(group),families=len({x["family"] for x in group}),
            heat_correct=sum(x["correct"]["heat"] for x in group),
            vibration_correct=sum(x["correct"]["vibration"] for x in group),
            joint_correct=sum(x["correct"]["joint"] for x in group),
            joint_correct_original=sum(x["correct_original"]["joint"] for x in group))
    pair_results={}
    lookup={(x["case_id"],x["condition"]):x for x in cases_results}
    for cond,changed in (("heat_swap","heat"),("vibration_swap","vibration"),("both_swap","both")):
        expected=[];flips=[]
        for i in seals:
            name=cases[i]["case_id"]
            a=lookup[(name,"genuine")]
            b=lookup[(name,cond)]
            ph=a["predicted_class"]//2!=b["predicted_class"]//2
            pv=a["predicted_class"]%2!=b["predicted_class"]%2
            flips.append(dict(case=name,heat_changed=ph,vibration_changed=pv))
            expected.append(((ph and not pv) if changed=="heat" else
                             ((pv and not ph) if changed=="vibration" else (ph and pv))))
        pair_results[cond]=dict(
            n=len(flips),selective_flip_count=sum(expected),
            changed_heat_count=sum(x["heat_changed"] for x in flips),
            changed_vibration_count=sum(x["vibration_changed"] for x in flips),
            paired=flips)
    creative=[]
    if with_creative:
        # Exactly four sealed source states, one of each class. No editing.
        chosen=[next(i for i in seals if cases[i]["answer_class"]==cls)
                for cls in range(4)]
        for i in chosen:
            for task in CREATIVE:
                fixed_prompt=format_prompt(tok,task)
                for cond in ("genuine","heat_swap","zero"):
                    pp,src,_=continuous_prefix(prefixes,scores,i,cases,cond,seed=3301+i)
                    generated,ids=generation_from_prefix(receiver,tok,fixed_prompt,pp,max_tokens=32)
                    creative.append(dict(case=cases[i]["case_id"],condition=cond,
                        task=task,numeric_donor=cases[src]["case_id"],
                        raw=generated,generated_token_ids=ids))
    return dict(method=method,ridge=ridge,conditions=by,paired_edits=pair_results,
                raw_cases=cases_results,creative=dict(
                    n=len(creative),samples=creative,unscored=True,not_edited=True),
                elapsed_seconds=round(time.monotonic()-started,2),
                claims=dict(
                    two_rule_pseudo_prefix_channel_is_supervised=True,
                    direct_cross_family_KV_cache_translation=False,
                    natural_language_world_memory_proven=False))

def run(source,prefix,method,out):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    report={"schema":"memplant-dg002-eval-v1","method":method,"status":"STARTED",
            "scientific_claim":"experimental two-attribute supervised numeric channel, not memory transplant"}
    try:
        torch.set_num_threads(min(torch.get_num_threads(),4))
        manifest,data,proto_meta,prototypes=load_inputs(source,prefix)
        report["provenance"]=dict(donor=manifest["donor"],
            donor_features_sha=manifest["channels"][method]["sha256"],
            prefix_sha=proto_meta["prototypes_sha256"])
        model,tok,recipient=load_pinned(RECEIVER,RECEIVER_REV)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        report["recipient"]=recipient
        report["evaluation"]=score_method(model,tok,data[method],manifest["cases"],
                                            prototypes,method,with_creative=method=="v")
        report["rss_end_mib"]=measure_rss()
        report["status"]="COMPLETED_EXPLORATORY"
        save_json(root/"report.json",report)
        print(json.dumps({k:v for k,v in report["evaluation"].items()
                          if k in ("ridge","conditions","elapsed_seconds")},
                         indent=2),flush=True)
    except Exception as exc:
        report["status"]="BLOCKED"
        report["error"]=dict(type=type(exc).__name__,message=str(exc),
                              trace=traceback.format_exc()[-4500:])
        save_json(root/"report.json",report)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="runtime/memplant/dg002/source")
    p.add_argument("--prefix",default="runtime/memplant/dg002/prototypes")
    p.add_argument("--method",choices=CHANNELS,required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.source,a.prefix,a.method,a.out)
