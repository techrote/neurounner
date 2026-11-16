"""Research-only CPU-trained residual action adapter on frozen d1 System One output.

This learns 55 scoring parameters from source-pinned *real* sparse-model
probabilities and local W0 expert targets; the 2B GGUF remains unmodified.
It is not a LoRA, a fine-tuned model, or CyberSand production policy.
"""
from __future__ import annotations
import hashlib
import json
import math
import random

ACTIONS=("STRAIGHT","LEFT","RIGHT","BRANCH","RETRACE","STOP")
DIR=((1,0),(0,1),(-1,0),(0,-1))
FEATURES_PER_ACTION=9
N_FEATURES=55
EPOCHS=80
TRAIN_SEED=1701
PRIOR_SCALE=0.45
OVERRIDE_MARGIN=0.30

def canonical(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()

def digest(data):
    return hashlib.sha256(canonical(data)).hexdigest()

def menu(obs):
    names=obs["legal"]
    if not isinstance(names,list) or not names or len(set(names))!=len(names) or any(a not in ACTIONS for a in names):
        raise ValueError("Malformed host-legal action menu")
    return names

def probs(raw,legal):
    if set(raw)!=set(legal):raise ValueError("Probability support differs from host menu")
    p={a:float(raw[a]) for a in legal}
    if any(not math.isfinite(x) or x<0 or x>1 for x in p.values()) or sum(p.values())<=0:
        raise ValueError("Invalid model probability distribution")
    z=sum(p.values())
    return {a:v/z for a,v in p.items()}

def neighbours(obs):
    cells=obs["local_5x5"]
    if len(cells)!=5 or any(len(row)!=5 for row in cells) or cells[2][2]!="@":
        raise ValueError("Malformed W0 5x5")
    h=obs["heading"]
    if type(h) is not int or h not in range(4):raise ValueError("Malformed W0 heading")
    return {a:cells[2+DIR[(h+r)%4][1]][2+DIR[(h+r)%4][0]]
            for a,r in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1))}

def novel_moves(obs):
    legal=menu(obs);n=neighbours(obs)
    result={a for a in ("STRAIGHT","LEFT","RIGHT") if a in legal and n[a]=="."}
    if "BRANCH" in legal and len(result)>=2:result.add("BRANCH")
    return result

def target(obs,model_action,p):
    legal=menu(obs)
    if model_action not in legal:raise ValueError("Non-legal model decision in train data")
    novel=novel_moves(obs)
    if model_action in novel or (model_action=="RETRACE" and not novel):return model_action
    if novel:return max(novel,key=lambda a:(p[a],-ACTIONS.index(a)))
    if "RETRACE" in legal:return "RETRACE"
    return max((a for a in legal if a!="STOP"),key=lambda a:p[a],default="STOP")

def features(obs,a,proposal,p):
    legal=menu(obs)
    if a not in legal or proposal not in legal:raise ValueError("Scoring illegal action")
    n=neighbours(obs);novel=novel_moves(obs);last=obs["last_actions"][-4:]
    b=ACTIONS.index(a)*FEATURES_PER_ACTION
    vals=(1.,float(a in novel),float(a in n and n[a]=="+"),
          float(a==proposal),last.count(a)/4.,len(novel)/4.,
          sum(r.count("#") for r in obs["local_5x5"])/24.,
          min(max(float(obs["tick"])/36.,0.),1.),
          float(a=="RETRACE" and not novel))
    out={b+i:v for i,v in enumerate(vals) if v}
    out[54]=max(-12.,math.log(max(1e-7,p[a])))/12.
    return out

def logits(obs,proposal,p,w):
    if len(w)!=N_FEATURES or any(not math.isfinite(v) or abs(v)>=1000 for v in w):
        raise ValueError("Malformed trained weights")
    return {a:PRIOR_SCALE*math.log(max(1e-7,p[a]))+
            sum(w[j]*v for j,v in features(obs,a,proposal,p).items())
            for a in menu(obs)}

def predict(obs,proposal,raw_probs,ckpt):
    if ckpt.get("schema")!="frankenlayers-frozen-d1-residual/1":
        raise ValueError("Untrusted residual model schema")
    p=probs(raw_probs,menu(obs))
    if proposal not in p:raise ValueError("Model proposal was not host legal")
    s=logits(obs,proposal,p,ckpt["weights"])
    best=max(s,key=lambda a:(s[a],-ACTIONS.index(a)))
    choice=best if s[best]>=s[proposal]+OVERRIDE_MARGIN else proposal
    assert choice in menu(obs)
    return choice,{"base":proposal,"adapted":choice,"overrode":choice!=proposal,
                   "base_logit":s[proposal],"adapted_logit":s[choice]}

def train(records):
    if not records:raise ValueError("No actual model training records")
    cases=[]
    for record in records:
        obs,raw=record["observation"],record["choice"]
        legal=menu(obs);p=probs(record["probabilities"],legal)
        cases.append((obs,raw,p,target(obs,raw,p)))
        if "STOP" in legal and len(legal)>1:
            # Synthetic corrupted-model response on an ACTUAL observed state.
            # Explicitly count this as augmentation, not another model inference.
            q={a:(0.6 if a=="STOP" else 0.4*p[a]/max(1e-8,1-p["STOP"])) for a in legal}
            q=probs(q,legal)
            cases.append((obs,"STOP",q,target(obs,"STOP",q)))
    w=[0.]*N_FEATURES
    rng=random.Random(TRAIN_SEED);order=list(range(len(cases)))
    for epoch in range(EPOCHS):
        rng.shuffle(order);rate=0.065/(1+epoch/16)
        for i in order:
            obs,proposal,p,gold=cases[i]
            scores=logits(obs,proposal,p,w);top=max(scores.values())
            ez={a:math.exp(v-top) for a,v in scores.items()};z=sum(ez.values())
            grad=[0.]*N_FEATURES
            for a,e in ez.items():
                d=e/z-float(a==gold)
                for j,v in features(obs,a,proposal,p).items():grad[j]+=d*v
            for j in range(N_FEATURES):w[j]-=rate*(grad[j]+0.0008*w[j])
    ckpt={"schema":"frankenlayers-frozen-d1-residual/1",
          "architecture":"55-parameter linear action-logit residual, not LoRA",
          "train_seed":TRAIN_SEED,"epochs":EPOCHS,"source_records":len(records),
          "augmented_cases":len(cases)-len(records),
          "training_data_sha256":digest(records),
          "weights":[round(x,10) for x in w],
          "base_systemone_frozen":True,
          "objective":"local novelty, RETRACE recovery and proposed-action retention",
          "override_margin":OVERRIDE_MARGIN}
    ckpt["weights_sha256"]=digest(ckpt["weights"])
    ckpt["train_accuracy"]=sum(predict(obs,proposal,p,ckpt)[0]==gold for obs,proposal,p,gold in cases)/len(cases)
    return ckpt
