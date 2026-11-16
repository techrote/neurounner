#!/usr/bin/env python3
"""W11: actual PyTorch W9/W10 checkpoint sampling; synthetic W0, not native.

No new model training or root forwards: the preceding W10 artifact contains both.
Host retains action legality, seeded draws and stopping authority.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, resource, sys, time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/"training"),str(HERE)]
import worm_v4 as objective
import worm_train_v9 as w9
import worm_train_v10 as w10
from common import profile,sha256_file,write_json
from numerics import configure_cpu

VERSION="sprocket-W11-head-sampling-v1"
W9_SHA="5c09762f77580064b23a49b1daefef1e181a69d1318b1cfbaa8ca071ee2e75d4"
W10_SHA="f984965d07c6774a5522269d7db9d61637427251c7871a8e37ebf4d67dc15d9f"
SEEDS=(1901,1913,1931)
TEMPS={"w9-greedy":0.,"w10-greedy":0.,"w10-t035":.35,"w10-t07":.7,"w10-t14":1.4}
ACTIONS=tuple(w9.ACTIONS)

def draw(seed,family,tick):
    return int.from_bytes(hashlib.sha256(f"{seed}/{family}/{tick}/w11-CRN".encode()).digest()[:8],"big")/2**64

def choose(scores, legal, temperature, unit):
    if len(scores)!=6 or not legal or set(legal)-set(ACTIONS):
        raise ValueError("invalid scores or host-legal actions")
    if not 0.<=unit<1. or temperature<0 or not math.isfinite(temperature):
        raise ValueError("invalid draw or temperature")
    if any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("nonfinite logits")
    if temperature==0:
        return max(legal,key=lambda a:scores[ACTIONS.index(a)])
    values=[float(scores[ACTIONS.index(a)])/temperature for a in legal]
    m=max(values)
    weights=[math.exp(x-m) for x in values]
    z=sum(weights); cumulative=0.
    for a,w in zip(legal,weights):
        cumulative+=w/z
        if unit<cumulative:return a
    return legal[-1]

def load(torch,path,sha,version):
    if sha256_file(path)!=sha: raise ValueError("original checkpoint SHA mismatch")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("version")!=version: raise ValueError("checkpoint version mismatch")
    model=w9.model_class(torch)()
    model.load_state_dict(payload["head"],strict=True)
    model.eval()
    return model

def probe(torch,w0,model,seed,family,mode):
    temp=TEMPS[mode];decisions=[]
    def selector(obs):
        with torch.no_grad():
            scores=model(torch.tensor([w9.features(obs)],dtype=torch.float32))[0].tolist()
        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=124]
        d=draw(seed,family,obs["tick"])
        action=choose(scores,legal,temp,d)
        decisions.append({"tick":obs["tick"],"raw_six_scores":scores,"legal":legal,
                          "host_draw":d,"temperature":temp,"selected":action,
                          "real_head_forward":True,"root_forward":False})
        return action
    rec=w0.run(seed,"callback",family,128,selector=selector)
    assert len(rec["events"])==len(decisions)
    assert all(e["raw_legal"] and e["applied"]==d["selected"]
               for e,d in zip(rec["events"],decisions))
    rec["mode"]=mode
    rec["head_decisions"]=decisions
    rec["engine"]="pinned-W0-PyTorch-head-not-native-CyberSand"
    return rec

def main(args):
    started=time.monotonic()
    p=profile("sprocket",require_supported=True)
    assert p["model_id"]=="kandivault/sprocket-500m"
    assert p["revision"]=="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert p["subfolder"]=="" and p["framing"]["eos_token_id"]==5
    assert sha256_file(args.w0)=="81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
    assert sha256_file(args.support)=="c555c55ea4fc4bc303159fd3268cef4a0b7d9e5c006e19918ea3433d1f7c95d6"
    assert not set(SEEDS)&(set(w9.TRAIN_SEEDS)|set(w9.EVAL_SEEDS)|
                            set(w10.TRAIN_SEEDS)|set(w10.HELDOUT))
    support=objective.load_support(args.support,args.w0)
    fixture=support.synthetic_fixture("greebles","junction")
    assert len(w9.features(fixture))==142
    assert choose([1,2,3,4,5,6],["LEFT","RIGHT"],0.,.5)=="RIGHT"
    assert choose([1,2,3,4,5,6],["LEFT"],1.,.5)=="LEFT"
    assert draw(1901,"greebles",4)!=draw(1901,"greebles",5)
    for u in (-.1,1.):
        try:choose([0]*6,["STRAIGHT"],.7,u)
        except ValueError:pass
        else:raise AssertionError("unsafe draw admitted")
    if args.smoke_only:
        print("W11 source/host legality/source pins PASS (not model training)",flush=True)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    configure_cpu(1)
    import torch
    torch.set_num_threads(1)
    old=load(torch,args.w9,W9_SHA,w9.FEATURE_VERSION)
    new=load(torch,args.w10,W10_SHA,w10.VERSION)
    models={"w9-greedy":old,"w10-greedy":new,"w10-t035":new,
            "w10-t07":new,"w10-t14":new}
    provenance={"source_sha":os.getenv("GITHUB_SHA","local"),
      "version":VERSION,"model_root":p,"W9_original_checkpoint_sha256":W9_SHA,
      "W10_original_checkpoint_sha256":W10_SHA,"W10_parent_run":43806919532,
      "W0_sha256":sha256_file(args.w0),"support_sha256":sha256_file(args.support),
      "script_sha256":sha256_file(__file__),"seed_disjoint":True,
      "root_forwards":0,"optimizer_steps":0,"NOT_native":True,"sealed_FINAL":False}
    write_json(args.output/"provenance.json",provenance)
    rows=[];folder=args.output/"png-and-json"
    for seed in SEEDS:
        for family in w9.FAMILIES:
            for name,model in models.items():
                rec=probe(torch,support.worm,model,seed,family,name)
                support.worm.write_result(rec,folder)
                rows.append({"seed":seed,"family":family,"mode":name,"temperature":TEMPS[name],
                   "metrics":rec["metrics"],"real_head_forwards":len(rec["head_decisions"]),
                   "raw_top":dict(Counter(ACTIONS[max(range(6),key=lambda i:d["raw_six_scores"][i])]
                         for d in rec["head_decisions"])),
                   "selected":dict(Counter(d["selected"] for d in rec["head_decisions"]))})
            for name in ("random","wall","fidget"):
                rec=support.worm.run(seed,name,family,128)
                support.worm.write_result(rec,folder)
                rows.append({"seed":seed,"family":family,"mode":name,"metrics":rec["metrics"],
                             "real_head_forwards":0,"control":True})
        write_json(args.output/"partial-results.json",rows)
    total=sum(r["real_head_forwards"] for r in rows)
    assert len(rows)==72 and 0 < total <= 5760
    write_json(args.output/"summary.json",{"status":"complete","version":VERSION,
       "original_pairs":72,"actual_head_forwards":total,"head_forward_capacity":5760,"root_forwards":0,"optimizer_steps":0,
       "results":rows,"wall_seconds":time.monotonic()-started,
       "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "synthetic_host":True,"native":False})
    print(json.dumps({"W11":"complete","pairs":len(rows),"head_forwards":total,
                      "optimizer_steps":0,"root_forwards":0}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("w0","support","w9","w10","output"):
        ap.add_argument("--"+n,type=Path,required=n in ("w0","support"))
    ap.add_argument("--smoke-only",action="store_true")
    a=ap.parse_args()
    try:main(a)
    except BaseException as e:
        if a.output and a.output.exists():
            write_json(a.output/"failure.json",{"failed":True,"error":str(e),"type":type(e).__name__})
        raise
