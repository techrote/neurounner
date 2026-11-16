#!/usr/bin/env python3
"""QL-008: learn a state-conditional gate between a Q1 worm and Q4-middle island.

Synthetic research only: immutable parent codes, external W0-like host, no game
modifications, GGUF IQ claims, or native CyberSand painting.
"""
import argparse
import json
import os
import random
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import ACTIONS, FAMILIES, WormWorld, draw, digest, timing
from ql002_branch_retrace import examples
from ql005_bitserial import verified_source
from ql006_activation_gates import GatedQ1
from ql007_mixed_bridges import Hybrid, PARENT_SHA


class Gate(nn.Module):
    """Only a 27-feature logistic mixing gate is learned; Q1 and Q4 codes freeze."""
    def __init__(self, first, second, weights=None, bias=0.):
        super().__init__()
        self.first, self.second = first, second
        self.weights = nn.Parameter(torch.zeros(27) if weights is None
                                    else torch.as_tensor(weights, dtype=torch.float32).clone())
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def mixture(self, features, a, b):
        alpha = torch.sigmoid(features @ self.weights + self.bias)[:, None]
        return a * (1. - alpha) + b * alpha

    def forward(self, x):
        with torch.no_grad():
            a, b = self.first(x), self.second(x)
        return self.mixture(x, a, b)

    def alpha(self, x):
        return torch.sigmoid(x @ self.weights + self.bias)


class ConstantBlend:
    def __init__(self, a, b, weight):
        self.a, self.b, self.weight = a, b, weight
    def __call__(self, x):
        return (1.-self.weight)*self.a(x)+self.weight*self.b(x)


def train_gate(a, b, data, steps, kl_weight, seed, cached):
    x, y, legal = data
    aa, bb = cached
    torch.manual_seed(seed)
    net = Gate(a, b)
    optim = torch.optim.AdamW(net.parameters(), lr=.06, weight_decay=0.)
    count = torch.bincount(y, minlength=len(ACTIONS)).float()
    probability = (1. / count[y]).double()
    probability /= probability.sum()
    trace = []
    for step in range(steps):
        ix = torch.multinomial(probability, 128, replacement=True)
        scores = net.mixture(x[ix], aa[ix], bb[ix]).masked_fill(~legal[ix], -1e9)
        target_prior = bb[ix].masked_fill(~legal[ix], -1e9)
        ce = F.cross_entropy(scores, y[ix])
        kl = F.kl_div(F.log_softmax(scores,dim=1), F.softmax(target_prior,dim=1),
                      reduction="batchmean")
        loss = ce + kl_weight * kl
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        trace.append({"step":step+1,"loss":float(loss.detach()),
                      "ce":float(ce.detach()),"kl":float(kl.detach())})
    moments = sum(int(torch.count_nonzero(v["exp_avg"]).item()>0)
                  for v in optim.state.values())
    assert moments == 2, moments
    summary = {"optimizer_updates":steps,"kl_weight":kl_weight,"seed":seed,
               "nonzero_adam_moment_tensors":moments,"first":trace[0],"last":trace[-1],
               "weights":net.weights.detach().tolist(),"bias":float(net.bias.detach()),
               "frozen_original_weight_codes":True}
    return net, optim.state_dict(), trace, summary


def episode(model, name, seed, family, stress, root, horizon):
    w = WormWorld(seed, family)
    if stress:
        # Held-out synthetic hazard shift: modifies only the EXTERNAL world.
        # Existing protected/solid cells stay intact; leave the spawn chamber clear.
        r = random.Random(seed*809+41)
        k = w.grid.shape[0]//2
        for y in range(w.grid.shape[0]):
            for x in range(w.grid.shape[1]):
                if (abs(x-k)<=2 and abs(y-k)<=2) or w.grid[y,x]:
                    continue
                v = r.random()
                if v < .14: w.grid[y,x] = 1
                elif v < .22: w.grid[y,x] = 2
    host = random.Random(1000003*seed + stress*23)
    steps = []
    for tick in range(horizon):
        legal = w.legal()
        features = w.features(tick)
        with torch.no_grad():
            logits = model(torch.from_numpy(features).unsqueeze(0))[0].tolist()
        raw_top = ACTIONS[int(np.argmax(logits))]
        masked = [v if action in legal else -1.e9 for action,v in zip(ACTIONS,logits)]
        random_override = host.random() < .10
        action = host.choice(legal) if random_override else ACTIONS[int(np.argmax(masked))]
        assert action in legal
        pos = list(w.pos)
        w.apply(action)
        steps.append({"tick":tick,"position":pos,"features":features.tolist(),
                      "legal":list(legal),"raw_top":raw_top,"chosen":action,
                      "host_random_override":random_override,
                      "scores":[round(float(v),7) for v in logits],
                      "next_position":list(w.pos)})
        if w.stopped:break
    png = root/"gallery"/f"{name}-{family.lower()}-{seed}-stress{stress}.png"
    draw(w,png)
    hist = Counter(w.actions)
    return ({"seed":seed,"family":family,"stress":stress,"turns":len(steps),
             "unique_cells":len(w.ink),"stopped":w.stopped,
             "branch":hist["BRANCH"],"retrace":hist["RETRACE"],
             "actions":dict(hist),"png":str(png.relative_to(root)),
             "png_sha256":digest(png)},steps)


def eval_states(net, held, reference):
    x, y, legal = held
    with torch.no_grad():
        logits = net(x)
        raw = logits.argmax(1)
        masked = logits.masked_fill(~legal,-1.e9).argmax(1)
        baseline = reference(x).masked_fill(~legal,-1.e9).argmax(1)
    return {"n":len(y),
            "raw_top_legal_rate":round(float(legal.gather(1,raw[:,None]).float().mean()),6),
            "masked_teacher_accuracy":round(float((masked==y).float().mean()),6),
            "masked_differences_vs_q4b":int((masked!=baseline).sum()),
            "teacher_denominators":{a:int((y==i).sum()) for i,a in enumerate(ACTIONS)},
            "masked_recall":{a:(None if not bool((y==i).any()) else
                              round(float((masked[y==i]==i).float().mean()),6))
                             for i,a in enumerate(ACTIONS)}}


def self_test():
    class Dummy:
        def __init__(self, value):self.value=value
        def __call__(self, x):return torch.ones((len(x),len(ACTIONS)))*self.value
    a,b=Dummy(0),Dummy(1)
    assert torch.allclose(Gate(a,b)(torch.randn(5,27)),
                          torch.full((5,len(ACTIONS)),.5))
    world=WormWorld(1,"GREEBLES")
    assert "STRAIGHT" in world.legal() and "STOP" in world.legal()
    print("QL008 gate / host legality self-test PASS")


def run(args):
    torch.set_num_threads(1)
    np.random.seed(80808)
    torch.manual_seed(80808)
    out=Path(args.out)
    (out/"gallery").mkdir(parents=True,exist_ok=True)
    (out/"models").mkdir(exist_ok=True)
    parent=Path(args.parent)
    _,verified=verified_source(parent,PARENT_SHA)
    q1_path=parent/"models"/"bal-q1.npz"
    q4_path=parent/"models"/"bal-q4.npz"
    q1=GatedQ1(q1_path,thresholds=(.45,)*3)
    q4b=Hybrid(q1_path,q4_path,"b")
    train,train_hist,train_bad=examples(range(2100,2180),33102)
    held,held_hist,held_bad=examples(range(9100,9132),33103)
    assert train_bad==held_bad==0 and len(held[1])==4352
    with torch.no_grad():
        cached=(q1(train[0]),q4b(train[0]))
    variants={"q1-threshold45":q1,"q1-q4middle":q4b,
              "blend-quarter":ConstantBlend(q1,q4b,.25),
              "blend-half":ConstantBlend(q1,q4b,.5),
              "blend-threequarter":ConstantBlend(q1,q4b,.75)}
    repairs={}
    for name,kl in (("learned-gate-ce",0.),("learned-gate-personality",1.5)):
        model,optim,trace,summary=train_gate(q1,q4b,train,args.steps,kl,80808,cached)
        variants[name]=model
        repairs[name]=summary
        torch.save({"optimizer":optim,"loss_trace":trace,"summary":summary,
                    "source_sha":os.environ.get("QL_SOURCE_SHA","local"),
                    "parent_source_sha":PARENT_SHA,"q1_sha256":digest(q1_path),
                    "q4_sha256":digest(q4_path)},
                   out/"models"/(name+"-optimizer.pt"))
    raw=[]
    reports={}
    seeds=(9611,9623,9647,9659)
    for name,model in variants.items():
        reports[name]={"heldout":eval_states(model,held,q4b)}
        episodes=[]
        for stress in (0,1):
            for seed in seeds:
                for family in FAMILIES:
                    ep,steps=episode(model,name,seed,family,stress,out,args.horizon)
                    episodes.append(ep)
                    raw.append({"variant":name,"first_pass":True,
                                "trajectory":ep,"decisions":steps})
        reports[name]["stress"]={}
        for stress in (0,1):
            sub=[e for e in episodes if e["stress"]==stress]
            reports[name]["stress"][str(stress)]={
                "n":len(sub),
                "mean_unique_cells":round(sum(e["unique_cells"] for e in sub)/len(sub),4),
                "mean_turns":round(sum(e["turns"] for e in sub)/len(sub),4),
                "stopped":sum(e["stopped"] for e in sub),
                "branch":sum(e["branch"] for e in sub),
                "retrace":sum(e["retrace"] for e in sub),
                "unique_pngs":len({e["png_sha256"] for e in sub}),
                "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in sub]}
        reports[name]["same_process_latency"]=timing(model,held[0],n=64)
        if isinstance(model,Gate):
            with torch.no_grad(): alphas=model.alpha(held[0]).numpy()
            reports[name]["gate"]={
                "mean":round(float(alphas.mean()),5),
                "std":round(float(alphas.std()),5),
                "min":round(float(alphas.min()),5),
                "max":round(float(alphas.max()),5)}
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    result={"schema":"ql008-state-conditional-precision-1",
            "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
            "parent_source_sha":PARENT_SHA,"parent_manifest_verified":verified,
            "parent_q1_sha256":digest(q1_path),"parent_q4_sha256":digest(q4_path),
            "train_n":len(train[1]),"heldout_n":len(held[1]),
            "train_histogram":train_hist,"heldout_histogram":held_hist,
            "optimizer_updates":2*args.steps,"repairs":repairs,
            "seeds":seeds,"horizon":args.horizon,"variants":reports,
            "original_png_n":len(list((out/"gallery").glob("*.png"))),
            "raw_decisions_sha256":digest(out/"raw_decisions.json"),
            "process_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "limitations":["External source-faithful synthetic W0 plus explicit hazard shift",
                "Frozen genuine Q1/Q4 parent codes; train only a logistic gate",
                "No actual GGUF IQ1/IQ2/IQ3 or optimized SIMD inference",
                "Host controls legal actions and STOP from first turn",
                "Different rollouts have unequal stopped horizons; no aesthetic score",
                "Process RSS includes full Python/Torch runtime; no model-only RSS",
                "No native Godot material painting or provider changes"]}
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    inventory=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,
                "sha256":digest(p)} for p in sorted(out.rglob("*"))
               if p.is_file() and p.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"status":"QL008_COMPLETE","source_sha":result["source_sha"],
                      "optimizer_updates":result["optimizer_updates"],
                      "original_png_n":result["original_png_n"],
                      "cells_base":{k:v["stress"]["0"]["mean_unique_cells"]
                                    for k,v in reports.items()},
                      "gate":{k:reports[k].get("gate") for k in repairs}},
                     sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",default="ql003-source")
    p.add_argument("--out",default="ql008-out")
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--horizon",type=int,default=96)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test() if args.self_test else run(args)
