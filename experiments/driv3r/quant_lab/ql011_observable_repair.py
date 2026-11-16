#!/usr/bin/env python3
"""QL-011: observable-neighbour counterfactual repair for extreme-quant game worms.

QL-010 showed that an adapter can respond to matched states while its teacher
still contains modulo-tick decisions not completely identified by observation.
Here the auxiliary novelty/recovery target is a *fully observed*, deterministic
function of real W0 features and host legal menus. No synthetic invented states,
host bypass, source mutation, native Game/Workbench changes, or GGUF IQ claims.

Compare CE-only, observable matched-state contrast, and personality-preserving
contrast under exactly the same original Q1 XNOR/Q4-middle parent and seeds.
"""
import argparse
import json
import os
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import ACTIONS, FAMILIES, WormWorld, digest, rollout, timing
from ql002_branch_retrace import examples
from ql003_surgery import morphology
from ql005_bitserial import verified_source
from ql006_activation_gates import GatedQ1
from ql007_mixed_bridges import Hybrid, PARENT_SHA
from ql009_sparse_lowrank import frozen_prior
from ql010_paired_contrastive import legal_pairs, pair_scores


def observable_targets(data):
    """Explicit state-only action law: local freshness, recovery and family style.

    Uses only 27 features and the true menu from actual WormWorld observations.
    It is an *experimental alternate objective*, not a claim of native W0 teacher
    agreement or ground-truth aesthetic quality. STOP always remains host legal.
    """
    x, _, legal = data
    labels = []
    for features, menu in zip(x.tolist(), legal.tolist()):
        local = [features[i] for i in (3, 7, 11)]
        family = int(np.argmax(features[16:19]))
        tick = features[19] * 64.0
        ink = features[20] * 64.0
        branches = features[22] * 16.0
        n_new = sum(v > .5 for v in local)
        if tick >= 40 - 1e-4 or ink >= 52 - 1e-4:
            selected = "STOP"
        elif menu[3] and n_new == 0:
            selected = "RETRACE"
        elif menu[4] and n_new >= 2 and ink >= 8 and branches < 3:
            selected = "BRANCH"
        else:
            order = ((1, 0, 2), (0, 2, 1), (2, 0, 1))[family]
            names = ("STRAIGHT", "LEFT", "RIGHT")
            selected = next((names[j] for j in order if menu[j] and local[j] > .5), None)
            if selected is None:
                selected = next((names[j] for j in order if menu[j]), None)
            if selected is None:
                selected = "RETRACE" if menu[3] else "STOP"
        index = ACTIONS.index(selected)
        assert menu[index], (selected,menu)
        labels.append(index)
    return torch.tensor(labels, dtype=torch.long)


class RankFour(nn.Module):
    """Frozen packed-parent inference with genuinely trained rank-4 residual.

    The start is exactly zero, without any constant class-bias parameter.
    Parent low-bit codes remain untouched; only u and v are optimizer-owned.
    """
    def __init__(self, parent, seed):
        super().__init__()
        self.parent = parent
        torch.manual_seed(seed)
        self.u = nn.Parameter(torch.randn(27,4) * .04)
        self.v = nn.Parameter(torch.zeros(4,len(ACTIONS)))

    def correction(self, x):
        return torch.tanh(x @ self.u) @ self.v

    def forward(self, x):
        with torch.no_grad():
            parent_logits = self.parent(x)
        return parent_logits + self.correction(x)

    def memory_bytes(self):
        return self.u.numel()*self.u.element_size() + self.v.numel()*self.v.element_size()


def fit(parent, train, oracle, pairs, steps, seed, contrast_weight, prior_weight):
    x, _, legal = train
    torch.manual_seed(seed)
    model = RankFour(parent,seed)
    with torch.no_grad():
        base = parent(x)
    optim = torch.optim.AdamW((model.u,model.v),lr=.025,weight_decay=0.)
    counts = torch.bincount(oracle,minlength=len(ACTIONS)).float().clamp(min=1)
    probs = (1. / counts[oracle]).double()
    probs /= probs.sum()
    trace=[]
    for step in range(steps):
        ids = torch.multinomial(probs,128,replacement=True)
        logits = (base[ids] + model.correction(x[ids])).masked_fill(~legal[ids],-1e9)
        base_masked = base[ids].masked_fill(~legal[ids],-1e9)
        ce = F.cross_entropy(logits,oracle[ids])
        kl = F.kl_div(F.log_softmax(logits,dim=1),
                      F.softmax(base_masked,dim=1),reduction="batchmean")
        pi,pj = pairs[torch.randint(len(pairs),(96,))].unbind(1)
        ca, cb = model.correction(x[pi]), model.correction(x[pj])
        margin = pair_scores(ca,cb,oracle[pi],oracle[pj])
        paired = F.softplus(.5-margin).mean()
        loss = ce + prior_weight*kl + contrast_weight*paired
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        trace.append({"step":step+1,"loss":float(loss.detach()),
                      "oracle_ce":float(ce.detach()),"parent_kl":float(kl.detach()),
                      "observable_pair_loss":float(paired.detach())})
    moments = sum(int(torch.count_nonzero(s["exp_avg"]))>0
                  for s in optim.state.values())
    assert moments == 2 and float(model.v.detach().norm()) > 0
    diag = {"updates":steps,"seed":seed,"contrast_weight":contrast_weight,
            "prior_weight":prior_weight,"trained_parameters":model.u.numel()+model.v.numel(),
            "optimizer_nonzero_moment_tensors":moments,
            "parent_packed_q1_q4_codes_unchanged":True,
            "first":trace[0],"last":trace[-1]}
    return model,optim.state_dict(),trace,diag


def score(model, held, original_labels, oracle_labels, base):
    x,_,legal=held
    with torch.no_grad():
        scores = model(x)
        pred_raw = scores.argmax(1)
        chosen = scores.masked_fill(~legal,-1e9).argmax(1)
        parent = base(x).masked_fill(~legal,-1e9).argmax(1)
        if isinstance(model,RankFour):
            static = base(x) + model.correction(x[:1]).expand(len(x),-1)
            static_choice = static.masked_fill(~legal,-1e9).argmax(1)
            adaptive_delta = int((chosen != static_choice).sum())
        else:
            adaptive_delta = None
    return {"n":len(x),"host_masked_legal":int(legal.gather(1,chosen[:,None]).sum()),
            "raw_top_legal":int(legal.gather(1,pred_raw[:,None]).sum()),
            "original_teacher_agreement":int((chosen==original_labels).sum()),
            "observable_oracle_agreement":int((chosen==oracle_labels).sum()),
            "changed_vs_parent":int((chosen != parent).sum()),
            "live_vs_constant_adapter_changes":adaptive_delta,
            "oracle_denominators":{a:int((oracle_labels==i).sum()) for i,a in enumerate(ACTIONS)},
            "original_denominators":{a:int((original_labels==i).sum()) for i,a in enumerate(ACTIONS)},
            "predicted_actions":{a:int((chosen==i).sum()) for i,a in enumerate(ACTIONS)},
            "observable_recall":{a:round(float((chosen[oracle_labels==i]==i).float().mean()),6)
                                 if bool((oracle_labels==i).any()) else None
                                 for i,a in enumerate(ACTIONS)}}


def paired_report(model, held, pairs, oracle):
    x,_,legal=held
    i,j=pairs.unbind(1)
    assert bool((legal[i]==legal[j]).all())
    assert bool(legal[i].gather(1,oracle[j,None]).all())
    with torch.no_grad():
        scores_i, scores_j=model(x[i]),model(x[j])
        margins=pair_scores(scores_i,scores_j,oracle[i],oracle[j])
        ai=scores_i.masked_fill(~legal[i],-1e9).argmax(1)
        aj=scores_j.masked_fill(~legal[j],-1e9).argmax(1)
        result={"n":len(pairs),"different_oracle_actions":int((oracle[i]!=oracle[j]).sum()),
                "both_host_actions_differ":int((ai!=aj).sum()),
                "teacher_aligned_total_margin":int((margins>1e-7).sum()),
                "mean_total_margin":round(float(margins.mean()),6)}
        if isinstance(model,RankFour):
            residual_i,residual_j=model.correction(x[i]),model.correction(x[j])
            m=pair_scores(residual_i,residual_j,oracle[i],oracle[j])
            result.update({"positive_residual_only_margin":int((m>1e-7).sum()),
                           "mean_residual_margin":round(float(m.mean()),6),
                           "fixed_residual_algebraic_zero":bool(torch.allclose(
                              pair_scores(residual_i,residual_i,oracle[i],oracle[j]),
                              torch.zeros_like(m),atol=1e-6))})
    return result


def self_test():
    w=WormWorld(11411,"GREEBLES")
    legal=torch.tensor([[a in w.legal() for a in ACTIONS]],dtype=torch.bool)
    one=torch.from_numpy(w.features(0))[None,:]
    data=(one,torch.zeros(1,dtype=torch.long),legal)
    assert bool(legal[0,observable_targets(data)[0]])
    class Fake:
        def __call__(self,x):
            return torch.ones(len(x),len(ACTIONS))
    parent=Fake()
    model=RankFour(parent,11)
    assert torch.equal(model(one),parent(one))
    with torch.no_grad():
        model.v[0,0]=.5
    assert not torch.equal(model(one),parent(one))
    fixed=model.correction(one)
    assert torch.allclose(pair_scores(fixed,fixed,torch.tensor([0]),
                                     torch.tensor([1])),torch.zeros(1),atol=1e-6)
    print("QL011 SELF-TEST PASS: observable legal targets, exact zero patch, genuine correction")


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(111011)
    np.random.seed(111011)
    out=Path(args.out)
    (out/"gallery").mkdir(parents=True,exist_ok=True)
    (out/"models").mkdir(exist_ok=True)
    parent_root=Path(args.parent)
    _,nverified=verified_source(parent_root,PARENT_SHA)
    q1=parent_root/"models"/"bal-q1.npz"
    q4=parent_root/"models"/"bal-q4.npz"
    parent=Hybrid(q1,q4,"b")
    q1_model=GatedQ1(q1,thresholds=(.45,)*3)
    train,train_hist,invalid=examples(range(2100,2180),33102)
    held,held_hist,held_invalid=examples(range(9100,9132),33103)
    assert invalid==held_invalid==0 and len(held[1])==4352
    assert not set(range(2100,2180)) & set(range(9100,9132))
    train_oracle=observable_targets(train)
    held_oracle=observable_targets(held)
    assert torch.all(train[2].gather(1,train_oracle[:,None]))
    assert torch.all(held[2].gather(1,held_oracle[:,None]))
    train_pairs=legal_pairs((train[0],train_oracle,train[2]),512,111013)
    held_pairs=legal_pairs((held[0],held_oracle,held[2]),128,111017)
    specs=(("oracle-ce-r4",0.,.3),("observable-paired-r4",1.75,.3),
           ("paired-personality-r4",1.75,1.8))
    variants={"q1-threshold45":q1_model,"q1-q4middle":parent}
    training={}
    for name,cweight,pweight in specs:
        m,opt,trace,diag=fit(parent,train,train_oracle,train_pairs,args.steps,
                              111021,cweight,pweight)
        variants[name]=m
        training[name]=diag
        torch.save({"state_dict":m.state_dict(),"optimizer":opt,"trace":trace,
                    "training":diag,"source_sha":os.environ.get("QL_SOURCE_SHA","local"),
                    "parent_source_sha":PARENT_SHA,
                    "parent_q1_sha256":digest(q1),"parent_q4_sha256":digest(q4)},
                   out/"models"/(name+"-optimizer.pt"))
    raw=[]
    seeds=(9911,9923,9947,9951)
    metrics={}
    for name,model in variants.items():
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(model,name,family,seed,out,horizon=args.horizon)
                ep["morphology"]=morphology(steps)
                episodes.append(ep)
                raw.append({"policy":name,"mode":"live","first_pass":True,
                            "seed":seed,"family":family,"steps":steps})
        actions=Counter()
        for ep in episodes:
            actions.update(ep["action_histogram"])
        metrics[name]={
            "heldout":score(model,held,held[1],held_oracle,parent),
            "observable_pairs":paired_report(model,held,held_pairs,held_oracle),
            "mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),5),
            "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),5),
            "branch":actions["BRANCH"],"retrace":actions["RETRACE"],"stop":actions["STOP"],
            "stop_episodes":sum(bool(e["stop_count"]) for e in episodes),
            "mean_repeat_fraction":round(sum(e["morphology"]["repeat_fraction"]
                                             for e in episodes)/len(episodes),5),
            "family_distinct_pngs":{str(seed):len({e["png_sha256"] for e in episodes
                                                  if e["seed"]==seed}) for seed in seeds},
            "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes],
            "working_adapter_bytes":model.memory_bytes() if isinstance(model,RankFour) else 0,
            "same_runner_latency":timing(model,held[0],n=80)}
    controls={}
    for name in ("random","wall","fidget"):
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(None,name,family,seed,out,horizon=args.horizon)
                episodes.append(ep)
                raw.append({"policy":name,"mode":"model-free","first_pass":True,
                            "seed":seed,"family":family,"steps":steps})
        controls[name]={"mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/12,5),
                        "mean_turns":round(sum(e["turns"] for e in episodes)/12,5),
                        "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]}
                                         for e in episodes]}
    for name in ("q1-q4middle","observable-paired-r4","paired-personality-r4"):
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(frozen_prior(variants[name],seed,family),
                                 name+"-opening",family,seed,out,horizon=args.horizon)
                episodes.append(ep)
                raw.append({"policy":name,"mode":"frozen-opening","first_pass":True,
                            "seed":seed,"family":family,"steps":steps})
        metrics[name]["frozen_opening_mean_unique_cells"]=round(
            sum(e["unique_cells"] for e in episodes)/12,5)
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    record={
       "schema":"ql011-observable-neighbour-counterfactual-1",
       "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
       "parent_source_sha":PARENT_SHA,"parent_manifest_entries":nverified,
       "parent_q1_sha256":digest(q1),"parent_q4_sha256":digest(q4),
       "synthetic_host":"W0 WormWorld real rollouts, NOT native Godot",
       "objective":"feature-identifiable neighbour/branch/recovery law, NOT native teacher or subjective aesthetics",
       "train_n":len(train[1]),"heldout_n":len(held[1]),
       "train_oracle_histogram":{a:int((train_oracle==i).sum()) for i,a in enumerate(ACTIONS)},
       "heldout_oracle_histogram":{a:int((held_oracle==i).sum()) for i,a in enumerate(ACTIONS)},
       "original_train_teacher_histogram":train_hist,"original_heldout_teacher_histogram":held_hist,
       "train_pairs":len(train_pairs),"heldout_pairs":len(held_pairs),
       "optimizer_updates":len(specs)*args.steps,"training":training,
       "variants":metrics,"controls":controls,"seeds":list(seeds),"horizon":args.horizon,
       "first_pass_trajectories":len(raw),"original_png_n":len(list((out/"gallery").glob("*.png"))),
       "raw_decisions_sha256":digest(out/"raw_decisions.json"),
       "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "limitations":[
         "Train and heldout observations arise from QL-002 teacher/exploration rollouts; the alternative observable label is derived, not ground truth quality",
         "Matched state pairs are observed, not controlled single-variable interventions",
         "The feature-grounded teacher adds novelty/branch/recovery biases and should not replace the original teacher denominator",
         "Original random action STOP can truncate every policy from its first turn",
         "Matched seed and host stream, but policy world states diverge after decisions",
         "Tiny synthetic packed-Q1 XNOR and decoded Q4 island model, NOT GGUF IQ1/IQ2/IQ3",
         "Rank-four fp32 adapters add working bytes; same-runner Python timings are decisions/s, not SLM tokens/s",
         "RSS is process-wide, not isolated quantized-weight RSS",
         "No production game, native runtime, Workbench or provider changes"]}
    (out/"results.json").write_text(json.dumps(record,indent=2)+"\n")
    manifest=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":digest(p)}
              for p in sorted(out.rglob("*")) if p.is_file() and p.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL011_COMPLETE","source":record["source_sha"],
        "updates":record["optimizer_updates"],"heldout":record["heldout_n"],
        "pairs":record["heldout_pairs"],"pngs":record["original_png_n"],
        "policies":{k:{"oracle":v["heldout"]["observable_oracle_agreement"],
                       "old_teacher":v["heldout"]["original_teacher_agreement"],
                       "adaptive_delta":v["heldout"]["live_vs_constant_adapter_changes"],
                       "cells":v["mean_unique_cells"],"branch":v["branch"],
                       "retrace":v["retrace"]}
                    for k,v in metrics.items()}},sort_keys=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",default="ql003-source")
    parser.add_argument("--out",default="ql011-out")
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--horizon",type=int,default=96)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else run(args)
