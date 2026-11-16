#!/usr/bin/env python3
"""QL-010: paired, menu-matched contrastive repair of extreme-quantized game worms.

Research-only CyberSand-like W0 host: real states, fixed protected-cell law,
six legal actions, immutable Q1 XNOR + Q4 middle-island parent. Contrastive
pairs come from different *observed* legal game states, never imaginary edits.
The pair objective cancels all constant-action bias, making a genuinely
state-sensitive residual necessary. This is not a native game or GGUF IQ test.
"""
import argparse
import json
import os
import random
import resource
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quant_worm import ACTIONS, FAMILIES, WormWorld, digest, rollout, timing
from ql002_branch_retrace import examples
from ql003_surgery import morphology
from ql005_bitserial import verified_source
from ql006_activation_gates import GatedQ1
from ql007_mixed_bridges import Hybrid, PARENT_SHA
from ql009_sparse_lowrank import SparseRankTwo, heldout, frozen_prior


def legal_pairs(data, requested, seed):
    """Same family and *identical* legal menu, different teacher and observation.

    Both labels must be independently legal in both states. The two observations
    are real features from explored WormWorld states; no holdout used in fitting.
    Nearest of 20 randomly nominated matching candidates, unique unordered pairs.
    """
    x, y, legal = data
    assert x.shape[1] == 27 and legal.shape[1] == len(ACTIONS)
    groups = defaultdict(list)
    keys = []
    for i in range(len(y)):
        fam = int(x[i, 16:19].argmax())
        key = (fam, tuple(bool(v) for v in legal[i].tolist()))
        keys.append(key)
        groups[key].append(i)
    rng = random.Random(seed)
    order = list(range(len(y)))
    rng.shuffle(order)
    pairs, used = [], set()
    for i in order:
        pool = groups[keys[i]]
        if len(pool) < 2:
            continue
        nominees = rng.sample(pool, min(32, len(pool)))
        viable = [j for j in nominees if y[j] != y[i] and
                  not torch.equal(x[i], x[j]) and
                  (min(i, j), max(i, j)) not in used]
        if not viable:
            continue
        j = min(viable, key=lambda k: float((x[i] - x[k]).square().sum()))
        assert y[i] != y[j] and torch.equal(legal[i], legal[j])
        assert bool(legal[i, y[i]]) and bool(legal[j, y[j]])
        used.add((min(i, j), max(i, j)))
        pairs.append((i, j))
        if len(pairs) == requested:
            break
    assert len(pairs) == requested, (len(pairs), requested)
    return torch.tensor(pairs, dtype=torch.long)


def pair_scores(logits_a, logits_b, labels_a, labels_b):
    """Teacher-aligned contrast of two action margins; constant vectors cancel."""
    a = logits_a.gather(1, labels_a[:, None]).squeeze(1)
    b = logits_a.gather(1, labels_b[:, None]).squeeze(1)
    c = logits_b.gather(1, labels_a[:, None]).squeeze(1)
    d = logits_b.gather(1, labels_b[:, None]).squeeze(1)
    return (a-b)-(c-d)


def pair_report(model, data, pairs, reference=None):
    x,y,legal = data
    i,j = pairs.unbind(1)
    with torch.no_grad():
        logits_i, logits_j = model(x[i]), model(x[j])
        gaps = pair_scores(logits_i, logits_j, y[i], y[j])
        choices_i = logits_i.masked_fill(~legal[i], -1e9).argmax(1)
        choices_j = logits_j.masked_fill(~legal[j], -1e9).argmax(1)
        result = {
            "n_pairs":len(pairs),
            "same_legal_menu_pairs":int((legal[i] == legal[j]).all(1).sum()),
            "teacher_action_diff_pairs":int((y[i] != y[j]).sum()),
            "legal_both_teachers":int((legal[i].gather(1,y[j,None]) &
                                      legal[j].gather(1,y[i,None])).sum()),
            "positive_teacher_margin_pairs":int((gaps > 1e-7).sum()),
            "mean_teacher_margin":round(float(gaps.mean()),6),
            "both_host_actions_different_pairs":int((choices_i != choices_j).sum())
        }
        if isinstance(model, SparseRankTwo):
            ci, cj = model.correction(x[i]), model.correction(x[j])
            cg = pair_scores(ci,cj,y[i],y[j])
            result.update({
                "positive_adapter_only_margin_pairs":int((cg > 1e-7).sum()),
                "mean_adapter_only_margin":round(float(cg.mean()),6),
                "adapter_change_l2_mean":round(float((ci-cj).norm(dim=1).mean()),6),
                "constant_correction_zero_gap_check":bool(torch.allclose(
                    pair_scores(ci,ci,y[i],y[j]),torch.zeros_like(cg),atol=1e-6))
            })
    return result


def fit(parent, data, pairs, mask, seed, steps, pair_weight):
    x,y,legal = data
    assert bool((legal.gather(1,y[:,None])).all())
    torch.manual_seed(seed)
    model = SparseRankTwo(parent, mask, seed=seed, train_bias=False)
    with torch.no_grad():
        parent_logits = parent(x)
    optimizer = torch.optim.AdamW([model.u,model.v],lr=.022,weight_decay=0.)
    counts = torch.bincount(y,minlength=len(ACTIONS)).float()
    probabilities = (1. / counts[y]).double()
    probabilities /= probabilities.sum()
    history=[]
    for tick in range(steps):
        ids = torch.multinomial(probabilities,128,replacement=True)
        current = (parent_logits[ids] + model.correction(x[ids])).masked_fill(
            ~legal[ids],-1e9)
        old = parent_logits[ids].masked_fill(~legal[ids],-1e9)
        ce = F.cross_entropy(current,y[ids])
        kl = F.kl_div(F.log_softmax(current,dim=1),
                      F.softmax(old,dim=1),reduction="batchmean")
        pair_ids = torch.randint(len(pairs),(96,))
        pa,pb = pairs[pair_ids].unbind(1)
        ca,cb = model.correction(x[pa]),model.correction(x[pb])
        contrast = pair_scores(ca,cb,y[pa],y[pb])
        pair_loss = F.softplus(0.5-contrast).mean()
        loss = ce + .65*kl + pair_weight*pair_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"step":tick+1,"loss":float(loss.detach()),
                        "cross_entropy":float(ce.detach()),"parent_kl":float(kl.detach()),
                        "paired_contrastive":float(pair_loss.detach())})
    moments=sum(int(torch.count_nonzero(state["exp_avg"]).item()>0)
                for state in optimizer.state.values())
    assert moments==2 and float(model.v.detach().norm())>0
    return model,optimizer.state_dict(),history,{
        "updates":steps,"seed":seed,"pair_weight":pair_weight,
        "paired_train_observations":len(pairs),"adam_nonzero_moment_tensors":moments,
        "actual_trainable_params":model.u.numel()+model.v.numel(),
        "frozen_class_bias":True,"parent_packed_codes_unchanged":True,
        "first_loss":history[0],"last_loss":history[-1]}


def self_test():
    class Parent:
        def __call__(self,x):
            return torch.zeros((len(x),len(ACTIONS)),dtype=x.dtype)
    torch.manual_seed(1)
    model=SparseRankTwo(Parent(),torch.ones(27),seed=1,train_bias=False)
    x=torch.randn(8,27)
    assert torch.equal(model(x),Parent()(x))
    ca=model.correction(x)
    ya=torch.tensor([0,1,2,3]);yb=torch.tensor([1,2,3,4])
    assert torch.allclose(pair_scores(ca[:4],ca[:4],ya,yb),torch.zeros(4))
    w=WormWorld(122,"GREEBLES")
    assert "STOP" in w.legal()
    print("QL010 TEST PASS: unmodified low-bit parent, algebraic constant-action cancellation, STOP legal")


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(101010)
    np.random.seed(101010)
    out=Path(args.out)
    for folder in ("gallery","models"):
        (out/folder).mkdir(parents=True,exist_ok=True)
    parent_root=Path(args.parent)
    _,verified=verified_source(parent_root,PARENT_SHA)
    q1_path=parent_root/"models"/"bal-q1.npz"
    q4_path=parent_root/"models"/"bal-q4.npz"
    q4=Hybrid(q1_path,q4_path,"b")
    q1=GatedQ1(q1_path,thresholds=(.45,.45,.45))
    train,train_hist,invalid=examples(range(2100,2180),33102)
    held,held_hist,held_invalid=examples(range(9100,9132),33103)
    assert invalid==held_invalid==0 and len(held[1])==4352
    train_pairs=legal_pairs(train,2048,101101)
    held_pairs=legal_pairs(held,512,101103)
    x,y,_=train
    sparse_mask=torch.zeros(27)
    selected=torch.argsort(x.std(0),descending=True)[:14]
    sparse_mask[selected]=1
    specs=(
        ("ce-no-bias",sparse_mask,101111,0.),
        ("paired-sparse",sparse_mask,101111,1.75),
        ("paired-dense",torch.ones(27),101111,1.75)
    )
    variants={"q1-threshold45":q1,"q1-q4middle":q4}
    training={}
    for name,mask,seed,weight in specs:
        model,optimizer,history,diag=fit(q4,train,train_pairs,mask,seed,args.steps,weight)
        variants[name]=model
        training[name]=diag
        torch.save({"model_state":model.state_dict(),"optimizer":optimizer,
                    "trace":history,"training":diag,
                    "source_sha":os.environ.get("QL_SOURCE_SHA","local"),
                    "parent_source_sha":PARENT_SHA,
                    "parent_q1_sha256":digest(q1_path),
                    "parent_q4_sha256":digest(q4_path)},
                   out/"models"/(name+"-optimizer.pt"))
    seeds=(9811,9827,9833,9851)
    results={}
    raw=[]
    for name,model in variants.items():
        metric=heldout(model,held,q4)
        response=pair_report(model,held,held_pairs)
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(model,name,family,seed,out,horizon=args.horizon)
                ep["morphology"]=morphology(steps)
                episodes.append(ep)
                raw.append({"policy":name,"mode":"live","seed":seed,
                            "family":family,"first_pass":True,"steps":steps})
        hist=Counter()
        for ep in episodes:
            hist.update(ep["action_histogram"])
        results[name]={
            "heldout":metric,"paired_heldout":response,
            "mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),5),
            "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),5),
            "stopped_episodes":sum(bool(e["stop_count"]) for e in episodes),
            "branch":hist["BRANCH"],"retrace":hist["RETRACE"],"stop":hist["STOP"],
            "total_applied_actions":sum(e["turns"] for e in episodes),
            "mean_repeat_fraction":round(sum(e["morphology"]["repeat_fraction"] for e in episodes)/len(episodes),5),
            "pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes],
            "same_runner_latency":timing(model,held[0],n=96),
            "adapter_working_bytes":model.memory_bytes() if isinstance(model,SparseRankTwo) else 0}
    controls={}
    for mode in ("random","wall","fidget"):
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(None,mode,family,seed,out,horizon=args.horizon)
                episodes.append(ep)
                raw.append({"policy":mode,"mode":"model-free","seed":seed,
                            "family":family,"first_pass":True,"steps":steps})
        controls[mode]={"mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),5),
                        "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),5),
                        "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes]}
    for name in ("q1-q4middle","ce-no-bias","paired-sparse","paired-dense"):
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(frozen_prior(variants[name],seed,family),
                                 name+"-opening",family,seed,out,horizon=args.horizon)
                episodes.append(ep)
                raw.append({"policy":name,"mode":"frozen-opening","seed":seed,
                            "family":family,"first_pass":True,"steps":steps})
        results[name]["opening_frozen_mean_cells"]=round(
            sum(e["unique_cells"] for e in episodes)/len(episodes),5)
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    meta={
        "schema":"ql010-paired-state-contrastive-1",
        "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
        "parent_source_sha":PARENT_SHA,"parent_manifest_verified":verified,
        "parent_q1_sha256":digest(q1_path),"parent_q4_sha256":digest(q4_path),
        "synthetic_host":"WormWorld W0, source-faithful Python; NOT Godot native",
        "train_n":len(train[1]),"heldout_n":len(held[1]),
        "train_histogram":train_hist,"heldout_histogram":held_hist,
        "train_matched_pairs":len(train_pairs),"heldout_matched_pairs":len(held_pairs),
        "pair_constraints":"same family, identical host-legal menu, distinct legal teacher actions and distinct real observations",
        "train_only_sparse_feature_ids":selected.tolist(),
        "optimizer_updates":len(specs)*args.steps,
        "training":training,"variants":results,"controls":controls,
        "seeds":list(seeds),"horizon":args.horizon,
        "first_pass_trajectories":len(raw),
        "original_png_n":len(list((out/"gallery").glob("*.png"))),
        "raw_decisions_sha256":digest(out/"raw_decisions.json"),
        "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations":[
            "QL-010 is a tiny trained synthetic game-policy MLP, not GGUF IQ1/IQ2/IQ3",
            "All pair observations arise from legal W0 world rollouts; pairs are matched across distinct observed states, not causal single-variable interventions",
            "Pair preference may be unidentifiable from visible features because the teacher uses modulo tick rules",
            "Adapter-only pair contrast cancels constant-action vectors but is not proof of closed-loop artistic quality",
            "Frozen-opening controls diverge after first different action; not same-state causal comparison",
            "STOP remains legal immediately; common host random STOP truncates episodes",
            "Controls have matched seeds but not always matched effective action horizons",
            "Process peak RSS includes Python/Torch/NumPy and all models rather than isolated model RSS",
            "No native Godot paintings or production provider changes",
            "Packed Q1 XNOR is Python bit-count reference; Q4 island float GEMV; no SIMD acceleration"]}
    (out/"results.json").write_text(json.dumps(meta,indent=2)+"\n")
    manifest=[{"path":str(path.relative_to(out)),"bytes":path.stat().st_size,
               "sha256":digest(path)}
              for path in sorted(out.rglob("*")) if path.is_file() and path.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL010_COMPLETE","source_sha":meta["source_sha"],
                      "optimizer_updates":meta["optimizer_updates"],
                      "heldout":meta["heldout_n"],"pngs":meta["original_png_n"],
                      "pairs":meta["heldout_matched_pairs"],
                      "comparisons":{name:{"teacher":v["heldout"]["teacher_agreement_n"],
                        "pair_correct":v["paired_heldout"]["positive_teacher_margin_pairs"],
                        "adapter_pair_correct":v["paired_heldout"].get("positive_adapter_only_margin_pairs"),
                        "state_delta":v["heldout"]["live_adapter_vs_constant_adapter_differences_n"],
                        "cells":v["mean_unique_cells"],"branch":v["branch"],
                        "retrace":v["retrace"]} for name,v in results.items()}},
                     sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",default="ql003-source")
    p.add_argument("--out",default="ql010-out")
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--horizon",type=int,default=96)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    self_test() if a.self_test else run(a)
