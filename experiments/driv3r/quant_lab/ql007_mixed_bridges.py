#!/usr/bin/env python3
"""QL-007: Q4 precision islands inside packed-XNOR Q1 game worms.

Research-only source-faithful synthetic game policies. No game modification,
GGUF IQ claim, SIMD acceleration, native painting, or provider promotion.
"""
import argparse
import json
import math
import os
import resource
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from quant_worm import (ACTIONS,FAMILIES,Policy,WormWorld,digest,load_packed,
                        save_packed,unpack_codes,rollout,timing)
from ql002_branch_retrace import examples,assess
from ql003_surgery import morphology
from ql004_integer_repair import meta
from ql005_bitserial import verified_source,q1_shapes
from ql006_activation_gates import GatedQ1,GatedQ1STE

PARENT_SHA="9b4103c48127e5c8538403e2a4d9e2525fbe8755"
LAYERS="abc"


def q4_islands(path,names):
    islands={}
    with np.load(path,allow_pickle=False) as f:
        for name in names:
            n,d=q1_shapes()[name]["shape"]
            code=unpack_codes(f[name+"_packed"],4,n*d).reshape(n,d)
            weights=(code.astype(np.float32)-7.)*f[name+"_scale"][:,None]
            islands[name]=(weights.astype(np.float32),f[name+"_bias"].astype(np.float32))
    return islands


class Hybrid(GatedQ1):
    """Q1 code XNOR/popcount plus unquantized activations at Q4 islands.

    Island intervention changes both Q4 weight precision and activation
    precision, not isolated weight quantization. Python reference only.
    """
    def __init__(self,q1,q4,islands="",thresholds=(.45,.45,.45),
                 gains=(1.,1.,1.),correction=None):
        super().__init__(q1,thresholds,gains,correction)
        self.island_names="".join(sorted(set(islands)))
        assert set(self.island_names).issubset(set(LAYERS))
        self.islands=q4_islands(q4,self.island_names)
        self.total_packed_bytes_with_redundant_q1=self.packed_bytes
        self.total_working_bytes_excluding_python=self.working_bytes
        for weight,bias in self.islands.values():
            n,d=weight.shape
            self.total_packed_bytes_with_redundant_q1+=math.ceil(n*d/2)+8*n
            self.total_working_bytes_excluding_python+=weight.nbytes+bias.nbytes

    def __call__(self,features):
        x=np.asarray(features.detach().cpu(),dtype=np.float32)
        if x.ndim==1:x=x[None,:]
        values=[]
        for sample in x:
            z=sample
            for i,(d,rows,scales,biases) in enumerate(self.layers):
                if LAYERS[i] in self.islands:
                    w,b=self.islands[LAYERS[i]]
                    z=w@z+b
                else:
                    threshold=float(self.thresholds[i])*float(np.max(np.abs(z)))
                    active=(z!=0)&(np.abs(z)>=threshold)
                    count=int(active.sum())
                    ab=int.from_bytes(np.packbits(active.astype(np.uint8),
                          bitorder="little").tobytes(),"little")
                    sb=int.from_bytes(np.packbits((z>=0).astype(np.uint8),
                          bitorder="little").tobytes(),"little")
                    amplitude=np.float32((np.abs(z[active]).mean() if count else 0.)*self.gains[i])
                    z=np.asarray([(count-2*((row^sb)&ab).bit_count())*scale*amplitude+bias
                                  for row,scale,bias in zip(rows,scales,biases)],dtype=np.float32)
                if i<2:z=np.tanh(z)
            values.append(z+self.correction)
        return torch.from_numpy(np.stack(values))


class HybridSTE(GatedQ1STE):
    """Frozen Q1/Q4 weight codes, train only STE gates, gains and correction."""
    def __init__(self,q1,q4,islands="b",threshold=.35):
        super().__init__(q1,thresholds=(threshold,)*3)
        self.island_names="".join(sorted(set(islands)))
        for name,(weight,bias) in q4_islands(q4,self.island_names).items():
            setattr(self,name+"_weight",torch.from_numpy(weight.copy()))
            setattr(self,name+"_bias",torch.from_numpy(bias.copy()))

    def forward(self,x):
        for i,name in enumerate(LAYERS):
            if name not in self.island_names:
                magnitude=x.abs()
                maximum=magnitude.amax(dim=-1,keepdim=True)
                threshold=self.thresholds[i]*maximum
                hard=((x!=0)&(magnitude>=threshold)).to(x.dtype)
                soft=torch.sigmoid((magnitude-threshold)/(.08*maximum+1e-6))*(x!=0).to(x.dtype)
                gate=soft+(hard-soft).detach()
                amplitude=(magnitude*hard).sum(dim=-1,keepdim=True)/hard.sum(
                          dim=-1,keepdim=True).clamp(min=1.)
                x=torch.where(x>=0,1.,-1.)*gate*amplitude*self.gains[i]
            x=F.linear(x,getattr(self,name+"_weight"),getattr(self,name+"_bias"))
            if i<2:x=torch.tanh(x)
        return x+self.correction

    def as_policy(self,q1,q4):
        return Hybrid(q1,q4,self.island_names,self.thresholds.detach().tolist(),
                      self.gains.detach().tolist(),self.correction.detach().tolist())


def train_repair(q1,q4,train,prior,nsteps,seed,kl_weight):
    torch.manual_seed(seed)
    model=HybridSTE(q1,q4)
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0.)
    x,y,_=train
    counts=torch.bincount(y,minlength=len(ACTIONS)).float()
    weights=(1./counts[y]).double()
    weights/=weights.sum()
    history=[]
    for step in range(nsteps):
        ix=torch.multinomial(weights,128,replacement=True)
        scores=model(x[ix])
        with torch.no_grad():before=prior(x[ix])
        ce=F.cross_entropy(scores,y[ix])
        kl=F.kl_div(F.log_softmax(scores,dim=-1),F.softmax(before,dim=-1),
                    reduction="batchmean")
        loss=ce+kl_weight*kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.thresholds.clamp_(0.,.8)
            model.gains.clamp_(.1,5.)
        history.append({"step":step+1,"loss":float(loss.detach()),"ce":float(ce.detach())})
    inferred=model.as_policy(q1,q4)
    with torch.no_grad():
        probe=torch.randn(32,27)
        probe[::2,::3]=0
        error=float((model(probe)-inferred(probe)).abs().max())
    assert error<5e-5,error
    moment_tensors=sum(int(torch.count_nonzero(v["exp_avg"])>0) for v in opt.state.values())
    assert moment_tensors>=2
    info={"updates":nsteps,"seed":seed,"kl_weight":kl_weight,
          "nonzero_moment_tensors":moment_tensors,
          "parity_max_abs":error,"fixed_q1_codes":True,"fixed_q4_codes":True,
          "parameter_count":12,"loss_first":history[0],"loss_last":history[-1],
          "thresholds":inferred.thresholds.tolist(),"gains":inferred.gains.tolist(),
          "output_correction":inferred.correction.tolist()}
    return inferred,opt.state_dict(),info,history


class FrozenOpening:
    def __init__(self,model,seed,family):
        start=WormWorld(seed,family)
        self.scores=model(torch.from_numpy(start.features(0)).unsqueeze(0))
    def __call__(self,x):
        return self.scores.expand(x.shape[0],-1)


def self_test():
    import tempfile
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        torch.manual_seed(19)
        baseline=Policy()
        q1,_,_=save_packed(root,"test-q1",baseline,(1,1,1))
        q4,_,_=save_packed(root,"test-q4",baseline,(4,4,4))
        probe=torch.randn(17,27)
        probe[::2,::4]=0
        for island in ("","a","b","c","ab"):
            a=Hybrid(q1,q4,island,thresholds=(.35,)*3)
            b=HybridSTE(q1,q4,island)
            difference=float((a(probe)-b(probe)).abs().max())
            assert difference<5e-5,(island,difference)
        net=HybridSTE(q1,q4,"b")
        net(probe).square().mean().backward()
        assert net.thresholds.grad is not None
        assert bool(torch.isfinite(net.thresholds.grad).all())
        assert any(abs(float(z))>1e-9 for z in net.thresholds.grad)
    print("QL007 PASS real Q1 XNOR and Q4 island parity; STE gradients")


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(70707)
    np.random.seed(70707)
    out=Path(args.out)
    (out/"gallery").mkdir(parents=True,exist_ok=True)
    (out/"models").mkdir(exist_ok=True)
    parent=Path(args.parent)
    pinned,n_verified=verified_source(parent,PARENT_SHA)
    q1=parent/"models"/"bal-q1.npz"
    q4=parent/"models"/"bal-q4.npz"
    frank=parent/"models"/"frank-reg-a-bal-bc-q1.npz"
    assert q1.is_file() and q4.is_file() and frank.is_file()
    teacher=load_packed(parent/"models"/"bal-fp32.npz",meta("bal-fp32",pinned))
    train,_,bad_train=examples(range(2100,2180),33102)
    held,held_hist,bad_held=examples(range(9100,9132),33103)
    assert bad_train==bad_held==0 and len(held[1])==4352
    assert all(held_hist.get(a,0)>0 for a in ACTIONS)
    baseline=GatedQ1(q1,thresholds=(.45,)*3)
    variants={
      "q1-threshold45":baseline,
      "q1-q4a":Hybrid(q1,q4,"a"),
      "q1-q4b":Hybrid(q1,q4,"b"),
      "q1-q4c":Hybrid(q1,q4,"c"),
      "q1-q4ab":Hybrid(q1,q4,"ab"),
      "frank-q1-q4b":Hybrid(frank,q4,"b")
    }
    repaired={}
    for name,kl in (("q1-q4b-ce-repair",0.),
                    ("q1-q4b-personality-repair",1.5)):
        net,optimizer,info,history=train_repair(q1,q4,train,baseline,
                                               args.steps,70707,kl)
        variants[name]=net
        repaired[name]=info
        torch.save({"optimizer":optimizer,"loss_trace":history,
          "diagnostic":info,"source_sha":os.environ.get("QL_SOURCE_SHA","local"),
          "parent_sha":PARENT_SHA,"parent_q1_sha256":digest(q1),
          "parent_q4_sha256":digest(q4)},out/"models"/(name+"-optimizer.pt"))
    x,y,legal=held
    with torch.no_grad():
        ref=teacher(x).masked_fill(~legal,-1.e9).argmax(dim=1)
    seeds=(9511,9523,9547,9551)
    raw=[]
    results={}
    for name,model in variants.items():
        with torch.no_grad():
            masked=model(x).masked_fill(~legal,-1.e9).argmax(dim=1)
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(model,name,family,seed,out,horizon=96)
                ep["morphology"]=morphology(steps)
                episodes.append(ep)
                raw.append({"variant":name,"mode":"live","seed":seed,
                            "family":family,"first_pass":True,"steps":steps})
        counts=Counter()
        for ep in episodes:counts.update(ep["action_histogram"])
        results[name]={"heldout":assess(model,held),
          "masked_disagreement_fp32_n":int((masked!=ref).sum()),
          "mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/12,4),
          "mean_turns":round(sum(e["turns"] for e in episodes)/12,4),
          "mean_repeat_fraction":round(sum(e["morphology"]["repeat_fraction"]
                                           for e in episodes)/12,5),
          "stop_episodes":sum(bool(e["stop_count"]) for e in episodes),
          "actions":dict(counts),
          "family_distinct_pngs_by_seed":[len({e["png_sha256"] for e in episodes
                                               if e["seed"]==seed}) for seed in seeds],
          "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]}
                            for e in episodes],
          "same_runner_latency":timing(model,x,n=64),
          "packed_payload_including_redundant_q1_codes":getattr(model,
                "total_packed_bytes_with_redundant_q1",model.packed_bytes),
          "working_weight_bytes_excluding_python":getattr(model,
                "total_working_bytes_excluding_python",model.working_bytes)}
    for name in ("q1-q4b","q1-q4b-personality-repair"):
        frozen=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(FrozenOpening(variants[name],seed,family),
                        name+"-frozen-opening",family,seed,out,horizon=96)
                frozen.append(ep)
                raw.append({"variant":name,"mode":"opening-prior",
                  "seed":seed,"family":family,"first_pass":True,"steps":steps})
        results[name]["frozen_mean_unique_cells"]=round(
            sum(e["unique_cells"] for e in frozen)/12,4)
        results[name]["frozen_original_pngs"]=[
            {"path":e["png"],"sha256":e["png_sha256"]} for e in frozen]
    controls={}
    for name in ("random","wall","fidget"):
        rows=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(None,name,family,seed,out,horizon=96)
                rows.append(ep)
                raw.append({"variant":name,"mode":"model-free",
                    "seed":seed,"family":family,"first_pass":True,"steps":steps})
        controls[name]={"mean_unique_cells":round(sum(e["unique_cells"] for e in rows)/12,4),
                        "mean_turns":round(sum(e["turns"] for e in rows)/12,4),
                        "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]}
                                         for e in rows]}
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    record={"schema":"ql007-q4-islands-q1-xnor-1",
      "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
      "parent_sha":PARENT_SHA,"verified_parent_manifest_entries":n_verified,
      "parent_q1_sha256":digest(q1),"parent_q4_sha256":digest(q4),
      "heldout_n":len(y),"train_n":len(train[1]),
      "heldout_histogram":held_hist,"new_optimizer_updates":2*args.steps,
      "repairs":repaired,"seeds":list(seeds),"horizon":96,
      "variants":results,"controls":controls,
      "first_pass_png_n":len(list((out/"gallery").glob("*.png"))),
      "raw_decisions_sha256":digest(out/"raw_decisions.json"),
      "process_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "limitations":[
        "Scalar trained Q1/Q4, not llama.cpp IQ1/IQ2/IQ3",
        "Q4 islands restore BOTH weight precision and full floating activation",
        "Q1 layers use Python XNOR/popcount and two activation bitplanes; Q4 uses FP32 GEMV",
        "Packed payload counts redundant Q1 islands in memory; not a compressed deployed model",
        "Python CPU same-process decision throughput; not token/s or isolated-model RSS",
        "Native Godot materials and production providers untouched",
        "STOP host-legal from first turn; no forced continuation",
        "Matched seeds and host random streams, divergent post-action states",
        "Synthetic oracle labels; no human aesthetic judgments"]}
    (out/"results.json").write_text(json.dumps(record,indent=2)+"\n")
    manifest=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,
               "sha256":digest(p)} for p in sorted(out.rglob("*"))
               if p.is_file() and p.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL007_COMPLETE","source":record["source_sha"],
       "optimizer_steps":record["new_optimizer_updates"],"heldout":len(y),
       "pngs":record["first_pass_png_n"],
       "variants":{k:{"cells":v["mean_unique_cells"],"stop":v["stop_episodes"],
                       "branch":v["actions"].get("BRANCH",0),
                       "retrace":v["actions"].get("RETRACE",0)}
                       for k,v in results.items()}},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",default="ql003-source")
    p.add_argument("--out",default="ql007-out")
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test() if args.self_test else run(args)
