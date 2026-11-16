#!/usr/bin/env python3
"""QL-006: two-bitplane Q1 activation-gating experiment, genuinely optimized.
Synthetic CyberSand-like host-legal worm research only. Not GGUF IQ, native
painting, a game change, or a promise of integer SIMD acceleration.
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
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import (ACTIONS, FAMILIES, WormWorld, digest, load_packed,
                        unpack_codes, rollout, timing)
from ql002_branch_retrace import examples, assess
from ql004_integer_repair import meta
from ql005_bitserial import (PackedBitSerialQ1, q1_shapes, verified_source)

PARENT_QL003 = "9b4103c48127e5c8538403e2a4d9e2525fbe8755"
PARENT_QL005 = "fab9f69b3b773bba25eb598ac600bd44eadf0cde"
LAYERS = "abc"


class GatedQ1:
    """Actual packed-XNOR, two activation bitplanes; dynamic sparsification.

    A feature survives iff nonzero and |value| >= threshold * maximum
    magnitude of that layer's input. Equal-amplitude surviving activations
    use mean |value|, like QL-005 A1Z. This is A1Z ternary, not IQ1.
    """
    def __init__(self, packed, thresholds=(0., 0., 0.),
                 gains=(1., 1., 1.), correction=None):
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.gains = np.asarray(gains, dtype=np.float32)
        self.correction = (np.zeros(len(ACTIONS), dtype=np.float32) if
                           correction is None else np.asarray(correction, dtype=np.float32))
        self.layers = []
        self.packed_bytes = 0
        self.working_bytes = self.thresholds.nbytes+self.gains.nbytes+self.correction.nbytes
        with np.load(packed, allow_pickle=False) as f:
            for name in LAYERS:
                n,d = q1_shapes()[name]["shape"]
                raw = unpack_codes(f[name+"_packed"], 1, n*d).reshape(n,d)
                rowbits = [int.from_bytes(np.packbits(row,bitorder="little").tobytes(),"little")
                           for row in raw]
                scale = f[name+"_scale"].astype(np.float32)
                bias = f[name+"_bias"].astype(np.float32)
                self.layers.append((d,rowbits,scale,bias))
                self.packed_bytes += math.ceil(n*d/8)+scale.nbytes+bias.nbytes
                self.working_bytes += n*math.ceil(d/8)+scale.nbytes+bias.nbytes

    def __call__(self, features):
        x = np.asarray(features.detach().cpu(), dtype=np.float32)
        if x.ndim == 1: x = x[None,:]
        results = []
        for feature in x:
            z=feature
            for layer,(d,rows,scales,biases) in enumerate(self.layers):
                assert len(z)==d
                threshold = float(self.thresholds[layer])*float(np.max(np.abs(z)))
                active = (z!=0)&(np.abs(z)>=threshold)
                n_active = int(active.sum())
                activebits = int.from_bytes(np.packbits(active.astype(np.uint8),
                          bitorder="little").tobytes(),"little")
                signbits = int.from_bytes(np.packbits((z>=0).astype(np.uint8),
                          bitorder="little").tobytes(),"little")
                gain = np.float32((np.abs(z[active]).mean() if n_active else 0.)*self.gains[layer])
                z = np.asarray([(n_active-2*((row^signbits)&activebits).bit_count())*s*gain+b
                      for row,s,b in zip(rows,scales,biases)],dtype=np.float32)
                if layer<2: z=np.tanh(z)
            results.append(z+self.correction)
        return torch.from_numpy(np.stack(results))


class GatedQ1STE(nn.Module):
    """Nine old Q1 weight codes remain immutable; learn only 3 gate fractions,
    3 gains and 6 output corrections, with soft-threshold STE gradients."""
    def __init__(self, packed, thresholds=(.05,.25,.25)):
        super().__init__()
        with np.load(packed,allow_pickle=False) as f:
            for name in LAYERS:
                shape=tuple(q1_shapes()[name]["shape"])
                raw=unpack_codes(f[name+"_packed"],1,math.prod(shape)).reshape(shape)
                weights=(raw.astype(np.float32)*2.-1.)*f[name+"_scale"][:,None]
                self.register_buffer(name+"_weight",torch.from_numpy(weights.copy()))
                self.register_buffer(name+"_bias",torch.from_numpy(f[name+"_bias"].copy()))
        self.thresholds=nn.Parameter(torch.tensor(thresholds,dtype=torch.float32))
        self.gains=nn.Parameter(torch.ones(3))
        self.correction=nn.Parameter(torch.zeros(len(ACTIONS)))

    def forward(self,features):
        x=features
        for i,name in enumerate(LAYERS):
            magnitude=x.abs()
            maximum=magnitude.amax(dim=-1,keepdim=True)
            thresh=self.thresholds[i]*maximum
            hard=((x!=0)&(magnitude>=thresh)).to(x.dtype)
            softness=0.08*maximum+1e-6
            soft=torch.sigmoid((magnitude-thresh)/softness)*(x!=0).to(x.dtype)
            gate=soft+(hard-soft).detach()
            # Preserve exact A1Z inference magnitude in forward, but allow
            # the straight-through gate to tune threshold on backward.
            amplitude=(magnitude*hard).sum(dim=-1,keepdim=True)/hard.sum(
                       dim=-1,keepdim=True).clamp(min=1.)
            q=torch.where(x>=0.,1.,-1.)*gate*amplitude*self.gains[i]
            x=F.linear(q,getattr(self,name+"_weight"),getattr(self,name+"_bias"))
            if i<2: x=torch.tanh(x)
        return x+self.correction

    def inference(self,path):
        return GatedQ1(path,self.thresholds.detach().tolist(),
                       self.gains.detach().tolist(),self.correction.detach().tolist())


def train_repair(path, data, original, steps, seed, penalty):
    torch.manual_seed(seed)
    net=GatedQ1STE(path)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.012,weight_decay=0.)
    x,y,_=data
    class_count=torch.bincount(y,minlength=len(ACTIONS)).float()
    sample_weight=(1./class_count[y]).double()
    sample_weight/=sample_weight.sum()
    traces=[]
    for step in range(steps):
        ix=torch.multinomial(sample_weight,128,replacement=True)
        out=net(x[ix])
        with torch.no_grad():
            prior=original(x[ix])
        loss=F.cross_entropy(out,y[ix])+penalty*F.kl_div(
             F.log_softmax(out,dim=-1),F.softmax(prior,dim=-1),reduction="batchmean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net.thresholds.clamp_(0.,.80)
            net.gains.clamp_(.10,5.)
        traces.append({"step":step+1,"loss":float(loss.detach()),
                       "supervised_nll":float(F.cross_entropy(out.detach(),y[ix])),
                       "thresholds":net.thresholds.detach().tolist()})
    saved=net.inference(path)
    with torch.no_grad():
        sample=torch.randn(32,27)
        sample[0,::3]=0
        parity=float((net(sample)-saved(sample)).abs().max())
    assert parity<3e-5,parity
    state=optimizer.state_dict()
    moments=sum(int(torch.count_nonzero(t["exp_avg"])>0) for t in optimizer.state.values())
    assert moments==3,moments
    report={"steps":steps,"seed":seed,"penalty":penalty,
       "parameters_updated":12,"nonzero_optimizer_moment_tensors":moments,
       "loss_first":traces[0]["loss"],"loss_last":traces[-1]["loss"],
       "packed_q1_bits_unchanged":True,
       "training_target":"host-legal synthetic oracle action labels, NOT logits of a pretrained teacher",
       "gates":saved.thresholds.tolist(),"gains":saved.gains.tolist(),
       "correction":saved.correction.tolist(),"inference_parity_max_abs":parity}
    return saved,state,report,traces


def run(args):
    torch.set_num_threads(1)
    np.random.seed(80606)
    torch.manual_seed(80606)
    parent=Path(args.parent)
    prior,n_verified=verified_source(parent,PARENT_QL003)
    # The QL-005 parent is rehashed independently; its existing PNGs are
    # never regenerated or rewritten by this child experiment.
    legacy,verified_005=verified_source(Path(args.ql005),PARENT_QL005)
    source=parent/"models"/"bal-q1.npz"
    original=GatedQ1(source)
    teacher=load_packed(source,meta("bal-q1",prior))
    out=Path(args.out)
    (out/"models").mkdir(parents=True,exist_ok=True)
    (out/"gallery").mkdir(exist_ok=True)
    train,hist,illegal=examples(range(2100,2180),33102)
    held,held_hist,bad=examples(range(9100,9132),33103)
    assert illegal==bad==0 and all(held_hist.get(a,0)>0 for a in ACTIONS)
    variants={
       "q1-a1z":original,
       "q1-gate-input-20":GatedQ1(source,(.20,0.,0.)),
       "q1-gate-hidden-20":GatedQ1(source,(0.,.20,.20)),
       "q1-gate-all-20":GatedQ1(source,(.20,.20,.20)),
       "q1-gate-all-45":GatedQ1(source,(.45,.45,.45)),
       "q1-gate-hidden-45":GatedQ1(source,(0.,.45,.45)),
    }
    training={}
    for label,penalty in (("q1-gate-ce-repair",0.),
                          ("q1-gate-personality-repair",1.5)):
        net,opt,diagnostic,trace=train_repair(source,train,original,
                                              args.steps,80606,penalty)
        variants[label]=net
        training[label]=diagnostic
        torch.save({"optimizer":opt,"diagnostic":diagnostic,"trace":trace,
           "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
           "parent_ql003_sha":PARENT_QL003,
           "parent_ql005_sha":PARENT_QL005,
           "immutable_parent_archive_sha256":digest(source)},
           out/"models"/(label+"-optimizer.pt"))

    x,y,legal=held
    teacher_scores=teacher(x)
    teacher_masked=teacher_scores.masked_fill(~legal,-1.e9).argmax(dim=1)
    reports={}
    raw=[]
    seeds=(9413,9431,9479,9491)
    for name,net in variants.items():
        predictions=net(x).masked_fill(~legal,-1.e9).argmax(dim=1)
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(net,name,family,seed,out,horizon=96)
                episodes.append(ep)
                raw.append({"variant":name,"mode":"live","seed":seed,
                            "family":family,"first_pass":True,"steps":steps})
        counted=Counter()
        for ep in episodes: counted.update(ep["action_histogram"])
        reports[name]={"heldout":assess(net,held),
          "host_masked_disagreement_q1_teacher_n":int((predictions!=teacher_masked).sum()),
          "heldout_n":len(y),
          "same_runner_latency":timing(net,x,n=50),
          "packed_weight_payload_bytes":net.packed_bytes,
          "working_weight_bytes":net.working_bytes,
          "gate_thresholds":net.thresholds.tolist(),
          "episodes":len(episodes),"mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),4),
          "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),4),
          "stop_episodes":sum(int(bool(e["stop_count"])) for e in episodes),
          "actions":dict(counted),
          "family_distinct_pngs_by_seed":[len({e["png_sha256"] for e in episodes
             if e["seed"]==seed}) for seed in seeds],
          "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes]}
    # Frozen-opening-logit control for the two repaired variants and A1Z.
    for name in ("q1-a1z","q1-gate-ce-repair","q1-gate-personality-repair"):
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                initial=WormWorld(seed,family)
                start=torch.from_numpy(initial.features(0)).unsqueeze(0)
                score=variants[name](start)
                class Frozen:
                    def __init__(self,score): self.score=score
                    def __call__(self,x): return self.score.expand(x.shape[0],-1)
                ep,steps=rollout(Frozen(score),name+"-opening-prior",family,seed,out,horizon=96)
                episodes.append(ep)
                raw.append({"variant":name,"mode":"opening-prior","seed":seed,
                            "family":family,"first_pass":True,"steps":steps})
        reports[name]["frozen_mean_unique_cells"]=round(
             sum(e["unique_cells"] for e in episodes)/len(episodes),4)
        reports[name]["frozen_original_pngs"]=[
             {"path":e["png"],"sha256":e["png_sha256"]} for e in episodes]

    controls={}
    for name in ("random","wall","fidget"):
        episodes=[]
        for seed in seeds:
            for family in FAMILIES:
                ep,steps=rollout(None,name,family,seed,out,horizon=96)
                episodes.append(ep)
                raw.append({"variant":name,"mode":"model-free",
                            "seed":seed,"family":family,"first_pass":True,"steps":steps})
        controls[name]={"mean_unique_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),4),
                        "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),4),
                        "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes]}
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    result={"schema":"ql006-thresholded-two-bitplane-q1-1",
      "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
      "pinned_sources":{"ql003":PARENT_QL003,"ql005":PARENT_QL005},
      "verified_parent_manifest_entries":[n_verified,verified_005],
      "original_q1_archive_sha256":digest(source),
      "new_optimizer_updates":2*args.steps,
      "train_n":len(train[1]),"heldout_n":len(held[1]),
      "heldout_histogram":held_hist,
      "heldout_metric_target":"synthetic oracle labels from examples(), not QL-003 neural logits",
      "visual_seeds":list(seeds),"visual_horizon":96,
      "variants":reports,"controls":controls,"repairs":training,
      "first_pass_png_n":len(list((out/"gallery").glob("*.png"))),
      "raw_decisions_sha256":digest(out/"raw_decisions.json"),
      "process_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "limitations":[
         "Two-bitplane ternary A1Z activations with dynamic threshold, not true 1-bit activations or GGUF IQ1/IQ2/IQ3",
         "Python integer bit_count/XNOR reference implementation; no low-bit SIMD acceleration",
         "Q1 code tensors never update; 12 floating gate/gain/output parameters train per repair",
         "Actual optimizer checkpoints remain only in immutable workflow artifact, not Git",
         "STOP legal from tick zero; longer output is not intrinsically better",
         "Synthetic CyberSand-like worlds, not native Godot material painting",
         "Matched starts and host seeds, not matched state histories after different actions",
         "Process RSS includes NumPy and Torch, not isolated single-model resident memory"]}
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    manifest=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,
               "sha256":digest(p)} for p in sorted(out.rglob("*"))
              if p.is_file() and p.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL006_COMPLETE","source_sha":result["source_sha"],
       "optimizer_updates":2*args.steps,"heldout":len(y),"pngs":result["first_pass_png_n"],
       "variants":{k:{"cells":v["mean_unique_cells"],"stop":v["stop_episodes"],
          "branch":v["actions"].get("BRANCH",0),"retrace":v["actions"].get("RETRACE",0),
          "distinct":v["family_distinct_pngs_by_seed"]} for k,v in reports.items()}},
       sort_keys=True))


def self_test():
    import tempfile
    from quant_worm import Policy, save_packed
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        torch.manual_seed(17)
        path,meta_info,_=save_packed(root,"test",Policy(),(1,1,1))
        plain=PackedBitSerialQ1(path,zero_aware=True)
        gate=GatedQ1(path)
        x=torch.randn(12,27)
        x[::2,::3]=0.
        assert float((plain(x)-gate(x)).abs().max())<3e-5
        ste=GatedQ1STE(path,thresholds=(0.,0.,0.))
        assert float((ste(x)-gate(x)).abs().max())<3e-5
        zero=GatedQ1(path,thresholds=(0.,.5,.5))
        scores=zero(x)
        assert scores.shape==(12,len(ACTIONS)) and bool(torch.isfinite(scores).all())
        model=GatedQ1STE(path)
        loss=model(x).square().mean()
        loss.backward()
        assert model.thresholds.grad is not None
        assert torch.isfinite(model.thresholds.grad).all()
        assert any(abs(float(t))>1e-9 for t in model.thresholds.grad)
    print("QL006 PASS: A1Z zero-gate parity; packed XNOR; genuine threshold STE gradients")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--parent",default="ql003-source")
    p.add_argument("--ql005",default="ql005-source")
    p.add_argument("--out",default="ql006-out")
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    self_test() if a.self_test else run(a)
