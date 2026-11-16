#!/usr/bin/env python3
"""QL-014: genuine 1/2/4-bit contextual action-table QAT on a frozen Q1/Q4 worm.

The state-conditioned residual uses the *observed* number of fresh legal
neighbours, not a new simulator feature. Compare an exact floating-point
calibrator, packed Q4/Q2/Q1 QAT revisions, QL-013's untrained novelty guard,
and the unchanged parent. The low-bit parent and host menus are immutable.
This is a tiny reference quantization codec, NOT llama.cpp IQ/GGUF.
"""
import argparse
import copy
import json
import os
import resource
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import ACTIONS, FAMILIES, digest, timing
from ql002_branch_retrace import examples
from ql005_bitserial import verified_source
from ql007_mixed_bridges import Hybrid, PARENT_SHA
from ql011_observable_repair import observable_targets
from ql013_contextual_retrace import (novelty_mask, ContextSuppress, cohort,
                                        conditional_report)

RETRACE = ACTIONS.index("RETRACE")


def codes_and_scales(table, bits):
    """Per-row midpoint codebook, with exact zero-only rows; genuine bit codes."""
    a = np.asarray(table, dtype=np.float32)
    assert a.shape == (4, len(ACTIONS)) and bits in (1, 2, 4)
    scales = np.max(np.abs(a), axis=1).astype(np.float32)
    safe = np.where(scales == 0., 1., scales).astype(np.float32)
    levels = (1 << bits) - 1
    codes = np.rint((np.clip(a / safe[:, None], -1., 1.) + 1.) * levels / 2.)
    codes = np.clip(codes, 0, levels).astype(np.uint8)
    codes[scales == 0] = 0
    return codes, scales


def pack_codes(codes, bits):
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    assert np.all(flat < 2**bits)
    packed = bytearray((len(flat)*bits + 7)//8)
    for i, c in enumerate(flat):
        offset = i*bits
        packed[offset//8] |= int(c) << (offset % 8)
    return bytes(packed)


def unpack_codes(packed, bits, shape=(4, len(ACTIONS))):
    count = int(np.prod(shape))
    assert len(packed) == (count*bits+7)//8
    mask = (1 << bits) - 1
    return np.array([(packed[(i*bits)//8] >> ((i*bits)%8)) & mask
                     for i in range(count)], dtype=np.uint8).reshape(shape)


def decoded_table(codes, scales, bits):
    levels = (1 << bits) - 1
    decoded = ((codes.astype(np.float32) * (2./levels)) - 1.) * scales[:, None]
    decoded[scales == 0] = 0.
    return decoded


class FrontierTable(nn.Module):
    """24-parameter residual, table indexed by observed fresh-neighbour count."""
    def __init__(self, parent, bits=32, init=None):
        super().__init__()
        self.parent = parent
        self.bits = bits
        self.table = nn.Parameter(torch.zeros(4,len(ACTIONS)))
        if init is not None:
            with torch.no_grad():
                self.table.copy_(init)

    def quant_table(self):
        if self.bits == 32:
            return self.table
        a = self.table
        scale = a.detach().abs().amax(1, keepdim=True)
        safe = scale.clamp_min(1e-6)
        levels = (1 << self.bits) - 1
        codes = ((a/safe).clamp(-1.,1.)+1.) * (levels/2.)
        rounded = codes.round()
        dequant = (rounded*(2./levels)-1.)*scale
        dequant = torch.where(scale > 0, dequant, torch.zeros_like(dequant))
        return a + (dequant-a).detach()

    def correction(self, x):
        return self.quant_table()[novelty_mask(x).long()]

    def forward(self, x):
        with torch.no_grad():
            parent = self.parent(x)
        return parent + self.correction(x)

    def payload(self):
        a = self.table.detach().numpy().astype(np.float32)
        if self.bits == 32:
            return {"bits":32,"bytes":int(a.nbytes),"table":a}
        codes,scales = codes_and_scales(a,self.bits)
        packed = pack_codes(codes,self.bits)
        recovered = unpack_codes(packed,self.bits)
        assert np.array_equal(recovered,codes)
        expected = decoded_table(codes,scales,self.bits)
        actual = self.quant_table().detach().numpy()
        assert np.allclose(expected,actual,rtol=0,atol=4e-7)
        return {"bits":self.bits,"bytes":len(packed)+scales.nbytes,
                "packed":np.frombuffer(packed,dtype=np.uint8).copy(),
                "scales":scales, "decoded":expected}


def fit(net, train, labels, updates, seed, learning_rate):
    """Real AdamW optimizer. Frozen parent logits, balanced visible target labels."""
    x, _, legal = train
    torch.manual_seed(seed)
    with torch.no_grad():
        base = net.parent(x).detach()
        base_dist = F.softmax(base.masked_fill(~legal,-1e9),dim=1)
    counts = torch.bincount(labels,minlength=len(ACTIONS)).float().clamp(min=1)
    prob = (1./counts[labels]).double()
    prob /= prob.sum()
    optimizer = torch.optim.AdamW([net.table],lr=learning_rate,weight_decay=0.)
    trace = []
    for step in range(updates):
        ix = torch.multinomial(prob,256,replacement=True)
        values = (base[ix] + net.correction(x[ix])).masked_fill(~legal[ix],-1e9)
        ce = F.cross_entropy(values,labels[ix])
        kl = F.kl_div(F.log_softmax(values,dim=1),base_dist[ix],reduction="batchmean")
        # Penalise backing away from parent RETRACE only when no fresh direction
        # exists; in novel-rich states the observed oracle handles the penalty.
        exhausted = (novelty_mask(x[ix]) == 0) & legal[ix,RETRACE]
        if bool(exhausted.any()):
            parent_p = base[ix][exhausted].masked_fill(
                ~legal[ix][exhausted],-1e9).softmax(1)[:,RETRACE]
            current_p = values[exhausted].softmax(1)[:,RETRACE]
            anch = F.relu(parent_p-current_p).mean()
        else:
            anch = values.sum()*0
        loss = ce + .35*kl + .55*anch
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.append({"step":step+1,"loss":float(loss.detach()),
                      "oracle_ce":float(ce.detach()),
                      "parent_kl":float(kl.detach()),
                      "exhausted_retrace_anchor":float(anch.detach())})
    state=optimizer.state_dict()
    assert len(state["state"]) == 1 and bool(torch.count_nonzero(net.table.detach()))
    assert bool(torch.count_nonzero(next(iter(state["state"].values()))["exp_avg"]))
    return state,trace


def heldout(net, held, oracle, parent):
    x, _, legal = held
    with torch.no_grad():
        logits=net(x)
        choice=logits.masked_fill(~legal,-1e9).argmax(1)
        pc=parent(x).masked_fill(~legal,-1e9).argmax(1)
        if isinstance(net,FrontierTable):
            first=net.quant_table()[0].expand(len(x),-1)
            frozen=parent(x)+first
            static=frozen.masked_fill(~legal,-1e9).argmax(1)
            adaptive=int((choice!=static).sum())
        else:
            adaptive=None
    return {"n":len(x),"host_masked_legal":int(legal.gather(1,choice[:,None]).sum()),
            "observable_correct":int((choice==oracle).sum()),
            "changed_vs_parent":int((choice!=pc).sum()),
            "live_vs_frozen_first_row":adaptive,
            "greedy_retrace":int((choice==RETRACE).sum()),
            "greedy_stop":int((choice==ACTIONS.index("STOP")).sum())}


def self_test():
    rng=np.random.default_rng(1414)
    for bits in (1,2,4):
        for a in (rng.normal(size=(4,len(ACTIONS))).astype("float32"),
                  np.zeros((4,len(ACTIONS)),dtype="float32")):
            c,s=codes_and_scales(a,bits)
            packed=pack_codes(c,bits)
            assert np.array_equal(c,unpack_codes(packed,bits))
            v=decoded_table(c,s,bits)
            assert np.isfinite(v).all()
            assert len(packed)==len(ACTIONS)*4*bits//8
            class Fake:
                def __call__(self,x):
                    return torch.zeros(len(x),len(ACTIONS))
            net=FrontierTable(Fake(),bits,torch.from_numpy(a))
            assert np.allclose(net.payload()["decoded"],v,atol=4e-7,rtol=0)
            x=torch.zeros(4,27)
            x[1,3]=1
            x[2,[3,7]]=1
            x[3,[3,7,11]]=1
            assert torch.allclose(net.correction(x),net.quant_table(),atol=1e-6)
    print("QL014 SELF-TEST PASS: real bitpack, exact unpack, zero rows, ST quant and fresh count")


def run(args):
    self_test()
    torch.set_num_threads(1)
    torch.manual_seed(14011)
    np.random.seed(14011)
    out=Path(args.out)
    (out/"models").mkdir(parents=True,exist_ok=True)
    (out/"gallery").mkdir(exist_ok=True)
    parent_dir=Path(args.parent)
    _,verified=verified_source(parent_dir,PARENT_SHA)
    q1=parent_dir/"models"/"bal-q1.npz"
    q4=parent_dir/"models"/"bal-q4.npz"
    parent=Hybrid(q1,q4,"b")
    train,_,invalid=examples(range(2100,2180),33102)
    held,_,invalid_held=examples(range(9100,9132),33103)
    assert invalid==invalid_held==0 and len(held[1])==4352
    oracle=observable_targets(train)
    oracle_held=observable_targets(held)
    assert bool(train[2].gather(1,oracle[:,None]).all())
    variants={"parent-q1-q4b":parent,"parent-context-guard":ContextSuppress(parent)}
    trained={}
    reference=FrontierTable(parent)
    state,trace=fit(reference,train,oracle,args.fp_steps,14101,.06)
    trained["fp32-table"]=(reference,state,trace)
    variants["fp32-table"]=reference
    for bits in (4,2,1):
        net=FrontierTable(parent,bits,reference.table.detach().clone())
        opt,log=fit(net,train,oracle,args.qat_steps,14100+bits,.025)
        name="qat-q"+str(bits)+"-table"
        trained[name]=(net,opt,log)
        variants[name]=net
    for name,(net,opt,trace) in trained.items():
        payload=net.payload()
        torch.save({"source_sha":os.environ.get("QL_SOURCE_SHA","local"),
                    "parent_source_sha":PARENT_SHA,"parent_q1_sha256":digest(q1),
                    "parent_q4_sha256":digest(q4),"bits":net.bits,
                    "table_fp32":net.table.detach().clone(),
                    "optimizer":opt,"updates":len(trace),"trace":trace,
                    "packed_payload":payload},
                   out/"models"/(name+"-optimizer.pt"))
        if net.bits != 32:
            np.savez_compressed(out/"models"/(name+"-packed.npz"),
                                bits=np.array(net.bits), packed=payload["packed"],
                                scales=payload["scales"])
    raw=[]
    seeds=(10311,10323,10347,10351,10367,10379)
    results={}
    for name,model in variants.items():
        result={"heldout":heldout(model,held,oracle_held,parent),
                "conditional":conditional_report(model,held,oracle_held),
                "random_10pct":cohort(model,name,seeds,out,.10,raw),
                "no_random_diagnostic":cohort(model,name,seeds,out,0.,raw),
                "same_runner_latency":timing(model,held[0],n=96)}
        if name in trained:
            result["trained_steps"]=len(trained[name][2])
            result["bits"]=model.bits
            result["packed_payload_bytes"]=model.payload()["bytes"]
            result["trainable_params"]=model.table.numel()
        results[name]=result
    (out/"raw_decisions.json").write_text(json.dumps(raw,separators=(",",":"))+"\n")
    report={"schema":"ql014-frontier-q-at-v1",
            "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
            "parent_source_sha":PARENT_SHA,
            "parent_q1_sha256":digest(q1),"parent_q4_sha256":digest(q4),
            "parent_manifest_entries":verified,
            "train_n":len(train[1]),"heldout_n":len(held[1]),
            "optimizer_updates":args.fp_steps+3*args.qat_steps,
            "seeds":seeds,"variants":results,
            "original_png_n":len(list((out/"gallery").glob("*.png"))),
            "first_pass_trajectories":len(raw),
            "raw_decisions_sha256":digest(out/"raw_decisions.json"),
            "rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "limitations":["Tiny 24-scalar QAT residual, not GGUF IQ1/IQ2/IQ3.",
                           "Decoded table for Python runtime; no low-bit speedup implied.",
                           "Training on synthetic W0 observable target, not native CyberSand.",
                           "STOP host-legal from tick zero; random overrides may truncate trails.",
                           "Disjoint train and heldout world seeds; no real-player aesthetic labels.",
                           "Parent Q1/XNOR and Q4-middle model codes unmodified.",
                           "Only synthetic research files and immutable Actions checkpoints."]}
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    manifest=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,
               "sha256":digest(p)} for p in sorted(out.rglob("*"))
              if p.is_file() and p.name!="MANIFEST.json"]
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"QL014_COMPLETE","head":report["source_sha"],
                      "updates":report["optimizer_updates"],
                      "pngs":report["original_png_n"],
                      "results":{name:{"oracle":v["heldout"]["observable_correct"],
                            "cells":v["random_10pct"]["mean_unique_cells"],
                            "retrace":v["random_10pct"]["total_retraces"],
                            "packed_bytes":v.get("packed_payload_bytes")}
                            for name,v in results.items()}},sort_keys=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",default="ql003-source")
    parser.add_argument("--out",default="ql014-out")
    parser.add_argument("--fp-steps",type=int,default=64)
    parser.add_argument("--qat-steps",type=int,default=48)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else run(args)
