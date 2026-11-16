#!/usr/bin/env python3
"""QL-005: real packed XNOR-popcount Q1-weight / sign-activation worm inference.

Research-only synthetic CyberSand-like action policy. NOT GGUF IQ1, native
CyberSand painting, SIMD acceleration, an architectural artist or game integration.
Frozen original QL-003/QL-004 model artifacts are independently rehashed.
"""
import argparse
import json
import math
import os
import random
import resource
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_worm import (ACTIONS, FAMILIES, Policy, digest, load_packed,
                        pack_codes, unpack_codes, save_packed, rollout, timing)
from ql002_branch_retrace import examples, assess
from ql004_integer_repair import Int8ActivationPackedPolicy, meta

PARENT_QL003 = "9b4103c48127e5c8538403e2a4d9e2525fbe8755"
PARENT_QL004 = "e421d69bf16d9e567f02815707c9743713e0d13b"
LAYERS = "abc"


def verified_source(root, expected):
    data = json.loads((root/"results.json").read_text())
    assert data["source_sha"] == expected, (root, data["source_sha"])
    manifest = json.loads((root/"MANIFEST.json").read_text())
    assert len(manifest) > 100
    for row in manifest:
        p = root/row["path"]
        assert p.is_file() and p.stat().st_size == row["bytes"]
        assert digest(p) == row["sha256"], row["path"]
    return data, len(manifest)


def q1_shapes():
    p = Policy()
    return {k: {"bits":1, "shape":list(getattr(p,k).weight.shape)} for k in LAYERS}


class PackedBitSerialQ1:
    """Actual packed XNOR/popcount; never expands code to float during a bit layer.

    Sign input ~= mean(abs(input)) * sign(input); zero has positive sign.
    Other selected layers are deliberately decoded-FP32 hybrid controls.
    Python integer bit_count is a REFERENCE kernel, not a SIMD speedup.
    """
    def __init__(self, archive, bit_layers="abc", gains=(1.,1.,1.), correction=None,
                 zero_aware=False):
        self.bit_layers = bit_layers
        self.zero_aware = zero_aware
        self.gains = np.asarray(gains,dtype=np.float32)
        self.correction = np.zeros(len(ACTIONS),np.float32) if correction is None else np.asarray(correction,np.float32)
        self.layers = []
        self.packed_bytes = 0
        self.working_bytes = self.gains.nbytes+self.correction.nbytes
        shape = q1_shapes()
        with np.load(archive,allow_pickle=False) as f:
            for k in LAYERS:
                n, d = shape[k]["shape"]
                scale = f[k+"_scale"].astype(np.float32)
                bias = f[k+"_bias"].astype(np.float32)
                packed = f[k+"_packed"]
                codes = unpack_codes(packed,1,n*d).reshape(n,d)
                row_bits = [int.from_bytes(np.packbits(row,bitorder="little").tobytes(),
                                           "little") for row in codes]
                weights = (codes.astype(np.float32)*2-1)*scale[:,None] if k not in bit_layers else None
                self.layers.append((d,row_bits,scale,bias,weights))
                self.packed_bytes += math.ceil(n*d/8)+scale.nbytes+bias.nbytes
                self.working_bytes += scale.nbytes+bias.nbytes + (0 if weights is None else weights.nbytes)
                self.working_bytes += n*math.ceil(d/8) if k in bit_layers else 0

    def __call__(self, features):
        x=np.asarray(features.detach().cpu(),dtype=np.float32)
        if x.ndim == 1: x=x[None,:]
        outputs=[]
        for entry in x:
            z=entry
            for index,(d,rows,scales,bias,weights) in enumerate(self.layers):
                if LAYERS[index] in self.bit_layers:
                    assert len(z)==d
                    sign_word=int.from_bytes(np.packbits((z>=0).astype(np.uint8),
                                             bitorder="little").tobytes(),"little")
                    if self.zero_aware:
                        # 2 bitplanes: activation sign and nonzero. Crucial for
                        # sparse one-hot material/legal/heading/family features.
                        active=z!=0
                        active_word=int.from_bytes(np.packbits(active.astype(np.uint8),
                                                 bitorder="little").tobytes(),"little")
                        count=active_word.bit_count()
                        multiplier=np.float32((np.abs(z[active]).mean() if count else 0.)*self.gains[index])
                        z=np.asarray([(count-2*((word ^ sign_word)&active_word).bit_count())*s*multiplier+b
                            for word,s,b in zip(rows,scales,bias)],dtype=np.float32)
                    else:
                        multiplier=np.float32(np.abs(z).mean()*self.gains[index])
                        z=np.asarray([(d-2*(word ^ sign_word).bit_count())*s*multiplier+b
                            for word,s,b in zip(rows,scales,bias)],dtype=np.float32)
                else:
                    z=weights @ z + bias
                if index<2: z=np.tanh(z)
            outputs.append(z+self.correction)
        return torch.from_numpy(np.stack(outputs))


class STEActivationRepair(nn.Module):
    """Only three activation gains and six output biases train; Q1 codes fixed."""
    def __init__(self, packed_path, zero_aware=False):
        super().__init__()
        self.zero_aware=zero_aware
        with np.load(packed_path,allow_pickle=False) as f:
            for k in LAYERS:
                shape=tuple(q1_shapes()[k]["shape"])
                raw=unpack_codes(f[k+"_packed"],1,math.prod(shape)).reshape(shape)
                weights=(raw.astype(np.float32)*2.-1.) * f[k+"_scale"][:,None]
                self.register_buffer(k+"_weight",torch.from_numpy(weights.copy()))
                self.register_buffer(k+"_bias",torch.from_numpy(f[k+"_bias"].copy()))
        self.gains=nn.Parameter(torch.ones(3))
        self.correction=nn.Parameter(torch.zeros(len(ACTIONS)))

    def forward(self,x):
        for i,k in enumerate(LAYERS):
            signed=torch.where(x>=0,torch.ones_like(x),-torch.ones_like(x))
            if self.zero_aware:
                active=(x!=0)
                mean_abs=x.abs().sum(dim=-1,keepdim=True)/active.sum(dim=-1,keepdim=True).clamp(min=1)
                q=signed*active.to(x.dtype)*mean_abs
            else:
                q=signed*x.abs().mean(dim=-1,keepdim=True)
            # Straight-through activation quantization, real quantized forward.
            x=F.linear((x+(q-x).detach())*self.gains[i],
                       getattr(self,k+"_weight"),getattr(self,k+"_bias"))
            if i<2: x=torch.tanh(x)
        return x+self.correction


def repair(source, train, reference, steps, seed, personality_kl, zero_aware=False):
    torch.manual_seed(seed)
    model=STEActivationRepair(source,zero_aware=zero_aware)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=0.)
    x,y,_=train
    counts=torch.bincount(y,minlength=len(ACTIONS)).float()
    sample_weights=(1./counts[y]).double()
    sample_weights/=sample_weights.sum()
    losses=[]
    for _ in range(steps):
        ix=torch.multinomial(sample_weights,128,replacement=True)
        forward=model(x[ix])
        with torch.no_grad(): prior=reference(x[ix])
        loss=F.cross_entropy(forward,y[ix])+personality_kl*F.kl_div(
            F.log_softmax(forward,dim=-1),F.softmax(prior,dim=-1),reduction="batchmean")
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        with torch.no_grad():model.gains.clamp_(.10,5.)
        losses.append(float(loss.detach()))
    moments=sum(int(torch.count_nonzero(s["exp_avg"]).item()>0)
                for s in opt.state.values())
    result=PackedBitSerialQ1(source,gains=model.gains.detach().tolist(),
                             correction=model.correction.detach().tolist(),zero_aware=zero_aware)
    with torch.no_grad():
        check=torch.randn(18,27)
        difference=float((model(check)-result(check)).abs().max())
    assert difference<2e-5,difference
    assert moments>=2
    return result, {"optimizer":"AdamW","steps":steps,"seed":seed,
        "loss_first":losses[0],"loss_last":losses[-1],
        "nonzero_moment_tensors":moments,"trained_parameters":9,
        "personality_kl":personality_kl,"zero_aware":zero_aware,
        "gains":model.gains.detach().tolist(),
        "output_correction":model.correction.detach().tolist(),
        "forward_parity_max_abs":difference,
        "packed_weight_codes_unchanged":True},opt.state_dict()


def self_test():
    import tempfile
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        torch.manual_seed(78)
        net=Policy()
        path,metadata,_=save_packed(root,"test",net,(1,1,1))
        f=load_packed(path,metadata)
        binary=PackedBitSerialQ1(path)
        # The bit-packed operation must equal dequant Q1 for constant-magnitude inputs.
        for width in (27,64,6):
            row=np.random.default_rng(width).choice([-1.,1.],size=width).astype(np.float32)
            with np.load(path,allow_pickle=False) as archive:
                layer={27:"a",64:"b",6:"c"}[width]
                if layer=="c":continue
                shape=tuple(metadata[layer]["shape"])
                code=unpack_codes(archive[layer+"_packed"],1,math.prod(shape)).reshape(shape)
                w=(code.astype(np.float32)*2.-1.)*archive[layer+"_scale"][:,None]
                p=int.from_bytes(np.packbits((row>=0).astype(np.uint8),bitorder="little").tobytes(),"little")
                rows=[int.from_bytes(np.packbits(z,bitorder="little").tobytes(),"little") for z in code]
                actual=np.asarray([(width-2*(b^p).bit_count())*s for b,s in zip(rows,archive[layer+"_scale"])])
                assert np.allclose(actual,w@row,atol=2e-5)
        out=binary(torch.randn(5,27))
        assert out.shape==(5,len(ACTIONS)) and bool(torch.isfinite(out).all())
        assert f(torch.randn(2,27)).shape==(2,len(ACTIONS))
        zero_aware=PackedBitSerialQ1(path,zero_aware=True)
        ste=STEActivationRepair(path,zero_aware=True)
        check=torch.randn(4,27);check[0,::2]=0
        assert float((zero_aware(check)-ste(check)).abs().max())<2e-5
    print("QL005 bitpacked XNOR/popcount and STE-parity self-test PASS")


def run(args):
    torch.set_num_threads(1)
    np.random.seed(77207)
    torch.manual_seed(77207)
    raw=Path(args.source_dir)
    prior,verified_a=verified_source(raw,PARENT_QL003)
    repair_dir=Path(args.repair_dir)
    repaired,verified_b=verified_source(repair_dir,PARENT_QL004)
    root=Path(args.out)
    (root/"gallery").mkdir(parents=True,exist_ok=True)
    (root/"models").mkdir(exist_ok=True)

    train,_,bad1=examples(range(2100,2180),33102)
    held,hist,bad2=examples(range(9100,9132),33103)
    assert bad1==bad2==0 and all(hist.get(a,0)>0 for a in ACTIONS)
    q1_path=raw/"models/bal-q1.npz"
    frank_path=raw/"models/frank-reg-a-bal-bc-q1.npz"
    distilled_path=repair_dir/"models/q1-scales-distilled.npz"
    ref=load_packed(q1_path,meta("bal-q1",prior))
    variants={
       "q1-fp32-decode":ref,
       "q1-int8-activation":Int8ActivationPackedPolicy(q1_path,meta("bal-q1",prior)),
       "q1-xnor-all":PackedBitSerialQ1(q1_path),
       "q1-xnor-zero-aware":PackedBitSerialQ1(q1_path,zero_aware=True),
       "q1-xnor-hidden":PackedBitSerialQ1(q1_path,"bc"),
       "q1-xnor-middle":PackedBitSerialQ1(q1_path,"b"),
       "franken-q1-xnor-all":PackedBitSerialQ1(frank_path),
       "franken-q1-xnor-zero-aware":PackedBitSerialQ1(frank_path,zero_aware=True),
       "prior-distilled-q1-xnor-all":PackedBitSerialQ1(distilled_path),
       "prior-distilled-q1-xnor-zero-aware":PackedBitSerialQ1(distilled_path,zero_aware=True),
    }
    opt_reports={}
    for name,kl,za in (("q1-xnor-ce-repair",0.,False),
                       ("q1-xnor-personality-repair",1.5,False),
                       ("q1-xnor-zero-aware-repair",1.5,True)):
        net,diag,optimizer=repair(q1_path,train,ref,args.steps,77207,kl,zero_aware=za)
        variants[name]=net
        opt_reports[name]=diag
        torch.save({"optimizer":optimizer,"diagnostic":diag,
           "source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
           "parent_ql003_sha":PARENT_QL003,"parent_ql004_sha":PARENT_QL004,
           "source_packed_archive_sha256":digest(q1_path)},
           root/"models"/(name+"-optimizer.pt"))

    x,y,legal=held
    fp_pred=ref(x).masked_fill(~legal,-1e9).argmax(1)
    all_receipts=[]
    reports={}
    visual_seeds=(9311,9323,9349,9371)
    for name,net in variants.items():
        scores=net(x)
        masked=scores.masked_fill(~legal,-1e9).argmax(1)
        episodes=[]
        for seed in visual_seeds:
            for fam in FAMILIES:
                ep,steps=rollout(net,name,fam,seed,root,horizon=96)
                episodes.append(ep)
                all_receipts.append({"variant":name,"seed":seed,"family":fam,
                    "first_pass":True,"steps":steps})
        counted=Counter()
        for ep in episodes:counted.update(ep["action_histogram"])
        reports[name]={"heldout":assess(net,held),
            "host_masked_difference_from_q1_dequant":int((masked!=fp_pred).sum()),
            "heldout_n":len(y),
            "latency":timing(net,x,n=120),
            "episodes":len(episodes),"stopped":sum(bool(e["stop_count"]) for e in episodes),
            "mean_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),4),
            "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),4),
            "actions":dict(counted),
            "family_distinct_pngs_by_seed":[len({e["png_sha256"] for e in episodes
                 if e["seed"]==seed}) for seed in visual_seeds],
            "working_weight_cache_bytes":getattr(net,"working_bytes",None),
            "packed_payload_bytes":getattr(net,"packed_bytes",None),
            "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes]}
    controls={}
    for name in ("random","fidget","wall"):
        episodes=[]
        for seed in visual_seeds:
            for fam in FAMILIES:
                ep,steps=rollout(None,name,fam,seed,root,horizon=96)
                episodes.append(ep)
                all_receipts.append({"variant":name,"seed":seed,"family":fam,
                     "first_pass":True,"steps":steps})
        controls[name]={"mean_cells":round(sum(e["unique_cells"] for e in episodes)/len(episodes),4),
            "mean_turns":round(sum(e["turns"] for e in episodes)/len(episodes),4),
            "original_pngs":[{"path":e["png"],"sha256":e["png_sha256"]} for e in episodes]}

    (root/"raw_decisions.json").write_text(json.dumps(all_receipts,separators=(",",":"))+"\n")
    results={"schema":"ql005-bitserial-1","source_sha":os.environ.get("QL_SOURCE_SHA",os.environ.get("GITHUB_SHA","local")),
       "pinned_sources":{"ql003":PARENT_QL003,"ql004":PARENT_QL004},
       "verified_original_manifest_entries":[verified_a,verified_b],
       "source_archive_hashes":{"q1":digest(q1_path),"frank":digest(frank_path),
                    "distilled":digest(distilled_path)},
       "new_optimizer_updates":3*args.steps,"repair":opt_reports,
       "heldout_n":len(y),"heldout_teacher_histogram":hist,
       "visual_seeds":list(visual_seeds),"horizon":96,
       "variants":reports,"controls":controls,
       "raw_decisions_sha256":digest(root/"raw_decisions.json"),
       "rss_process_max_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "limitations":[
          "This is actual packed Python integer XNOR/popcount, not an optimized C/C++ SIMD kernel",
          "Q1 per-row scalar weight codes, not GGUF IQ1/IQ2/IQ3 and not a pretrained SLM",
          "Sign-only A1 destroys zero-valued sparse features; A1Z two-bitplane sign+nonzero recovers sparsity at extra activation-bit cost",
          "A1Z is ternary activation quantization, NOT a one-bit activation or an IQ1 codec",
          "Gains and correction remain float32 and are included in runtime working memory",
          "No native Godot/CyberSand painting; external synthetic host-legal W0-like worlds",
          "STOP is never suppressed; differing episode lengths constrain aesthetic comparisons",
          "Same world start and host RNG stream, but later trajectories diverge",
          "Process-wide max RSS includes Python, NumPy, Torch and all simultaneously loaded variants",
       ]}
    (root/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    inventory=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,
                "sha256":digest(p)} for p in sorted(root.rglob("*"))
                if p.is_file() and p.name!="MANIFEST.json"]
    (root/"MANIFEST.json").write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"status":"QL005_COMPLETE","source_sha":results["source_sha"],
        "optimizer_updates":3*args.steps,"heldout":len(y),"pngs":len(list((root/"gallery").glob("*.png"))),
        "variant_summary":{k:{"cells":v["mean_cells"],"stopped":v["stopped"],
          "branch":v["actions"].get("BRANCH",0),"retrace":v["actions"].get("RETRACE",0),
          "ms":v["latency"]["same_process_p50_ms"]} for k,v in reports.items()}},
          sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",default="ql003-source")
    p.add_argument("--repair-dir",default="ql004-source")
    p.add_argument("--out",default="ql005-out")
    p.add_argument("--steps",type=int,default=40)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    self_test() if a.self_test else run(a)
