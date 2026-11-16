"""Pinned pretrained tiny transformer -> frozen CPU embeddings -> trained legal head.

Game-facing SYNTHETIC research only: no native readback, gameplay source edits,
pretrained-backbone gradients, provider changes, or sealed TRAIN/FINAL.
"""
from __future__ import annotations
import argparse,hashlib,json,resource,statistics,time,traceback
from pathlib import Path
import numpy as np
from .core import ACTIONS,FAMILIES,World,teacher,choose_random,choose_wall,raster,run,replay,digest
from .train import generate

MODELS={
 "smollm2-135m":("HuggingFaceTB/SmolLM2-135M-Instruct","12fd25f77366fa6b3b4b768ec3050bf629380bac","apache-2.0"),
 "opt-125m":("facebook/opt-125m","27dcfa74d334bc871f3234de431e71c6eeba5dd6","other; non-commercial research restriction"),
 "pythia-160m":("EleutherAI/pythia-160m","50f5173d932e8e61f858120bcb800b97af589f46","apache-2.0")}
SPLITS={"TRAIN":list(range(1001,1009)),"DEV":list(range(2001,2005)),
        "EVAL":[5001,5003,5009,5011]}
assert len(set(sum(SPLITS.values(),[])))==sum(map(len,SPLITS.values()))

def text_observation(v, legal):
    v=np.asarray(v,np.float32)
    if v.shape!=(162,) or not np.isfinite(v).all(): raise ValueError("Bad 162-vector")
    if not legal or any(type(i) is not int or i not in range(5) for i in legal):
        raise ValueError("Bad host menu")
    planes=v[:147].reshape(3,7,7)
    grid="/".join("".join("X" if planes[2,y,x] else "#" if planes[0,y,x]
                  else "o" if planes[1,y,x] else "." for x in range(7)) for y in range(7))
    return (f"synthetic 96x96 worm; theme {FAMILIES[int(v[151:154].argmax())]};"
            f" heading {ACTIONS[int(v[147:151].argmax())]}; turn {int(v[154]*96)};"
            f" legal {' '.join(ACTIONS[i] for i in legal)}; view {grid}; choose:")

def samples(name):
    x,y,m,ids=generate(SPLITS[name])
    rows=[]
    for i,identity in enumerate(ids):
        if identity[2]%12: continue
        legal=tuple(int(j) for j in np.flatnonzero(m[i]))
        if int(y[i]) not in legal:raise RuntimeError("Illegal teacher label")
        rows.append({"text":text_observation(x[i],legal),"y":int(y[i]),
                     "legal":legal,"id":identity})
    return rows

def encode(torch,backbone,tok,rows):
    pieces=[];timings=[]
    for offset in range(0,len(rows),4):
        tokens=tok([x["text"] for x in rows[offset:offset+4]],
                   return_tensors="pt",padding=True,truncation=True,max_length=128)
        start=time.perf_counter()
        with torch.inference_mode():
            hidden=backbone(**tokens).last_hidden_state
            last=tokens["attention_mask"].sum(1)-1
            pieces.append(hidden[torch.arange(hidden.shape[0]),last].float().cpu())
        timings.append((time.perf_counter()-start)/len(tokens["input_ids"]))
    return torch.cat(pieces),timings

def fit(torch,x,dev,train_rows,dev_rows):
    torch.manual_seed(522)
    mu=x.mean(0);sd=x.std(0).clamp_min(.05)
    x=((x-mu)/sd).clamp(-8,8);dev=((dev-mu)/sd).clamp(-8,8)
    labels=torch.tensor([r["y"] for r in train_rows])
    val_labels=torch.tensor([r["y"] for r in dev_rows])
    mask=torch.tensor([[j in r["legal"] for j in range(5)] for r in train_rows])
    val_mask=torch.tensor([[j in r["legal"] for j in range(5)] for r in dev_rows])
    head=torch.nn.Linear(x.shape[1],5);untouched=torch.nn.Linear(x.shape[1],5)
    untouched.load_state_dict(head.state_dict())
    opt=torch.optim.AdamW(head.parameters(),lr=.012,weight_decay=.005)
    best=-1;chosen=None;history=[]
    for ep in range(36):
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(999+ep))
        for ids in order.split(64):
            loss=torch.nn.functional.cross_entropy(head(x[ids]).masked_fill(~mask[ids],-1e6),labels[ids])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            logits=head(dev).masked_fill(~val_mask,-1e6)
            acc=float((logits.argmax(-1)==val_labels).float().mean())
            ce=float(torch.nn.functional.cross_entropy(logits,val_labels))
        history.append({"epoch":ep+1,"dev_acc":acc,"dev_ce":ce})
        if acc>best:
            best=acc;chosen={k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(chosen)
    return head,untouched,mu,sd,history

def score(torch,head,x,mu,sd,rows):
    with torch.inference_mode():
        logits=head(((x-mu)/sd).clamp(-8,8))
        allowed=torch.tensor([[j in r["legal"] for j in range(5)] for r in rows])
        choice=logits.masked_fill(~allowed,-1e6).argmax(-1)
        target=torch.tensor([r["y"] for r in rows])
    return {"n":len(rows),"teacher_agreement":round(float((choice==target).float().mean()),5),
            "illegal_choices":int((~allowed).gather(1,choice[:,None]).sum()),
            "action_counts":{ACTIONS[i]:int((choice==i).sum()) for i in range(5)}}

def manifest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name!="manifest.json"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=MODELS)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error("Evidence directory already exists")
    args.out.mkdir(parents=True)
    model_id,revision,license=MODELS[args.model]
    report={"status":"STARTED","protocol":"tiny-policy-frozen-pretrained-m1",
            "model_id":model_id,"revision":revision,"license":license,
            "synthetic_only":True,"native":False,"backbone_optimizer_steps":0,
            "split_seeds":SPLITS,"action_vocabulary":ACTIONS}
    started=time.perf_counter()
    try:
        import torch
        from transformers import AutoTokenizer,AutoModel
        from huggingface_hub import HfApi
        torch.set_num_threads(2)
        info=HfApi().model_info(model_id,revision=revision)
        if info.sha!=revision:raise RuntimeError("Model revision mismatch")
        load=time.perf_counter()
        tok=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        backbone=AutoModel.from_pretrained(model_id,revision=revision,
                   trust_remote_code=False,use_safetensors=args.model!="opt-125m")
        backbone.eval().requires_grad_(False)
        report["load_seconds"]=round(time.perf_counter()-load,3)
        report["parameters"]=sum(p.numel() for p in backbone.parameters())
        report["model_class"]=type(backbone).__name__
        data={split:samples(split) for split in SPLITS}
        report["splits"]={k:{"n":len(v),"identity_sha256":digest([r["id"] for r in v])}
                          for k,v in data.items()}
        vec={};timings={}
        for part in SPLITS:vec[part],timings[part]=encode(torch,backbone,tok,data[part])
        report["warm_sample_latency_ms"]=round(statistics.median(timings["EVAL"])*1000,2)
        head,untrained,mu,sd,history=fit(torch,vec["TRAIN"],vec["DEV"],data["TRAIN"],data["DEV"])
        report["head_optimizer_steps"]=36*((len(data["TRAIN"])+63)//64)
        report["head_params"]=sum(p.numel() for p in head.parameters())
        report["best_dev_accuracy"]=max(row["dev_acc"] for row in history)
        report["eval_trained"]=score(torch,head,vec["EVAL"],mu,sd,data["EVAL"])
        report["eval_untrained"]=score(torch,untrained,vec["EVAL"],mu,sd,data["EVAL"])
        (args.out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        torch.save({"head":head.state_dict(),"mean":mu,"std":sd,
                    "model_id":model_id,"revision":revision,
                    "identities":report["splits"]},args.out/"trained-decision-head.pt")
        def policy(w,rng):
            inp=tok([text_observation(w.observe(),w.legal())],
                    return_tensors="pt",truncation=True,max_length=128)
            with torch.inference_mode():
                hidden=backbone(**inp).last_hidden_state
                pos=int(inp["attention_mask"].sum()-1)
                logits=head(((hidden[0,pos].float()-mu)/sd).clamp(-8,8))
            return max(w.legal(),key=lambda a:(float(logits[a]),-a))
        policy.__name__="trained_frozen_backbone"
        rows=[]
        for seed in [5001,5003]:
            for family in FAMILIES:
                for name,selector in [("trained",policy),("teacher",teacher),
                                      ("random",choose_random),("wall",choose_wall)]:
                    w=World(seed,family,budget=48)
                    receipt=run(w,selector,seed*21+FAMILIES.index(family))
                    replay(receipt)
                    folder=args.out/"rollouts";folder.mkdir(exist_ok=True)
                    ident=f"{seed}-{family}-{name}"
                    (folder/(ident+".json")).write_text(json.dumps(receipt,sort_keys=True)+"\n")
                    raster(w,3).save(folder/(ident+".png"))
                    rows.append({"seed":seed,"family":family,"policy":name,
                                 "steps":len(receipt["events"]),"unique":receipt["unique_cells"],
                                 "branches":receipt["branch_nodes"],"retraces":receipt["retrace_steps"],
                                 "stop":receipt["stop"]})
        (args.out/"closed-loop-rows.json").write_text(json.dumps(rows,indent=2)+"\n")
        report["rollouts"]={name:{"n":len(group),"mean_unique":round(statistics.mean(r["unique"] for r in group),2),
                                "mean_branch":round(statistics.mean(r["branches"] for r in group),2),
                                "early_stops":sum(r["stop"] for r in group)}
                            for name in ["trained","teacher","random","wall"]
                            for group in [[r for r in rows if r["policy"]==name]]}
        report["peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["status"]="COMPLETED"
    except Exception:
        report["status"]="FAILED";report["error"]=traceback.format_exc()
        (args.out/"failure.txt").write_text(report["error"])
        raise
    finally:
        report["wall_seconds"]=round(time.perf_counter()-started,2)
        (args.out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        (args.out/"manifest.json").write_text(json.dumps(manifest(args.out),indent=2,sort_keys=True)+"\n")
        print(json.dumps({"status":report["status"],"model":model_id,
               "eval_trained":report.get("eval_trained"),"rollouts":report.get("rollouts")},indent=2))

if __name__=="__main__":main()
