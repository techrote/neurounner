"""M15: genuinely pretrained frozen backbones, goal-conditioned, host-masked CPU game policies.
Research-only; no native/Workbench access, no model-weight redistribution, no EVAL optimizer exposure.
"""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, time, traceback
from pathlib import Path
import numpy as np
from .core import ACTIONS, DELTA, FAMILIES, World, raster
from .m14_dagger import reachable, advance, oracle, greedy
from .pretrained_cpu import MODELS, text_observation

PROTO = "tiny-policy-m15-pretrained-goal-ood-v1"
SEEDS = {"train": (81001,81002,81003,81004),
         "dev": (82001,82002), "eval": (83001,83002,83003)}
BUDGET = 64

def sha(data):
    return hashlib.sha256(data).hexdigest()

def world(seed, fam, split):
    if seed not in SEEDS[split] or fam not in FAMILIES:
        raise ValueError("split and family are host-authoritative")
    w = World(seed, fam, 96)
    w.blocked[:] = True
    w.protected[:] = False
    w.blocked[27:76,27:76] = False
    r = random.Random(seed*39)
    if split != "eval":
        for i in range(7):
            x = 33 + 5*i
            for y in range(31,70):
                if (y + 3*i + seed)%17 < 7:
                    continue
                w.blocked[y,x] = True
        for i in range(8):
            x,y = r.randrange(31,72),r.randrange(31,72)
            if abs(x-48)+abs(y-48)>9:
                w.blocked[y:y+2,x:x+2] = True
    else:
        variant = (seed-83001)
        if variant == 0:  # offset rings and narrow diagonal gates
            for lo,hi in ((33,67),(39,61)):
                w.blocked[lo,lo:hi+1] = True
                w.blocked[hi,lo:hi+1] = True
                w.blocked[lo:hi+1,lo] = True
                w.blocked[lo:hi+1,hi] = True
                w.blocked[lo+5:lo+9,lo] = False
                w.blocked[hi,hi-10:hi-6] = False
        elif variant == 1:  # multiple U pockets
            for x in (34,45,56,67):
                w.blocked[35:67,x] = True
                w.blocked[35,x:x+7] = True
                w.blocked[53:57,x] = False
                w.blocked[37:42,x] = False
        else:  # staggered horizontal bars and offset passages
            for k,y in enumerate((34,41,50,59,67)):
                w.blocked[y,31:72] = True
                a = 34 + ((k*11+seed)%25)
                w.blocked[y,a:a+6] = False
            w.blocked[30:68,35] = True
            w.blocked[30:68,64] = True
            w.blocked[43:50,35] = False
            w.blocked[55:60,64] = False
    for i in range(9):
        x,y=r.randrange(31,71),r.randrange(31,71)
        if abs(x-48)+abs(y-48)>12:
            w.protected[y:y+2,x:x+2]=True
    w.blocked[45:52,45:52]=False
    w.protected[45:52,45:52]=False
    return w

def prompt(w,goals,idx):
    if idx>=len(goals):
        gx,gy=w.x,w.y
    else:
        gx,gy=goals[idx]
    text=text_observation(w.observe(),w.legal())
    return text.replace("; choose:",f"; goal dx {gx-w.x} dy {gy-w.y}; waypoint {idx}; choose:")

def collect(split):
    rows=[]
    for seed in SEEDS[split]:
        for fam in FAMILIES:
            for episode in (0,1):
                w=world(seed,fam,split)
                goals=reachable(w); idx=0
                rng=random.Random(seed*101+episode+FAMILIES.index(fam))
                for t in range(48):
                    idx=advance(w,goals,idx)
                    label=int(oracle(w,goals,idx))
                    legal=tuple(int(i) for i in w.legal())
                    assert label in legal
                    rows.append({"text":prompt(w,goals,idx),"label":label,
                                 "legal":legal,"id":[seed,fam,episode,t],
                                 "goal_index":idx})
                    if episode==0 or rng.random()<.56:
                        action=label
                    elif rng.random()<.5:
                        action=int(greedy(w,goals,idx))
                    else:
                        action=rng.choice([i for i in legal if i!=4] or [4])
                    w.apply(int(action))
                    if w.stopped:break
    return rows

def embed(torch,backbone,tok,rows):
    chunks=[]
    for i in range(0,len(rows),8):
        token=tok([x["text"] for x in rows[i:i+8]],padding=True,
                  truncation=True,max_length=128,return_tensors="pt")
        with torch.inference_mode():
            h=backbone(**token).last_hidden_state
            last=token["attention_mask"].sum(-1)-1
            chunks.append(h[torch.arange(len(last)),last].float().cpu())
    return torch.cat(chunks)

def masks(torch,rows):
    return torch.tensor([[a in r["legal"] for a in range(5)] for r in rows],
                        dtype=torch.bool)

def fit(torch,X,Y,M,DX,DY,DM,kind):
    import torch.nn as nn
    dim=X.shape[-1]
    if kind=="linear":net=nn.Linear(dim,5)
    else:net=nn.Sequential(nn.Linear(dim,96),nn.ReLU(),nn.Linear(96,5))
    mu=X.mean(0);sigma=X.std(0).clamp_min(.1)
    X=((X-mu)/sigma).clamp(-8,8);DX=((DX-mu)/sigma).clamp(-8,8)
    opt=torch.optim.AdamW(net.parameters(),lr=.003 if kind=="mlp" else .008,
                          weight_decay=.003)
    rng=torch.Generator().manual_seed(15007 if kind=="mlp" else 15009)
    best=float("inf");weights=None;history=[]
    for step in range(240):
        ix=torch.randint(len(X),(48,),generator=rng)
        logits=net(X[ix]).masked_fill(~M[ix],-1e6)
        loss=nn.functional.cross_entropy(logits,Y[ix])
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if step%24==23:
            with torch.inference_mode():
                v=nn.functional.cross_entropy(net(DX).masked_fill(~DM,-1e6),DY)
            metric=float(v)
            history.append({"step":step+1,"dev_masked_ce":round(metric,5)})
            if metric<best:
                best=metric
                weights={k:t.detach().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(weights);net.eval().requires_grad_(False)
    return net,mu,sigma,history

def trajectory(torch,model,seed,fam,kind,heads,tok,backbone,out):
    w=world(seed,fam,"eval");goals=reachable(w);idx=0;events=[]
    rng=random.Random(seed*99+FAMILIES.index(fam))
    for t in range(BUDGET):
        idx=advance(w,goals,idx)
        legal=tuple(int(a) for a in w.legal())
        before=w.state_sha256()
        if kind=="oracle":
            a=int(oracle(w,goals,idx))
        elif kind=="greedy":
            a=int(greedy(w,goals,idx))
        elif kind=="random":
            a=rng.choice([j for j in legal if j!=4] or [4])
        else:
            net,mu,sd=heads[kind]
            token=tok([prompt(w,goals,idx)],truncation=True,max_length=128,
                      return_tensors="pt")
            with torch.inference_mode():
                h=backbone(**token).last_hidden_state[0,-1].float().cpu()
                logits=net(((h-mu)/sd).clamp(-8,8))
            # Host no-premature-STOP rule; explicit stop only if done/trapped.
            permitted=legal if idx==len(goals) or not any(a!=4 for a in legal) else tuple(a for a in legal if a!=4)
            a=max(permitted,key=lambda k:(float(logits[k]),-k))
        if a not in legal:raise AssertionError("Illegal action")
        row={"before_sha256":before,"legal":[ACTIONS[j] for j in legal],
             "goal_index":idx,"observation_sha256":sha(prompt(w,goals,idx).encode())}
        row.update(w.apply(int(a)))
        events.append(row)
        if w.stopped:break
    idx=advance(w,goals,idx)
    rec={"protocol":PROTO,"model":model,"seed":seed,"family":fam,"policy":kind,
         "events":events,"state":w.state_sha256(),"cells":int(w.visited.sum()),
         "branches":w.branch_nodes(),"retraces":w.revisit_landings,"waypoints":idx,
         "geometry":sha(w.blocked.tobytes()+w.protected.tobytes()),"synthetic_only":True}
    # Independent host-state replay, including legal menus and per-turn state.
    test=world(seed,fam,"eval");target=reachable(test);position=0
    for row in events:
        position=advance(test,target,position)
        assert test.state_sha256()==row["before_sha256"]
        assert [ACTIONS[i] for i in test.legal()]==row["legal"]
        assert row["goal_index"]==position
        assert sha(prompt(test,target,position).encode())==row["observation_sha256"]
        got=test.apply(ACTIONS.index(row["action"]))
        assert all(got[k]==row[k] for k in got)
    assert test.state_sha256()==rec["state"] and int(test.visited.sum())==rec["cells"]
    assert test.branch_nodes()==rec["branches"] and test.revisit_landings==rec["retraces"]
    raster(w,2).save(out.with_suffix(".png"))
    out.with_suffix(".json").write_text(json.dumps(rec,sort_keys=True)+"\n")
    return {k:rec[k] for k in ("seed","family","policy","cells","branches","retraces","waypoints")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,choices=MODELS)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error("Immutable output path already exists")
    a.out.mkdir(parents=True);(a.out/"checkpoints").mkdir();(a.out/"rollouts").mkdir()
    result={"protocol":PROTO,"model":a.model,"model_source":MODELS[a.model],
            "seeds":SEEDS,"synthetic_only":True,"status":"STARTED",
            "backbone_optimizer_updates":0,"new_head_optimizer_updates":0}
    try:
        import torch
        from transformers import AutoTokenizer,AutoModel
        from huggingface_hub import HfApi
        torch.manual_seed(15001);torch.set_num_threads(2)
        model_id,revision,_=MODELS[a.model]
        assert HfApi().model_info(model_id,revision=revision).sha==revision
        tok=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        backbone=AutoModel.from_pretrained(model_id,revision=revision,
                   trust_remote_code=False,use_safetensors=a.model!="opt-125m")
        backbone.eval().requires_grad_(False)
        source={split:collect(split) for split in SEEDS}
        result["split_sizes"]={s:len(v) for s,v in source.items()}
        result["split_sha256"]={s:sha(json.dumps([r["id"] for r in v]).encode())
                                for s,v in source.items()}
        H={s:embed(torch,backbone,tok,source[s]) for s in SEEDS}
        Y={s:torch.tensor([r["label"] for r in source[s]],dtype=torch.long) for s in SEEDS}
        M={s:masks(torch,source[s]) for s in SEEDS}
        heads={}
        for kind in ("linear","mlp"):
            net,mu,sd,history=fit(torch,H["train"],Y["train"],M["train"],
                                  H["dev"],Y["dev"],M["dev"],kind)
            heads[kind]=(net,mu,sd)
            checkpoint=a.out/"checkpoints"/(kind+".pt")
            torch.save({"protocol":PROTO,"model_id":model_id,"revision":revision,
                        "kind":kind,"network":net.state_dict(),"mean":mu,"std":sd,
                        "training_split_sha256":result["split_sha256"]["train"]},checkpoint)
            with torch.inference_mode():
                masked=net(((H["eval"]-mu)/sd).clamp(-8,8)).masked_fill(~M["eval"],-1e6)
                valid=masked.argmax(-1)
            result[kind]={"head_params":sum(t.numel() for t in net.parameters()),
                          "eval_teacher_agreement":round(float((valid==Y["eval"]).float().mean()),4),
                          "eval_illegal":int((~M["eval"]).gather(1,valid[:,None]).sum()),
                          "history":history,"checkpoint_sha256":sha(checkpoint.read_bytes())}
            result["new_head_optimizer_updates"]+=240
        rows=[]
        for seed in SEEDS["eval"]:
            for fam in FAMILIES:
                for kind in ("linear","mlp","oracle","greedy","random"):
                    stem=f"{seed}-{fam}-{kind}"
                    rows.append(trajectory(torch,a.model,seed,fam,kind,heads,tok,backbone,
                                           a.out/"rollouts"/stem))
        (a.out/"rows.json").write_text(json.dumps(rows,sort_keys=True,indent=2)+"\n")
        result["rollouts"]=len(rows)
        result["summary"]={kind:{metric:round(statistics.mean(r[metric] for r in rows if r["policy"]==kind),3)
                          for metric in ("cells","branches","retraces","waypoints")}
                           for kind in ("linear","mlp","oracle","greedy","random")}
        from PIL import Image
        for seed in SEEDS["eval"]:
            img=Image.new("RGB",(192*5,192*3),(243,243,243))
            for fi,fam in enumerate(FAMILIES):
                for pi,kind in enumerate(("linear","mlp","oracle","greedy","random")):
                    with Image.open(a.out/"rollouts"/f"{seed}-{fam}-{kind}.png") as pix:
                        img.paste(pix,(pi*192,fi*192))
            img.save(a.out/f"gallery-{seed}.png")
        result["status"]="COMPLETED"
    except Exception:
        result["status"]="FAILED";result["error"]=traceback.format_exc()
        (a.out/"failure.txt").write_text(result["error"])
        raise
    finally:
        (a.out/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        manifest={str(p.relative_to(a.out)):sha(p.read_bytes()) for p in sorted(a.out.rglob("*"))
                  if p.is_file() and p.name!="manifest.json"}
        (a.out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
        print(json.dumps({"model":a.model,"status":result["status"],
                          "updates":result["new_head_optimizer_updates"],
                          "summary":result.get("summary")},sort_keys=True))
if __name__=="__main__":
    main()
