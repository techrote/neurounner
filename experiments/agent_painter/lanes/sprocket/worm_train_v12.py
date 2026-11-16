#!/usr/bin/env python3
"""W12: two actual AdamW waves to counter W10 greedy BRANCH collapse.

Frozen root Instruct; train a 142x64x64x6 auxiliary W10-continuation head.
W0 observations are synthetic game-like host-owned local geometry, never native.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, sys, time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/"training"),str(HERE)]
import worm_train_v9 as w9
import worm_train_v10 as w10
import worm_head_sample_v11 as w11
import worm_v4 as objective
from common import profile, sha256_file, write_json
from numerics import configure_cpu

VERSION="sprocket-w12-rare-action-rehearsal-v1"
PARENT_SHA=w11.W10_SHA
PARENT_W9_SHA=w11.W9_SHA
TRAIN_SEEDS=(233,239,241,251)
HELDOUT=(2203,2213,2221)
UPDATES=120
QUOTA={"STRAIGHT":24,"LEFT":12,"RIGHT":12,"BRANCH":6,"RETRACE":8,"STOP":2}
assert sum(QUOTA.values())==64

def class_rows(rows):
    groups={a:[] for a in w9.ACTIONS}
    for i,r in enumerate(rows):
        if r["action"] not in r["observation"]["legal"]:
            raise AssertionError("illegal teacher action")
        groups[r["action"]].append(i)
    missing=[a for a in w9.ACTIONS if not groups[a]]
    if missing:raise ValueError("missing curriculum actions "+str(missing))
    return groups

def balanced(groups,rng):
    indices=[]
    for name,count in QUOTA.items():
        indices.extend(rng.choices(groups[name],k=count))
    rng.shuffle(indices)
    return indices

def class_recall(torch,model,rows):
    x,y=w9.as_tensors(torch,rows)
    with torch.no_grad():
        out=model(x)
        legal=torch.tensor([[a in r["observation"]["legal"] for a in w9.ACTIONS]
                            for r in rows],dtype=torch.bool)
        pred=out.masked_fill(~legal,-1e9).argmax(dim=1)
    result={}
    for a in w9.ACTIONS:
        j=w9.ACTIONS.index(a)
        truth=(y==j)
        result[a]={"true":int(truth.sum()),"predicted":int((pred==j).sum()),
                   "correct":int((truth & (pred==j)).sum())}
    result["macro_recall"]=round(sum(result[a]["correct"]/max(1,result[a]["true"])
                                      for a in w9.ACTIONS)/6,6)
    return result

def stage_checkpoint(torch,head,opt,folder,stage,measure):
    path=folder/(stage+".pt")
    torch.save({"version":VERSION,"stage":stage,"root_frozen":True,
                "head":head.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng_state":torch.random.get_rng_state(),
                "dev":measure},path)
    return {"file":path.name,"sha256":sha256_file(path),"bytes":path.stat().st_size}

def replay_data(torch,w0,support,models):
    result=[];manifest=[]
    for seed in TRAIN_SEEDS:
        for family in w9.FAMILIES:
            for mode,head in models.items():
                rec=w11.probe(torch,w0,head,seed,family,mode)
                manifest.append([seed,family,mode,len(rec["events"])])
                for event in rec["events"]:
                    obs=event["observation"]
                    target=w10.motif_teacher(obs,support)
                    result.append({"observation":obs,"action":target,"family":family,
                                   "legal":obs["legal"],"seed":seed,
                                   "source":"closed-loop-"+mode})
    return result,manifest

def train(torch,w0,support,args,out):
    torch.manual_seed(121212)
    rng=random.Random(121212)
    w9_head=w11.load(torch,args.w9,PARENT_W9_SHA,w9.FEATURE_VERSION)
    frozen=w11.load(torch,args.parent,PARENT_SHA,w10.VERSION)
    model=w9.model_class(torch)()
    model.load_state_dict(frozen.state_dict())
    source,source_manifest=w10.gather(w0,support,TRAIN_SEEDS,steps=128)
    heldout,heldout_manifest=w10.gather(w0,support,HELDOUT,steps=128)
    assert not set(TRAIN_SEEDS)&set(HELDOUT)
    groups=class_rows(source)
    sx,sy=w9.as_tensors(torch,source)
    write_json(out/"dataset-manifest.json",{
        "schema":VERSION,"train_worlds":source_manifest,"heldout_worlds":heldout_manifest,
        "train_counts":dict(Counter(r["action"] for r in source)),
        "heldout_counts":dict(Counter(r["action"] for r in heldout)),
        "quota_per_64":QUOTA,"synthetic":True,"native":False})
    opt=torch.optim.AdamW(model.parameters(),lr=0.00035,weight_decay=0.002)
    stages=[{"name":"frozen-W10","teacher":w9.accuracy(torch,frozen,heldout),
             "class_recall":class_recall(torch,frozen,heldout)}]
    hist=[]
    for stage in ("rare-action-rehearsal","sampled-recovery-replay"):
        if stage=="sampled-recovery-replay":
            replay,manifest=replay_data(torch,w0,support,{"w9-greedy":w9_head,
                                                           "w10-t07":model})
            rgroups=class_rows(replay)
            rx,ry=w9.as_tensors(torch,replay)
            write_json(out/"replay-manifest.json",{
                "source_controllers":list(("w9-greedy","w10-t07")),
                "closed_loop_worlds":manifest,
                "actual_replay_rows":len(replay),
                "label_counts":dict(Counter(r["action"] for r in replay)),
                "synthetic":True})
        for step in range(UPDATES):
            opt.zero_grad(set_to_none=True)
            ids=balanced(groups,rng)
            loss=torch.nn.functional.cross_entropy(model(sx[ids]),sy[ids])
            if stage=="sampled-recovery-replay":
                rids=balanced(rgroups,rng)
                rloss=torch.nn.functional.cross_entropy(model(rx[rids]),ry[rids])
                loss=.55*loss+.45*rloss
            if not torch.isfinite(loss):raise RuntimeError("nonfinite objective")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            if not torch.isfinite(norm) or float(norm)<=0:raise RuntimeError("bad optimizer gradient")
            opt.step()
            if (step+1)%30==0:
                item={"stage":stage,"update":len(hist)+1,"global_update":len(hist)*30+30,
                      "stage_step":step+1,"loss":round(float(loss.detach()),6),
                      "gradient_norm":round(float(norm),6)}
                hist.append(item)
                write_json(out/"partial-optimizer-history.json",hist)
        measure={"name":stage,"updates":UPDATES,"teacher":w9.accuracy(torch,model,heldout),
                 "class_recall":class_recall(torch,model,heldout)}
        measure["checkpoint"]=stage_checkpoint(torch,model,opt,out,stage,measure)
        stages.append(measure)
        write_json(out/"stage-progress.json",stages)
    return w9_head,frozen,model,stages,hist

def main(args):
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    configure_cpu(1)
    import torch
    torch.set_num_threads(1)
    p=profile("sprocket",require_supported=True)
    assert p["model_id"]=="kandivault/sprocket-500m"
    assert p["revision"]=="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert p["subfolder"]=="" and p["framing"]["eos_token_id"]==5
    assert sha256_file(args.parent)==PARENT_SHA
    assert sha256_file(args.w9)==PARENT_W9_SHA
    assert sha256_file(args.w0)=="81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
    assert sha256_file(args.support)=="c555c55ea4fc4bc303159fd3268cef4a0b7d9e5c006e19918ea3433d1f7c95d6"
    assert not set(TRAIN_SEEDS)&(set(w10.TRAIN_SEEDS)|set(w10.HELDOUT)|set(w11.SEEDS))
    assert not set(HELDOUT)&(set(w10.TRAIN_SEEDS)|set(w10.HELDOUT)|set(w11.SEEDS)|set(TRAIN_SEEDS))
    support=objective.load_support(args.support,args.w0)
    write_json(args.output/"provenance.json",{
        "schema":VERSION,"source":os.environ.get("GITHUB_SHA","local"),
        "root_profile":p,"root_frozen":True,"root_updates":0,"lora_updates":0,
        "parent_W10_sha256":PARENT_SHA,"parent_W9_sha256":PARENT_W9_SHA,
        "W0_sha256":sha256_file(args.w0),"support_sha256":sha256_file(args.support),
        "script_sha256":sha256_file(__file__),"planned_adamw_steps":2*UPDATES,
        "heldout_seeds":HELDOUT,"NOT_native":True,"sealed_final_opened":False})
    old,w10head,w12head,stages,hist=train(torch,support.worm,support,args,args.output)
    rows=[];folder=args.output/"png-and-json"
    for seed in HELDOUT:
        for fam in w9.FAMILIES:
            for name,head,mode in (
                ("w9-greedy",old,"w9-greedy"),
                ("w10-greedy",w10head,"w10-greedy"),
                ("w10-t07",w10head,"w10-t07"),
                ("w12-greedy",w12head,"w10-greedy"),
                ("w12-t07",w12head,"w10-t07")):
                r=w11.probe(torch,support.worm,head,seed,fam,mode)
                r["mode"]=name
                support.worm.write_result(r,folder)
                rows.append({"seed":seed,"family":fam,"mode":name,
                    "metrics":r["metrics"],"head_forwards":len(r["head_decisions"])})
            for mode in ("random","wall","fidget"):
                r=support.worm.run(seed,mode,fam,128)
                support.worm.write_result(r,folder)
                rows.append({"seed":seed,"family":fam,"mode":mode,
                             "metrics":r["metrics"],"head_forwards":0})
        write_json(args.output/"partial-original-rollouts.json",rows)
    assert len(rows)==72
    checkpoints=[s["checkpoint"] for s in stages if "checkpoint" in s]
    assert len(checkpoints)==2 and all(
        sha256_file(args.output/x["file"])==x["sha256"] for x in checkpoints)
    final={"status":"complete","version":VERSION,
       "real_aux_optimizer_steps":UPDATES*2,"root_updates":0,
       "real_head_forwards":sum(x["head_forwards"] for x in rows),
       "root_forwards":0,"original_png_json_pairs":len(rows),
       "two_checkpoints_valid":True,"stages":stages,"history":hist,
       "results":rows,"wall_seconds":time.monotonic()-start,
       "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       "native":False,"sealed_final_opened":False}
    write_json(args.output/"summary.json",final)
    print(json.dumps({"status":"complete","optimizer_steps":UPDATES*2,
        "png_pairs":len(rows),"head_forwards":final["real_head_forwards"]}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("parent","w9","w0","support","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    try:main(a)
    except BaseException as e:
        if a.output.exists():
            write_json(a.output/"failure.json",{"error":str(e),"type":type(e).__name__,
                                                "partial_or_failed":True})
        raise
