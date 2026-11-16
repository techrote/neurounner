#!/usr/bin/env python3
"""W15: contrastive short-history motif continuation and branch timing.

Root-Instruct pinned, entirely frozen. The learned object is a small auxiliary
head continued from SHA-admitted W14. W0 is an external, read-only synthetic host.
Counterfactual short histories are labeled synthetic, never native observations.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, resource, sys, time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/"training"),str(HERE)]
import worm_train_v9 as w9
import worm_train_v10 as w10
import worm_train_v13 as w13
import worm_train_v14 as w14
import worm_head_sample_v11 as w11
import worm_v4 as objective
from common import profile,sha256_file,write_json
from numerics import configure_cpu

VERSION="sprocket-w15-short-history-motif-contrast-v1"
W14_SHA="1c4decfe00fb89f8ccd44269cb6693c5706dcd793705f925a75143fbc3392c35"
W13_SHA=w14.W13_SHA
W9_SHA=w14.W9_SHA
W0_SHA=w14.W0_SHA
SUPPORT_SHA=w14.SUPPORT_SHA
TRAIN_SEEDS=(313,317,331,337)
HELDOUT=(2503,2521,2531)
UPDATES=120
MOTIF={"greebles":"STRAIGHT","pipework":"LEFT","architecture":"RIGHT"}
TARGET={"greebles":"LEFT","pipework":"RIGHT","architecture":"LEFT"}

def teacher(obs,support):
    """A modest history-sensitive motif teacher, NOT native outcome feedback."""
    legal=obs["legal"]
    default=w10.motif_teacher(obs,support)
    family=obs["family"]
    last=obs["last_actions"]
    # A branch followed by another immediate branch tends to create tangled
    # repetition. Veto it only when an ordinary legal fresh route exists.
    h=obs["heading"]
    dirs=((1,0),(0,1),(-1,0),(0,-1))
    fresh=[]
    for a,rot in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1)):
        dx,dy=dirs[(h+rot)%4]
        if a in legal and obs["local_5x5"][2+dy][2+dx]==".":
            fresh.append(a)
    if default=="BRANCH" and last and last[-1]=="BRANCH" and fresh:
        return next(a for a in ("STRAIGHT","LEFT","RIGHT") if a in fresh)
    if len(last)>=3 and last[-3:]==[MOTIF[family]]*3 and TARGET[family] in fresh:
        # Distinct families switch their local motif after short persisted runs:
        # hatching, elbows, and corridor-counterturns respectively.
        return TARGET[family]
    return default

def worlds(w0,support,seeds):
    rows,manifest=w10.gather(w0,support,seeds,steps=128)
    for r in rows:
        r["action"]=teacher(r["observation"],support)
    assert rows and all(r["action"] in r["observation"]["legal"] for r in rows)
    return rows,manifest

def contrast_pairs(rows,support):
    """Explicit counterfactual last-four histories; separate from W0 states."""
    pairs=[]
    for row in rows:
        o=row["observation"]
        if o["tick"]>=124:continue
        a=copy.deepcopy(o);b=copy.deepcopy(o)
        a["last_actions"]=[]
        b["last_actions"]=[MOTIF[o["family"]]]*3
        ta,tb=teacher(a,support),teacher(b,support)
        if ta!=tb and ta in a["legal"] and tb in b["legal"]:
            pairs.append(({"observation":a,"action":ta,"family":o["family"]},
                          {"observation":b,"action":tb,"family":o["family"]}))
    if len(pairs)<24:
        raise RuntimeError("insufficient contrast pairs: "+str(len(pairs)))
    return pairs

def pair_measure(torch,head,pairs):
    a,b=zip(*pairs)
    xa,ya=w9.as_tensors(torch,a);xb,yb=w9.as_tensors(torch,b)
    legal=w13.legal_mask(torch,a)
    with torch.no_grad():
        pa=head(xa).masked_fill(~legal,-1e4).argmax(1)
        pb=head(xb).masked_fill(~legal,-1e4).argmax(1)
    return {"pairs":len(pairs),"argmax_switched":int((pa!=pb).sum()),
            "both_targets":int(((pa==ya)&(pb==yb)).sum()),
            "a_accuracy":round(float((pa==ya).float().mean()),6),
            "b_accuracy":round(float((pb==yb).float().mean()),6)}

def pair_contrast(torch,head,pairs,rng,n=32):
    picks=[pairs[rng.randrange(len(pairs))] for _ in range(n)]
    a,b=zip(*picks)
    xa,ya=w9.as_tensors(torch,a);xb,yb=w9.as_tensors(torch,b)
    sa=head(xa);sb=head(xb)
    mask=w13.legal_mask(torch,a)
    sa=sa.masked_fill(~mask,-1e4)
    sb=sb.masked_fill(~mask,-1e4)
    aa=sa.gather(1,ya[:,None]).squeeze(1)-sa.gather(1,yb[:,None]).squeeze(1)
    bb=sb.gather(1,yb[:,None]).squeeze(1)-sb.gather(1,ya[:,None]).squeeze(1)
    return (torch.nn.functional.softplus(1-aa)+torch.nn.functional.softplus(1-bb)).mean()

def collect_replay(torch,w0,support,heads):
    rows=[];manifest=[]
    for seed in TRAIN_SEEDS:
        for fam in w9.FAMILIES:
            for name,head,temp in heads:
                rec=w14.rollout(torch,w0,head,seed,fam,name,temp)
                manifest.append({"seed":seed,"family":fam,"mode":name,"steps":len(rec["events"])})
                for event in rec["events"]:
                    obs=event["observation"]
                    rows.append({"observation":obs,"action":teacher(obs,support),
                                 "family":fam,"legal":obs["legal"],"seed":seed,
                                 "source":"closed-loop-"+name})
    return rows,manifest

def checkpoint(torch,head,opt,out,name,dev):
    path=out/(name+".pt")
    torch.save({"version":VERSION,"stage":name,"root_frozen":True,
                "head":head.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng_state":torch.random.get_rng_state(),
                "heldout":dev},path)
    return {"file":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path)}

def eval_model(torch,head,rows,pairs):
    return {"teacher":w9.accuracy(torch,head,rows),
            "branch":w13.branch_diagnostic(torch,head,rows),
            "classes":w13.class_recall(torch,head,rows),
            "short_history_contrast":pair_measure(torch,head,pairs)}

def train(torch,w0,support,w14head,out):
    torch.manual_seed(151515)
    rng=random.Random(151515)
    train_rows,train_worlds=worlds(w0,support,TRAIN_SEEDS)
    heldout,heldout_worlds=worlds(w0,support,HELDOUT)
    train_pairs=contrast_pairs(train_rows,support)
    heldout_pairs=contrast_pairs(heldout,support)
    assert set(TRAIN_SEEDS).isdisjoint(HELDOUT)
    x,y=w9.as_tensors(torch,train_rows)
    mask=w13.legal_mask(torch,train_rows)
    groups=w13.class_rows(train_rows)
    pos,neg=w13.eligible_indices(train_rows)
    write_json(out/"dataset-manifest.json",{
        "version":VERSION,"train_worlds":train_worlds,"heldout_worlds":heldout_worlds,
        "train_counts":dict(Counter(r["action"] for r in train_rows)),
        "heldout_counts":dict(Counter(r["action"] for r in heldout)),
        "train_counterfactual_pairs":len(train_pairs),
        "heldout_counterfactual_pairs":len(heldout_pairs),
        "train_obs_sha256":hashlib.sha256(json.dumps(
            [r["observation"] for r in train_rows],sort_keys=True,
            separators=(",",":")).encode()).hexdigest(),
        "pairs_change_only":"last_actions; engine-legal list unchanged",
        "counterfactual_reachability_unproven":True,"native":False})
    model=w9.model_class(torch)()
    model.load_state_dict(w14head.state_dict(),strict=True)
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.00013,weight_decay=.003)
    stages=[{"name":"frozen-W14","evaluation":eval_model(torch,w14head,heldout,heldout_pairs)}]
    history=[];stage1=None
    for stidx,stage in enumerate(("history-contrast","onpolicy-motif-replay")):
        if stidx:
            stage1=w9.model_class(torch)()
            stage1.load_state_dict(model.state_dict(),strict=True);stage1.eval()
            replay,rm=collect_replay(torch,w0,support,[
                ("frozen-W14",w14head,.7),
                ("W15-stage1",stage1,.7)])
            rx,ry=w9.as_tensors(torch,replay)
            rmask=w13.legal_mask(torch,replay)
            write_json(out/"replay-manifest.json",{
                "sources":rm,"actual_rows":len(replay),
                "labels":dict(Counter(r["action"] for r in replay)),
                "synthetic_host":True,"not_native":True})
        for step in range(UPDATES):
            opt.zero_grad(set_to_none=True)
            ids=w13.balanced(groups,rng)
            ce=torch.nn.functional.cross_entropy(
                model(x[ids]).masked_fill(~mask[ids],-1e4),y[ids])
            branch=w14.calibrated_branch_loss(torch,model,x,mask,pos,neg,rng)
            pair=pair_contrast(torch,model,train_pairs,rng)
            if stidx:
                rid=[rng.randrange(len(replay)) for _ in range(64)]
                rce=torch.nn.functional.cross_entropy(
                    model(rx[rid]).masked_fill(~rmask[rid],-1e4),ry[rid])
                loss=.25*ce+.20*branch+.30*pair+.25*rce
            else:
                loss=.40*ce+.20*branch+.40*pair
            if not bool(torch.isfinite(loss)):raise ValueError("nonfinite W15 loss")
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            if not bool(torch.isfinite(grad)) or float(grad)<=0:
                raise ValueError("nonfinite/zero W15 gradient")
            opt.step()
            if (step+1)%30==0:
                history.append({"stage":stage,"stage_step":step+1,
                    "global_step":stidx*UPDATES+step+1,
                    "loss":round(float(loss.detach()),6),
                    "gradient_norm":round(float(grad),6)})
                write_json(out/"partial-optimizer-history.json",history)
        measure={"name":stage,"real_updates":UPDATES,
                 "evaluation":eval_model(torch,model,heldout,heldout_pairs)}
        measure["checkpoint"]=checkpoint(torch,model,opt,out,stage,measure)
        stages.append(measure)
        write_json(out/"stage-progress.json",stages)
    return stage1,model,stages,history

def motifs(rec):
    events=rec["events"]
    actions=[e["applied"] for e in events]
    turns=sum(a in ("LEFT","RIGHT") for a in actions)
    runs=sum(actions[i]==actions[i-1]==actions[i-2]
             for i in range(2,len(actions)))
    branch_ticks=[i for i,a in enumerate(actions) if a=="BRANCH"]
    clustered=sum(b-a<=3 for a,b in zip(branch_ticks,branch_ticks[1:]))
    blocked=sum(e["observation"]["local_5x5"][2][3]=="#" for e in events)
    blocked_recovery=sum(e["observation"]["local_5x5"][2][3]=="#" and
                         e["applied"] in ("LEFT","RIGHT","RETRACE") for e in events)
    return {"turns":turns,"three_action_repetitions":runs,
            "branch_bursts_gap_le_3":clustered,
            "blocked_front_observations":blocked,
            "blocked_front_recovery_actions":blocked_recovery}

def main(args):
    if args.smoke_only:
        assert all(len(s)==64 for s in (W14_SHA,W13_SHA,W9_SHA,W0_SHA,SUPPORT_SHA))
        assert len(TRAIN_SEEDS)==4 and len(HELDOUT)==3
        assert set(TRAIN_SEEDS).isdisjoint(set(HELDOUT+w14.TRAIN_SEEDS+w14.HELDOUT+
                w13.TRAIN_SEEDS+w13.HELDOUT+w10.TRAIN_SEEDS+w10.HELDOUT))
        print("W15 source-only smoke PASS; zero optimizer updates",flush=True)
        return
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    configure_cpu(1)
    import torch
    torch.set_num_threads(1)
    p=profile("sprocket",require_supported=True)
    assert p["model_id"]=="kandivault/sprocket-500m"
    assert p["revision"]=="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert p["subfolder"]=="" and p["framing"]["eos_token_id"]==5
    for f,sha in ((args.w14,W14_SHA),(args.w13,W13_SHA),(args.w9,W9_SHA),
                  (args.w0,W0_SHA),(args.support,SUPPORT_SHA)):
        assert sha256_file(f)==sha,(f,"parent/source SHA mismatch")
    sup=objective.load_support(args.support,args.w0)
    write_json(args.output/"provenance.json",{
        "version":VERSION,"GITHUB_SHA":os.getenv("GITHUB_SHA","local"),
        "run_id":os.getenv("GITHUB_RUN_ID","local"),
        "run_attempt":os.getenv("GITHUB_RUN_ATTEMPT","local"),
        "script_sha256":sha256_file(__file__),"root_profile":p,
        "W14_final_sha256":W14_SHA,"W13_sha256":W13_SHA,"W9_sha256":W9_SHA,
        "W0_sha256":W0_SHA,"support_sha256":SUPPORT_SHA,
        "actual_root_forwards":0,"root_updates":0,"lora_updates":0,
        "planned_aux_AdamW_steps":UPDATES*2,
        "native":False,"sealed_final_opened":False})
    h14=w11.load(torch,args.w14,W14_SHA,w14.VERSION)
    h13=w11.load(torch,args.w13,W13_SHA,w13.VERSION)
    h9=w11.load(torch,args.w9,W9_SHA,w9.FEATURE_VERSION)
    stage1,model,stages,hist=train(torch,sup.worm,sup,h14,args.output)
    entries=[];gallery=args.output/"png-and-json"
    for seed in HELDOUT:
        for fam in w9.FAMILIES:
            for name,head,temp in (
                ("w9-greedy",h9,0.),("w13-greedy",h13,0.),
                ("w14-greedy",h14,0.),("w14-t07",h14,.7),
                ("w15-stage1-greedy",stage1,0.),
                ("w15-final-greedy",model,0.),("w15-final-t07",model,.7)):
                rec=w14.rollout(torch,sup.worm,head,seed,fam,name,temp)
                sup.worm.write_result(rec,gallery)
                entries.append({"seed":seed,"family":fam,"mode":name,
                    "metrics":rec["metrics"],"motif":motifs(rec),
                    "actual_head_forwards":len(rec["head_decisions"])})
            for mode in ("random","wall","fidget"):
                rec=sup.worm.run(seed,mode,fam,128)
                sup.worm.write_result(rec,gallery)
                entries.append({"seed":seed,"family":fam,"mode":mode,
                    "metrics":rec["metrics"],"motif":motifs(rec),"actual_head_forwards":0})
        write_json(args.output/"partial-original-rollouts.json",entries)
    assert len(entries)==90 and all(e["metrics"]["invalid_raw_choices"]==0 for e in entries)
    cps=[x["checkpoint"] for x in stages if "checkpoint" in x]
    assert len(cps)==2 and all(sha256_file(args.output/c["file"])==c["sha256"] for c in cps)
    write_json(args.output/"summary.json",{
        "status":"complete","version":VERSION,"real_aux_optimizer_steps":UPDATES*2,
        "root_forwards":0,"root_updates":0,"lora_updates":0,
        "real_head_forwards":sum(x["actual_head_forwards"] for x in entries),
        "original_png_json_pairs":len(entries),"stages":stages,"history":hist,
        "results":entries,"wall_seconds":time.monotonic()-start,
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoints_verified":True,"native":False})
    print(json.dumps({"status":"complete","real_optimizer_steps":UPDATES*2,
        "png_pairs":len(entries)}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("w14","w13","w9","w0","support","output"):
        ap.add_argument("--"+n,type=Path)
    ap.add_argument("--smoke-only",action="store_true")
    a=ap.parse_args()
    try:
        if not a.smoke_only and any(getattr(a,n) is None for n in
                                   ("w14","w13","w9","w0","support","output")):
            ap.error("all paths required")
        main(a)
    except BaseException as exc:
        if a.output and a.output.exists():
            write_json(a.output/"failure.json",{
                "failed":True,"type":type(exc).__name__,"error":str(exc)})
        raise
