#!/usr/bin/env python3
"""W14: correct W13's BRANCH false-positive excess without killing motif branching.

Two genuinely optimized PyTorch auxiliary-head stages on pinned synthetic W0.
W13 source and checkpoints preserved. Root-Instruct frozen; no game/native edits.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, sys, time
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/"training"),str(HERE)]
import worm_train_v9 as w9
import worm_train_v10 as w10
import worm_train_v13 as w13
import worm_head_sample_v11 as w11
import worm_v4 as objective
from common import profile,sha256_file,write_json
from numerics import configure_cpu

VERSION="sprocket-w14-branch-precision-motif-v1"
W13_SHA="7852d845d6c388502ad72b2d699bac3cd69f156d1385ae5f5bc1295551fbc49e"
W9_SHA=w11.W9_SHA
W0_SHA="81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
SUPPORT_SHA="c555c55ea4fc4bc303159fd3268cef4a0b7d9e5c006e19918ea3433d1f7c95d6"
TRAIN_SEEDS=(281,283,293,307)
HELDOUT=(2401,2411,2423)
UPDATES=120
# Deliberately skew legal BRANCH training to hard negatives: W13 recall .54,
# precision .24. Evaluation guards against trading this for W12's zero recall.
BRANCH_POS=20
BRANCH_NEG=44

def calibrated_branch_loss(torch,head,x,mask,pos,neg,rng):
    ids=rng.choices(pos,k=BRANCH_POS)+rng.choices(neg,k=BRANCH_NEG)
    scores=head(x[ids]);legal=mask[ids];j=w9.ACTIONS.index("BRANCH")
    if not bool(legal[:,j].all()):raise AssertionError("hard negative must be BRANCH-legal")
    rivals=scores.masked_fill(~legal,-1e4).clone()
    rivals[:,j]=-1e4
    gap=scores[:,j]-rivals.logsumexp(1)
    target=torch.tensor([1.]*BRANCH_POS+[0.]*BRANCH_NEG,dtype=torch.float32)
    return torch.nn.functional.binary_cross_entropy_with_logits(gap,target)

def legal_ce(torch,head,x,y,mask,ids):
    return torch.nn.functional.cross_entropy(
        head(x[ids]).masked_fill(~mask[ids],-1e4),y[ids])

def head_snapshot(torch,source):
    out=w9.model_class(torch)()
    out.load_state_dict(source.state_dict(),strict=True)
    out.eval()
    return out

def data_receipt(rows):
    return {"rows":len(rows),"labels":dict(Counter(r["action"] for r in rows)),
        "observations_sha256":hashlib.sha256(json.dumps(
            [r["observation"] for r in rows],sort_keys=True,
            separators=(",",":")).encode()).hexdigest()}

def save_checkpoint(torch,head,opt,out,stage,metrics):
    path=out/(stage+".pt")
    torch.save({"version":VERSION,"stage":stage,"root_frozen":True,
                "head":head.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng_state":torch.random.get_rng_state(),
                "heldout_metrics":metrics},path)
    return {"file":path.name,"sha256":sha256_file(path),"bytes":path.stat().st_size}

def train(torch,w0,support,parent,w9head,out):
    rng=random.Random(140014)
    torch.manual_seed(140014)
    frozen=head_snapshot(torch,parent)
    model=head_snapshot(torch,parent);model.train()
    train_rows,train_worlds=w10.gather(w0,support,TRAIN_SEEDS,steps=128)
    heldout,heldout_worlds=w10.gather(w0,support,HELDOUT,steps=128)
    assert not set(TRAIN_SEEDS)&set(HELDOUT)
    x,y=w9.as_tensors(torch,train_rows)
    mask=w13.legal_mask(torch,train_rows)
    groups=w13.class_rows(train_rows)
    pos,neg=w13.eligible_indices(train_rows)
    write_json(out/"dataset-manifest.json",{
        "schema":VERSION,"train_worlds":train_worlds,"heldout_worlds":heldout_worlds,
        "train":data_receipt(train_rows),"heldout":data_receipt(heldout),
        "branch_legal_train_positives":len(pos),"branch_legal_train_negatives":len(neg),
        "branch_positive_per_64":BRANCH_POS,"branch_hard_negative_per_64":BRANCH_NEG,
        "synthetic":True,"native":False})
    opt=torch.optim.AdamW(model.parameters(),lr=0.00010,weight_decay=0.003)
    stages=[{"name":"frozen-W13","teacher":w9.accuracy(torch,frozen,heldout),
         "branch":w13.branch_diagnostic(torch,frozen,heldout),
         "classes":w13.class_recall(torch,frozen,heldout)}]
    hist=[];stage1=None
    for stage_idx,stage in enumerate(("legal-hard-negative-calibration",
                                      "diverse-motif-replay")):
        if stage_idx:
            stage1=head_snapshot(torch,model)
            replay,manifest=w13.replay_data(torch,w0,support,
                     {"w9-greedy":w9head,"w13-stage1-t07":model})
            rx,ry=w9.as_tensors(torch,replay)
            rmask=w13.legal_mask(torch,replay)
            rgroups=w13.class_rows(replay)
            rp,rn=w13.eligible_indices(replay)
            write_json(out/"replay-manifest.json",{
                "source_controllers":["w9-greedy","w13-stage1-t07"],
                "worlds":manifest,"actual":data_receipt(replay),
                "teacher":"W10 motif teacher, not native game feedback",
                "synthetic":True})
        for step in range(UPDATES):
            opt.zero_grad(set_to_none=True)
            ids=w13.balanced(groups,rng)
            ce=legal_ce(torch,model,x,y,mask,ids)
            bce=calibrated_branch_loss(torch,model,x,mask,pos,neg,rng)
            if stage_idx:
                rids=w13.balanced(rgroups,rng)
                rce=legal_ce(torch,model,rx,ry,rmask,rids)
                rbce=calibrated_branch_loss(torch,model,rx,rmask,rp,rn,rng)
                # Rehearse W13's non-BRANCH distribution to avoid erasing
                # LEFT/RIGHT/RETRACE while suppressing false-positive BRANCH.
                with torch.no_grad():old=frozen(x[ids])[:,[0,1,2,4,5]]
                new=model(x[ids])[:,[0,1,2,4,5]]
                reg=torch.nn.functional.mse_loss(new-new.mean(1,keepdim=True),
                                                   old-old.mean(1,keepdim=True))
                loss=.30*ce+.25*bce+.20*rce+.20*rbce+.05*reg
            else:
                loss=.45*ce+.55*bce
            if not bool(torch.isfinite(loss)):raise ValueError("nonfinite W14 loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            if not bool(torch.isfinite(norm)) or float(norm)<=0:
                raise ValueError("nonfinite or zero W14 gradient")
            opt.step()
            if (step+1)%30==0:
                hist.append({"stage":stage,"stage_step":step+1,
                     "global_step":stage_idx*UPDATES+step+1,
                     "loss":round(float(loss.detach()),6),
                     "grad_norm":round(float(norm),6)})
                write_json(out/"partial-optimizer-history.json",hist)
        measure={"name":stage,"genuine_adamw_steps":UPDATES,
            "teacher":w9.accuracy(torch,model,heldout),
            "branch":w13.branch_diagnostic(torch,model,heldout),
            "classes":w13.class_recall(torch,model,heldout)}
        measure["checkpoint"]=save_checkpoint(torch,model,opt,out,stage,measure)
        stages.append(measure)
        write_json(out/"stage-progress.json",stages)
    return frozen,stage1,model,stages,hist

def rollout(torch,w0,head,seed,family,name,temp):
    decisions=[]
    def selector(obs):
        with torch.no_grad():
            scores=head(torch.tensor([w9.features(obs)],dtype=torch.float32))[0].tolist()
        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=124]
        unit=w11.draw(seed,family,obs["tick"])
        action=w11.choose(scores,legal,temp,unit)
        decisions.append({"tick":obs["tick"],"raw_six_scores":scores,
            "host_legal":legal,"host_draw":unit,"temperature":temp,
            "selected":action,"actual_head_forward":True,"root_forward":False})
        return action
    rec=w0.run(seed,"callback",family,128,selector=selector)
    assert len(rec["events"])==len(decisions)
    assert all(e["raw_legal"] and e["applied"]==d["selected"]
               for e,d in zip(rec["events"],decisions))
    rec["mode"]=name;rec["head_decisions"]=decisions
    rec["engine"]="SHA-pinned-synthetic-W0-PyTorch-head-not-native"
    return rec

def main(args):
    if args.smoke_only:
        assert len(w9.ACTIONS)==6 and w9.ACTIONS[3]=="BRANCH"
        assert set(TRAIN_SEEDS).isdisjoint(HELDOUT)
        assert set(TRAIN_SEEDS+HELDOUT).isdisjoint(
            set(w9.TRAIN_SEEDS+w9.EVAL_SEEDS+w10.TRAIN_SEEDS+w10.HELDOUT+
                w11.SEEDS+w13.TRAIN_SEEDS+w13.HELDOUT))
        assert len(W13_SHA)==64 and len(W0_SHA)==64
        print("W14 source/seed/preflight checks PASS; no optimization",flush=True)
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
    assert sha256_file(args.parent)==W13_SHA
    assert sha256_file(args.w9)==W9_SHA
    assert sha256_file(args.w0)==W0_SHA
    assert sha256_file(args.support)==SUPPORT_SHA
    support=objective.load_support(args.support,args.w0)
    write_json(args.output/"provenance.json",{
        "schema":VERSION,"source_sha":os.getenv("GITHUB_SHA","local"),
        "run_id":os.getenv("GITHUB_RUN_ID","local"),"run_attempt":os.getenv("GITHUB_RUN_ATTEMPT","local"),
        "script_sha256":sha256_file(__file__),"pinned_root":p,
        "W13_parent_sha256":W13_SHA,"W9_parent_sha256":W9_SHA,
        "W0_sha256":W0_SHA,"support_sha256":SUPPORT_SHA,
        "new_aux_head_adamw_steps_planned":UPDATES*2,
        "root_forwards":0,"root_optimizer_steps":0,"lora_updates":0,
        "native":False,"sealed_final_opened":False})
    frozen=w11.load(torch,args.parent,W13_SHA,w13.VERSION)
    old=w11.load(torch,args.w9,W9_SHA,w9.FEATURE_VERSION)
    frozen,stage1,trained,stages,hist=train(torch,support.worm,support,frozen,old,args.output)
    entries=[];pngdir=args.output/"png-and-json"
    for seed in HELDOUT:
        for family in w9.FAMILIES:
            for name,head,temp in (
                ("w9-greedy",old,0.),("w13-greedy",frozen,0.),
                ("w13-t07",frozen,.7),
                ("w14-stage1-greedy",stage1,0.),
                ("w14-final-greedy",trained,0.),
                ("w14-final-t07",trained,.7)):
                rec=rollout(torch,support.worm,head,seed,family,name,temp)
                support.worm.write_result(rec,pngdir)
                entries.append({"seed":seed,"family":family,"mode":name,
                    "metrics":rec["metrics"],"actual_head_forwards":len(rec["head_decisions"]),
                    "trajectory_sha256":hashlib.sha256(json.dumps(
                        rec["trajectory"],separators=(",",":")).encode()).hexdigest()})
            for name in ("random","wall","fidget"):
                rec=support.worm.run(seed,name,family,128)
                support.worm.write_result(rec,pngdir)
                entries.append({"seed":seed,"family":family,"mode":name,
                    "metrics":rec["metrics"],"actual_head_forwards":0,
                    "trajectory_sha256":hashlib.sha256(json.dumps(
                        rec["trajectory"],separators=(",",":")).encode()).hexdigest()})
        write_json(args.output/"partial-original-rollouts.json",entries)
    assert len(entries)==81
    assert all(x["metrics"]["invalid_raw_choices"]==0 for x in entries)
    cps=[x["checkpoint"] for x in stages if "checkpoint" in x]
    assert len(cps)==2 and all(sha256_file(args.output/c["file"])==c["sha256"] for c in cps)
    final={"status":"complete","version":VERSION,
        "real_aux_adamw_updates":UPDATES*2,"root_forwards":0,"root_updates":0,
        "head_forward_count":sum(x["actual_head_forwards"] for x in entries),
        "original_png_json_pairs":len(entries),"stages":stages,
        "history":hist,"results":entries,
        "wall_seconds":time.monotonic()-start,
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_integrity":True,"synthetic":True,"native":False}
    write_json(args.output/"summary.json",final)
    print(json.dumps({"status":"complete","steps":UPDATES*2,
        "png_pairs":len(entries),"head_forwards":final["head_forward_count"]}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("parent","w9","w0","support","output"):
        ap.add_argument("--"+n,type=Path,required=False)
    ap.add_argument("--smoke-only",action="store_true")
    a=ap.parse_args()
    try:
        if not a.smoke_only and any(getattr(a,n) is None for n in ("parent","w9","w0","support","output")):
            ap.error("all source paths and output required for actual optimization")
        main(a)
    except BaseException as exc:
        if a.output and a.output.exists():
            write_json(a.output/"failure.json",{"failed":True,
                "type":type(exc).__name__,"error":str(exc)})
        raise
