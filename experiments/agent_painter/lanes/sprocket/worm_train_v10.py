#!/usr/bin/env python3
"""W10: genuine W9-checkpoint continuation with closed-loop motif/recovery replay.

Synthetic pinned Python W0 game-like situations, never native CyberSand.
W9 root Instruct stays frozen; only the 142->64->64->6 auxiliary steering
head receives optimizer updates. Compare W9, W10 and untuned root fairly.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, sys, time
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parents[1]/"training"),str(HERE)]
import worm_train_v9 as w9
import worm_v4 as objective
from common import profile, sha256_file, write_json
from lane_adapters import adapter_for
from numerics import configure_cpu

VERSION="sprocket-w10-motif-replay-v1"
PARENT="9a6c7956377c22fda3c0599da7645fd796d03569"
PARENT_SOURCE_SHA256="215dedd4b3ab8d2ca6885e08f9ca51a59e683d97b0adafce6f054a7c10025552"
PARENT_CHECKPOINT_SHA256="5c09762f77580064b23a49b1daefef1e181a69d1318b1cfbaa8ca071ee2e75d4"
TRAIN_SEEDS=(181,191,193,197)
HELDOUT=(1709,1721,1733)
ROOT_SEED=1709
UPDATES_PER_WAVE=160
ACTIONS=w9.ACTIONS
FAMILIES=w9.FAMILIES

def motif_teacher(obs,support):
    """Legal, locally responsive, explicitly constructed teacher; NOT learned."""
    support.validate_observation(obs)
    legal=obs["legal"]
    if obs["tick"]>=124: return "STOP"
    h=obs["heading"]
    dirs=((1,0),(0,1),(-1,0),(0,-1))
    nearby={}
    for name,rotation in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1)):
        dx,dy=dirs[(h+rotation)%4]
        nearby[name]=obs["local_5x5"][2+dy][2+dx]
    fresh=[a for a in ("STRAIGHT","LEFT","RIGHT")
           if a in legal and nearby[a]=="."]
    if not fresh:
        return "RETRACE" if "RETRACE" in legal else "STOP"
    family,tick=obs["family"],obs["tick"]
    if family=="greebles":
        # Edge hatching and deliberately repeated local ornament.
        if "BRANCH" in legal and tick%7 in (0,1) and ("#" in "".join(obs["local_5x5"]) or tick%14==0):
            return "BRANCH"
        pref=("LEFT","RIGHT","STRAIGHT") if tick%8 in (1,2,5) else ("RIGHT","LEFT","STRAIGHT")
    elif family=="pipework":
        # Mostly long runs; occasional branch and elbow at junctions.
        if "BRANCH" in legal and tick%19 in (0,1) and tick>0: return "BRANCH"
        pref=("RIGHT","LEFT","STRAIGHT") if tick%17==16 else ("STRAIGHT","LEFT","RIGHT")
    else:
        # Repeated corridor corners, with front-block recovery.
        if "BRANCH" in legal and tick%29==0 and tick>0: return "BRANCH"
        pref=("LEFT","STRAIGHT","RIGHT") if tick%16==15 else ("STRAIGHT","RIGHT","LEFT")
    return next((a for a in pref if a in fresh),fresh[0])

def gather(w0,support,seeds,steps=128,modes=("random","fidget","wall")):
    rows=[]; manifest=[]
    for seed in seeds:
        for family in FAMILIES:
            for mode in modes:
                trace=w0.run(seed,mode,family,steps)
                manifest.append([seed,family,mode,len(trace["events"])])
                for event in trace["events"]:
                    obs=event["observation"]
                    if obs["tick"]%2: continue
                    target=motif_teacher(obs,support)
                    if target not in obs["legal"]: raise AssertionError("illegal motif target")
                    rows.append({"observation":obs,"action":target,"family":family,
                                 "legal":obs["legal"],"seed":seed,"source":mode})
    return rows,manifest

def full_eval(torch,head,rows):
    return w9.accuracy(torch,head,rows)

def train(torch,w0,support,base,output):
    torch.manual_seed(101016)
    rng=random.Random(101016)
    frozen=w9.model_class(torch)()
    payload=torch.load(base,map_location="cpu",weights_only=True)
    if payload.get("version")!=w9.FEATURE_VERSION or payload.get("stage")!="onpolicy-replay":
        raise ValueError("W9 checkpoint identity mismatch")
    frozen.load_state_dict(payload["head"],strict=True)
    frozen.eval()
    head=w9.model_class(torch)()
    head.load_state_dict(payload["head"],strict=True)
    old,dev_manifest=gather(w0,support,HELDOUT)
    rows,train_manifest=gather(w0,support,TRAIN_SEEDS)
    assert not set(TRAIN_SEEDS)&set(HELDOUT)
    manifest={"version":VERSION,"trainer":"counterfactual motif-distillation synthetic Python W0",
              "train_worlds":train_manifest,"development_worlds":dev_manifest,
              "world_seed_disjoint":True,"training_rows":len(rows),"heldout_rows":len(old),
              "train_actions":dict(Counter(r["action"] for r in rows)),
              "train_observation_sha256":hashlib.sha256(json.dumps(
                [r["observation"] for r in rows],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
              "NOT_native":True}
    write_json(output/"dataset-manifest.json",manifest)
    opt=torch.optim.AdamW(head.parameters(),lr=0.0008,weight_decay=0.002)
    src_x,src_y=w9.as_tensors(torch,rows)
    stages=[{"stage":"W9-frozen","heldout_motif_teacher":full_eval(torch,frozen,old)}]
    histories=[]
    for wave in ("motif-balanced","closed-loop-recovery-replay"):
        if wave=="closed-loop-recovery-replay":
            replay=[]
            for seed in TRAIN_SEEDS:
                for family in FAMILIES:
                    def selector(obs):
                        with torch.no_grad():
                            logits=head(torch.tensor([w9.features(obs)],dtype=torch.float32))[0]
                        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=124]
                        return max(legal,key=lambda a:float(logits[ACTIONS.index(a)]))
                    trace=w0.run(seed,"callback",family,128,selector=selector)
                    for ev in trace["events"]:
                        ob=ev["observation"]
                        replay.append({"observation":ob,"action":motif_teacher(ob,support),
                                       "family":family,"legal":ob["legal"]})
            rx,ry=w9.as_tensors(torch,replay)
            write_json(output/"replay-manifest.json",{
                "worlds":list(TRAIN_SEEDS),"rows":len(replay),
                "action_counts":dict(Counter(r["action"] for r in replay)),
                "synthetic_host":True})
        for step in range(UPDATES_PER_WAVE):
            opt.zero_grad(set_to_none=True)
            ids=w9.balanced_indices(rows,64,rng)
            logits=head(src_x[ids])
            loss=torch.nn.functional.cross_entropy(logits,src_y[ids])
            if wave=="closed-loop-recovery-replay":
                rid=[rng.randrange(len(replay)) for _ in range(64)]
                replay_loss=torch.nn.functional.cross_entropy(head(rx[rid]),ry[rid])
                loss=0.50*loss+0.50*replay_loss
            if not torch.isfinite(loss).item(): raise ValueError("nonfinite real loss")
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(head.parameters(),3.)
            if not torch.isfinite(grad).item() or float(grad)==0: raise ValueError("zero/nonfinite gradient")
            opt.step()
            if (step+1)%40==0:
                row={"stage":wave,"global_update":len(histories)*40+40,
                     "loss":round(float(loss.detach()),6),"grad_norm":round(float(grad),6)}
                histories.append(row)
                write_json(output/"partial-optimizer-history.json",histories)
        stage={"stage":wave,"updates":UPDATES_PER_WAVE,
               "heldout_motif_teacher":full_eval(torch,head,old)}
        check=output/(wave+".pt")
        torch.save({"version":VERSION,"stage":wave,"root_frozen":True,
                    "head":head.state_dict(),"optimizer":opt.state_dict(),
                    "metadata":stage,"torch_rng_state":torch.random.get_rng_state()},check)
        stage["checkpoint"]={"name":check.name,"sha256":sha256_file(check),"bytes":check.stat().st_size}
        stages.append(stage)
        write_json(output/"stage-progress.json",stages)
    return frozen,head,stages,histories,old

def head_rollout(w0,torch,head,seed,family,name):
    decisions=[]
    def selector(obs):
        with torch.no_grad():
            raw=head(torch.tensor([w9.features(obs)],dtype=torch.float32))[0].tolist()
        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=124]
        selected=max(legal,key=lambda a:raw[ACTIONS.index(a)]) if legal else "STOP"
        decisions.append({"tick":obs["tick"],"raw_six_scores":raw,
                          "legal":list(obs["legal"]),"selected":selected})
        return selected
    r=w0.run(seed,"callback",family,128,selector=selector)
    assert all(e["raw_legal"] for e in r["events"])
    r["mode"]=name
    r["head_decisions"]=decisions
    r["engine"]="pinned-synthetic-Python-W0-trained-head-not-native"
    return r

def root_rollout(w0,torch,head,root,tok,adapter,token_ids,seed,family,name,weight):
    from worm_score_v5_eval import host_draw,sample_scores
    decisions=[]
    def selector(obs):
        prefix=adapter.encode_prompt(tok,objective.prompt(obs),objective.SYSTEM)
        x=torch.tensor([prefix],dtype=torch.long)
        with torch.no_grad():
            raw=root(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0,-1,token_ids].float().tolist()
            aux=head(torch.tensor([w9.features(obs)],dtype=torch.float32))[0].tolist()
        combined=[raw[i]+weight*(aux[i]-sum(aux)/6) for i in range(6)]
        legal=[a for a in obs["legal"] if a!="STOP" or obs["tick"]>=44]
        draw=host_draw(family,seed,obs["tick"],2.25)
        if not legal: legal=["STOP"]
        selected,_=sample_scores(combined,legal,2.25,draw)
        decisions.append({"tick":obs["tick"],"root_logits":raw,"head_logits":aux,
                          "selection_logits":combined,"host_legal":legal,
                          "host_draw":draw,"selected":selected,"root_real_forward":True})
        return selected
    r=w0.run(seed,"callback",family,48,selector=selector)
    assert all(e["raw_legal"] and e["applied"]==d["selected"]
               for e,d in zip(r["events"],decisions))
    r["mode"]=name
    r["model_score_decisions"]=decisions
    r["engine"]="pinned-root-Instruct-real-forward-Python-W0-not-native"
    return r

def main(args):
    started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    configure_cpu(1)
    import torch
    torch.set_num_threads(1)
    p=profile("sprocket",require_supported=True)
    assert p["model_id"]=="kandivault/sprocket-500m"
    assert p["revision"]=="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
    assert p["subfolder"]=="" and p["framing"]["eos_token_id"]==5
    assert sha256_file(HERE/"worm_train_v9.py")==PARENT_SOURCE_SHA256
    assert sha256_file(args.parent)==PARENT_CHECKPOINT_SHA256
    assert hashlib.sha256((args.w0).read_bytes()).hexdigest()=="81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
    assert hashlib.sha256((args.support).read_bytes()).hexdigest()=="c555c55ea4fc4bc303159fd3268cef4a0b7d9e5c006e19918ea3433d1f7c95d6"
    support=objective.load_support(args.support,args.w0)
    w0=support.worm
    write_json(args.output/"preflight.json",{
        "schema":VERSION,"source_sha":os.getenv("GITHUB_SHA","local"),
        "parent_producer":PARENT,"parent_checkpoint_sha256":PARENT_CHECKPOINT_SHA256,
        "root":p,"W0_sha256":sha256_file(args.w0),"support_sha256":sha256_file(args.support),
        "source_sha256":sha256_file(__file__),"root_optimizer_updates":0,
        "head_updates_planned":UPDATES_PER_WAVE*2,"NOT_native":True,"sealed_final_opened":False})
    frozen,trained,stages,histories,heldout=train(torch,w0,support,args.parent,args.output)
    entries=[]
    out=args.output/"png-and-json"
    for seed in HELDOUT:
        for family in FAMILIES:
            for mode in ("random","wall","fidget"):
                rec=w0.run(seed,mode,family,128)
                w0.write_result(rec,out)
                entries.append({"seed":seed,"family":family,"mode":mode,
                                "metrics":rec["metrics"],"steps":rec["steps_executed"]})
            for name,head in (("w9-head-frozen",frozen),("w10-head-trained",trained)):
                rec=head_rollout(w0,torch,head,seed,family,name)
                w0.write_result(rec,out)
                entries.append({"seed":seed,"family":family,"mode":name,
                                "metrics":rec["metrics"],"steps":rec["steps_executed"]})
    # Pinned root forward is mandatory, not a synthetic projection replacement.
    if not args.model: raise ValueError("pinned root required")
    import train as shared
    from worm_score_v5_eval import action_tokens
    root,tok=shared.load_base(args.model,p);root.eval()
    adapter=adapter_for(p);token_ids,token_map=action_tokens(tok)
    for family in FAMILIES:
        for name,head,weight in (("root-untuned",frozen,0.0),
                                  ("root-w9-fused",frozen,2.0),
                                  ("root-w10-fused",trained,2.0)):
            rec=root_rollout(w0,torch,head,root,tok,adapter,token_ids,
                             ROOT_SEED,family,name,weight)
            w0.write_result(rec,out)
            entries.append({"seed":ROOT_SEED,"family":family,"mode":name,
                            "metrics":rec["metrics"],"steps":rec["steps_executed"],
                            "real_root_forwards":len(rec["model_score_decisions"])})
        write_json(args.output/"partial-root-results.json",entries)
    final={"status":"complete","version":VERSION,"producer_sha":os.getenv("GITHUB_SHA"),
           "root_model":p,"real_head_optimizer_updates":UPDATES_PER_WAVE*2,
           "root_optimizer_updates":0,"stages":stages,"original_png_count":len(entries),
           "root_forward_count":sum(e.get("real_root_forwards",0) for e in entries),
           "root_action_token_ids":token_map,"checkpoint_integrity":
           all(sha256_file(args.output/x["checkpoint"]["name"])==x["checkpoint"]["sha256"]
               for x in stages if x.get("checkpoint")),
           "results":entries,"wall_seconds":time.monotonic()-started,
           "cpu_seconds":time.process_time(),
           "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "native":False,"sealed_final_opened":False}
    write_json(args.output/"summary.json",final)
    print(json.dumps({"status":"complete","optimizer_steps":320,
                      "original_pngs":len(entries),"root_forwards":final["root_forward_count"]}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("parent","model","w0","support","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    try:main(a)
    except BaseException as e:
        if a.output.exists():write_json(a.output/"failure.json",{
            "type":type(e).__name__,"error":str(e),"partial_or_failed":True})
        raise
