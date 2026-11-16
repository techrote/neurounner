#!/usr/bin/env python3
"""GP350-010: selective branching with held-out legal counterexamples.
All states are genuine source-pinned W0 synthetic-host observations. Labels
are synthetic teacher preferences, not native/human supervision. Exact full
LFM350 + GP006 rank8 adapter weights; fresh AdamW, NOT exact resume.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import time

import lfm350 as c
import gp350_paired_margin_train as prev

SCHEMA = "driv3r.lfm350.gp350010-selective-branch/1"
PARENT_RUN = 43806503356
PARENT_ARTIFACT = 14199280586
PARENT_ADAPTER = "436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c"
TRAIN_SEEDS = tuple(range(3101, 3113))
DEV_SEEDS = (9103, 9119, 9137)
EVAL_SEEDS = (9503, 9521)
SIZES = (32, 48)
STEPS = 72
CHECKPOINTS = (24, 48, 72)
DEADLINE = 1950
CANCELLED = False

def stop(_sig, _frame):
    global CANCELLED
    CANCELLED = True

def at2(obs, turn):
    dx, dy = ((1,0),(0,1),(-1,0),(0,-1))[(obs["heading"]+turn)%4]
    return obs["local_5x5"][2+2*dy][2+2*dx]

def teacher(obs):
    """Transparent synthetic W0 geometry/cadence teacher, host-legal."""
    legal=set(obs["legal"])
    open_arms=[a for a,t in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1))
               if a in legal and c.nearby(obs,t)=="."]
    safe_arms=[a for a,t in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1))
               if a in open_arms and at2(obs,t)=="."]
    tick,family=obs["tick"],obs["family"]
    ray=prev.ray2(obs)
    # Avoid actual W0-observed two-cell obstacles and visited trails first.
    if ray in ("#","+") and "STRAIGHT" in open_arms:
        order=("RIGHT","LEFT") if family=="pipework" else ("LEFT","RIGHT")
        turn=next((a for a in order if a in safe_arms),
                  next((a for a in order if a in open_arms),None))
        if turn is not None:
            return turn
    # A legal BRANCH is insufficient: require genuinely open spatial context.
    junction=("BRANCH" in legal and ray=="." and
              len(open_arms)>=2 and len(safe_arms)>=2)
    if junction and family=="greebles" and tick%7 in (0,1):
        return "BRANCH"
    if junction and family=="pipework" and tick%17==16:
        return "BRANCH"
    if family=="greebles":
        order=("LEFT","RIGHT","STRAIGHT") if tick%9 in (4,5) else ("STRAIGHT","RIGHT","LEFT")
    elif family=="pipework":
        order=("STRAIGHT","RIGHT","LEFT")
    else:
        order=("RIGHT","STRAIGHT","LEFT") if tick%13==12 else ("STRAIGHT","LEFT","RIGHT")
    result=next((a for a in order if a in safe_arms),
                next((a for a in order if a in open_arms),None))
    return result if result is not None else ("RETRACE" if "RETRACE" in legal else "STOP")

def observed(w0, seeds):
    """Real W0 event observations from model-free walks, never invented maps."""
    rows = []
    seen = set()
    for seed in seeds:
        for size in SIZES:
            for family in c.FAMILIES:
                for mode in ("fidget", "random"):
                    record = w0.run(seed,mode=mode,family=family,size=size,steps=80)
                    c.check(record["metrics"]["invalid_raw_choices"]==0,"source host illegal")
                    for event in record["events"]:
                        obs=event["observation"]
                        mark=prev.ray2(obs)
                        target=teacher(obs)
                        c.check(mark in (".","#","+"),"invalid ray2 state")
                        c.check(target in obs["legal"],"synthetic teacher was illegal")
                        key=(seed,size,family,mark,c.canonical(obs))
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append({"seed":seed,"size":size,"mode":mode,
                                     "observation":obs,"ray2":mark,"target":target,
                                     "W0_observed_not_counterfactual":True,
                                     "teacher_synthetic_not_native":True})
    return rows

def stratified(rows, per=8):
    buckets=collections.defaultdict(list)
    for row in rows:
        buckets[(row["observation"]["family"],row["ray2"])].append(row)
    selected=[]
    for family in c.FAMILIES:
        for mark in (".","#","+"):
            pool=buckets[(family,mark)]
            c.check(len(pool)>=per,
                    "insufficient observed DEV exposure "+family+"/"+mark)
            # Sample many different seeds and local grids, not one repeated prompt.
            uniq={}
            for row in pool:
                o=row["observation"]
                key=(tuple(o["local_5x5"]),o["heading"],tuple(o["legal"]),
                     o["tick"]%13,tuple(o["last_actions"][-2:]))
                uniq.setdefault(key,row)
            pool=list(uniq.values())
            c.check(len(pool)>=per,"insufficient distinct DEV state structures")
            selected.extend(pool[(i*len(pool))//per] for i in range(per))
    c.check(len(selected)==72,"stratified probe budget drift")
    return selected

def make_pairs(rows):
    grouped=collections.defaultdict(lambda:collections.defaultdict(list))
    branches=collections.defaultdict(list)
    for r in rows:
        fam=r["observation"]["family"]
        grouped[fam][r["ray2"]].append(r)
        if r["target"]=="BRANCH":
            branches[fam].append(r)
    pairs=collections.defaultdict(list)
    for family in c.FAMILIES:
        frees=grouped[family]["."]
        c.check(bool(frees),"no free training examples")
        if family in ("greebles","pipework"):
            c.check(bool(branches[family]),"no family-specific branch training examples")
        for marker in ("#","+"):
            pressures=grouped[family][marker]
            c.check(len(pressures)>=8,"insufficient real pressure states "+family+marker)
            for i,press in enumerate(pressures):
                # Preference changes between DIFFERENT actually-observed states,
                # not identical-geometry marker substitution. Match legal menus
                # where feasible, but never mislabel as causal counterfactuals.
                normals=[f for f in frees
                         if f["target"] != press["target"]
                         and f["observation"]["legal"]==press["observation"]["legal"]]
                if not normals:
                    normals=[f for f in frees if f["target"] != press["target"]]
                if normals:
                    pairs[(family,marker)].append((normals[i%len(normals)],press))
            c.check(len(pairs[(family,marker)])>=6,"too few labelled real-state pairs")
    return pairs,branches

def probe(model,tok,ids,examples):
    import torch
    model.eval()
    out=[]
    with torch.inference_mode():
        for row in examples:
            obs=row["observation"]
            z=prev.rich_score(model,tok,ids,obs)
            raw=c.ACTIONS[int(z.argmax().item())]
            legal=max(obs["legal"],key=lambda a:float(z[c.ACTIONS.index(a)]))
            out.append({"seed":row["seed"],"size":row["size"],
                        "family":obs["family"],"marker":row["ray2"],
                        "target":row["target"],"legal":legal,"raw":raw,
                        "match":legal==row["target"],"raw_legal":raw in obs["legal"],
                        "six_logits":[float(v) for v in z]})
    return {"n":len(out),"teacher_matches":sum(r["match"] for r in out),
            "raw_illegal":sum(not r["raw_legal"] for r in out),
            "marker_target_matches":{m:sum(r["match"] for r in out if r["marker"]==m)
                                     for m in (".","#","+")},
            "actual_W0_observations":True,"rows":out}

def episode(w0,model,tok,ids,out,stage,size,seed,family,started):
    import torch
    scored=[]
    def choose(obs):
        c.check(not CANCELLED and time.monotonic()-started<DEADLINE-200,
                "bounded inference or STOP")
        with torch.inference_mode():
            z=prev.rich_score(model,tok,ids,obs)
        raw=c.ACTIONS[int(z.argmax().item())]
        legal=max(obs["legal"],key=lambda a:float(z[c.ACTIONS.index(a)]))
        scored.append({"obs":obs,"ray2":prev.ray2(obs),"raw":raw,"applied":legal,
                       "raw_legal":raw in obs["legal"],"six_logits":[float(v) for v in z]})
        return legal
    record=w0.run(seed,mode="callback",family=family,size=size,steps=64,selector=choose)
    c.check(len(record["events"])==len(scored),"scored/event mismatch")
    for d,e in zip(scored,record["events"]):
        c.check(d["applied"]==e["applied"]==e["raw_choice"] and e["raw_legal"],
                "illegal host application")
    replay=w0.run(seed,mode="replay",family=family,size=size,steps=64,
                  raw_actions=[d["applied"] for d in scored])
    png=w0.png_bytes(record)
    c.check(png==w0.png_bytes(replay),"original W0 PNG replay mismatch")
    stem=f"{stage}-s{size}-{seed}-{family}"
    (out/(stem+".png")).write_bytes(png)
    c.save_json(out/(stem+".json"),{"record":record,"scores":scored})
    m=record["metrics"]
    return {"stage":stage,"size":size,"seed":seed,"family":family,
            "decisions":len(scored),"unique_cells":m["unique_trail_cells"],
            "branches":m["branches"],"retraces":m["retraces"],
            "stopped":m["terminated_by_stop"],
            "raw_illegal":sum(not d["raw_legal"] for d in scored),
            "host_illegal":m["invalid_raw_choices"],
            "ray2_counts":dict(collections.Counter(d["ray2"] for d in scored)),
            "choices":dict(collections.Counter(d["applied"] for d in scored)),
            "png":stem+".png","png_sha256":c.sha_bytes(png)}

def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi,snapshot_download
    torch.set_num_threads(2)
    torch.manual_seed(3510010)
    started=time.monotonic()
    out=args.out
    c.check(not out.exists(),"immutable experiment lease already exists")
    out.mkdir(parents=True)
    c.save_json(out/"lease.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
       "run_id":os.getenv("GITHUB_RUN_ID"),"attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
       "parent_run":PARENT_RUN,"parent_artifact":PARENT_ARTIFACT,
       "parent_adapter_sha256":PARENT_ADAPTER,"new_optimizer_state":True,
       "initialization":"GP350006 rank8 LoRA weights, NOT exact optimizer resume",
       "model":c.MODEL,"model_revision":c.REVISION,
       "steps":STEPS,"native":False,"draft_hold":True})
    c.check(c.sha(args.adapter/"adapter_model.safetensors")==PARENT_ADAPTER,
            "original parent LoRA artifact custody mismatch")
    w0=c.load_w0(args.w0)
    training=observed(w0,TRAIN_SEEDS)
    dev=observed(w0,DEV_SEEDS)
    c.check({r["seed"] for r in training}.isdisjoint({r["seed"] for r in dev}),
            "training and DEV worlds overlap")
    sample=stratified(dev)
    pairs,branches=make_pairs(training)
    raw=("\n".join(c.canonical(r) for r in training)+"\n").encode()
    (out/"real-W0-observation-train.jsonl").write_bytes(raw)
    c.save_json(out/"curriculum.json",{"train_worlds":list(TRAIN_SEEDS),
        "dev_worlds":list(DEV_SEEDS),"real_observed_training_rows":len(training),
        "actual_heldout_probe_rows":len(sample),"train_sha256":c.sha_bytes(raw),
        "teacher_labels_explicitly_synthetic":True,
        "model_free_source_policies":["fidget","random"],
        "training_ray2":dict(collections.Counter(r["ray2"] for r in training)),
        "heldout_ray2":dict(collections.Counter(r["ray2"] for r in sample)),
        "branch_training_rows":sum(map(len,branches.values())),
        "selective_branching_and_GP006_retention":True,
        "matched_except_marker":False,
        "pairs_are_distinct_actual_W0_observations":True,
        "pair_buckets":{f"{family}/{mark}":len(v)
                        for (family,mark),v in pairs.items()}})
    info=HfApi().model_info(c.MODEL,revision=c.REVISION)
    c.check(info.sha==c.REVISION,"full upstream model drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=args.cache,max_workers=2,
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
            "generation_config.json","tokenizer*","special_tokens_map.json",
            "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    inventory={str(p.relative_to(args.cache)):c.sha(p)
               for p in sorted(args.cache.rglob("*")) if p.is_file() and ".cache" not in p.parts}
    c.check(any(k.endswith(".safetensors") for k in inventory),"full model weights missing")
    c.save_json(out/"model-download.json",{"revision":info.sha,"sha256":inventory})
    tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,trust_remote_code=False)
    ids=c.id_map(tok)
    base=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
                         trust_remote_code=False,dtype=torch.float32)
    c.check(base.config.model_type=="lfm2" and base.config.num_hidden_layers==16,
            "expected full LFM350 not loaded")
    base.requires_grad_(False)
    base.eval()
    model=PeftModel.from_pretrained(base,args.adapter,is_trainable=True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    c.check(len(trainable)==24 and all("lora_" in n for n,p in model.named_parameters()
            if p.requires_grad),"unexpected non-LoRA trainability")
    c.check(sum(p.numel() for p in trainable)==172032,"rank8 q/v size drift")
    frozen=c.base_fingerprint(model)
    before={n:c.sha_bytes(p.detach().cpu().numpy().tobytes())
            for n,p in model.named_parameters() if "lora_" in n}
    c.save_json(out/"preflight.json",{"full_model":c.MODEL,"revision":c.REVISION,
        "adapter":PARENT_ADAPTER,"rank8_LoRA_trainable":172032,"torch":torch.__version__,
        "optimizer":"fresh AdamW, 2e-4, betas .9/.95, decay .01",
        "objective":"host-legal CE plus positive/negative BRANCH contrast and GP006 KL preservation",
        "comparison":"unchanged GP350006 weights versus selective 72-step continuation"})
    initial=probe(model,tok,ids,sample)
    c.save_json(out/"probe-initial.json",initial)
    # Unchanged baseline policy uses exactly original GP350006 weights before updates.
    baseline=[]
    model.eval()
    for size in SIZES:
        for seed in EVAL_SEEDS:
            for family in c.FAMILIES:
                baseline.append(episode(w0,model,tok,ids,out,"gp350006",size,seed,family,started))
    rng=random.Random(3510011)
    for bucket in pairs.values():rng.shuffle(bucket)
    for bucket in branches.values():rng.shuffle(bucket)
    # Real BRANCH-positive versus branch-legal nonbranch observations.
    negatives={}
    for fam in ("greebles","pipework"):
        options=[r for r in training if r["observation"]["family"]==fam
                 and "BRANCH" in r["observation"]["legal"] and r["target"]!="BRANCH"]
        c.check(len(options)>=8 and branches[fam],
                "need source-W0 branch positives and legal negatives "+fam)
        period=7 if fam=="greebles" else 17
        windows=(0,1) if fam=="greebles" else (16,)
        matched=[r for r in options if r["observation"]["tick"]%period in windows]
        negatives[fam]=matched if matched else options
        rng.shuffle(negatives[fam])
    # Snapshot GP006 action distributions on real W0 pressure observations.
    # Frozen parent logits are stored once, then no second model is resident.
    retention=[r for r in training if r["ray2"] in ("#","+")
               and r["target"]!="BRANCH"]
    c.check(len(retention)>=24,"too few W0 parent pressure anchors")
    rng.shuffle(retention)
    retention=retention[:24]
    model.eval()
    with torch.inference_mode():
        parent_logits=[prev.rich_score(model,tok,ids,r["observation"]).detach().cpu()
                       for r in retention]
    parent_digest=c.sha_bytes(c.canonical(
        [[float(v) for v in z] for z in parent_logits]).encode())
    c.save_json(out/"frozen-gp006-parent-logits.json",{
        "parent_adapter_sha256":PARENT_ADAPTER,"real_W0_anchors":len(retention),
        "parent_logits_sha256":parent_digest,"one_resident_model":True})
    optimizer=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=.01,betas=(.9,.95))
    visits=collections.Counter()
    progress=[]
    for step in range(STEPS):
        if CANCELLED or time.monotonic()-started>args.deadline-540:
            break
        fam=c.FAMILIES[step%3]
        is_branch=(step%2==0) and fam in ("greebles","pipework")
        marker=("#","+")[(step//3)%2]
        if is_branch:
            data=branches[fam]
            positive=data[visits[(fam,"branch")]%len(data)]
            negative=negatives[fam][visits[(fam,"neg")]%len(negatives[fam])]
            c.check(positive["target"]=="BRANCH" and negative["target"]!="BRANCH"
                    and "BRANCH" in negative["observation"]["legal"],
                    "illegal or misleading branch counterexample")
            batch=[positive,negative]
            visits[(fam,"neg")]+=1
            visits[(fam,"branch")]+=1
        else:
            data=pairs[(fam,marker)]
            batch=list(data[visits[(fam,marker)]%len(data)])
            visits[(fam,marker)]+=1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores=[prev.rich_score(model,tok,ids,row["observation"]) for row in batch]
        ce=[]
        for z,row in zip(scores,batch):
            legal=row["observation"]["legal"]
            selected=z.index_select(0,torch.tensor([c.ACTIONS.index(a) for a in legal]))
            ce.append(F.cross_entropy(selected.unsqueeze(0),
                                     torch.tensor([legal.index(row["target"])])))
        celoss=torch.stack(ce).mean()
        if len(batch)==2:
            ia,ib=(c.ACTIONS.index(row["target"]) for row in batch)
            diff_a=scores[0][ia]-scores[0][ib]
            diff_b=scores[1][ib]-scores[1][ia]
            margin=(F.softplus(0.6-diff_a)+F.softplus(0.6-diff_b))/2
            loss=0.5*celoss+0.5*margin
        else:
            margin=None
            loss=celoss
        # KL regularizes legal GP006 logits on original W0 observed pressure
        # states, reducing collapse of proven anticipatory obstacle steering.
        idx=step%len(retention)
        obs=retention[idx]["observation"]
        anchor_z=prev.rich_score(model,tok,ids,obs)
        selection=torch.tensor([c.ACTIONS.index(a) for a in obs["legal"]])
        temperature=1.7
        ref=parent_logits[idx].index_select(0,selection)/temperature
        pred=anchor_z.index_select(0,selection)/temperature
        preserve=F.kl_div(F.log_softmax(pred,dim=0),
                          F.softmax(ref,dim=0),reduction="sum")*temperature**2
        loss=.65*loss+.35*preserve
        c.check(torch.isfinite(loss).item(),"nonfinite replay CE/margin/KL")
        loss.backward()
        grads=[p.grad for p in trainable if p.grad is not None]
        c.check(grads and any(torch.count_nonzero(g).item() for g in grads),
                "optimizer did not update any LoRA gradients")
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.0))
        c.check(math.isfinite(norm) and norm>0,"nonfinite/zero grad norm")
        optimizer.step()
        rec={"step":step+1,"family":fam,"real_state_markers":[r["ray2"] for r in batch],
             "targets":[r["target"] for r in batch],
             "real_gradient_forwards":len(batch)+1,"actual_adamw_updates":1,
              "branch_pos_neg":is_branch,"GP006_pressure_KL":float(preserve.detach()),
             "backward_passes":1,"loss":float(loss.detach()),
             "ce":float(celoss.detach()),"margin":None if margin is None else float(margin.detach()),
             "grad_norm":norm}
        progress.append(rec)
        with (out/"optimizer.jsonl").open("a") as handle:
            handle.write(c.canonical(rec)+"\n")
        print(c.canonical({"GP350010_REAL_ADAMW":rec}),flush=True)
        if step+1 in CHECKPOINTS:
            model.save_pretrained(out/f"adapter-step-{step+1}",safe_serialization=True)
    c.check(progress,"zero optimizer steps")
    c.check(frozen==c.base_fingerprint(model),"frozen full parent changed")
    after={n:c.sha_bytes(p.detach().cpu().numpy().tobytes())
           for n,p in model.named_parameters() if "lora_" in n}
    c.check(any(before[n]!=after[n] for n in before),"LoRA unchanged")
    if len(progress) not in CHECKPOINTS:
        model.save_pretrained(out/f"adapter-step-{len(progress)}",safe_serialization=True)
    final=probe(model,tok,ids,sample)
    c.save_json(out/"probe-final.json",final)
    learned=[]
    model.eval()
    for size in SIZES:
        for seed in EVAL_SEEDS:
            for family in c.FAMILIES:
                learned.append(episode(w0,model,tok,ids,out,"gp350010",size,seed,family,started))
    controls=[]
    for size in SIZES:
        for seed in EVAL_SEEDS:
            for family in c.FAMILIES:
                for mode in ("random","wall","fidget"):
                    r=w0.run(seed,mode=mode,family=family,size=size,steps=64)
                    b=w0.png_bytes(r)
                    name=f"control-{mode}-s{size}-{seed}-{family}.png"
                    (out/name).write_bytes(b)
                    controls.append({"png":name,"png_sha256":c.sha_bytes(b),
                      "family":family,"size":size,"seed":seed,"mode":mode,
                      "unique_cells":r["metrics"]["unique_trail_cells"],
                      "branches":r["metrics"]["branches"],
                      "host_illegal":r["metrics"]["invalid_raw_choices"]})
    images=baseline+learned+controls
    c.check(len(images)==60 and all(c.sha(out/r["png"])==r["png_sha256"] for r in images),
            "original W0 PNG count/hash mismatch")
    adapters={str(p.relative_to(out)):c.sha(p) for p in out.rglob("adapter-step-*/*")
              if p.is_file()}
    c.check(adapters,"checkpoint missing")
    c.save_json(out/"summary.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
        "status":"complete" if len(progress)==STEPS else "partial",
        "model":c.MODEL,"revision":c.REVISION,
        "parent_adapter_sha256":PARENT_ADAPTER,
        "optimizer_is_new_not_exact_resume":True,
        "updates":len(progress),"backwards":len(progress),
        "real_gradient_forwards":sum(x["real_gradient_forwards"] for x in progress),
        "initial_probe":{k:v for k,v in initial.items() if k!="rows"},
        "final_probe":{k:v for k,v in final.items() if k!="rows"},
        "baseline":baseline,"trained":learned,"controls":controls,
        "original_pngs":len(images),"adapter_sha256":adapters,
        "frozen_full_parent_unchanged":True,"adapter_updated":True,
        "native":False,"host":"original synthetic W0 only",
        "d1_matched":"NOT RUN","vision_conditioned":False,
        "parent_logits_sha256":parent_digest,"parent_W0_anchors":len(retention),
        "branch_positive_negative_updates":sum(r["branch_pos_neg"] for r in progress),
        "wall_seconds":time.monotonic()-started,
        "cpu_seconds":time.process_time(),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    c.check(len(progress)==STEPS,"deadline before 72 updates")

def self_test(w0_path):
    w0=c.load_w0(w0_path)
    a=observed(w0,TRAIN_SEEDS)
    d=observed(w0,DEV_SEEDS)
    x=stratified(d)
    pairs,branches=make_pairs(a)
    c.check({r["seed"] for r in a}.isdisjoint({r["seed"] for r in d}),
            "train/dev leak")
    c.check(len(x)==72 and len(pairs)==6,"bad curriculum shape")
    c.check(all(branches[f] for f in ("greebles","pipework")),
            "no real W0 positive branch samples")
    c.check(all(r["target"] in r["observation"]["legal"] for r in a+d),
            "illegal teacher")
    c.check(any(r["ray2"]=="+" for r in a) and
            any(r["ray2"]=="#" for r in a),"no real visited or obstacle exposure")
    print(c.canonical({"GP350010_SELFTEST":"PASS","training_rows":len(a),
        "dev_rows":len(d),"real_ray_counts":dict(collections.Counter(r["ray2"] for r in a)),
        "heldout_ray_counts":dict(collections.Counter(r["ray2"] for r in x)),
        "paired_pressure_buckets":{str(k):len(v) for k,v in pairs.items()},
        "branch_rows":sum(len(v) for v in branches.values()),
        "synthetic_teacher_on_real_W0_observations":True}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--w0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--adapter",type=Path)
    p.add_argument("--deadline",type=int,default=DEADLINE)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    c.check(a.deadline==DEADLINE,"deadline recipe drift")
    if a.self_test:
        self_test(a.w0)
    else:
        c.check(a.adapter is not None and a.adapter.is_dir(),"original adapter missing")
        try:run(a)
        except BaseException as exc:
            a.out.mkdir(parents=True,exist_ok=True)
            f=a.out/"failure.json"
            if not f.exists():
                c.save_json(f,{"type":type(exc).__name__,"message":str(exc),
                    "source_sha":os.getenv("GITHUB_SHA"),"no_success_claim":True})
            raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    main()
