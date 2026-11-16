#!/usr/bin/env python3
"""GP350-003: fresh-parent, balanced three-theme/geometry LFM350 rank8 CPU LoRA.

Isolated synthetic W0 game-policy experiment. No native game writes.
Unlike GP350-001 (which optimized Greebles-vs-Pipework pairs), every
optimizer step backpropagates all THREE family targets for the SAME state.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, resource, signal, time
from pathlib import Path
import lfm350 as c

SCHEMA="driv3r.lfm350.gp350003-triad/1"
STEPS=72
SEEDS=tuple(range(3101,3113))
DEV=(9103,9119,9137)
ROLLOUT=(9503,9521,9539)
FAMILIES=c.FAMILIES
STOP=False

def sig(_s,_f):
    global STOP
    STOP=True

def target(obs):
    """Explicit synthetic motif/geometry preference, NOT native teacher truth."""
    legal=set(obs["legal"])
    tick=obs["tick"]
    def open_turn(turn):
        return c.nearby(obs,turn)=="."
    def first(names):
        return next((x for x in names if x in legal), "STOP")
    if obs["family"]=="greebles":
        if "BRANCH" in legal and tick%4 in (0,1):
            return "BRANCH"
        return first([x for x,t in (("RIGHT",1),("LEFT",-1),("STRAIGHT",0))
                      if open_turn(t)]+["RETRACE","STOP"])
    if obs["family"]=="pipework":
        if tick%11==9 and "BRANCH" in legal:
            return "BRANCH"
        return first([x for x,t in (("STRAIGHT",0),("RIGHT",1),("LEFT",-1))
                      if open_turn(t)]+["RETRACE","STOP"])
    # Architecture: a right-angle corridor entry, then long straight segments,
    # re-route locally around blocked/old-trail cells. No game authority.
    if tick%12==0 and open_turn(1) and "RIGHT" in legal:
        return "RIGHT"
    return first([x for x,t in (("STRAIGHT",0),("LEFT",-1),("RIGHT",1))
                  if open_turn(t)]+["RETRACE","STOP"])

def make_triples(w0,seeds):
    triples=[]
    seen=set()
    for seed in seeds:
        for family in FAMILIES:
            for e in c.collect_world(w0,seed,family,steps=18):
                base=e["observation"]
                # Counterfactual same-observation family labels; no altered legal set.
                key=(seed,c.canonical({k:v for k,v in base.items() if k!="family"}))
                if key in seen: continue
                seen.add(key)
                triple=[]
                for name in FAMILIES:
                    obs=dict(base,family=name)
                    action=target(obs)
                    c.check(action in obs["legal"],"unsafe triad label")
                    triple.append({"seed":seed,"observation":obs,"target":action,
                                   "synthetic":True,"W0_observed_state":True})
                c.check(all({k:v for k,v in x["observation"].items() if k!="family"}==
                            {k:v for k,v in triple[0]["observation"].items() if k!="family"}
                            for x in triple),"triad input drift")
                triples.append(triple)
    return triples

def case_label(triple):
    state=triple[0]["observation"]
    distinct=len({x["target"] for x in triple})
    blocked=sum(row.count("#")+row.count("+") for row in state["local_5x5"])
    return {"distinct":distinct,"blocked":blocked,"seed":triple[0]["seed"],
            "tick":state["tick"],"legal":state["legal"]}

def probe(model,tok,ids,triples,n=18):
    import torch
    model.eval()
    results=[]
    # Explicitly span distinct seeds, ticks and obstacle structures.
    order=sorted(triples,key=lambda t:(t[0]["observation"]["tick"],t[0]["seed"]))
    picks=order[::max(1,len(order)//n)][:n]
    with torch.inference_mode():
        for trio in picks:
            answers=[]
            for row in trio:
                obs=row["observation"]
                z=c.score(model,tok,ids,obs)
                raw=c.ACTIONS[int(z.argmax().item())]
                legal=max(obs["legal"],key=lambda a:float(z[c.ACTIONS.index(a)]))
                answers.append({"family":obs["family"],"target":row["target"],
                                "raw":raw,"legal":legal,"match":legal==row["target"],
                                "scores":[float(v) for v in z]})
            results.append({"seed":trio[0]["seed"],"tick":trio[0]["observation"]["tick"],
                            "blocked":case_label(trio)["blocked"],"choices":answers,
                            "distinct_raw":len({a["raw"] for a in answers}),
                            "distinct_legal":len({a["legal"] for a in answers})})
    return {"cases":results,"matches":sum(x["match"] for a in results for x in a["choices"]),
            "total":sum(len(a["choices"]) for a in results),
            "family_matches":{f:sum(x["match"] for a in results for x in a["choices"] if x["family"]==f)
                              for f in FAMILIES}}

def rollout(w0,model,tok,ids,out,stage):
    import torch
    model.eval()
    results=[]
    for seed in ROLLOUT:
        for family in FAMILIES:
            decisions=[]
            def choose(obs):
                with torch.inference_mode():
                    z=c.score(model,tok,ids,obs)
                raw=c.ACTIONS[int(z.argmax().item())]
                applied=max(obs["legal"],key=lambda a:float(z[c.ACTIONS.index(a)]))
                decisions.append({"raw":raw,"raw_legal":raw in obs["legal"],
                                  "applied":applied,"logits":[float(x) for x in z],
                                  "observation":obs})
                return applied
            rec=w0.run(seed,mode="callback",family=family,steps=24,selector=choose)
            c.check(len(decisions)==len(rec["events"]),"model/W0 forward disagreement")
            c.check(all(e["applied"]==x["applied"]==e["raw_choice"] and e["raw_legal"]
                        for x,e in zip(decisions,rec["events"])),"host action violation")
            rep=w0.run(seed,mode="replay",family=family,steps=24,
                       raw_actions=[x["applied"] for x in decisions])
            png=w0.png_bytes(rec)
            c.check(png==w0.png_bytes(rep),"source W0 replay mismatch")
            stem=f"{stage}-{seed}-{family}"
            (out/(stem+".png")).write_bytes(png)
            c.save_json(out/(stem+".json"),{"record":rec,"decisions":decisions})
            results.append({"seed":seed,"family":family,"stage":stage,
                            "first_raw":decisions[0]["raw"],
                            "first_legal":decisions[0]["applied"],
                            "decisions":len(decisions),"raw_illegal":sum(not a["raw_legal"] for a in decisions),
                            "unique_cells":rec["metrics"]["unique_trail_cells"],
                            "branches":rec["metrics"]["branches"],
                            "stopped":rec["metrics"]["terminated_by_stop"],
                            "png":stem+".png","png_sha256":c.sha_bytes(png)})
    return results

def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    from huggingface_hub import HfApi,snapshot_download
    torch.set_num_threads(2)
    torch.manual_seed(3503003)
    random.seed(3503003)
    started=time.monotonic()
    out=args.out
    c.check(not out.exists(),"immutable evidence lease already exists")
    out.mkdir(parents=True)
    c.save_json(out/"lease.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
      "run_id":os.getenv("GITHUB_RUN_ID"),"attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
      "fresh_parent":c.MODEL,"revision":c.REVISION,"steps":STEPS,
      "backward_passes":STEPS*3,"native":False,"draft_hold":True})
    w0=c.load_w0(args.w0)
    tr=make_triples(w0,SEEDS)
    dev=make_triples(w0,DEV)
    c.check(len(tr)>=80 and len(dev)>=20,"insufficient genuine W0-state triples")
    c.check({e[0]["seed"] for e in tr}.isdisjoint({e[0]["seed"] for e in dev}),"world split leaked")
    labels=[case_label(t) for t in tr]
    c.check(sum(x["distinct"]==3 for x in labels)>=24,"insufficient three-way target contrasts")
    digest=(("\n".join(c.canonical(t) for t in tr))+"\n").encode()
    (out/"train-triples.jsonl").write_bytes(digest)
    c.save_json(out/"curriculum.json",{"train_worlds":list(SEEDS),"dev_worlds":list(DEV),
      "train_triples":len(tr),"dev_triples":len(dev),"train_sha256":c.sha_bytes(digest),
      "three_distinct_target_triples":sum(x["distinct"]==3 for x in labels),
      "obstacle_neighbour_triples":sum(x["blocked"]>0 for x in labels),
      "synthetic_teacher_not_native":True})
    info=HfApi().model_info(c.MODEL,revision=c.REVISION)
    c.check(info.sha==c.REVISION,"model revision drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=args.cache,max_workers=2,
      allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
       "generation_config.json","tokenizer*","special_tokens_map.json","added_tokens.json",
       "vocab.*","merges.txt","README.md","LICENSE*"])
    filehash={str(p.relative_to(args.cache)):c.sha(p) for p in sorted(args.cache.rglob("*"))
              if p.is_file() and ".cache" not in p.parts}
    c.check(any(n.endswith(".safetensors") for n in filehash),"no real pinned weights")
    c.save_json(out/"download.json",{"model":c.MODEL,"revision":c.REVISION,"sha256":filehash})
    tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,trust_remote_code=False)
    ids=c.id_map(tok)
    base=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
                                               trust_remote_code=False,dtype=torch.float32)
    c.check(base.config.model_type=="lfm2" and base.config.num_hidden_layers==16,
            "unexpected model architecture")
    targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear)
             and ".self_attn." in n and n.endswith((".q_proj",".v_proj"))]
    c.check(len(targets)==12,"rank8 target path inventory changed")
    base.requires_grad_(False)
    base.eval()
    model=get_peft_model(base,LoraConfig(task_type="CAUSAL_LM",r=8,lora_alpha=16,
           lora_dropout=0.0,bias="none",target_modules=targets))
    trainable=[p for p in model.parameters() if p.requires_grad]
    c.check(len(trainable)==24 and all("lora_" in n for n,p in model.named_parameters()
             if p.requires_grad),"non-LoRA trainable parameter")
    before=c.base_fingerprint(model)
    pre_adapter={n:c.sha_bytes(p.detach().cpu().numpy().tobytes())
                 for n,p in model.named_parameters() if "lora_" in n}
    c.save_json(out/"preflight.json",{"projection_paths":targets,"rank":8,
      "trainable_parameters":sum(p.numel() for p in trainable),
      "token_ids":ids,"torch":torch.__version__,"fp32":True})
    stockprobe=probe(model,tok,ids,dev)
    c.save_json(out/"probe-stock.json",stockprobe)
    stock=rollout(w0,model,tok,ids,out,"stock")
    optimizer=torch.optim.AdamW(trainable,lr=3.0e-4,weight_decay=.01,betas=(.9,.95))
    rnd=random.Random(3503004)
    triple_unique=[t for t in tr if len({x["target"] for x in t})==3]
    triple_other=[t for t in tr if len({x["target"] for x in t})<3]
    c.check(len(triple_unique)>=24 and triple_other,"strata missing")
    rnd.shuffle(triple_unique);rnd.shuffle(triple_other)
    progress=[]
    for index in range(STEPS):
        if STOP or time.monotonic()-started>args.deadline-420:break
        source=triple_unique if index%4!=3 else triple_other
        triad=source[(index//4 if index%4==3 else index-index//4)%len(source)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for row in triad:
            z=c.score(model,tok,ids,row["observation"])
            target=c.ACTIONS.index(row["target"])
            loss=F.cross_entropy(z.unsqueeze(0),torch.tensor([target]))
            c.check(torch.isfinite(loss).item(),"nonfinite triad CE")
            (loss/3).backward()
            losses.append(float(loss.item()))
        gradients=[p.grad for p in trainable if p.grad is not None]
        c.check(gradients and any(torch.count_nonzero(g).item()>0 for g in gradients),
                "LoRA optimizer received no real gradients")
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.0).item())
        c.check(math.isfinite(norm),"nonfinite gradient norm")
        optimizer.step()
        row={"update":index+1,"backward_passes":3,"world":triad[0]["seed"],
             "targets":[x["target"] for x in triad],"losses":losses,"grad_norm":norm}
        progress.append(row)
        with (out/"optimizer.jsonl").open("a") as fd: fd.write(c.canonical(row)+"\n")
        print(c.canonical({"GP350003_REAL_ADAMW":row}),flush=True)
        if (index+1)%24==0:
            model.save_pretrained(out/f"adapter-step-{index+1}",safe_serialization=True)
    c.check(progress,"zero optimizer steps")
    c.check(before==c.base_fingerprint(model),"frozen base changed")
    after_adapter={n:c.sha_bytes(p.detach().cpu().numpy().tobytes())
                   for n,p in model.named_parameters() if "lora_" in n}
    c.check(any(after_adapter[n]!=pre_adapter[n] for n in pre_adapter),"LoRA numerically unchanged")
    if len(progress)%24:model.save_pretrained(out/f"adapter-step-{len(progress)}",safe_serialization=True)
    trainedprobe=probe(model,tok,ids,dev)
    c.save_json(out/"probe-trained.json",trainedprobe)
    trained=rollout(w0,model,tok,ids,out,"trained")
    checks={str(p.relative_to(out)):c.sha(p) for p in out.rglob("adapter-step-*/*") if p.is_file()}
    c.save_json(out/"summary.json",{"schema":SCHEMA,"source_sha":os.getenv("GITHUB_SHA"),
      "status":"complete" if len(progress)==STEPS else "partial","updates":len(progress),
      "backward_passes":3*len(progress),"original_parent_model":c.MODEL,
      "model_revision":c.REVISION,"adapter_sha256":checks,
      "frozen_reference_unchanged":True,"adapter_numerically_changed":True,
      "heldout_stock":stockprobe,"heldout_trained":trainedprobe,
      "stock":stock,"trained":trained,"original_pngs":len(stock)+len(trained),
      "native_paint":0,"host":"synthetic_W0_only",
      "wall_seconds":time.monotonic()-started,
      "cpu_seconds":time.process_time(),
      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    c.check(len(progress)==STEPS,"optimizer deadline stopped before requested steps")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--w0",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--deadline",type=int,default=1600)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    c.check(args.deadline==1600,"deadline altered")
    if args.self_test:
        open_state={"family":"greebles","tick":0,"heading":0,
                    "local_5x5":[".....",".....","..@..",".....","....."],
                    "legal":["STRAIGHT","LEFT","RIGHT","BRANCH","STOP"],"last_actions":[]}
        assert [target(dict(open_state,family=f)) for f in FAMILIES]==[
            "BRANCH","STRAIGHT","RIGHT"]
        wall=dict(open_state,tick=3,local_5x5=[".....",".....","..@#.",".....","....."])
        assert all(target(dict(wall,family=f)) in wall["legal"] for f in FAMILIES)
        print("GP350003 TRUE THREE-THEME CONTRAST AND WALL SAFETY SELF-TEST PASS")
        return
    try:
        run(args)
    except BaseException as exc:
        args.out.mkdir(parents=True,exist_ok=True)
        p=args.out/"failure.json"
        if not p.exists():
            p.write_text(json.dumps({"kind":type(exc).__name__,"message":str(exc),
                "source":os.getenv("GITHUB_SHA"),"no_success_claim":True})+"\n")
        raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,sig)
    signal.signal(signal.SIGINT,sig)
    main()
