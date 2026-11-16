"""M5: topology-diverse synthetic tiny decision-head training and held-out evaluation.

Explicitly NOT CyberSand native gameplay. All legal action menus and 96x96 pixel
receipts are enforced by the unmodified core.World interface. All optims bounded.
No legacy M4 EVAL geometry appears in TRAIN/DEV datasets or checkpoint selection.
"""
from __future__ import annotations
import argparse,hashlib,json,random,statistics,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from .core import ACTIONS,DELTA,FAMILIES,FEATURES,World,choose_random,choose_wall,raster
from .m3_curriculum import BUDGET,models as m3_models
from .m4_ood import world as m4_world,SEEDS as M4_SEEDS,novel as m4_novel,reachable as reachable_global

PROTOCOL="tiny-policy-m5-topology-v1"
TRAIN_SEEDS=tuple(range(21000,21036))
DEV_SEEDS=tuple(range(22000,22008))
EVAL_SEEDS=tuple(range(23000,23012))
TURNS=64
POLICIES=("oracle","novel","wall","random","m3-linear","m3-mlp","m3-conv","m5-linear","m5-mlp","m5-conv")
ORIGINAL_M3_SHA={
    "linear":"a5d1ec8f5f956c05968f7b5b538952e68d89b861d337a48559e0de759fc32435",
    "mlp":"1b2c9cea0b99fffde0b3ebd39942835540880d93a6e70ce1518ff39c508ceeb5",
    "conv":"b44a97d0a6af84010280bf83e22c481afa15546d8b1575ef4c855d5bb6661d6d",
}

def validated_m3_lineage(checkpoint_dir, source_rebuilt=False):
    """Allow a different CPU checkpoint only if its own source-run report proves lineage.

    Default remains the exact archived M3 bytes. Source rebuild is explicit,
    SHA-recorded, synthetic-only and never represented as the original weights.
    """
    hashes={name:hashlib.sha256((checkpoint_dir/f"{name}.pt").read_bytes()).hexdigest()
            for name in ORIGINAL_M3_SHA}
    if not source_rebuilt:
        if hashes!=ORIGINAL_M3_SHA:raise ValueError("Original M3 checkpoint SHA mismatch")
        return hashes,None
    report_path=checkpoint_dir.parent/"report.json"
    report=json.loads(report_path.read_text())
    if (report.get("protocol")!="tiny-policy-m3-curriculum-cpu-v1"
            or report.get("synthetic_only") is not True
            or set(report.get("models",{}))!=set(ORIGINAL_M3_SHA)
            or report.get("illegal_applied_actions")!=0):
        raise ValueError("Missing/invalid regenerated M3 research source report")
    for name,sha in hashes.items():
        if report["models"][name].get("head_checkpoint_sha256")!=sha:
            raise ValueError("Rebuilt M3 checkpoint disagrees with original producer report")
    return hashes,hashlib.sha256(report_path.read_bytes()).hexdigest()

def geometry_sha(w):
    return hashlib.sha256(w.blocked.tobytes()+w.protected.tobytes()).hexdigest()

def maze_world(seed,family):
    """Seed-disjoint, source-contained DFS/braided/grid terrain; never a M4 seed."""
    if seed not in (*TRAIN_SEEDS,*DEV_SEEDS,*EVAL_SEEDS):
        raise ValueError("Unallocated world seed")
    w=World(seed,family,budget=BUDGET)
    w.blocked[1:95,1:95]=True
    w.protected[:]=False
    # Graph nodes at 3 native pixels separation, centred on the starting pixel.
    rng=random.Random(seed*3907)
    origin=(6,6)
    cells={origin}; stack=[origin]; edges=set()
    while stack:
        x,y=stack[-1]
        candidates=[(x+dx,y+dy) for dx,dy in DELTA[:4]
                    if 0<=x+dx<13 and 0<=y+dy<13 and (x+dx,y+dy) not in cells]
        if not candidates:
            stack.pop();continue
        n=rng.choice(candidates)
        edges.add(tuple(sorted(((x,y),n))))
        cells.add(n);stack.append(n)
    # Controls for novelty, recovery, long corridor transitions, and loops.
    variant=seed%4
    chance=(.01,.06,.16,.27)[variant]
    for x in range(13):
        for y in range(13):
            for dx,dy in ((1,0),(0,1)):
                n=(x+dx,y+dy)
                if n[0]>=13 or n[1]>=13:continue
                if rng.random()<chance:edges.add(tuple(sorted(((x,y),n))))
    def xy(p):return (48+3*(p[0]-6),48+3*(p[1]-6))
    for p in cells:
        x,y=xy(p);w.blocked[y,x]=False
    for a,b in edges:
        x0,y0=xy(a);x1,y1=xy(b)
        if x0==x1:w.blocked[min(y0,y1):max(y0,y1)+1,x0]=False
        else:w.blocked[y0,min(x0,x1):max(x0,x1)+1]=False
    # Scarce protected gates may terminate corridors; never obscure the origin.
    possibilities=[(x,y) for x,y in zip(*np.where(~w.blocked)[::-1])
                   if abs(x-48)+abs(y-48)>10 and rng.random()<.018]
    for x,y in possibilities[:9]:w.protected[y,x]=True
    w.blocked[48,48]=False;w.protected[48,48]=False
    if not any(a!=4 for a in w.legal()):raise ValueError("Bad initial trap")
    return w

def oracle(w,rng=None):
    """Local 7x7 lookahead only. Novelty + texture habits, with dead-end recovery.

    No global reachable grid, lookahead beyond observed radius, target art JSON,
    or hidden world identity passed to policy. Both success and failure labels retained.
    """
    moves=[a for a in w.legal() if a!=4]
    if not moves:return 4
    t=len(w.actions)
    options=[]
    for a in moves:
        dx,dy=DELTA[a];x,y=w.x+dx,w.y+dy
        visited=bool(w.visited[y,x])
        # 2-step visible local future access; no privileged global pathfinding.
        futures=0
        walls=0
        for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1),
                      (x-2,y),(x+2,y),(x,y-2),(x,y+2)):
            if not 0<=xx<96 or not 0<=yy<96: walls+=1;continue
            if w.blocked[yy,xx] or w.protected[yy,xx]:walls+=1
            elif not w.visited[yy,xx]:futures+=1
        straight=a==w.heading
        # Bounded aesthetic deviations, requiring a legal action. Some retracing is intentional.
        if w.family=="Greebles":
            style=.45*(not straight)+.25*(t%17 in (0,1) and visited)
        elif w.family=="Pipework":
            style=.6*straight+.08*walls
        else:
            style=.50*(straight if (t//9)%2==0 else not straight)+.17*(a in (0,3))
        score=(3.2*(not visited)-1.5*visited +.25*futures +style
               -.11*(a==(w.heading+2)%4))
        options.append((score,-a,a))
    return max(options)[2]

def novel(w,_rng=None):
    return m4_novel(w,_rng)

def mixed_examples(seeds,turns=56):
    xx=[];yy=[];mm=[];ww=[];ids=[];classes=Counter()
    for seed in seeds:
        for family in FAMILIES:
            for exploration in ("oracle","random","wall","novel"):
                w=maze_world(seed,family)
                rng=random.Random(seed*1543+FAMILIES.index(family)*23)
                for tick in range(turns):
                    legal=w.legal()
                    if not legal:break
                    obs=w.observe()
                    label=oracle(w)
                    if label not in legal:raise ValueError("Illegal label")
                    mask=np.zeros(5,dtype=np.bool_);mask[list(legal)]=True
                    xx.append(obs);yy.append(label);mm.append(mask)
                    ww.append(1.2 if label!=w.heading else 1.0)
                    ids.append((seed,family,exploration,tick,w.state_sha256(),ACTIONS[label]))
                    classes[ACTIONS[label]]+=1
                    if exploration=="oracle":a=label
                    elif exploration=="random":a=choose_random(w,rng)
                    elif exploration=="wall":a=choose_wall(w,rng)
                    else:a=novel(w,rng)
                    w.apply(int(a))
                    if w.stopped:break
            # Actual locally protected forced STOP-only context, not cosmetic suppression.
            trap=maze_world(seed,family)
            for dx,dy in DELTA[:4]:trap.protected[trap.y+dy,trap.x+dx]=True
            if trap.legal()!=(4,):raise ValueError("STOP fixture not protected")
            xx.append(trap.observe());yy.append(4)
            mask=np.zeros(5,dtype=np.bool_);mask[4]=True;mm.append(mask);ww.append(1.)
            ids.append((seed,family,"protected-stop",0,trap.state_sha256(),"STOP"));classes["STOP"]+=1
    X=np.asarray(xx,np.float32);Y=np.asarray(yy,np.int64)
    M=np.asarray(mm,np.bool_);W=np.asarray(ww,np.float32)
    identity=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    return (X,Y,M,W),{"identity_sha256":identity,"examples":len(Y),
                      "actions":dict(sorted(classes.items())),
                      "world_seeds":list(seeds)}

def train_heads(torch,train,dev,out,epochs=5):
    torch.manual_seed(541);torch.set_num_threads(2)
    heads=m3_models(torch)
    X,Y,M,W=[torch.from_numpy(a) for a in train]
    DX,DY,DM=[torch.from_numpy(a) for a in dev[:3]]
    reports={}
    (out/"checkpoints").mkdir(parents=True)
    for name,model in heads.items():
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.002)
        best=None;bestloss=float("inf");hist=[];updates=0
        for epoch in range(epochs):
            model.train()
            indices=torch.randperm(len(X),generator=torch.Generator().manual_seed(9001+epoch))
            for batch in indices.split(512):
                logits=model(X[batch]).masked_fill(~M[batch],-1e6)
                loss=(torch.nn.functional.cross_entropy(logits,Y[batch],reduction="none")*W[batch]).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                optimizer.step();updates+=1
            model.eval()
            with torch.inference_mode():
                all_logits=[]
                for batch in DX.split(2048):all_logits.append(model(batch))
                logits=torch.cat(all_logits).masked_fill(~DM,-1e6)
                val=float(torch.nn.functional.cross_entropy(logits,DY))
                acc=float((logits.argmax(1)==DY).float().mean())
            hist.append({"epoch":epoch+1,"dev_ce":round(val,6),"dev_agreement":round(acc,5)})
            if val<bestloss:
                bestloss=val;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best);model.eval()
        chk={"protocol":PROTOCOL,"architecture":name,"train_identity":train_meta_global["identity_sha256"],
             "dev_identity":dev_meta_global["identity_sha256"],"epochs":epochs,"optimizer_steps":updates,"state_dict":best}
        filename=out/"checkpoints"/f"{name}.pt"
        torch.save(chk,filename)
        reports[name]={"parameters":sum(p.numel() for p in model.parameters()),
                       "updates":updates,"dev":hist,"best_dev_ce":round(bestloss,6),
                       "checkpoint_sha256":hashlib.sha256(filename.read_bytes()).hexdigest()}
    return heads,reports

def evaluate(torch,heads,old_heads,out,seeds,world_builder,tag,policies):
    from PIL import Image,ImageDraw
    rows=[]
    (out/"rollouts").mkdir(parents=True,exist_ok=True)
    examples={}
    for seed in seeds:
        for family in FAMILIES:
            base=world_builder(seed,family)
            orig_geom=geometry_sha(base)
            family_rows=[]
            for name in policies:
                w=world_builder(seed,family)
                if geometry_sha(w)!=orig_geom:raise ValueError("Family geometry drift")
                rng=random.Random(seed*977+FAMILIES.index(family)*71)
                actions=[]
                for _ in range(TURNS):
                    legal=w.legal()
                    if not legal:break
                    obs=w.observe()
                    if name=="oracle":action=oracle(w)
                    elif name=="novel":action=novel(w,rng)
                    elif name=="wall":action=choose_wall(w,rng)
                    elif name=="random":action=choose_random(w,rng)
                    else:
                        pack=old_heads if name.startswith("m3-") else heads
                        model=pack[name[3:]]
                        with torch.inference_mode():
                            logits=model(torch.from_numpy(obs[None,:]))[0]
                        action=max(legal,key=lambda a:(float(logits[a]),-a))
                    if action not in legal:raise ValueError("Illegal proposal")
                    before=w.state_sha256()
                    record={"before_sha256":before,"obs_sha256":hashlib.sha256(obs.tobytes()).hexdigest(),
                            "legal":[ACTIONS[a] for a in legal]}
                    record.update(w.apply(int(action)))
                    actions.append(record)
                    if w.stopped:break
                receipt={"protocol":PROTOCOL,"synthetic_only":True,"tag":tag,
                         "world_seed":seed,"family":family,"policy":name,
                         "geometry_sha256":orig_geom,"window":TURNS,"budget":BUDGET,
                         "events":actions,"final_sha256":w.state_sha256(),
                         "cells":int(w.visited.sum()),"branches":w.branch_nodes(),
                         "retraces":w.revisit_landings,"dead_ends":w.dead_ends,
                         "steps":len(actions),"early_stop":bool(w.actions and w.actions[-1]=="STOP")}
                # Rebuild and fully validate all receipts before considering a result.
                check=world_builder(seed,family)
                for r in receipt["events"]:
                    if check.state_sha256()!=r["before_sha256"] or hashlib.sha256(check.observe().tobytes()).hexdigest()!=r["obs_sha256"]:
                        raise ValueError("Replay pre-state/observation corruption")
                    if [ACTIONS[a] for a in check.legal()]!=r["legal"]:raise ValueError("Replay legality corruption")
                    if r["action"] not in ACTIONS:raise ValueError("Unknown action")
                    a=ACTIONS.index(r["action"]);got=check.apply(a)
                    if got!={k:r[k] for k in got}:raise ValueError("Replay state corruption")
                if check.state_sha256()!=receipt["final_sha256"] or not np.array_equal(check.visited,w.visited):
                    raise ValueError("Replay image corruption")
                filename=f"{tag}-{seed}-{family}-{name}"
                (out/"rollouts"/f"{filename}.json").write_text(json.dumps(receipt,sort_keys=True)+"\n")
                raster(w,2).save(out/"rollouts"/f"{filename}.png")
                rows.append({k:receipt[k] for k in ("world_seed","family","policy","cells","branches","retraces","dead_ends","steps","early_stop")}|{"tag":tag})
                family_rows.append((name,w))
            examples[(seed,family)]=family_rows
    # Produce original-host pixel galleries, not invented art.
    for seed in seeds[:2]:
        cell=192;names=list(policies)
        art=Image.new("RGB",(len(names)*cell,len(FAMILIES)*(cell+30)),(250,250,247))
        draw=ImageDraw.Draw(art)
        for fi,family in enumerate(FAMILIES):
            for pi,(name,w) in enumerate(examples[(seed,family)]):
                art.paste(raster(w,2),(pi*cell,fi*(cell+30)+30))
                draw.text((pi*cell+4,fi*(cell+30)+8),f"{family[:4]} {name}",fill=(0,0,0))
        art.save(out/f"m5-{tag}-{seed}-original.png")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--m3-checkpoints",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--m3-rebuilt-from-source",action="store_true")
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError("M5 must not overwrite evidence")
    if not 1<=args.epochs<=12:raise ValueError("Bounded epoch deadline")
    args.output.mkdir(parents=True)
    import torch
    torch.set_num_threads(2)
    # Default exact M3 checkpoints; explicit source rebuild has independent lineage.
    m3_hashes,m3_report_sha=validated_m3_lineage(args.m3_checkpoints,args.m3_rebuilt_from_source)
    torch.manual_seed(730)
    old=m3_models(torch)
    for name,net in old.items():
        f=args.m3_checkpoints/f"{name}.pt"
        if hashlib.sha256(f.read_bytes()).hexdigest()!=m3_hashes[name]:
            raise ValueError("Original M3 checkpoint SHA mismatch")
        chk=torch.load(f,map_location="cpu",weights_only=True)
        if chk["architecture"]!=name or chk["protocol"]!="tiny-policy-m3-curriculum-cpu-v1":
            raise ValueError("M3 identity mismatch")
        net.load_state_dict(chk["state_dict"],strict=True);net.eval()
    assert set(TRAIN_SEEDS).isdisjoint(DEV_SEEDS+EVAL_SEEDS+M4_SEEDS)
    assert set(DEV_SEEDS).isdisjoint(EVAL_SEEDS+M4_SEEDS)
    global train_meta_global,dev_meta_global
    train,train_meta_global=mixed_examples(TRAIN_SEEDS)
    dev,dev_meta_global=mixed_examples(DEV_SEEDS)
    heads,models_report=train_heads(torch,train,dev,args.output,epochs=args.epochs)
    test,eval_meta=mixed_examples(EVAL_SEEDS,turns=24) # diagnostic-only held-out teacher agreement
    diagnostic={}
    for name,model in heads.items():
        with torch.inference_mode():
            logits=model(torch.from_numpy(test[0])).masked_fill(~torch.from_numpy(test[2]),-1e6)
            diagnostic[name]=round(float((logits.argmax(1)==torch.from_numpy(test[1])).float().mean()),5)
    rows=evaluate(torch,heads,old,args.output,EVAL_SEEDS,maze_world,"unseen-maze",POLICIES)
    # True OOD M4 12 worlds held out of all optimizer/DEV data.
    # This comparison is research OOD, not a pristine blind final gate: M4 results were known.
    rows+=evaluate(torch,heads,old,args.output,M4_SEEDS,m4_world,"m4-ood",POLICIES)
    aggregates={}
    for p in POLICIES:
        z=[r for r in rows if r["policy"]==p]
        def group(data):
            return {"n":len(data),"cells_mean":round(statistics.mean(r["cells"] for r in data),3),
                    "branches_mean":round(statistics.mean(r["branches"] for r in data),3),
                    "retraces_mean":round(statistics.mean(r["retraces"] for r in data),3),
                    "early_stop":sum(r["early_stop"] for r in data)}
        aggregates[p]={"maze":group([r for r in z if r["tag"]=="unseen-maze"]),
                       "m4_ood":group([r for r in z if r["tag"]=="m4-ood"])}
    report={"protocol":PROTOCOL,"synthetic_only":True,"native_validated":False,"optimizer_updates":sum(v["updates"] for v in models_report.values()),
            "world_size":96,"host_action_strings":list(ACTIONS),"budget":BUDGET,"window":TURNS,
            "m3_checkpoint_sha256":m3_hashes,"m3_source_rebuild":args.m3_rebuilt_from_source,"m3_report_sha256":m3_report_sha,
            "seeds":{"train":TRAIN_SEEDS,"dev":DEV_SEEDS,"eval":EVAL_SEEDS,"m4_ood":M4_SEEDS},
            "datasets":{"train":train_meta_global,"dev":dev_meta_global,"eval":eval_meta},
            "models":models_report,"eval_oracle_agreement":diagnostic,"aggregate":aggregates,
            "control_note":"STOP never suppressed; legal actions host-enforced; synthetic_only",
            "ood_caveat":"M4 archetypes were known from prior research and do NOT form blind final qualification"}
    (args.output/"rows.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    sha={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output/"manifest.json").write_text(json.dumps(sha,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"optimizer_updates":report["optimizer_updates"],"aggregate":aggregates},indent=2))
if __name__=="__main__":main()
