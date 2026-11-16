"""Live-path versus executed-action memory: real bounded 36/72/128 Adam ablation.

Only existing synthetic DRIV3R World host menus, model-only training, DEVELOPMENT
rollouts and exact checkpoint/event verification. Not native CyberSand PAINT.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np
from . import history_motif_lab as h
from .train import HORIZONS, MAX_RECOVERIES, sha_file, source_pins
from .world import World, MOVE, digest, replay, specs

VERSION = "driv3r-live-stack-github/1"
ARMS = ("executed", "live")
STYLES = ("wave", "jag")
FORCED = {9:"alternate", 12:"BACKTRACK", 20:"RELOCATE_START",
          34:"alternate", 37:"BACKTRACK", 45:"RELOCATE_START",
          68:"alternate", 72:"BACKTRACK", 92:"RELOCATE_START",
          110:"BACKTRACK"}

def heads(w, arm):
    if arm == "executed":
        return [e["chosen"][-1] for e in w.decisions if e["chosen"] in MOVE]
    if arm != "live": raise ValueError("arm")
    reverse = {v:k[-1] for k,v in MOVE.items()}
    return [reverse[(b[0]-a[0],b[1]-a[1])] for a,b in zip(w.path,w.path[1:])]

def memory(w, arm):
    seq = heads(w,arm)
    if not seq: return 0,0,w.heading
    run,last = 1,0
    for a,b in zip(seq,seq[1:]):
        if a==b: run+=1
        else:
            d=(h.HEADINGS.index(b)-h.HEADINGS.index(a))%4
            last = 1 if d==1 else (-1 if d==3 else 0)
            run=1
    return run,last,seq[-1]

def menu_features(w,style,arm):
    if style not in STYLES: raise ValueError("style")
    menu,X = h.matrix(w,style,False)
    run,last,hd=memory(w,arm)
    target=h.STYLES[style][0]
    X=X.copy()
    for i,act in enumerate(menu):
        if act in MOVE:
            rot=(h.HEADINGS.index(act[-1])-h.HEADINGS.index(hd))%4
            straight,right,left=(float(rot==k) for k in (0,1,3))
        else: straight=right=left=0.
        X[i,4:7]=[straight,right,left]
        X[i,15:18]=[min(run/target,2)*q for q in (straight,right,left)]
        X[i,18]=float(style=="wave")*last*(right-left)
    return menu,X

def teacher(w,style):
    run,last,hd=memory(w,"live")
    length,kind=h.STYLES[style]
    if run<length: wanted="MOVE_"+hd
    else:
        sign = 1 if kind=="right" else (-1 if kind=="left" else (-last if last else 1))
        wanted="MOVE_"+h.HEADINGS[(h.HEADINGS.index(hd)+sign)%4]
    menu,X=menu_features(w,style,"live")
    if wanted in menu: return wanted
    moves=[(a,i) for i,a in enumerate(menu) if a in MOVE]
    if moves: return max(moves,key=lambda ai:(X[ai[1],6]+X[ai[1],8]-.4*X[ai[1],10],-ai[1]))[0]
    return next((a for a in ("BACKTRACK","RELOCATE_START","STOP") if a in menu),"STOP")

def force(w,preferred,t,seed,enabled):
    if not enabled or t not in FORCED: return preferred,None
    kind=FORCED[t];menu=w.menu(MAX_RECOVERIES)
    if kind=="alternate":
        moves=[x for x in menu if x in MOVE and x!=preferred]
        actual=moves[(seed+w.spec.seed+t)%len(moves)] if moves else preferred
    else: actual=kind if kind in menu else preferred
    return actual,{"attempt":kind,"executed":actual,"changed":actual!=preferred}

def corpus(seed,n,steps):
    if steps not in HORIZONS or n not in (1,2): raise ValueError("bounded curriculum")
    X={a:[] for a in ARMS};M=[];Y=[];ids=[];recovery=0
    for spec in specs("train",n):
        for style in STYLES:
            w=World.create(spec)
            for t in range(steps):
                if w.stopped: break
                label=teacher(w,style);menu=w.menu(MAX_RECOVERIES)
                for arm in ARMS:
                    offered,f=menu_features(w,style,arm)
                    if offered!=menu:raise ValueError("changed menu")
                    pad=np.zeros((7,h.F));pad[:len(menu)]=f;X[arm].append(pad)
                mask=np.zeros(7,dtype=bool);mask[:len(menu)]=True
                M.append(mask);Y.append(menu.index(label))
                actual,_=force(w,label,t,seed,True)
                recovery+=int(actual in ("BACKTRACK","RELOCATE_START"))
                w.apply(actual,MAX_RECOVERIES)
            if replay(spec,w.decisions,MAX_RECOVERIES).state_sha()!=w.state_sha():
                raise ValueError("training replay mismatch")
            ids.append([spec.ident,style,w.base_sha(),digest(w.decisions)])
    return {a:np.stack(x) for a,x in X.items()},np.stack(M),np.array(Y),digest(ids),recovery

def distribution(w,style,arm,m):
    menu,X=menu_features(w,style,arm)
    z=X@m.weights;z-=z.max();p=np.exp(z)
    return menu,p/p.sum()

def episode(spec,style,arm,m,steps,forced):
    if spec.split!="development" or steps not in (128,160):raise ValueError("DEV only")
    w=World.create(spec);actions=[];lat=[];forced_log=[];matches=[]
    recovery_end=-1
    for t in range(steps):
        if w.stopped:break
        t0=time.perf_counter_ns();menu,p=distribution(w,style,arm,m)
        lat.append((time.perf_counter_ns()-t0)/1e6)
        preferred=menu[int(p.argmax())];ref=teacher(w,style)
        actual,intervention=force(w,preferred,t,m.seed,forced)
        if intervention:forced_log.append([t,intervention])
        if t<=recovery_end and intervention is None and any(a in MOVE for a in menu):
            matches.append(preferred==ref)
        actions.append({"t":t,"preferred":preferred,"teacher":ref,
                       "prob":{a:round(float(v),8) for a,v in zip(menu,p)}})
        w.apply(actual,MAX_RECOVERIES)
        if intervention and intervention["changed"] and actual in ("BACKTRACK","RELOCATE_START"):
            recovery_end=t+8
    if replay(spec,w.decisions,MAX_RECOVERIES).state_sha()!=w.state_sha():
        raise ValueError("evaluation replay mismatch")
    return {"version":VERSION,"world_id":spec.ident,"split":spec.split,"style":style,
            "arm":arm,"horizon":steps,"forced":forced,"base_sha":w.base_sha(),
            "events":w.decisions,"actions":actions,"forced_log":forced_log,
            "post_recovery_n":len(matches),
            "post_recovery_match":round(sum(matches)/len(matches),6) if matches else None,
            "metrics":w.metrics(steps),"ink":[list(x) for x in sorted(w.ink)],
            "final_sha":w.state_sha(),"trajectory_sha":digest(w.decisions),
            "counts":dict(sorted(Counter(x["chosen"] for x in w.decisions).items())),
            "mean_latency_ms":round(statistics.mean(lat),6),"native_receipts":0}

def verify_episode(e,m):
    if (e["version"]!=VERSION or e["split"]!="development" or e["arm"] not in ARMS or
        e["style"] not in STYLES or e["horizon"] not in (128,160) or e["native_receipts"]!=0):
        raise ValueError("invalid episode")
    found=[s for s in specs("development",1) if s.ident==e["world_id"]]
    if len(found)!=1:raise ValueError("unapproved fixture")
    w=World.create(found[0]);end=-1;matches=[];injections=[]
    if e["base_sha"]!=w.base_sha() or e["trajectory_sha"]!=digest(e["events"]):
        raise ValueError("source mismatch")
    if len(e["events"])!=len(e["actions"]):raise ValueError("event length mismatch")
    for t,(row,log) in enumerate(zip(e["events"],e["actions"])):
        menu,p=distribution(w,e["style"],e["arm"],m)
        preferred=menu[int(p.argmax())];ref=teacher(w,e["style"])
        if (log["t"]!=t or log["preferred"]!=preferred or log["teacher"]!=ref or
            set(log["prob"])!=set(menu) or
            abs(sum(log["prob"].values())-1)>1e-6 or
            any(abs(log["prob"][a]-float(v))>2e-8 for a,v in zip(menu,p))):
            raise ValueError("action probability tampering")
        actual,intervention=force(w,preferred,t,m.seed,e["forced"])
        if intervention:injections.append([t,intervention])
        if t<=end and intervention is None and any(a in MOVE for a in menu):
            matches.append(preferred==ref)
        if actual!=row["chosen"] or w.apply(actual,MAX_RECOVERIES)!=row:
            raise ValueError("illegal/falsified host transition")
        if intervention and intervention["changed"] and actual in ("BACKTRACK","RELOCATE_START"):
            end=t+8
    observed=round(sum(matches)/len(matches),6) if matches else None
    if (injections!=e["forced_log"] or observed!=e["post_recovery_match"] or
        len(matches)!=e["post_recovery_n"] or w.metrics(e["horizon"])!=e["metrics"] or
        w.state_sha()!=e["final_sha"] or [list(x) for x in sorted(w.ink)]!=e["ink"] or
        dict(sorted(Counter(x["chosen"] for x in w.decisions).items()))!=e["counts"]):
        raise ValueError("forged recovery or geometry")
    return len(w.decisions)

def run(output,seed=579,worlds=1,updates=40):
    path=Path(output)
    if path.exists():raise FileExistsError("refuse checkpoint overwrite")
    if not 0<=seed<100000 or worlds not in (1,2) or not 1<=updates<=200:
        raise ValueError("resource cap")
    path.mkdir(parents=True);start=time.perf_counter()
    models={a:h.Adapter(seed,False) for a in ARMS};logs=[]
    for stage,horizon in enumerate(HORIZONS):
        X,M,Y,pin,recoveries=corpus(seed,worlds,horizon)
        for arm in ARMS:
            result=models[arm].optimize(X[arm],M,Y,updates,seed+101*stage)
            logs.append({"arm":arm,"horizon":horizon,"rows":len(Y),
                         "corpus_sha":pin,"recoveries":recoveries,"loss":result})
    checks={}
    for arm,m in models.items():
        m.save(path/(arm+".npz"))
        checks[arm]={"sha256":sha_file(path/(arm+".npz")),"params":h.F,"updates":m.updates}
    episodes=[];turns=0
    for arm,m in models.items():
        for spec in specs("development",1):
            for style in STYLES:
                for horizon in (128,160):
                    for forced in (False,True):
                        e=episode(spec,style,arm,m,horizon,forced)
                        turns+=verify_episode(e,m)
                        name=f"{arm}-{spec.ident}-{style}-{horizon}-force{int(forced)}.json"
                        (path/name).write_text(json.dumps(e,sort_keys=True))
                        episodes.append(e)
    summary={}
    for arm in ARMS:
        for horizon in (128,160):
            for forced in (False,True):
                pack=[e for e in episodes if e["arm"]==arm and e["horizon"]==horizon and e["forced"]==forced]
                rates=[e["post_recovery_match"] for e in pack if e["post_recovery_match"] is not None]
                summary[f"{arm}/{horizon}/force{int(forced)}"]={
                    "episodes":len(pack),
                    "mean_ink":round(statistics.mean(e["metrics"]["unique_painted_cells"] for e in pack),4),
                    "post_recovery_teacher_match":round(statistics.mean(rates),4) if rates else None,
                    "early_stops":sum(e["metrics"]["stopped_early"] for e in pack),
                    "mean_latency_ms":round(statistics.mean(e["mean_latency_ms"] for e in pack),6),
                    "action_distribution":dict(sum((Counter(e["counts"]) for e in pack),Counter()))}
    source={**source_pins(),"history_motif_lab.py":sha_file(Path(h.__file__)),
            "live_path_probe.py":sha_file(Path(__file__))}
    report={"version":VERSION,"source_sha256":source,"seed":seed,"train_only":True,
            "sealed_heldout_unread":True,"native_executed":False,"native_limit_unchanged":64,
            "training":logs,"checkpoints":checks,"episodes":len(episodes),
            "verified_decisions":turns,"summary":summary,
            "elapsed_s":round(time.perf_counter()-start,3),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (path/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True))
    (path/"SHA256SUMS.json").write_text(json.dumps({p.name:sha_file(p) for p in path.iterdir()},indent=2,sort_keys=True))
    return report

def verify(path):
    path=Path(path);r=json.loads((path/"report.json").read_text())
    if r["version"]!=VERSION or not r["sealed_heldout_unread"] or r["native_executed"] or r["native_limit_unchanged"]!=64:
        raise ValueError("research boundary")
    hashes=json.loads((path/"SHA256SUMS.json").read_text())
    actual={p.name:sha_file(p) for p in path.iterdir() if p.name!="SHA256SUMS.json"}
    if hashes!=actual:raise ValueError("artifact checksum mismatch")
    expected={**source_pins(),"history_motif_lab.py":sha_file(Path(h.__file__)),
              "live_path_probe.py":sha_file(Path(__file__))}
    if r["source_sha256"]!=expected:raise ValueError("exact source mismatch")
    models={}
    for arm in ARMS:
        m=h.Adapter.load(path/(arm+".npz"))
        if sha_file(path/(arm+".npz"))!=r["checkpoints"][arm]["sha256"] or m.updates!=r["checkpoints"][arm]["updates"]:
            raise ValueError("checkpoint tampering")
        models[arm]=m
    count=turns=0
    for p in path.glob("*.json"):
        if p.name in ("report.json","SHA256SUMS.json"):continue
        e=json.loads(p.read_text());turns+=verify_episode(e,models[e["arm"]]);count+=1
    if count!=r["episodes"] or turns!=r["verified_decisions"]:raise ValueError("missing episode")
    return {"verified_episodes":count,"verified_decisions":turns,"files_sha256":len(actual)}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path)
    p.add_argument("--verify",type=Path);p.add_argument("--seed",type=int,default=579)
    p.add_argument("--worlds",type=int,default=1);p.add_argument("--updates",type=int,default=40)
    a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.output,a.seed,a.worlds,a.updates),sort_keys=True))
