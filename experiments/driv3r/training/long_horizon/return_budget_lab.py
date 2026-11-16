"""RETURN-BUDGET-001: competing host-legal recovery routes under finite capacity.

Matched CPU-trained action rankers compare capacity-blind vs capacity-aware
features on the *same* TRAIN teacher labels. There is no lookahead planner,
native execution, structural HELDOUT evaluation or production game change.
This experiment is a deliberately narrow information/control ablation, not a
claim of general long-horizon gameplay competence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time

import numpy as np

from . import policy
from .train import _png, sha_file
from .world import MOVE, World, digest, replay, specs

VERSION = "driv3r-return-budget/1"
ARMS = ("blind", "capacity_visible")
CAPS = (2, 8)
STAGES = (36, 72, 128)
HORIZONS = (128, 160)
DIM = len(policy.FEATURES) + 5
MAX_STEPS = 96
TRAIN_WORLDS = 8
DEV_WORLDS = 8


def matrix(w: World, horizon: int, cap: int, arm: str) -> tuple[list[str], np.ndarray]:
    """Only actions already authorized by the synthetic host are scored.

    The visible arm can compare recovery capacity with live path depth. This
    is an explicit host-observable planning *feature*, not a supplied answer.
    At the paired fork, menus and material state are identical across caps.
    """
    if arm not in ARMS or cap not in CAPS:
        raise ValueError("unknown arm or recovery cap")
    menu = w.menu(cap)
    X = np.zeros((policy.ACTION_MAX, DIM), dtype=np.float64)
    remaining = max(0, cap - w.recoveries)
    depth = len(w.path)-1
    for i, action in enumerate(menu):
        X[i, :len(policy.FEATURES)] = policy.features(w, action, horizon)
        if arm == "capacity_visible":
            X[i, -5:] = [
                float(action == "BACKTRACK")*min(1,remaining/max(1,depth)),
                float(action == "BACKTRACK")*float(remaining < depth),
                float(action == "RELOCATE_START")*float(remaining < depth),
                float(action == "RELOCATE_START")*float(remaining >= depth),
                float(action in MOVE)*remaining/12.,
            ]
    return menu, X


def correction_teacher(w: World, horizon: int, cap: int, rng: random.Random) -> str:
    """TRAIN-only teacher: different *strategy* for cheap vs deep return.

    The BACKTRACK alternative needs one recovery for every remaining path
    edge. Under inadequate budget choose RELOCATE_START while still legal.
    Neither preference is asserted to be optimal for artistic quality.
    """
    menu = w.menu(cap)
    if not any(a in MOVE for a in menu):
        remaining = cap-w.recoveries
        if "BACKTRACK" in menu and remaining >= len(w.path)-1:
            return "BACKTRACK"
        if "RELOCATE_START" in menu:
            return "RELOCATE_START"
        if "BACKTRACK" in menu:
            return "BACKTRACK"
        return "STOP"
    base = np.stack([policy.features(w,a,horizon) for a in menu])
    scores=policy.teacher_scores(w,menu,base)
    scores += np.asarray([rng.uniform(-.13,.13) for _ in menu])
    return menu[int(scores.argmax())]


def initial_corridor(w: World, cap: int) -> list[dict]:
    """Six *legal* externally forced MOVE_E mistakes, never imitation labels."""
    if w.spec.challenge != "trap_corridor" or w.spec.split == "heldout":
        raise ValueError("only permitted synthetic TRAIN/DEV corridor")
    out = []
    for _ in range(6):
        if "MOVE_E" not in w.menu(cap):
            raise ValueError("corridor forced action not host-legal")
        out.append(w.apply("MOVE_E",cap))
    if w.cursor != (54,48) or set(w.menu(cap)) != {"BACKTRACK","RELOCATE_START","STOP"}:
        raise ValueError("competing-return fixture no longer valid")
    return out


def corpus(seed: int, horizon: int):
    if horizon not in STAGES:
        raise ValueError("invalid curriculum")
    rows=[]; labels=Counter(); fingerprints=[]; forks={}
    for sp0 in specs("train",2):
        sp=replace(sp0,challenge="trap_corridor")
        for cap in CAPS:
            w=World.create(sp); initial_corridor(w,cap)
            fork=(w.state_sha(), tuple(w.menu(cap)))
            forks.setdefault(sp.ident, {})[cap]=fork
            rng=random.Random(seed*100003+sp.seed*17+horizon*11)
            for t in range(6,horizon):
                if w.stopped: break
                label=correction_teacher(w,horizon,cap,rng)
                menu=w.menu(cap)
                if label not in menu:
                    raise ValueError("teacher violates host menu")
                xs={arm:matrix(w,horizon,cap,arm)[1] for arm in ARMS}
                weight=18. if t==6 else (6. if label in ("BACKTRACK","RELOCATE_START") else 1.)
                rows.append((xs, np.arange(policy.ACTION_MAX)<len(menu),menu.index(label),weight))
                labels[label]+=1
                # A bounded source-created action disturbance, no policy teacher at DEV.
                actual=label
                if t in (31,67,103):
                    alternatives=[a for a in menu if a in MOVE and a!=label]
                    if alternatives: actual=alternatives[(seed+sp.seed+t)%len(alternatives)]
                w.apply(actual,cap)
            if replay(sp,w.decisions,cap).state_sha()!=w.state_sha():
                raise ValueError("training replay failure")
            fingerprints.append([sp.ident,cap,digest(w.decisions)])
    for both in forks.values():
        if both[2]!=both[8]:
            raise ValueError("counterfactual fork states or menus not identical")
    xs={arm:np.stack([x[0][arm] for x in rows]) for arm in ARMS}
    mask=np.stack([x[1] for x in rows]);Y=np.asarray([x[2] for x in rows],int)
    weights=np.asarray([x[3] for x in rows],float)
    return xs,mask,Y,weights,{"examples":len(Y),"source_digest":digest(fingerprints),
            "fork_pairs":len(forks),"teacher_actions":dict(sorted(labels.items())),
            "split":"train", "pairing":"identical fork state/menu, opposite desired route"}


class Ranker:
    def __init__(self, seed: int, arm: str):
        if arm not in ARMS: raise ValueError("unknown arm")
        self.seed=seed; self.arm=arm; self.updates=0
        self.w=np.random.default_rng(seed).normal(0,.045,DIM)
        self.m=np.zeros(DIM);self.v=np.zeros(DIM)

    def decide(self,w:World,horizon:int,cap:int):
        menu,X=matrix(w,horizon,cap,self.arm)
        scores=X[:len(menu)]@self.w;scores-=scores.max()
        ps=np.exp(scores);ps/=ps.sum()
        return menu[int(ps.argmax())],{a:float(p) for a,p in zip(menu,ps)}

    def fit(self,X,M,Y,W,steps,seed):
        rng=np.random.default_rng(seed)
        p=W/W.sum();losses=[];sample=Counter()
        for _ in range(steps):
            inds=rng.choice(len(Y),size=64,replace=True,p=p)
            x=X[inds];mask=M[inds];y=Y[inds]
            z=x@self.w;z=np.where(mask,z,-1e8);z-=z.max(axis=1,keepdims=True)
            q=np.exp(z)*mask;q/=q.sum(axis=1,keepdims=True)
            losses.append(float(-np.log(np.maximum(q[np.arange(64),y],1e-25)).mean()))
            isrec=(x[np.arange(64),y,1]+x[np.arange(64),y,2])>0
            sample["recovery"]+=int(isrec.sum()); sample["other"]+=int((~isrec).sum())
            q[np.arange(64),y]-=1
            g=np.einsum("baf,ba->f",x,q)/64
            g*=min(1.,4./max(1e-12,float(np.linalg.norm(g))))
            self.updates+=1
            self.m=.9*self.m+.1*g;self.v=.999*self.v+.001*g*g
            self.w-=.03*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        return {"updates":steps,"first_loss":round(losses[0],5),
                "last_10_loss":round(statistics.mean(losses[-10:]),5),
                "draws":dict(sorted(sample.items()))}

    def save(self,path:Path):
        np.savez_compressed(path,w=self.w,m=self.m,v=self.v,seed=self.seed,
                            arm=self.arm,version=VERSION,updates=self.updates)

    @classmethod
    def load(cls,path:Path):
        with np.load(path,allow_pickle=False) as f:
            if str(f["version"])!=VERSION:raise ValueError("checkpoint version")
            m=cls(int(f["seed"]),str(f["arm"]))
            for k in ("w","m","v"):
                arr=f[k]
                if arr.shape!=(DIM,) or not np.isfinite(arr).all():raise ValueError("bad model")
                setattr(m,k,arr.copy())
            m.updates=int(f["updates"])
        return m


def intervention(w:World,selected:str,t:int,seed:int,cap:int):
    if t not in (35,71,109):return selected,None
    options=[a for a in w.menu(cap) if a in MOVE and a!=selected]
    wanted=options[(seed+w.spec.seed+t)%len(options)] if options else selected
    return wanted,{"requested":wanted,"changed":wanted!=selected}


def metric(sp,cap,rows):
    first=rows[6]["event"]["chosen"] if len(rows)>6 else None
    target="RELOCATE_START" if cap==2 else "BACKTRACK"
    outside=[i for i,r in enumerate(rows) if i>=6 and r["event"]["chosen"] in MOVE and
             tuple(r["event"]["cursor_after"]) not in {(x,48) for x in range(49,55)}]
    return {"first_strategy":first,"desired_first_strategy":target,
            "fork_correct":first==target,"first_paint_outside_at":outside[0] if outside else None,
            "new_ink_outside_by_46":sum(i<46 for i in outside),
            "successful_resumption_8":sum(i<46 for i in outside)>=8,
            "paint_after_64":sum(i>=64 for i in outside),
            "premature_resource_exhaustion":first=="BACKTRACK" and cap==2 and not outside}


def rollout(sp,model:Ranker,cap:int,horizon:int):
    if sp.split!="development" or sp.challenge!="trap_corridor" or horizon not in HORIZONS:
        raise ValueError("sealed heldout or unsupported evaluation")
    w=World.create(sp);rows=[];lat=[]
    for ev in initial_corridor(w,cap):
        rows.append({"preferred":None,"probabilities":None,"intervention":"external_initial_error","event":ev})
    for t in range(6,horizon):
        if w.stopped:break
        now=time.perf_counter_ns();selected,ps=model.decide(w,horizon,cap)
        lat.append((time.perf_counter_ns()-now)/1e6)
        actual,inject=intervention(w,selected,t,model.seed,cap)
        ev=w.apply(actual,cap)
        rows.append({"preferred":selected,"probabilities":{a:round(p,8) for a,p in ps.items()},
                     "intervention":inject,"event":ev})
    return {"version":VERSION,"world":sp.ident,"base_sha":w.base_sha(),
            "split":sp.split,"challenge":sp.challenge,"cap":cap,"horizon":horizon,
            "model_seed":model.seed,"arm":model.arm,"rows":rows,
            "metrics":w.metrics(horizon),"return_metrics":metric(sp,cap,rows),
            "actions":dict(sorted(Counter(r["event"]["chosen"] for r in rows).items())),
            "trace_sha256":digest([r["event"] for r in rows]),"final_sha":w.state_sha(),
            "ink":[list(p) for p in sorted(w.ink)],"mean_decision_ms":round(statistics.mean(lat),6),
            "native_receipts":0}


def verify_episode(ep,model:Ranker,sp):
    if (ep["version"]!=VERSION or ep["world"]!=sp.ident or ep["split"]!="development"
        or ep["arm"]!=model.arm or ep["model_seed"]!=model.seed or ep["native_receipts"]!=0):
        raise ValueError("trajectory identity")
    cap=ep["cap"];w=World.create(sp)
    if cap not in CAPS or ep["horizon"] not in HORIZONS or ep["base_sha"]!=w.base_sha():
        raise ValueError("world base/budget")
    pref=initial_corridor(w,cap)
    if [r["event"] for r in ep["rows"][:6]]!=pref:
        raise ValueError("forged forced error")
    for t,row in enumerate(ep["rows"][6:],start=6):
        if w.stopped:raise ValueError("post STOP actions")
        selected,ps=model.decide(w,ep["horizon"],cap)
        if (row["preferred"]!=selected or set(ps)!=set(row["probabilities"])
            or any(abs(v-row["probabilities"][a])>2e-8 for a,v in ps.items())):
            raise ValueError("checkpoint action distribution disagreement")
        action,inject=intervention(w,selected,t,model.seed,cap)
        if inject!=row["intervention"] or w.apply(action,cap)!=row["event"]:
            raise ValueError("host action/state mismatch")
    if (w.state_sha()!=ep["final_sha"] or ep["trace_sha256"]!=digest([r["event"] for r in ep["rows"]])
        or ep["ink"]!=[list(p) for p in sorted(w.ink)]
        or ep["return_metrics"]!=metric(sp,cap,ep["rows"])
        or ep["actions"]!=dict(sorted(Counter(r["event"]["chosen"] for r in ep["rows"]).items()))
        or ep["metrics"]!=w.metrics(ep["horizon"])):
        raise ValueError("geometry or result mismatch")
    return len(ep["rows"])


def aggregate(episodes):
    if not episodes:raise ValueError("empty aggregation")
    return {"episodes":len(episodes),"mean_ink":round(statistics.mean(x["metrics"]["unique_painted_cells"] for x in episodes),3),
            "early_stop":sum(x["metrics"]["stopped_early"] for x in episodes),
            "fork_correct":sum(x["return_metrics"]["fork_correct"] for x in episodes),
            "resumption_8":sum(x["return_metrics"]["successful_resumption_8"] for x in episodes),
            "mean_outside_by_46":round(statistics.mean(x["return_metrics"]["new_ink_outside_by_46"] for x in episodes),3),
            "mean_after_64":round(statistics.mean(x["return_metrics"]["paint_after_64"] for x in episodes),3),
            "mean_latency_ms":round(statistics.mean(x["mean_decision_ms"] for x in episodes),6)}


def dep_hashes():
    base=Path(__file__).parent
    return {name:sha_file(base/name) for name in ("return_budget_lab.py","world.py","policy.py","train.py")}


def run(out:Path,seed:int=594,updates:int=50):
    if out.exists(): raise FileExistsError("immutable artifact location")
    if not 1<=updates<=90 or not 0<=seed<100000:raise ValueError("optimizer budget")
    out.mkdir(parents=True)
    start=time.perf_counter(); models={a:Ranker(seed,a) for a in ARMS};curr=[]
    for h in STAGES:
        X,M,Y,W,meta=corpus(seed,h)
        results={a:m.fit(X[a],M,Y,W,updates,seed*10007+h*13) for a,m in models.items()}
        curr.append({"horizon":h,"corpus":meta,"training":results})
    checkpoints={}
    for arm,m in models.items():
        file=f"{arm}-seed{seed}.npz"; m.save(out/file)
        r=Ranker.load(out/file)
        if not np.array_equal(r.w,m.w) or r.updates!=m.updates:raise ValueError("checkpoint restore")
        checkpoints[arm]={"file":file,"sha256":sha_file(out/file),"updates":m.updates,"parameters":DIM}
    (out/"trajectories").mkdir();episodes=[];total=0
    dev=[replace(sp,challenge="trap_corridor") for sp in specs("development",2)]
    for sp in dev:
        for cap in CAPS:
            for h in HORIZONS:
                for arm,m in models.items():
                    e=rollout(sp,m,cap,h)
                    total+=verify_episode(e,m,sp)
                    fn=f"{arm}-cap{cap}-h{h}-{sp.ident}"
                    (out/"trajectories"/(fn+".json")).write_text(json.dumps(e,sort_keys=True)+"\n")
                    if h==160 and sp==dev[0]:
                        _png(out/"trajectories"/(fn+".png"),replay(sp,[r["event"] for r in e["rows"]],cap))
                    episodes.append(e)
    table={}
    for h in HORIZONS:
        for cap in CAPS:
            for arm in ARMS:
                table[f"{arm}-cap{cap}-h{h}"]=aggregate([e for e in episodes if (e["arm"],e["cap"],e["horizon"])==(arm,cap,h)])
    result={"version":VERSION,"seed":seed,"source_sha256":dep_hashes(),
            "training":curr,"checkpoints":checkpoints,"table":table,"episodes":len(episodes),
            "decisions":total,"elapsed_s":round(time.perf_counter()-start,3),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "heldout_uninspected":True,"native_receipts":0,"paid_compute":False,
            "native_64_limit_unchanged":True}
    (out/"report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob("*") if p.is_file()}
    (out/"SHA256SUMS.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    return result


def verify(out:Path):
    result=json.loads((out/"report.json").read_text())
    if (result["version"]!=VERSION or result["source_sha256"]!=dep_hashes()
        or not result["heldout_uninspected"] or result["native_receipts"]!=0):raise ValueError("source provenance")
    stored=json.loads((out/"SHA256SUMS.json").read_text())
    actual={str(p.relative_to(out)):sha_file(p) for p in out.rglob("*") if p.is_file() and p.name!="SHA256SUMS.json"}
    if stored!=actual:raise ValueError("SHA256 manifest")
    models={a:Ranker.load(out/c["file"]) for a,c in result["checkpoints"].items()}
    if set(models)!=set(ARMS):raise ValueError("model arms")
    for arm,model in models.items():
        c=result["checkpoints"][arm]
        if (model.seed!=result["seed"] or model.arm!=arm or model.updates!=c["updates"]
            or c["parameters"]!=DIM or sha_file(out/c["file"])!=c["sha256"]):raise ValueError("checkpoint identity")
    dev={s.ident:s for s in [replace(sp,challenge="trap_corridor") for sp in specs("development",2)]}
    episodes=[];decisions=0
    for p in (out/"trajectories").glob("*.json"):
        e=json.loads(p.read_text())
        if e["world"] not in dev: raise ValueError("unrecognized development world")
        decisions+=verify_episode(e,models[e["arm"]],dev[e["world"]]);episodes.append(e)
    if (len(episodes),decisions)!=(result["episodes"],result["decisions"]):raise ValueError("episode counts")
    for key,expected in result["table"].items():
        arm,cap,h=key.rsplit("-",2)
        obs=aggregate([e for e in episodes if (e["arm"],e["cap"],e["horizon"])==(arm,int(cap[3:]),int(h[1:]))])
        for v in (obs,expected):v.pop("mean_latency_ms",None)
        if obs!=expected:raise ValueError("aggregate disagreement")
    return {"episodes_replayed":len(episodes),"decisions_replayed":decisions,
            "hashes_verified":len(stored),"checkpoints_restored":len(models)}


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);p.add_argument("--verify",type=Path)
    p.add_argument("--seed",type=int,default=594);p.add_argument("--updates",type=int,default=50)
    args=p.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.updates),sort_keys=True))
