"""DRIV3R TRAP-PREVENT-001: matched trainable proactive escape-risk sensor ablation.

Research only. Synthetic 96x96 host-legal materials, no native receipts, game edits,
external models, paid compute or structural HELDOUT opening. CPU-trained checkpoints
and traceable original trajectories go into artifact output, NEVER Git history.

Two linear-softmax action rankers use identical teacher labels, examples, optimizer
budgets and initialization. The control masks nine predictive geometric features;
the experimental arm sees bounded 1/2-turn local geometry, not an A* route oracle.
A teacher uses the same public host menu and a fixed local-frontier heuristic,
never rollout hindsight. The teacher is TRAIN only, never live supervisor in eval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import time

import numpy as np

from . import policy
from .train import _png, MAX_RECOVERIES, sha_file
from .world import MOVE, Spec, World, digest, replay, specs

VERSION = "driv3r-trap-prevention/1"
ARMS = ("baseline", "local_escape_sensor")
STAGES = (36, 72, 128)
EVAL_HORIZONS = (36, 72, 128, 160)
LOCAL_FEATURES = (
    "free_successor_degree", "two_turn_frontier", "forced_dead_end",
    "one_exit_only", "forward_corridor", "adjacent_material_next",
    "adjacent_ink_next", "recoveries_remaining", "frontier_best_gap",
)
DIM = len(policy.FEATURES) + len(LOCAL_FEATURES)
TRAIN_INTERVENTIONS = .22
FORCED_INDICES = frozenset((12, 35, 71))


def sensor(w: World, action: str) -> np.ndarray:
    """No hidden goals, no flood-fill, no A*; 1/2-ply host occupancy only."""
    if action not in w.menu(MAX_RECOVERIES):
        raise ValueError("Sensor called for non-host-legal action")
    out = np.zeros(len(LOCAL_FEATURES), dtype=np.float64)
    if action not in MOVE:
        out[7] = max(0, 12-w.recoveries)/12.
        return out
    dx,dy = MOVE[action]
    target = (w.cursor[0]+dx,w.cursor[1]+dy)
    if w.occupied(target):
        raise ValueError("Sensor called for illegal target")
    successors = []
    for sx,sy in MOVE.values():
        p = (target[0]+sx,target[1]+sy)
        if not w.occupied(p): successors.append(p)
    # Distinct available destinations in exactly two future move steps.
    two_step = set()
    for p in successors:
        for sx,sy in MOVE.values():
            p2 = (p[0]+sx,p[1]+sy)
            if p2 != target and not w.occupied(p2): two_step.add(p2)
    out[:4] = [len(successors)/4., min(len(two_step),12)/12.,
               float(len(successors)==0),float(len(successors)==1)]
    out[4] = sum(not w.occupied((target[0]+dx*i,target[1]+dy*i)) for i in (1,2,3))/3.
    nbrs = [(target[0]+sx,target[1]+sy) for sx,sy in MOVE.values()]
    out[5] = sum(p in w.materials or p in w.protected for p in nbrs)/4.
    out[6] = sum(p in w.ink for p in nbrs)/4.
    out[7] = max(0,12-w.recoveries)/12.
    peers = []
    for a,(ax,ay) in MOVE.items():
        near = (w.cursor[0]+ax,w.cursor[1]+ay)
        if a != action and not w.occupied(near):
            peers.append(sum(not w.occupied((near[0]+sx,near[1]+sy)) for sx,sy in MOVE.values()))
    out[8] = (len(successors)-max(peers,default=len(successors)))/4.
    return out


def matrix(w: World, horizon: int, arm: str) -> tuple[list[str],np.ndarray]:
    if arm not in ARMS:
        raise ValueError("Invalid arm")
    legal, base = policy.menu_matrix(w,horizon)
    X = np.zeros((policy.ACTION_MAX,DIM),np.float64)
    X[:,:len(policy.FEATURES)] = base
    if arm == "local_escape_sensor":
        for i,a in enumerate(legal): X[i,len(policy.FEATURES):] = sensor(w,a)
    return legal,X


def teacher_action(w: World,horizon:int,rng:random.Random) -> str:
    """Shared offline TRAIN labels: imperfect stylistic teacher + local trap risk.

    The tutor has no route-planner/global connectivity search, and never acts
    during evaluation. The test isolates access to the tutor's LOCAL risk input.
    """
    legal,base = policy.menu_matrix(w,horizon)
    scores = policy.teacher_scores(w,legal,base)
    viable = [a for a in legal if a in MOVE]
    for i,a in enumerate(legal):
        if a in MOVE:
            s = sensor(w,a)
            scores[i] += 2.1*s[0]+1.35*s[1]-2.7*s[2]-.45*s[3]+.35*s[4]
            scores[i] += .75*s[8]
        elif a == "BACKTRACK" and viable:
            # Backtracking with a free exit is possible but costly, deliberately not vetoed.
            scores[i] -= 1.2
    scores += np.array([rng.uniform(-.24,.24) for _ in legal])
    return legal[int(scores.argmax())]


class LinearRanker:
    def __init__(self,seed:int,arm:str):
        if arm not in ARMS: raise ValueError("Invalid model arm")
        self.seed,self.arm,self.updates = seed,arm,0
        rng=np.random.default_rng(seed)
        self.w = rng.normal(0,.05,DIM).astype(np.float64)
        self.m=np.zeros(DIM);self.v=np.zeros(DIM)

    def probabilities(self,X:np.ndarray, n:int) -> np.ndarray:
        z=X[:n] @ self.w
        z-=z.max();p=np.exp(z)
        return p/p.sum()

    def select(self,w:World,horizon:int):
        legal,X=matrix(w,horizon,self.arm)
        p=self.probabilities(X,len(legal))
        return legal[int(p.argmax())],dict(zip(legal,map(float,p)))

    def train(self,rows:list[tuple[World,int,str]],steps:int,stage_seed:int):
        # Source-created feature arrays use identical states and labels for each arm.
        xx=[];mm=[];yy=[]
        for w,h,label in rows:
            actions,X=matrix(w,h,self.arm)
            xx.append(X); mm.append([j<len(actions) for j in range(policy.ACTION_MAX)])
            yy.append(actions.index(label))
        X=np.stack(xx);M=np.asarray(mm,bool);Y=np.asarray(yy,int)
        rng=np.random.default_rng(stage_seed)
        first=last=None
        for _ in range(steps):
            ids=rng.integers(0,len(Y),size=64)
            z=X[ids]@self.w
            z=np.where(M[ids],z,-1e8)
            z-=z.max(axis=1,keepdims=True)
            p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
            loss=float(-np.log(np.maximum(p[np.arange(len(ids)),Y[ids]],1e-14)).mean())
            if first is None:first=loss
            last=loss
            p[np.arange(len(ids)),Y[ids]]-=1
            grad=np.einsum('baf,ba->f',X[ids],p)/len(ids)
            norm=float(np.linalg.norm(grad))
            grad*=min(1.,4./max(norm,1e-12))
            self.updates+=1
            self.m=.9*self.m+.1*grad
            self.v=.999*self.v+.001*grad*grad
            self.w-=.035*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        return {"updates":steps,"samples":len(Y),"first_loss":round(first,6),"last_loss":round(last,6)}

    def save(self,path:Path):
        np.savez_compressed(path,w=self.w,m=self.m,v=self.v,seed=self.seed,
                            updates=self.updates,arm=self.arm,version=VERSION)

    @classmethod
    def load(cls,path:Path):
        with np.load(path,allow_pickle=False) as d:
            if str(d['version'])!=VERSION: raise ValueError('checkpoint version')
            out=cls(int(d['seed']),str(d['arm']))
            for n in ('w','m','v'):
                v=d[n]
                if v.shape!=(DIM,) or not np.isfinite(v).all(): raise ValueError('invalid checkpoint')
                setattr(out,n,v.copy())
            out.updates=int(d['updates'])
        return out


def corpus(horizon:int,run_seed:int):
    rows=[]; counts=Counter(); interventions=0
    plans=specs('train',3)
    plans += [replace(s,challenge='trap_corridor') for s in plans if s.seed%2==0]
    for sp in plans:
        w=World.create(sp)
        rng=random.Random(run_seed*100011+sp.seed*397+horizon*17)
        for _ in range(horizon):
            if w.stopped: break
            choice=teacher_action(w,horizon,rng)
            # Snapshot the actual environment, never a hindsight label.
            import copy
            rows.append((copy.deepcopy(w),horizon,choice));counts[choice]+=1
            moves=[a for a in w.menu() if a in MOVE and a!=choice]
            action=rng.choice(moves) if moves and rng.random()<TRAIN_INTERVENTIONS else choice
            interventions+=int(action!=choice)
            w.apply(action,MAX_RECOVERIES)
    return rows,{"examples":len(rows),"teacher_actions":dict(counts),
                 "interventions":interventions,"world_count":len(plans),
                 "split":"train","horizon":horizon}


def forced_alternative(w:World,base_action:str,seed:int,index:int):
    choices=[a for a in sorted(w.menu()) if a in MOVE and a!=base_action]
    return choices[(seed+index*7)%len(choices)] if choices else None


def rollout(sp:Spec,model:LinearRanker,horizon:int,forced:bool,seed:int,
            output:Path|None=None):
    w=World.create(sp)
    actions=Counter(); near_traps=0; avoidable=0; forced_applied=0
    timings=[]; events=[]; loss_of_options=0
    for index in range(horizon):
        if w.stopped:break
        t=time.perf_counter_ns()
        selected,probs=model.select(w,horizon)
        lat=(time.perf_counter_ns()-t)/1e6
        alternative=forced_alternative(w,selected,seed,index) if forced and index in FORCED_INDICES else None
        chosen=alternative or selected
        ss=sensor(w,chosen) if chosen in MOVE else None
        move_options=[a for a in w.menu() if a in MOVE]
        if chosen in MOVE and ss[2] and len(move_options)>1:near_traps+=1
        if chosen in MOVE and ss[0]<.5 and any(sensor(w,a)[0]>ss[0] for a in move_options):avoidable+=1
        if chosen in MOVE and ss[0]<.25:loss_of_options+=1
        event=w.apply(chosen,MAX_RECOVERIES)
        events.append({"event":event,"probabilities":probs,"model_choice":selected,
                       "intervention":alternative is not None,
                       "local_risk":None if ss is None else [round(float(v),8) for v in ss]})
        timings.append(lat);actions[chosen]+=1
        forced_applied+=int(alternative is not None)
    raw=[x['event'] for x in events]
    rr=replay(sp,raw,MAX_RECOVERIES)
    assert rr.state_sha()==w.state_sha()
    result={"version":VERSION,"arm":model.arm,"checkpoint_updates":model.updates,
            "seed":seed,"spec":{"split":sp.split,"family":sp.family,"seed":sp.seed,
            "mode":sp.mode,"challenge":sp.challenge,"ident":sp.ident},
            "horizon":horizon,"forced":forced,"forced_applied":forced_applied,
            "base_sha256":World.create(sp).base_sha(),"metrics":w.metrics(horizon),
            "action_distribution":dict(sorted(actions.items())),
            "avoidable_low_exit_choices":avoidable,"forced_dead_end_choices":near_traps,
            "zero_fresh_successor_choices":loss_of_options,
            "mean_decision_latency_ms":round(statistics.mean(timings),6),
            "p95_decision_latency_ms":round(sorted(timings)[int(.95*(len(timings)-1))],6),
            "final_state_sha256":w.state_sha(),"trajectory_sha256":digest(raw),
            "native_receipts":0,"events":events}
    if output:
        output.mkdir(parents=True,exist_ok=True)
        prefix=f"{model.arm}-{horizon}-{'forced' if forced else 'intact'}-{sp.ident}"
        (output/(prefix+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
        _png(output/(prefix+'.png'),w)
    return result


def aggregate(items):
    return {"episodes":len(items),"mean_ink":round(statistics.mean(x['metrics']['unique_painted_cells'] for x in items),3),
            "early_stops":sum(x['metrics']['stopped_early'] for x in items),
            "mean_branch":round(statistics.mean(x['metrics']['branch_cells'] for x in items),3),
            "mean_recoveries":round(statistics.mean(x['metrics']['recoveries'] for x in items),3),
            "avoidable_low_exit_choices":sum(x['avoidable_low_exit_choices'] for x in items),
            "forced_dead_end_choices":sum(x['forced_dead_end_choices'] for x in items),
            "action_distribution":dict(sorted(sum((Counter(x['action_distribution']) for x in items),Counter()).items())),
            "mean_decision_latency_ms":round(statistics.mean(x['mean_decision_latency_ms'] for x in items),6)}


def run(seed:int,out:Path,updates:int=70,n_development_per_family:int=2):
    if updates<1 or updates>200 or n_development_per_family<1 or n_development_per_family>3:
        raise ValueError('bounded resource envelope')
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    models={arm:LinearRanker(seed,arm) for arm in ARMS}
    stages=[]
    for h in STAGES:
        rows,meta=corpus(h,seed)
        fits={arm:models[arm].train(rows,updates,seed*1009+h*17) for arm in ARMS}
        stages.append({"horizon":h,"corpus":meta,"fits":fits})
    cps={}
    for arm in ARMS:
        name=arm+'-seed'+str(seed)+'.npz'
        models[arm].save(out/name)
        restored=LinearRanker.load(out/name)
        if not np.array_equal(restored.w,models[arm].w): raise AssertionError('failed restore')
        cps[arm]={"name":name,"sha256":sha_file(out/name),"parameters":DIM,"optimizer_updates":models[arm].updates,
                   "weight_l2":round(float(np.linalg.norm(models[arm].w)),9)}
    trials=[]
    development=specs('development',n_development_per_family)
    development += [replace(s,challenge='trap_corridor') for s in development if s.seed%2==0]
    for h in EVAL_HORIZONS:
        for sp in development:
            for forced in (False,True):
                for arm in ARMS:
                    trials.append(rollout(sp,models[arm],h,forced,seed,out/'trajectories'))
    groups={}
    for h in EVAL_HORIZONS:
        for forced in (False,True):
            for arm in ARMS:
                key=f"{arm}-{h}-{'forced' if forced else 'intact'}"
                groups[key]=aggregate([x for x in trials if x['arm']==arm and x['horizon']==h and x['forced']==forced])
    src=Path(__file__)
    report={"version":VERSION,"seed":seed,"source_sha256":sha_file(src),
            "source_pins":{"world.py":sha_file(src.parent/'world.py'),"policy.py":sha_file(src.parent/'policy.py'),
                           "trap_avoidance_lab.py":sha_file(src)},
            "training":stages,"checkpoints":cps,"development_groups":groups,
            "episodes":len(trials),"sequential_choices":sum(x['metrics']['decisions'] for x in trials),
            "runtime_sec":round(time.perf_counter()-start,3),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "no_heldout_opened":True,"native_receipts":0,"native_ceiling_unchanged":64,
            "no_slm_weights_executed":True}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    manifests={str(x.relative_to(out)):sha_file(x) for x in sorted(out.rglob('*')) if x.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifests,sort_keys=True,indent=2)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=586);ap.add_argument('--updates',type=int,default=70)
    ap.add_argument('--dev-per-family',type=int,default=2)
    ns=ap.parse_args()
    r=run(ns.seed,Path(ns.out),ns.updates,ns.dev_per_family)
    print(json.dumps({k:r[k] for k in ('seed','checkpoints','episodes','sequential_choices','runtime_sec','peak_rss_kib','development_groups')},sort_keys=True))

if __name__=='__main__': main()
