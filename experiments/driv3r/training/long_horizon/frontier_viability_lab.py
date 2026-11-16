"""FRONTIER-VIABILITY-001: matched 36/72/128 synthetic worm policy ablation.

Research-only on host-authoritative World states, not the native game. No paid
models, no A* routes, no off-menu actions, no structural HELDOUT evaluation.

Question: does a bounded connected-frontier estimate improve continuation after
bad moves over the same local escape sensor, given identical teacher-labelled
TRAIN states, initialization and optimizer budgets? A third blind control is
included; all policies use a 39-parameter linear softmax action ranker.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np

from . import policy, trap_avoidance_lab as trap
from .train import _png, MAX_RECOVERIES, sha_file
from .world import GRID, MOVE, Spec, World, digest, replay, specs

VERSION = "driv3r-frontier-viability/2"
ARMS = ("blind", "local", "frontier")
HORIZONS = (36,72,128)
EVAL_HORIZONS = (128,160)
DEVIATIONS = (12,35,71,109)
FUTURE_CAP = 96
FUTURE_RADIUS = 12
FRONTIER_FEATURES = (
    "reachable_free_area_96", "reachable_ring_8", "max_reachable_radius_12",
    "pocket_under_12", "multiple_quadrants", "free_frontier_boundary",
)
BASE = len(policy.FEATURES)
LOCAL = len(trap.LOCAL_FEATURES)
DIM = BASE+LOCAL+len(FRONTIER_FEATURES)
assert DIM == 39


def frontier_features(world:World, action:str)->np.ndarray:
    """No route assignment: bounded flood-fill of currently empty host cells.

    Treat chosen target as newly painted/blocked, then start from its legal
    successor cells. Never walk through solids, protected regions or own ink.
    The 96-node/12-radius cap is intentionally a *local topological sensor*,
    not a proof of 128-step reachability or a substitute for host permissions.
    """
    if action not in world.menu(MAX_RECOVERIES):
        raise ValueError("non-legal frontier feature")
    out=np.zeros(len(FRONTIER_FEATURES),dtype=np.float64)
    if action not in MOVE:return out
    dx,dy=MOVE[action]
    target=(world.cursor[0]+dx,world.cursor[1]+dy)
    if world.occupied(target): raise ValueError("target is occupied")
    queue=deque()
    seen={target}
    for sx,sy in MOVE.values():
        p=(target[0]+sx,target[1]+sy)
        if p not in seen and not world.occupied(p):
            seen.add(p)
            queue.append(p)
    cells=[]; radii=[]; quadrants=set();boundary=0
    while queue and len(cells)<FUTURE_CAP:
        pos=queue.popleft()
        dist=abs(pos[0]-target[0])+abs(pos[1]-target[1])
        if dist>FUTURE_RADIUS:continue
        cells.append(pos);radii.append(dist)
        if dist>=8:
            quadrants.add((int(pos[0]>target[0]),int(pos[1]>target[1])))
        for sx,sy in MOVE.values():
            nbr=(pos[0]+sx,pos[1]+sy)
            nd=abs(nbr[0]-target[0])+abs(nbr[1]-target[1])
            if nd==FUTURE_RADIUS+1:
                if not world.occupied(nbr):boundary+=1
            elif nd<=FUTURE_RADIUS and nbr not in seen and not world.occupied(nbr):
                seen.add(nbr);queue.append(nbr)
    n=len(cells)
    out[:]=[n/FUTURE_CAP,
            min(sum(d>=8 for d in radii),48)/48.,
            max(radii,default=0)/FUTURE_RADIUS,
            float(n<12),
            len(quadrants)/4.,
            min(boundary,24)/24.]
    return out


def features(world:World,horizon:int,arm:str)->tuple[list[str],np.ndarray]:
    if arm not in ARMS:raise ValueError("unknown arm")
    legal,base=policy.menu_matrix(world,horizon)
    X=np.zeros((policy.ACTION_MAX,DIM),np.float64)
    X[:,:BASE]=base
    if arm in ("local","frontier"):
        for i,a in enumerate(legal):
            X[i,BASE:BASE+LOCAL]=trap.sensor(world,a)
            if arm=="frontier":X[i,BASE+LOCAL:]=frontier_features(world,a)
    return legal,X


def teacher(world:World,horizon:int,rng:random.Random)->str:
    """Teacher mixture: stylistic preference plus *state-only* frontier safety.

    A soft risk mixture, not a planner, and never active at evaluation time.
    It selects one of the exact current host-legal actions. No lookahead into
    future policy actions, no hidden privileged map, no teacher on DEV worlds.
    """
    menu,X=policy.menu_matrix(world,horizon)
    scores=policy.teacher_scores(world,menu,X)
    for i,a in enumerate(menu):
        if a in MOVE:
            local=trap.sensor(world,a)
            f=frontier_features(world,a)
            scores[i]+=1.1*local[0]+.8*local[1]-1.0*local[3]
            scores[i]+=3.4*f[0]+.8*f[1]-.5*f[2]-4.2*f[3]+.25*f[4]
        elif a=="BACKTRACK" and any(z in MOVE for z in menu):
            scores[i]-=.8
    scores +=np.asarray([rng.uniform(-.22,.22) for _ in menu])
    return menu[int(scores.argmax())]


def training_corpus(seed:int,horizon:int,train_worlds:int=2):
    if horizon not in HORIZONS or not 1<=train_worlds<=3:
        raise ValueError("invalid training curriculum")
    states=[];masks=[];labels=[];ids=[];interventions=0;counts=Counter()
    source=specs("train",train_worlds)
    source +=[replace(s,challenge="trap_corridor") for s in source if s.seed%2==0]
    for sp in source:
        w=World.create(sp)
        rng=random.Random(seed*100009+sp.seed*30011+horizon*101)
        for t in range(horizon):
            if w.stopped:break
            correct=teacher(w,horizon,rng)
            menu=w.menu(MAX_RECOVERIES)
            row={};
            for arm in ARMS:
                offered,X=features(w,horizon,arm)
                if offered!=menu:raise ValueError("host menu altered")
                row[arm]=X
            states.append(row)
            mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
            masks.append(mask);labels.append(menu.index(correct));counts[correct]+=1
            other=[a for a in menu if a in MOVE and a!=correct]
            actual=rng.choice(other) if other and rng.random()<.24 else correct
            interventions+=int(actual!=correct)
            w.apply(actual,MAX_RECOVERIES)
        if replay(sp,w.decisions,MAX_RECOVERIES).state_sha()!=w.state_sha():
            raise ValueError("training world replay failure")
        ids.append([sp.ident,sp.challenge,digest(w.decisions)])
    xs={arm:np.stack([r[arm] for r in states]) for arm in ARMS}
    return xs,np.stack(masks),np.asarray(labels,int),{
        "worlds":len(source),"states":len(labels),"teacher_actions":dict(sorted(counts.items())),
        "offpolicy_movements":interventions,"source_digest":digest(ids),"split":"train"}


class Ranker:
    def __init__(self,seed:int,arm:str):
        if arm not in ARMS:raise ValueError("unknown arm")
        self.seed=seed;self.arm=arm;self.updates=0
        self.w=np.random.default_rng(seed).normal(0,.04,DIM)
        self.m=np.zeros(DIM);self.v=np.zeros(DIM)

    def probs(self,X:np.ndarray,n:int)->np.ndarray:
        z=X[:n]@self.w;z-=z.max()
        p=np.exp(z);return p/p.sum()

    def decide(self,world:World,horizon:int):
        menu,X=features(world,horizon,self.arm)
        p=self.probs(X,len(menu))
        return menu[int(p.argmax())],{a:float(v) for a,v in zip(menu,p)}

    def optimize(self,X,mask,Y,steps,seed):
        rng=np.random.default_rng(seed);losses=[]
        for _ in range(steps):
            inds=rng.integers(0,len(Y),size=64)
            xb=X[inds];mb=mask[inds];yb=Y[inds]
            z=xb@self.w;z=np.where(mb,z,-1e8);z-=z.max(axis=1,keepdims=True)
            p=np.exp(z)*mb;p/=p.sum(axis=1,keepdims=True)
            losses.append(float(-np.log(np.maximum(1e-20,p[np.arange(len(yb)),yb])).mean()))
            p[np.arange(len(yb)),yb]-=1
            grad=np.einsum('baf,ba->f',xb,p)/len(yb)
            grad *= min(1.,4./max(1e-12,float(np.linalg.norm(grad))))
            self.updates+=1
            self.m=.9*self.m+.1*grad;self.v=.999*self.v+.001*grad*grad
            self.w-=.03*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        return {"steps":steps,"first_loss":round(losses[0],6),
                "last_mean_10":round(statistics.mean(losses[-10:]),6)}

    def save(self,path:Path):
        np.savez_compressed(path,w=self.w,m=self.m,v=self.v,seed=self.seed,
                            arm=self.arm,version=VERSION,updates=self.updates)

    @classmethod
    def load(cls,path:Path):
        with np.load(path,allow_pickle=False) as f:
            if str(f['version'])!=VERSION:raise ValueError('model version mismatch')
            m=cls(int(f['seed']),str(f['arm']))
            for key in ('w','m','v'):
                array=f[key]
                if array.shape!=(DIM,) or not np.isfinite(array).all():
                    raise ValueError('invalid tensor shape')
                setattr(m,key,array.copy())
            m.updates=int(f['updates'])
        return m


def deviation(world:World,selected:str,index:int,seed:int,enabled:bool)->tuple[str,dict|None]:
    if not enabled:return selected,None
    menu=world.menu(MAX_RECOVERIES)
    # Deliberately induce the eastward cul-de-sac from its starting position,
    # before allowing a controller to attempt recovery using legal host moves.
    if world.spec.challenge=='trap_corridor' and index in (0,1,2,3,4):
        wanted='MOVE_E'
    elif index in DEVIATIONS:
        candidates=[a for a in menu if a in MOVE and a!=selected]
        wanted=candidates[(seed+world.spec.seed+index)%len(candidates)] if candidates else selected
    else:return selected,None
    actual=wanted if wanted in menu else selected
    return actual,{"requested":wanted,"executed":actual,"changed":actual!=selected}


def rollout(sp:Spec,m:Ranker,horizon:int,forced:bool):
    if sp.split!='development' or horizon not in EVAL_HORIZONS:
        raise ValueError('unsealed DEV only')
    w=World.create(sp);logs=[];times=[];chosen_risks=[];forced_count=0;avoid=0
    for index in range(horizon):
        if w.stopped:break
        t0=time.perf_counter_ns();selected,probs=m.decide(w,horizon)
        times.append((time.perf_counter_ns()-t0)/1e6)
        actual,injection=deviation(w,selected,index,m.seed,forced)
        if injection and injection['changed']:forced_count+=1
        risk=None
        if actual in MOVE:
            f=frontier_features(w,actual);risk=round(float(f[0]),8)
            others=[a for a in w.menu(MAX_RECOVERIES) if a in MOVE]
            avoid+=int(f[0]<.125 and any(frontier_features(w,a)[0]>=.5 for a in others))
            chosen_risks.append(risk)
        event=w.apply(actual,MAX_RECOVERIES)
        logs.append({"event":event,"preferred":selected,"probabilities":{a:round(p,8) for a,p in probs.items()},
                     "intervention":injection,"frontier_free_frac":risk})
    # verify_episode will independently replay this exact transcript against the host.
    return {"version":VERSION,"arm":m.arm,"model_seed":m.seed,"world_id":sp.ident,
            "split":sp.split,"challenge":sp.challenge,"base_sha":w.base_sha(),
            "horizon":horizon,"forced":forced,"events":logs,
            "metrics":w.metrics(horizon),"final_sha":w.state_sha(),
            "trajectory_sha":digest([x['event'] for x in logs]),
            "ink":[list(pos) for pos in sorted(w.ink)],
            "action_distribution":dict(sorted(Counter(x['event']['chosen'] for x in logs).items())),
            "avoidably_small_pocket_entries":avoid,"forced_changes":forced_count,
            "mean_selected_frontier":round(statistics.mean(chosen_risks),6) if chosen_risks else 0.,
            "mean_decision_latency_ms":round(statistics.mean(times),6),"native_receipts":0}


def verify_episode(e:dict,m:Ranker,sp:Spec)->int:
    if e['version']!=VERSION or e['arm']!=m.arm or e['model_seed']!=m.seed or e['split']!='development' or e['native_receipts']!=0 or e['world_id']!=sp.ident:
        raise ValueError('episode provenance')
    w=World.create(sp)
    if e['base_sha']!=w.base_sha() or e['trajectory_sha']!=digest([z['event'] for z in e['events']]):
        raise ValueError('world identity or trace digest')
    if e['horizon'] not in EVAL_HORIZONS or len(e['events'])>e['horizon']:
        raise ValueError('episode horizon')
    risks=[]; avoid=0;forced=0;lat=[]
    for i,row in enumerate(e['events']):
        choice,probs=m.decide(w,e['horizon'])
        if row['preferred']!=choice or set(row['probabilities'])!=set(probs):
            raise ValueError('model policy mismatch')
        for a,val in probs.items():
            if abs(row['probabilities'][a]-val)>2e-8:
                raise ValueError('checkpoint logits changed')
        actual,inj=deviation(w,choice,i,m.seed,e['forced'])
        if inj!=row['intervention']:raise ValueError('forced deviation not reproducible')
        forced+=int(inj is not None and inj['changed'])
        risk=None
        if actual in MOVE:
            f=frontier_features(w,actual);risk=round(float(f[0]),8)
            options=[a for a in w.menu(MAX_RECOVERIES) if a in MOVE]
            avoid+=int(f[0]<.125 and any(frontier_features(w,a)[0]>=.5 for a in options))
            risks.append(risk)
        if risk!=row['frontier_free_frac']:raise ValueError('risk evidence falsified')
        if w.apply(actual,MAX_RECOVERIES)!=row['event']:
            raise ValueError('illegal host action / state mutation')
    metrics=w.metrics(e['horizon'])
    if (metrics!=e['metrics'] or w.state_sha()!=e['final_sha'] or
        [list(x) for x in sorted(w.ink)]!=e['ink'] or
        dict(sorted(Counter(z['chosen'] for z in w.decisions).items()))!=e['action_distribution'] or
        forced!=e['forced_changes'] or avoid!=e['avoidably_small_pocket_entries'] or
        abs((statistics.mean(risks) if risks else 0.)-e['mean_selected_frontier'])>1e-6):
        raise ValueError('falsified geometry/recovery statistics')
    return len(e['events'])


def aggregate(episodes):
    return {"episodes":len(episodes),
            "mean_ink":round(statistics.mean(x['metrics']['unique_painted_cells'] for x in episodes),3),
            "early_stops":sum(x['metrics']['stopped_early'] for x in episodes),
            "mean_branches":round(statistics.mean(x['metrics']['branch_cells'] for x in episodes),3),
            "avoidable_pocket_entries":sum(x['avoidably_small_pocket_entries'] for x in episodes),
            "forced_changes":sum(x['forced_changes'] for x in episodes),
            "mean_frontier":round(statistics.mean(x['mean_selected_frontier'] for x in episodes),5),
            "mean_latency_ms":round(statistics.mean(x['mean_decision_latency_ms'] for x in episodes),6),
            "action_distribution":dict(sorted(sum((Counter(x['action_distribution']) for x in episodes),Counter()).items()))}


def source_hashes():
    root=Path(__file__).parent
    return {n:sha_file(root/n) for n in ('world.py','policy.py','train.py','trap_avoidance_lab.py','frontier_viability_lab.py')}


def run(out:Path,seed:int=588,updates:int=60,train_worlds:int=2,dev_worlds:int=1):
    if not (0<=seed<100000 and 1<=updates<=150 and 1<=train_worlds<=3 and 1<=dev_worlds<=2):
        raise ValueError('budget cap')
    if out.exists():raise FileExistsError('immutable experiment output')
    out.mkdir(parents=True)
    started=time.perf_counter()
    models={a:Ranker(seed,a) for a in ARMS};stages=[]
    for horizon in HORIZONS:
        xs,mask,Y,meta=training_corpus(seed,horizon,train_worlds)
        fits={a:models[a].optimize(xs[a],mask,Y,updates,seed+horizon*103) for a in ARMS}
        stages.append({"horizon":horizon,"corpus":meta,"fits":fits})
    cps={}
    for arm,m in models.items():
        file=arm+'-seed'+str(seed)+'.npz';m.save(out/file)
        restored=Ranker.load(out/file)
        if not np.array_equal(restored.w,m.w):raise ValueError('failed checkpoint restore')
        cps[arm]={"name":file,"sha256":sha_file(out/file),"parameters":DIM,"updates":m.updates}
    trials=[];decisions=0
    scenes=specs('development',dev_worlds)[:2]  # bounded paired study: pickets, ridges
    scenes +=[replace(scenes[0],challenge='trap_corridor')]
    (out/'trajectories').mkdir()
    for sp in scenes:
        for horizon in EVAL_HORIZONS:
            for forced in (False,True):
                for arm,m in models.items():
                    e=rollout(sp,m,horizon,forced)
                    decisions+=verify_episode(e,m,sp)
                    name=f"{arm}-{horizon}-f{int(forced)}-{sp.ident}"
                    (out/'trajectories'/(name+'.json')).write_text(json.dumps(e,sort_keys=True)+'\n')
                    if horizon==160 and (sp==scenes[0] or sp.challenge=='trap_corridor') and forced:
                        _png(out/'trajectories'/(name+'.png'),replay(sp,[z['event'] for z in e['events']],MAX_RECOVERIES))
                    trials.append(e)
    summary={}
    for h in EVAL_HORIZONS:
        for forced in (False,True):
            for arm in ARMS:
                pack=[x for x in trials if (x['horizon'],x['forced'],x['arm'])==(h,forced,arm)]
                summary[f"{arm}-{h}-f{int(forced)}"]=aggregate(pack)
    report={"version":VERSION,"seed":seed,"source_sha256":source_hashes(),
            "train_stages":stages,"checkpoints":cps,"development_summary":summary,
            "episodes":len(trials),"verified_decisions":decisions,
            "elapsed_s":round(time.perf_counter()-started,3),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "heldout_sealed":True,"native_receipts":0,"native_move_ceiling":64,"paid_compute":False}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    sha={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(sha,indent=2,sort_keys=True)+'\n')
    return report


def verify(out:Path):
    report=json.loads((out/'report.json').read_text())
    if (report['version']!=VERSION or report['source_sha256']!=source_hashes() or
        not report['heldout_sealed'] or report['native_receipts'] or report['native_move_ceiling']!=64):
        raise ValueError('report source/boundary mismatch')
    sha=json.loads((out/'SHA256SUMS.json').read_text())
    current={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if current!=sha:raise ValueError('manifest tampered')
    models={}
    for arm,c in report['checkpoints'].items():
        m=Ranker.load(out/c['name'])
        if m.arm!=arm or m.updates!=c['updates'] or sha_file(out/c['name'])!=c['sha256']:
            raise ValueError('checkpoint identity')
        models[arm]=m
    scenes=specs('development',2)
    scenes +=[replace(s,challenge='trap_corridor') for s in scenes if s.seed%2==0]
    world_lookup={s.ident:s for s in scenes}
    count=steps=0;episodes=[]
    for path in (out/'trajectories').glob('*.json'):
        e=json.loads(path.read_text());sp=world_lookup.get(e['world_id'])
        if sp is None:raise ValueError('unknown world id')
        steps+=verify_episode(e,models[e['arm']],sp)
        count+=1;episodes.append(e)
    if (count,steps)!=(report['episodes'],report['verified_decisions']):
        raise ValueError('missing replay data')
    for key,s in report['development_summary'].items():
        arm,h,f=key.split('-');h=int(h);forced=bool(int(f[1:]))
        actual=aggregate([x for x in episodes if (x['arm'],x['horizon'],x['forced'])==(arm,h,forced)])
        # latency is deliberately hardware/run dependent, not replayable.
        a={k:v for k,v in actual.items() if k!='mean_latency_ms'}
        b={k:v for k,v in s.items() if k!='mean_latency_ms'}
        if a!=b:raise ValueError('summary mismatch')
    return {"episodes_verified":count,"decisions_verified":steps,"files_sha256":len(sha),"checkpoints":len(models)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=588);p.add_argument('--updates',type=int,default=60)
    p.add_argument('--train-worlds',type=int,default=2);p.add_argument('--dev-worlds',type=int,default=1)
    args=p.parse_args()
    result=verify(args.verify) if args.verify else run(args.out,args.seed,args.updates,args.train_worlds,args.dev_worlds)
    print(json.dumps(result if args.verify else {
        k:result[k] for k in ('seed','checkpoints','episodes','verified_decisions','elapsed_s','peak_rss_kib','development_summary')},sort_keys=True))

if __name__=='__main__':main()
