"""RETURN-GEN-001: variable-depth budget recovery, exact host-menu synthetic training.

Development families/directions/depths are distinct from training configurations.
No native execution, sealed HELDOUT inspection, production changes, or teacher
supervisor during evaluation. Engineered comparator intentionally discloses
teacher's local capacity inequality; it is a diagnostic control, not planning.
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
from .world import MOVE, World, digest, specs

VERSION = 'return-generalization/1'
ARMS = ('blind', 'budget_only', 'raw_counters', 'engineered')
STAGES = (36,72,128)
HORIZONS = (128,160)
TRAIN_DEPTHS = (5,7,9)
DEV_DEPTHS = (6,8,10)
TRAIN_CAPS = (2,6,10,12)
DEV_CAPS = (2,5,9,12)
EVAL_CAP_PAIRS = {6:(5,9),8:(5,9),10:(9,12)}
TRAIN_DIRECTIONS = ('E','N')
DEV_DIRECTIONS = ('W','S')
DIRECTIONS = {a[-1]: v for a,v in MOVE.items()}
DIM = len(policy.FEATURES)+6


def fixture(spec, depth:int, direction:str):
    """Overlay a reproducible synthetic cul-de-sac on the unmodified host World.

    Uses the actual World.menu/apply legality and one-cell material invariants.
    Overlay hashes are incorporated into World.base_sha/state_sha.
    """
    if spec.split not in ('train','development'):
        raise ValueError('sealed/unsupported split')
    allowed = (TRAIN_DEPTHS,TRAIN_DIRECTIONS) if spec.split=='train' else (DEV_DEPTHS,DEV_DIRECTIONS)
    if depth not in allowed[0] or direction not in allowed[1]:
        raise ValueError('configuration partition breach')
    w = World.create(spec)
    dx,dy = DIRECTIONS[direction]
    sx,sy = -dy,dx
    start = (48,48)
    for k in range(1,depth+1):
        p=(48+k*dx,48+k*dy)
        w.materials.pop(p,None); w.protected.discard(p)
        for sign in (-1,1):
            side=(p[0]+sign*sx,p[1]+sign*sy)
            w.materials[side]=37; w.protected.discard(side)
    front=(48+(depth+1)*dx,48+(depth+1)*dy)
    w.materials[front]=1; w.protected.add(front)
    for p in [start,(48-dx,48-dy),(48+sx,48+sy),(48-sx,48-sy)]:
        w.materials.pop(p,None);w.protected.discard(p)
    # Duplicated source seed/overlay is sufficient to reconstruct any replay.
    return w


def force_error(w:World, depth:int, direction:str, cap:int):
    events=[]; move='MOVE_'+direction
    for k in range(depth):
        if move not in w.menu(cap):raise ValueError('forced corridor movement illegal')
        events.append(w.apply(move,cap))
    if set(w.menu(cap)) != {'BACKTRACK','RELOCATE_START','STOP'}:
        raise ValueError('corridor endpoint is not the defined competing fork')
    return events


def matrix(w:World,horizon:int,cap:int,arm:str):
    if arm not in ARMS or cap<2 or cap>12:raise ValueError('invalid arm/cap')
    menu=w.menu(cap)
    X=np.zeros((policy.ACTION_MAX,DIM),dtype=np.float64)
    remain=max(0,cap-w.recoveries);depth=len(w.path)-1
    for i,a in enumerate(menu):
        X[i,:len(policy.FEATURES)]=policy.features(w,a,horizon)
        f=np.array([float(a=='BACKTRACK')*remain/12.,
                    float(a=='RELOCATE_START')*remain/12.,
                    float(a=='BACKTRACK')*depth/12.,
                    float(a=='RELOCATE_START')*depth/12.,
                    float(a=='BACKTRACK')*float(remain>=depth),
                    float(a=='RELOCATE_START')*float(remain<depth)])
        k={'blind':0,'budget_only':2,'raw_counters':4,'engineered':6}[arm]
        X[i,len(policy.FEATURES):len(policy.FEATURES)+k]=f[:k]
    return menu,X


def teacher(w:World,horizon:int,cap:int,rng):
    menu=w.menu(cap)
    if not any(a in MOVE for a in menu):
        if 'BACKTRACK' in menu and cap-w.recoveries >= len(w.path)-1:
            return 'BACKTRACK'
        if 'RELOCATE_START' in menu:return 'RELOCATE_START'
        if 'BACKTRACK' in menu:return 'BACKTRACK'
        return 'STOP'
    f=np.stack([policy.features(w,a,horizon) for a in menu])
    z=policy.teacher_scores(w,menu,f)
    z+=np.asarray([rng.uniform(-.13,.13) for _ in menu])
    return menu[int(z.argmax())]


def corpus(seed,horizon,quick=False):
    if horizon not in STAGES:raise ValueError('curriculum stage')
    records=[];labels=Counter();fingerprints=[];forks=Counter()
    for sp in specs('train',1)[:1 if quick else None]:
        for direction in (TRAIN_DIRECTIONS[:1] if quick else TRAIN_DIRECTIONS):
            for depth in (TRAIN_DEPTHS[:1] if quick else TRAIN_DEPTHS):
                for cap in ((2,12) if quick else TRAIN_CAPS):
                    w=fixture(sp,depth,direction);force_error(w,depth,direction,cap)
                    rng=random.Random(seed*100003+sp.seed*13+depth*97+cap*101+horizon)
                    for t in range(depth,horizon):
                        if w.stopped:break
                        menu=w.menu(cap)
                        label=teacher(w,horizon,cap,rng)
                        if label not in menu:raise ValueError('teacher illegal')
                        # Real source-faithful 36/72/128 teacher trajectories;
                        # slightly upweight fork decisions without overriding data.
                        xs={a:matrix(w,horizon,cap,a)[1] for a in ARMS}
                        weight=35. if t==depth else (4. if label in ('BACKTRACK','RELOCATE_START') else 1.)
                        records.append((xs,len(menu),menu.index(label),weight))
                        labels[label]+=1
                        if t==depth:forks[f'{depth}:{cap}:{label}']+=1
                        actual=label
                        if t in (37,79,107):
                            opts=[a for a in menu if a in MOVE and a!=label]
                            if opts:actual=opts[(seed+sp.seed+t)%len(opts)]
                        w.apply(actual,cap)
                    fingerprints.append((sp.ident,direction,depth,cap,digest(w.decisions)))
    X={a:np.stack([r[0][a] for r in records]) for a in ARMS}
    M=np.arange(policy.ACTION_MAX)[None,:]<np.asarray([r[1] for r in records])[:,None]
    Y=np.array([r[2] for r in records]);W=np.array([r[3] for r in records])
    return X,M,Y,W,{'examples':len(Y),'fork_labels':dict(sorted(forks.items())),
          'teacher_actions':dict(sorted(labels.items())),'source_digest':digest(fingerprints),'split':'train'}


class Ranker:
    def __init__(self,seed,arm):
        if arm not in ARMS:raise ValueError('arm')
        self.seed=seed;self.arm=arm;self.updates=0
        self.w=np.random.default_rng(seed).normal(0,.045,DIM)
        self.m=np.zeros(DIM);self.v=np.zeros(DIM)

    def decide(self,w,horizon,cap):
        menu,X=matrix(w,horizon,cap,self.arm)
        z=X[:len(menu)]@self.w;z-=z.max();p=np.exp(z);p/=p.sum()
        return menu[int(p.argmax())],{a:float(v) for a,v in zip(menu,p)}

    def fit(self,X,M,Y,W,steps,seed):
        rng=np.random.default_rng(seed);p=W/W.sum();losses=[]
        for _ in range(steps):
            indices=rng.choice(len(Y),size=64,replace=True,p=p)
            x=X[indices];mask=M[indices];y=Y[indices]
            z=x@self.w;z=np.where(mask,z,-1e8);z-=z.max(axis=1,keepdims=True)
            q=np.exp(z)*mask;q/=q.sum(axis=1,keepdims=True)
            losses.append(float(-np.log(np.maximum(q[np.arange(64),y],1e-20)).mean()))
            q[np.arange(64),y]-=1
            g=np.einsum('baf,ba->f',x,q)/64
            g*=min(1.,4./max(float(np.linalg.norm(g)),1e-12))
            self.updates+=1;self.m=.9*self.m+.1*g;self.v=.999*self.v+.001*g*g
            self.w-=.03*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        return {'updates':steps,'first_loss':round(losses[0],5),
                'final_10_loss':round(statistics.mean(losses[-10:]),5)}

    def save(self,path):
        np.savez_compressed(path,w=self.w,m=self.m,v=self.v,seed=self.seed,
                            arm=self.arm,version=VERSION,updates=self.updates)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            if str(f['version'])!=VERSION:raise ValueError('version mismatch')
            m=cls(int(f['seed']),str(f['arm']))
            for k in ('w','m','v'):
                v=f[k]
                if v.shape!=(DIM,) or not np.isfinite(v).all():raise ValueError('checkpoint corrupt')
                setattr(m,k,v.copy())
            m.updates=int(f['updates'])
        return m


def inject(w,selected,t,cap,seed):
    if t not in (37,79,109):return selected,None
    opts=[a for a in w.menu(cap) if a in MOVE and a!=selected]
    if not opts:return selected,{'changed':False,'requested':selected}
    actual=opts[(seed+w.spec.seed+t)%len(opts)]
    return actual,{'changed':True,'requested':actual}


def rollout(sp,model,depth,direction,cap,horizon):
    if sp.split!='development' or depth not in DEV_DEPTHS or direction not in DEV_DIRECTIONS or cap not in DEV_CAPS or horizon not in HORIZONS:
        raise ValueError('sealed or unsupported rollout')
    w=fixture(sp,depth,direction);start=w.base_sha();rows=[];lat=[]
    for event in force_error(w,depth,direction,cap):
        rows.append({'preferred':None,'probabilities':None,'injected':'external_legal_error','event':event})
    fork_menu=w.menu(cap)
    selected_fork=None
    for t in range(depth,horizon):
        if w.stopped:break
        t0=time.perf_counter_ns();selected,ps=model.decide(w,horizon,cap)
        lat.append((time.perf_counter_ns()-t0)/1e6)
        if t==depth:selected_fork=selected
        actual,inj=inject(w,selected,t,cap,model.seed)
        ev=w.apply(actual,cap)
        rows.append({'preferred':selected,'probabilities':{a:round(v,8) for a,v in ps.items()},
                     'injected':inj,'event':ev})
    correct='BACKTRACK' if cap>=depth else 'RELOCATE_START'
    return {'version':VERSION,'world':sp.ident,'split':'development','depth':depth,
            'direction':direction,'cap':cap,'horizon':horizon,'arm':model.arm,
            'seed':model.seed,'base_sha':start,'fork_menu':fork_menu,
            'fork':selected_fork,'expected_fork':correct,'fork_correct':selected_fork==correct,
            'rows':rows,'metrics':w.metrics(horizon),'actions':dict(sorted(Counter(r['event']['chosen'] for r in rows).items())),
            'trace_sha256':digest([r['event'] for r in rows]),'final_sha':w.state_sha(),
            'ink':[list(p) for p in sorted(w.ink)],'mean_decision_ms':round(statistics.mean(lat),6),
            'native_receipts':0}


def verify_episode(ep,model,sp):
    if (ep['version']!=VERSION or ep['split']!='development' or ep['world']!=sp.ident
        or ep['seed']!=model.seed or ep['arm']!=model.arm or ep['native_receipts']!=0):raise ValueError('identity')
    w=fixture(sp,ep['depth'],ep['direction']);cap=ep['cap'];h=ep['horizon']
    if cap not in DEV_CAPS or h not in HORIZONS or ep['base_sha']!=w.base_sha():raise ValueError('fixture')
    events=force_error(w,ep['depth'],ep['direction'],cap)
    if [r['event'] for r in ep['rows'][:ep['depth']]]!=events or ep['fork_menu']!=w.menu(cap):
        raise ValueError('forced source invalid')
    fork=None
    for t,row in enumerate(ep['rows'][ep['depth']:],start=ep['depth']):
        if w.stopped:raise ValueError('continued after stop')
        preferred,ps=model.decide(w,h,cap)
        if t==ep['depth']:fork=preferred
        if (row['preferred']!=preferred or set(row['probabilities'])!=set(ps)
            or any(abs(v-row['probabilities'][a])>2e-8 for a,v in ps.items())):
            raise ValueError('checkpoint logits disagree')
        actual,inj=inject(w,preferred,t,cap,model.seed)
        if inj!=row['injected'] or w.apply(actual,cap)!=row['event']:
            raise ValueError('host replay mismatch')
    expected='BACKTRACK' if cap>=ep['depth'] else 'RELOCATE_START'
    if (ep['fork']!=fork or ep['fork_correct']!=(fork==expected) or ep['expected_fork']!=expected
        or ep['final_sha']!=w.state_sha() or ep['trace_sha256']!=digest([r['event'] for r in ep['rows']])
        or ep['ink']!=[list(p) for p in sorted(w.ink)]
        or ep['metrics']!=w.metrics(h)
        or ep['actions']!=dict(sorted(Counter(r['event']['chosen'] for r in ep['rows']).items()))):
        raise ValueError('result replay mismatch')
    return len(ep['rows'])


def summarize(episodes):
    return {'n':len(episodes),'correct_fork':sum(e['fork_correct'] for e in episodes),
            'early_stop':sum(e['metrics']['stopped_early'] for e in episodes),
            'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in episodes),3),
            'mean_branches':round(statistics.mean(e['metrics']['branch_cells'] for e in episodes),3),
            'mean_decision_ms':round(statistics.mean(e['mean_decision_ms'] for e in episodes),6)}


def dep_hashes():
    base=Path(__file__).parent
    return {p:sha_file(base/p) for p in ('return_generalization_lab.py','return_budget_lab.py','world.py','policy.py','train.py')}


def run(out,seed=596,updates=30,quick=False):
    if out.exists():raise FileExistsError('immutable output')
    if not 1<=updates<=80 or not 0<=seed<100000:raise ValueError('optimizer budget')
    out.mkdir(parents=True)
    t0=time.perf_counter();models={arm:Ranker(seed,arm) for arm in ARMS};train=[]
    for h in (STAGES[:1] if quick else STAGES):
        X,M,Y,W,meta=corpus(seed,h,quick=quick)
        train.append({'horizon':h,'corpus':meta,'arms':{a:m.fit(X[a],M,Y,W,updates,seed*17+h*23) for a,m in models.items()}})
    checkpoints={}
    for arm,m in models.items():
        name=f'{arm}-seed{seed}.npz';m.save(out/name)
        r=Ranker.load(out/name)
        if not np.array_equal(r.w,m.w) or r.updates!=m.updates:raise ValueError('restore')
        checkpoints[arm]={'file':name,'sha256':sha_file(out/name),'parameters':DIM,'updates':m.updates}
    (out/'trajectories').mkdir();ep=[];decisions=0
    dev=specs('development',1)[:1 if quick else 2]
    for j,sp in enumerate(dev):
        direction=DEV_DIRECTIONS[j%len(DEV_DIRECTIONS)]
        for depth in (DEV_DEPTHS[:1] if quick else DEV_DEPTHS):
            for cap in ((2,12) if quick else EVAL_CAP_PAIRS[depth]):
                for h in ((128,) if quick else HORIZONS):
                    for arm,m in models.items():
                        e=rollout(sp,m,depth,direction,cap,h)
                        decisions+=verify_episode(e,m,sp)
                        fname=f'{arm}-d{depth}-{direction}-c{cap}-h{h}-{sp.ident}'
                        (out/'trajectories'/(fname+'.json')).write_text(json.dumps(e,sort_keys=True)+'\n')
                        if j==0 and h==160 and (depth,cap) in ((6,2),(10,12)):
                            original=fixture(sp,depth,direction)
                            for row in e['rows']:original.apply(row['event']['chosen'],cap)
                            _png(out/'trajectories'/(fname+'.png'),original)
                        ep.append(e)
    table={}
    for arm in ARMS:
        table[arm]=summarize([e for e in ep if e['arm']==arm])
    by_margin={}
    for d in (DEV_DEPTHS[:1] if quick else DEV_DEPTHS):
        for cap in ((2,12) if quick else EVAL_CAP_PAIRS[d]):
            key=f'depth{d}-cap{cap}'
            by_margin[key]={a:summarize([e for e in ep if (e['depth'],e['cap'],e['arm'])==(d,cap,a)]) for a in ARMS}
    result={'version':VERSION,'seed':seed,'source_sha256':dep_hashes(),'training':train,
            'checkpoints':checkpoints,'table':table,'by_depth_and_cap':by_margin,
            'episodes':len(ep),'decisions':decisions,'quick':bool(quick),'elapsed_s':round(time.perf_counter()-t0,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'train_depths':TRAIN_DEPTHS,'dev_depths':DEV_DEPTHS,
            'train_orientations':TRAIN_DIRECTIONS,'dev_orientations':DEV_DIRECTIONS,
            'sealed_heldout_uninspected':True,'native_receipts':0,'paid_compute':False,
            'native_64_limit_unchanged':True}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return result


def verify(out):
    r=json.loads((out/'report.json').read_text())
    if (r['version']!=VERSION or r['source_sha256']!=dep_hashes() or
        not r['sealed_heldout_uninspected'] or r['native_receipts']!=0):raise ValueError('provenance')
    manifest=json.loads((out/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if actual!=manifest:raise ValueError('artifact hash mismatch')
    models={a:Ranker.load(out/c['file']) for a,c in r['checkpoints'].items()}
    if set(models)!=set(ARMS):raise ValueError('missing model')
    for arm,m in models.items():
        c=r['checkpoints'][arm]
        if m.seed!=r['seed'] or m.arm!=arm or m.updates!=c['updates'] or c['parameters']!=DIM or sha_file(out/c['file'])!=c['sha256']:
            raise ValueError('checkpoint identity')
    dev={s.ident:s for s in specs('development',1)[:1 if r['quick'] else 2]}
    ep=[];decisions=0
    for p in (out/'trajectories').glob('*.json'):
        e=json.loads(p.read_text())
        if e['world'] not in dev:raise ValueError('unknown dev world')
        decisions+=verify_episode(e,models[e['arm']],dev[e['world']]);ep.append(e)
    if (len(ep),decisions)!=(r['episodes'],r['decisions']):raise ValueError('episode count')
    for key,arms in r['by_depth_and_cap'].items():
        for arm,expected in arms.items():
            sample=[e for e in ep if f"depth{e['depth']}-cap{e['cap']}"==key and e['arm']==arm]
            obs=summarize(sample);exp=dict(expected)
            obs.pop('mean_decision_ms');exp.pop('mean_decision_ms')
            if obs!=exp:raise ValueError('depth/cap aggregate tamper')
    for arm in ARMS:
        obs=summarize([e for e in ep if e['arm']==arm]);exp=dict(r['table'][arm])
        obs.pop('mean_decision_ms');exp.pop('mean_decision_ms')
        if obs!=exp:raise ValueError('aggregate tampering')
    return {'episodes_replayed':len(ep),'decisions_replayed':decisions,
            'hashes_verified':len(manifest),'checkpoints_restored':len(models)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=596);p.add_argument('--updates',type=int,default=30)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.updates,args.quick),sort_keys=True))
