"""Real bounded history-conditioned policy adapter vs history-blind control.

Original synthetic host transitions only. Model-side style commands do not mutate
materials, legality, or the authoritative World specification. DEVELOPMENT only;
structural HELDOUT and native execution remain untouched.
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
from .policy import menu_matrix
from .train import HORIZONS, MAX_RECOVERIES, _png, sha_file, source_pins
from .world import MOVE, World, digest, replay, specs

VERSION = 'driv3r-history-motif-adapter/1'
STYLES = {'arc':(9,'right'), 'wave':(4,'alternate'), 'jag':(2,'left')}
ARMS = ('history_blind', 'history_aware')
HEADINGS = 'NESW'
F = 19
PROBES = (12,35,71)
EVAL_HORIZONS = (36,72,128,160)

def history(w):
    moves=[e['chosen'][-1] for e in w.decisions if e['chosen'] in MOVE]
    if not moves: return 0,0,0
    straight=1; last_turn=0; turns=0
    for a,b in zip(moves,moves[1:]):
        if a==b: straight+=1
        else:
            last_turn=1 if (HEADINGS.index(b)-HEADINGS.index(a))%4==1 else -1
            turns+=1;straight=1
    return straight,last_turn,turns

def desired(w,style):
    run,last,turns=history(w)
    target,form=STYLES[style]
    if run<target:return 'MOVE_'+w.heading
    direction = 1 if form=='right' else (-1 if form=='left' else (-last if last else 1))
    return 'MOVE_'+HEADINGS[(HEADINGS.index(w.heading)+direction)%4]

def matrix(w,style,blind):
    if style not in STYLES: raise ValueError('unknown style')
    menu,X=menu_matrix(w,128)
    run,last,turns=history(w)
    target,_=STYLES[style]
    result=np.zeros((len(menu),F),dtype=np.float64)
    for i,a in enumerate(menu):
        b=X[i]
        row=result[i]
        row[:9]=[float(a in MOVE),b[4],b[5],b[6],float(a=='BACKTRACK'),
                 float(a=='STOP'),b[8],b[10],b[11]]
        row[9:12]=[float(style==s and a in MOVE) for s in STYLES]
        row[12:15]=[float(style==s)*(b[5]-b[6]) for s in STYLES]
        if not blind:
            row[15]=min(run/target,2.)*b[4]
            row[16]=min(run/target,2.)*b[5]
            row[17]=min(run/target,2.)*b[6]
            row[18]=float(style=='wave')*last*(b[5]-b[6])
    return menu,result

def teacher(w,style,rng):
    menu,X=matrix(w,style,False)
    viable=[(a,i) for i,a in enumerate(menu) if a in MOVE]
    if not viable:
        return 'BACKTRACK' if 'BACKTRACK' in menu else ('RELOCATE_START' if 'RELOCATE_START' in menu else 'STOP')
    goal=desired(w,style)
    if goal in menu:return goal
    # Deterministic geometry fallback uses only nearby host-visible occupancy.
    return max(viable,key=lambda ai:(X[ai[1],6]-.4*X[ai[1],8]+rng.random()*.000001,
                                  -ai[1]))[0]

def corpus(seed,worlds,horizon):
    if horizon not in HORIZONS or not 1<=worlds<=3:raise ValueError('caps')
    rows=[]; masks=[]; labels=[]; provenance=[]
    for spec in specs('train',worlds):
        for style in STYLES:
            w=World.create(spec);rng=random.Random(seed*3407+spec.seed*17+horizon)
            for t in range(horizon):
                if w.stopped:break
                menu,X=matrix(w,style,False)
                label=teacher(w,style,rng)
                padded=np.zeros((7,F));padded[:len(menu)]=X
                mask=np.zeros(7,dtype=bool);mask[:len(menu)]=True
                rows.append(padded);masks.append(mask);labels.append(menu.index(label))
                action=label
                if t in PROBES:
                    alternatives=[a for a in menu if a in MOVE and a!=label]
                    if alternatives:action=alternatives[0]
                w.apply(action,MAX_RECOVERIES)
            assert replay(spec,w.decisions,MAX_RECOVERIES).state_sha()==w.state_sha()
            provenance.append((spec.ident,w.base_sha(),style,digest(w.decisions)))
    return np.stack(rows),np.stack(masks),np.asarray(labels),digest(provenance)

class Adapter:
    def __init__(self,seed,blind):
        self.seed=seed;self.blind=blind;self.weights=np.random.default_rng(seed).normal(0,.03,F)
        self.m=np.zeros(F);self.v=np.zeros(F);self.updates=0
    def distribution(self,w,style):
        menu,X=matrix(w,style,self.blind)
        z=X@self.weights;z-=max(z);v=np.exp(z);p=v/sum(v)
        return menu,p
    def optimize(self,X,M,Y,updates,seed):
        if not 1<=updates<=200:raise ValueError('optimizer cap')
        rng=np.random.default_rng(seed)
        if self.blind:X=X.copy();X[:,:,15:]=0
        losses=[]
        for _ in range(updates):
            ii=rng.integers(0,len(Y),size=64);x=X[ii]; mask=M[ii];y=Y[ii]
            z=x@self.weights;z=np.where(mask,z,-1e6);z-=z.max(axis=1,keepdims=True)
            p=np.exp(z)*mask;p/=p.sum(axis=1,keepdims=True)
            losses.append(float(-np.log(np.maximum(1e-30,p[np.arange(len(y)),y])).mean()))
            p[np.arange(len(y)),y]-=1
            grad=np.einsum('baf,ba->f',x,p)/len(y)
            self.updates+=1;self.m=.9*self.m+.1*grad;self.v=.999*self.v+.001*grad**2
            self.weights-=.04*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        return {'first_loss':round(losses[0],6),'last_20_loss':round(statistics.mean(losses[-20:]),6)}
    def save(self,path):
        np.savez_compressed(path,weights=self.weights,seed=self.seed,blind=self.blind,updates=self.updates)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            o=cls(int(f['seed']),bool(f['blind']))
            if f['weights'].shape!=(F,) or not np.isfinite(f['weights']).all():raise ValueError('invalid checkpoint')
            o.weights=f['weights'].copy();o.updates=int(f['updates']);return o

def episode(spec,style,model,horizon,switch=None,force=False):
    if spec.split!='development' or horizon not in EVAL_HORIZONS or style not in STYLES or (switch is not None and (switch not in STYLES or horizon<128)):
        raise ValueError('development-only fixed experiment')
    w=World.create(spec);probs=[];lat=[]; actual=[]
    for t in range(horizon):
        if w.stopped:break
        command=style if switch is None or t<64 else switch
        start=time.perf_counter_ns();menu,p=model.distribution(w,command);lat.append((time.perf_counter_ns()-start)/1e6)
        preferred=menu[int(p.argmax())]; chosen=preferred
        if force and t in PROBES:
            alternatives=[a for a in menu if a in MOVE and a!=preferred]
            if alternatives:chosen=alternatives[0]
        probs.append({'turn':t,'command':command,'preferred':preferred,'distribution':{a:round(float(p[i]),8) for i,a in enumerate(menu)}})
        actual.append(chosen);w.apply(chosen,MAX_RECOVERIES)
    assert replay(spec,w.decisions,MAX_RECOVERIES).state_sha()==w.state_sha()
    moves=[e['chosen'][-1] for e in w.decisions if e['chosen'] in MOVE]
    seg=[];current=0;last=None
    for hd in moves:
        if hd==last:current+=1
        else:
            if current:seg.append(current)
            current=1;last=hd
    if current:seg.append(current)
    tgt=STYLES[style][0]
    quality=sum(abs(x-tgt)<=1 for x in seg)/max(1,len(seg))
    return {'version':VERSION,'world_id':spec.ident,'split':spec.split,'style':style,
            'switch_to':switch,'force':force,'horizon':horizon,'base_sha256':w.base_sha(),
            'events':w.decisions,'probabilities':probs,'metrics':w.metrics(horizon),
            'segment_lengths':seg,'segment_target_fraction':round(quality,6),
            'ink_cells':[list(x) for x in sorted(w.ink)],'trajectory_sha256':digest(w.decisions),
            'final_sha256':w.state_sha(),'action_distribution':dict(sorted(Counter(actual).items())),
            'mean_latency_ms':round(statistics.mean(lat),6),'native_receipts':0}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=575);p.add_argument('--updates',type=int,default=60)
    p.add_argument('--worlds',type=int,default=1);a=p.parse_args(argv)
    if not 0<=a.seed<=100000 or not 1<=a.worlds<=3 or not 1<=a.updates<=200:raise ValueError('caps')
    if a.output.exists():raise FileExistsError('immutable original output exists')
    a.output.mkdir(parents=True)
    begin=time.perf_counter();models={};train_log=[]
    for arm in ARMS:models[arm]=Adapter(a.seed,arm=='history_blind')
    for stage,h in enumerate(HORIZONS):
        X,M,Y,provenance=corpus(a.seed,a.worlds,h)
        for arm in ARMS:
            result=models[arm].optimize(X,M,Y,a.updates,a.seed+stage*101)
            train_log.append({'arm':arm,'horizon':h,'rows':len(Y),'corpus_sha256':provenance,'loss':result})
    cps={}
    for arm,m in models.items():
        path=a.output/f'{arm}.npz';m.save(path)
        restored=Adapter.load(path)
        assert np.array_equal(restored.weights,m.weights) and restored.updates==a.updates*3
        cps[arm]={'sha256':sha_file(path),'updates':restored.updates,'params':F}
    episodes=[];counts={}
    for arm,m in models.items():
        for spec in specs('development',1):
            for h in EVAL_HORIZONS:
                for style in STYLES:
                    obj=episode(spec,style,m,h,force=(h==128))
                    name=f'{arm}-{spec.ident}-{style}-{h}'
                    (a.output/f'{name}.json').write_text(json.dumps(obj,sort_keys=True))
                    episodes.append((arm,h,style,obj))
                    if h>=128:_png(a.output/f'{name}.png',replay(spec,obj['events'],MAX_RECOVERIES))
            for switch_horizon in (128,160):
                obj=episode(spec,'arc',m,switch_horizon,switch='jag',force=True)
                (a.output/f'{arm}-{spec.ident}-arc_to_jag-{switch_horizon}.json').write_text(json.dumps(obj,sort_keys=True))
                episodes.append((arm,switch_horizon,'switch',obj))
    for arm in ARMS:
        for h in EVAL_HORIZONS:
            for style in (*STYLES,'switch'):
                pack=[e for aa,hh,ss,e in episodes if (aa,hh,ss)==(arm,h,style)]
                if pack: counts[f'{arm}/{h}/{style}']={'episodes':len(pack),'ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in pack),3),
                    'turns':round(statistics.mean(e['metrics']['turns'] for e in pack),3),'motif':round(statistics.mean(e['segment_target_fraction'] for e in pack),4),
                    'early_stops':sum(e['metrics']['stopped_early'] for e in pack),
                    'latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in pack),5),
                    'action_distribution':dict(sorted(sum((Counter(e['action_distribution']) for e in pack),Counter()).items()))}
    summary={'version':VERSION,'source_sha256':{**source_pins(),'history_motif_lab.py':sha_file(Path(__file__))},
             'train_only':True,'sealed_heldout_unread':True,'native_executed':False,'native_limit_unchanged':64,
             'seed':a.seed,'checkpoints':cps,'training':train_log,'summary':counts,'episodes':len(episodes),
             'elapsed_seconds':round(time.perf_counter()-begin,3),
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    files={q.name:sha_file(q) for q in a.output.iterdir() if q.is_file()}
    (a.output/'SHA256SUMS.json').write_text(json.dumps(files,indent=2,sort_keys=True))
    return summary

if __name__=='__main__':
    print(json.dumps(main(),sort_keys=True))
