"""RECOVERY-BALANCE-001: controlled recovery-positive replay curriculum.

Train two same-seed, same-architecture, same-update local rankers on identical
source-faithful TRAIN states. Only the minibatch sampling schedule differs.
First six eastward moves into a cul-de-sac are *legal* external interventions;
recovery and resumed painting are trained from the actual host menu. No A*, no
future privileged observations, no game-side changes, no structural holdout.
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
from . import frontier_viability_lab as fv, policy
from .train import _png, MAX_RECOVERIES, sha_file
from .world import MOVE, World, Spec, digest, replay, specs

VERSION = 'driv3r-recovery-balance/1'
ARMS = ('uniform','recovery_balanced')
STAGES = (36,72,128)
HORIZONS = (128,160)
DIM = fv.DIM
RECOVERY_WEIGHT = 16.
POST_RETURN_WEIGHT = 4.


def hard_cases(seed,horizon,n=2):
    """TRAIN only: deliberate bad-but-legal actions then legal corrective labels.

    Exclude the six externally imposed eastward actions from the imitation
    corpus: these are mistakes to recover from, NOT desired controller moves.
    """
    xs=[]; masks=[]; ys=[]; weights=[]; labels=Counter(); provenance=[]
    for sp0 in specs('train',n):
        sp=replace(sp0,challenge='trap_corridor')
        w=World.create(sp);rng=random.Random(seed*10009+sp.seed*101+horizon)
        for j in range(6):
            if 'MOVE_E' not in w.menu(MAX_RECOVERIES):raise ValueError('corridor no longer host-legal')
            w.apply('MOVE_E',MAX_RECOVERIES)
        if w.cursor!=(54,48):raise ValueError('forced corridor contract')
        strategy='BACKTRACK' if sp.seed%2 else 'RELOCATE_START'
        for t in range(6,horizon):
            if w.stopped:break
            menu,X=fv.features(w,horizon,'local')
            if strategy=='BACKTRACK' and t<12 and w.cursor[0]>48 and w.cursor[1]==48:
                label='BACKTRACK'
            elif t==6 and strategy=='RELOCATE_START':
                label='RELOCATE_START'
            else:
                label=fv.teacher(w,horizon,rng)
            if label not in menu:raise ValueError(f'training teacher off host menu t={t} strategy={strategy} pos={w.cursor} rec={w.recoveries} label={label} menu={menu} spec={sp}')
            mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
            xs.append(X);masks.append(mask);ys.append(menu.index(label));labels[label]+=1
            # Remaining history is observed state, not privileged outcome.
            if label in ('BACKTRACK','RELOCATE_START'): weight=RECOVERY_WEIGHT
            elif w.recoveries and w.steps<24: weight=POST_RETURN_WEIGHT
            else: weight=1.
            weights.append(weight)
            w.apply(label,MAX_RECOVERIES)
        if replay(sp,w.decisions,MAX_RECOVERIES).state_sha()!=w.state_sha():
            raise ValueError('hard-case source replay failure')
        provenance.append([sp.ident,strategy,digest(w.decisions)])
    return np.stack(xs),np.stack(masks),np.asarray(ys,int),np.asarray(weights,float),{
        'examples':len(ys),'teacher_actions':dict(sorted(labels.items())),
        'forcing':'six legal MOVE_E actions per training cul-de-sac',
        'world_digest':digest(provenance),'split':'train'}


def dataset(seed,horizon):
    base,mask,y,meta=fv.training_corpus(seed,horizon,2)
    x=base['local']
    hard,hm,hy,hw,hmeta=hard_cases(seed,horizon)
    X=np.concatenate((x,hard))
    M=np.concatenate((mask,hm))
    Y=np.concatenate((y,hy))
    W=np.concatenate((np.ones(len(y)),hw))
    labels=[('BACKTRACK' if X[i,Y[i],1] else 'RELOCATE_START' if X[i,Y[i],2]
             else 'MOVE' if X[i,Y[i],0] else 'STOP') for i in range(len(Y))]
    return X,M,Y,W,{'ordinary':meta,'hard':hmeta,'examples':len(Y),
        'label_class_distribution':dict(sorted(Counter(labels).items())),
        'recovery_examples':sum(a in ('BACKTRACK','RELOCATE_START') for a in labels),
        'mean_weight':float(W.mean()),'source_sha256':digest([meta['source_digest'],hmeta['world_digest']])}


def fit(model,X,M,Y,W,steps,seed,balanced):
    rng=np.random.default_rng(seed)
    pdraw=W/W.sum() if balanced else None
    losses=[];sample=Counter()
    for _ in range(steps):
        inds=rng.choice(len(Y),size=64,replace=True,p=pdraw)
        xb=X[inds];mb=M[inds];yb=Y[inds]
        recovery_mask=(xb[np.arange(64),yb,1]+xb[np.arange(64),yb,2])>0
        sample['recovery']+=int(recovery_mask.sum());sample['ordinary']+=int((~recovery_mask).sum())
        z=xb@model.w;z=np.where(mb,z,-1e8);z-=z.max(axis=1,keepdims=True)
        prob=np.exp(z)*mb;prob/=prob.sum(axis=1,keepdims=True)
        losses.append(float(-np.log(np.maximum(prob[np.arange(64),yb],1e-20)).mean()))
        prob[np.arange(64),yb]-=1
        grad=np.einsum('baf,ba->f',xb,prob)/64
        grad*=min(1.,4./max(1e-12,float(np.linalg.norm(grad))))
        model.updates+=1
        model.m=.9*model.m+.1*grad;model.v=.999*model.v+.001*grad*grad
        model.w-=.03*(model.m/(1-.9**model.updates))/(np.sqrt(model.v/(1-.999**model.updates))+1e-8)
    return {'first_loss':round(losses[0],6),'last_10_loss':round(statistics.mean(losses[-10:]),6),
            'sampling':dict(sorted(sample.items())),'updates':steps}


def intervene(w,selected,t,seed,forced):
    if not forced:return selected,None
    menu=w.menu(MAX_RECOVERIES)
    if w.spec.challenge=='trap_corridor' and t<6:
        desired='MOVE_E'
    elif t in (12,35,71,109):
        other=[a for a in menu if a in MOVE and a!=selected]
        desired=other[(seed+w.spec.seed+t)%len(other)] if other else selected
    else:return selected,None
    actual=desired if desired in menu else selected
    return actual,{'requested':desired,'executed':actual,'changed':actual!=selected}


def rescue_metrics(sp,rows):
    if sp.challenge!='trap_corridor':return None
    recovery=[i for i,r in enumerate(rows) if i>=6 and r['event']['chosen'] in ('BACKTRACK','RELOCATE_START')]
    outside=[i for i,r in enumerate(rows) if i>=6 and r['event']['chosen'] in MOVE and
             r['event']['cursor_after'] not in [[x,48] for x in range(49,55)]]
    first_outside=outside[0] if outside else None
    return {'recovery_started':bool(recovery),'first_recovery_at':recovery[0] if recovery else None,
            'resumed_new_paint':bool(outside),'first_post_trap_new_paint_at':first_outside,
            'post_trap_outside_ink':len(outside),'new_ink_within_40':sum(i<46 for i in outside),
            'successful_return_8':sum(i<46 for i in outside)>=8}


def episode(sp,model,seed,arm,horizon,forced):
    if sp.split!='development' or horizon not in HORIZONS or arm not in ARMS:raise ValueError('sealed structural holdout')
    w=World.create(sp);rows=[];times=[]
    for t in range(horizon):
        if w.stopped:break
        start=time.perf_counter_ns();selected,ps=model.decide(w,horizon)
        times.append((time.perf_counter_ns()-start)/1e6)
        action,injection=intervene(w,selected,t,seed,forced)
        evt=w.apply(action,MAX_RECOVERIES)
        rows.append({'chosen':selected,'probabilities':{a:round(float(p),8) for a,p in ps.items()},
                     'injection':injection,'event':evt})
    return {'version':VERSION,'world':sp.ident,'base_sha':w.base_sha(),
      'split':'development','challenge':sp.challenge,'arm':arm,'model_seed':seed,
      'horizon':horizon,'forced':forced,'rows':rows,
      'trace_sha256':digest([r['event'] for r in rows]),
      'final_sha':w.state_sha(),'ink':[list(p) for p in sorted(w.ink)],
      'metrics':w.metrics(horizon),'rescue':rescue_metrics(sp,rows),
      'actions':dict(sorted(Counter(r['event']['chosen'] for r in rows).items())),
      'mean_latency_ms':round(statistics.mean(times),6),'native_receipts':0}


def check_episode(e,model,sp):
    if (e['world']!=sp.ident or e['split']!='development' or e['version']!=VERSION or
            e['base_sha']!=World.create(sp).base_sha() or e['native_receipts']!=0):raise ValueError('identity')
    w=World.create(sp)
    for t,r in enumerate(e['rows']):
        selected,prob=model.decide(w,e['horizon'])
        if selected!=r['chosen'] or set(prob)!=set(r['probabilities']) or any(abs(v-r['probabilities'][k])>2e-8 for k,v in prob.items()):
            raise ValueError('checkpoint decision probability mismatch')
        a,inj=intervene(w,selected,t,e['model_seed'],e['forced'])
        if inj!=r['injection'] or w.apply(a,MAX_RECOVERIES)!=r['event']:
            raise ValueError('illegal action or forged state transition')
    if (w.state_sha()!=e['final_sha'] or e['trace_sha256']!=digest([r['event'] for r in e['rows']])
            or e['ink']!=[list(p) for p in sorted(w.ink)] or e['rescue']!=rescue_metrics(sp,e['rows'])
            or e['metrics']!=w.metrics(e['horizon'])
            or e['actions']!=dict(sorted(Counter(r['event']['chosen'] for r in e['rows']).items()))):
        raise ValueError('forged trajectory or metrics')
    return w.steps


def summarize(rows):
    if not rows:return None
    return {'episodes':len(rows),'mean_ink':round(statistics.mean(r['metrics']['unique_painted_cells'] for r in rows),3),
            'mean_branches':round(statistics.mean(r['metrics']['branch_cells'] for r in rows),3),
            'early_stop':sum(r['metrics']['stopped_early'] for r in rows),
            'mean_recoveries':round(statistics.mean(r['metrics']['recoveries'] for r in rows),3),
            'rescue_success_8':sum(bool(r['rescue'] and r['rescue']['successful_return_8']) for r in rows),
            'mean_post_trap_ink':round(statistics.mean(r['rescue']['post_trap_outside_ink'] for r in rows if r['rescue']),3)
                if any(r['rescue'] for r in rows) else None,
            'mean_latency_ms':round(statistics.mean(r['mean_latency_ms'] for r in rows),6)}


def deps():
    d=Path(__file__).parent
    return {name:sha_file(d/name) for name in ('recovery_balance_lab.py','frontier_viability_lab.py',
           'trap_avoidance_lab.py','world.py','policy.py','train.py')}


def run(out,seed=592,updates=45):
    if not 1<=updates<=100 or not 0<=seed<100000:raise ValueError('budget')
    if out.exists():raise FileExistsError('immutable artifact')
    out.mkdir(parents=True);t0=time.perf_counter()
    models={a:fv.Ranker(seed,'local') for a in ARMS}
    curriculum=[]
    for h in STAGES:
        X,M,Y,W,meta=dataset(seed,h)
        perf={a:fit(m,X,M,Y,W,updates,seed*10007+h*11,a=='recovery_balanced') for a,m in models.items()}
        curriculum.append({'horizon':h,'dataset':meta,'loss':perf})
    ck={}
    for a,m in models.items():
        fn=a+'-'+str(seed)+'.npz';m.save(out/fn)
        if not np.array_equal(fv.Ranker.load(out/fn).w,m.w):raise ValueError('checkpoint restoration')
        ck[a]={'file':fn,'sha256':sha_file(out/fn),'parameters':DIM,'updates':m.updates}
    dev=specs('development',2)
    eval_worlds=dev[:4]+[replace(s,challenge='trap_corridor') for s in dev[:4]]
    lookup={sp.ident:sp for sp in eval_worlds}
    (out/'trajectories').mkdir(); episodes=[];total=0
    for sp in eval_worlds:
        for h in HORIZONS:
            for forced in (False,True):
                for a,m in models.items():
                    e=episode(sp,m,seed,a,h,forced);total+=check_episode(e,m,sp)
                    fn=f'{a}-{h}-f{int(forced)}-{sp.ident}'
                    (out/'trajectories'/(fn+'.json')).write_text(json.dumps(e,sort_keys=True)+'\n')
                    if h==160 and forced and sp==eval_worlds[-1]:
                        _png(out/'trajectories'/(fn+'.png'),replay(sp,[r['event'] for r in e['rows']],MAX_RECOVERIES))
                    episodes.append(e)
    summaries={}
    for h in HORIZONS:
        for forced in (False,True):
            for a in ARMS:
                rs=[e for e in episodes if (e['horizon'],e['forced'],e['arm'])==(h,forced,a)]
                summaries[f'{a}-{h}-f{int(forced)}']=summarize(rs)
    rescue_table={}
    for a in ARMS:
        for h in HORIZONS:
            rs=[e for e in episodes if e['arm']==a and e['horizon']==h and e['forced'] and e['challenge']=='trap_corridor']
            rescue_table[f'{a}-{h}']=summarize(rs)
    report={'version':VERSION,'seed':seed,'source_sha256':deps(),'checkpoints':ck,
        'curriculum':curriculum,'summary':summaries,'forced_trap_rescue':rescue_table,
        'episodes':len(episodes),'decisions':total,'elapsed_s':round(time.perf_counter()-t0,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'structural_heldout_sealed':True,'native_receipts':0,'paid_compute':False,
        'native_64_limit_unchanged':True,'eval_world_hashes':{sp.ident:World.create(sp).base_sha() for sp in eval_worlds}}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=deps() or not rep['structural_heldout_sealed'] or rep['native_receipts']!=0:raise ValueError('source')
    manifest=json.loads((out/'SHA256SUMS.json').read_text())
    found={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if manifest!=found:raise ValueError('manifest')
    models={a:fv.Ranker.load(out/c['file']) for a,c in rep['checkpoints'].items()}
    if set(models)!=set(ARMS):raise ValueError('checkpoint set')
    for a,m in models.items():
        c=rep['checkpoints'][a]
        if m.seed!=rep['seed'] or m.arm!='local' or m.updates!=c['updates'] or sha_file(out/c['file'])!=c['sha256']:
            raise ValueError('checkpoint identity')
    dev=specs('development',2)
    lookup={sp.ident:sp for sp in dev[:4]+[replace(sp,challenge='trap_corridor') for sp in dev[:4]]}
    episodes=[];total=0
    for p in (out/'trajectories').glob('*.json'):
        e=json.loads(p.read_text());sp=lookup[e['world']]
        if e['arm'] not in ARMS or e['model_seed']!=rep['seed']:raise ValueError('episode model')
        total+=check_episode(e,models[e['arm']],sp);episodes.append(e)
    if (len(episodes),total)!=(rep['episodes'],rep['decisions']):raise ValueError('missing episodes')
    for key,saved in rep['summary'].items():
        arm,h,f=key.rsplit('-',2);h=int(h);forced=bool(int(f[1:]))
        obs=summarize([e for e in episodes if (e['arm'],e['horizon'],e['forced'])==(arm,h,forced)])
        obs.pop('mean_latency_ms');saved={k:v for k,v in saved.items() if k!='mean_latency_ms'}
        if obs!=saved:raise ValueError('summary forged')
    for key,saved in rep['forced_trap_rescue'].items():
        arm,h=key.rsplit('-',1)
        obs=summarize([e for e in episodes if e['arm']==arm and e['horizon']==int(h) and e['forced'] and e['challenge']=='trap_corridor'])
        obs.pop('mean_latency_ms');saved={k:v for k,v in saved.items() if k!='mean_latency_ms'}
        if obs!=saved:raise ValueError('rescue aggregate forged')
    return {'episodes_replayed':len(episodes),'decisions_replayed':total,'files_sha256':len(manifest),'checkpoints_restored':len(models)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=592);p.add_argument('--updates',type=int,default=45)
    args=p.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.updates),sort_keys=True))
