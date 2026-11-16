"""Train matched tiny dual-clock and executed-clock controls, synthetic DEV only.

Recovery is represented as a live-path clock, not hindsight. Training labels
come from a deterministic, imperfect live-path motif teacher; both arms see
identical states, labels, samples and optimizer schedules. The host always
checks the legal action menu. This is not native CyberSand policy success.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np
from .policy import menu_matrix, FEATURES
from .train import HORIZONS, MAX_RECOVERIES, sha_file, source_pins, _png
from .world import MOVE, World, digest, replay, specs

VERSION = 'driv3r-dual-clock/1'
ARMS = ('executed_clock', 'dual_clock')
HEADINGS = 'NESW'
F = len(FEATURES) + 12
STYLES = {'wave':(4,'alternate'), 'jag':(2,'left')}
SCHEDULES = ('wave','jag','wave_jag')
def current_style(schedule,t):
    if schedule not in SCHEDULES:raise ValueError('invalid schedule')
    return ('jag' if t>=64 else 'wave') if schedule=='wave_jag' else schedule
# Fixed before any rollout, never adapted to development outcomes.
INJECTIONS = {14:'alternate', 19:'BACKTRACK', 32:'RELOCATE_START',
              49:'alternate', 68:'BACKTRACK', 83:'RELOCATE_START',
              109:'alternate'}

def headings(world, mode):
    if mode == 'executed':
        return [e['chosen'][-1] for e in world.decisions if e['chosen'] in MOVE]
    reverse = {v:k[-1] for k,v in MOVE.items()}
    return [reverse[(b[0]-a[0],b[1]-a[1])] for a,b in zip(world.path,world.path[1:])]

def clock(world, mode):
    hs = headings(world,mode)
    if not hs:return (0,0,world.heading)
    run,last = 1,0
    for prev,cur in zip(hs,hs[1:]):
        if prev==cur:run+=1
        else:
            diff=(HEADINGS.index(cur)-HEADINGS.index(prev))%4
            last=1 if diff==1 else (-1 if diff==3 else 0)
            run=1
    return run,last,hs[-1]

def features(world,style,arm):
    if style not in STYLES or arm not in ARMS:raise ValueError('invalid control')
    menu,base=menu_matrix(world,128)
    x=np.zeros((len(menu),F),dtype=np.float64)
    x[:,:len(FEATURES)]=base[:len(menu)]
    run,last,heading=clock(world,'executed')
    live_run,live_last,live_heading=clock(world,'live')
    target=STYLES[style][0]
    for i,act in enumerate(menu):
        if act in MOVE:
            rot=(HEADINGS.index(act[-1])-HEADINGS.index(heading))%4
            lrot=(HEADINGS.index(act[-1])-HEADINGS.index(live_heading))%4
            straight=float(rot==0);left=float(rot==3);right=float(rot==1)
            ls=float(lrot==0);ll=float(lrot==3);lr=float(lrot==1)
            x[i,24:28]=[float(style=='wave'),float(style=='jag'),
                         min(run/target,2)*straight,last*(right-left)]
            if arm=='dual_clock':
                x[i,28:36]=[min(live_run/target,2)*ls,live_last*(lr-ll),
                             ls,ll,lr, float(len(world.path)>1),
                             float(world.last_action in ('BACKTRACK','RELOCATE_START')),
                             min(world.recoveries/12,1.)]
        else:
            x[i,24:26]=[float(style=='wave'),float(style=='jag')]
            if arm=='dual_clock':
                x[i,34:36]=[float(world.last_action in ('BACKTRACK','RELOCATE_START')),
                             min(world.recoveries/12,1.)]
    return menu,x

def wanted(world,style):
    run,last,heading=clock(world,'live')
    target,kind=STYLES[style]
    if run<target:return 'MOVE_'+heading
    turn=(-last if last else 1) if kind=='alternate' else -1
    return 'MOVE_'+HEADINGS[(HEADINGS.index(heading)+turn)%4]

def teacher(world,style):
    menu,_=features(world,style,'executed_clock')
    pref=wanted(world,style)
    if pref in menu:return pref
    alternatives=[a for a in menu if a in MOVE]
    if alternatives:
        # Stable, geometry-aware, non-hindsight fallback. STOP remains offered.
        old=world.heading
        return max(alternatives,key=lambda a:(int(a[-1]==old),
                   sum(not world.occupied((world.cursor[0]+MOVE[a][0]*k,
                                            world.cursor[1]+MOVE[a][1]*k)) for k in (2,3)),
                   -menu.index(a)))
    return next((a for a in ('BACKTRACK','RELOCATE_START','STOP') if a in menu),'STOP')

def intervene(world,chosen,t,seed,enabled):
    if not enabled or t not in INJECTIONS:return chosen,None
    kind=INJECTIONS[t];menu=world.menu(MAX_RECOVERIES)
    if kind=='alternate':
        alternatives=[a for a in menu if a in MOVE and a!=chosen]
        actual=alternatives[(seed+world.spec.seed+t)%len(alternatives)] if alternatives else chosen
    else:actual=kind if kind in menu else chosen
    return actual,{'requested':kind,'executed':actual,'changed':actual!=chosen}

class Adapter:
    def __init__(self,seed):
        self.seed=seed
        self.w=np.random.default_rng(seed).normal(0,.025,F)
        self.m=np.zeros(F);self.v=np.zeros(F);self.updates=0
    def distribution(self,x):
        z=x@self.w;z-=z.max();p=np.exp(z);return p/p.sum()
    def train(self,xs,masks,ys,updates,seed):
        rng=np.random.default_rng(seed);loss=[]
        for _ in range(updates):
            idx=rng.integers(0,len(ys),size=96)
            xb=xs[idx];mb=masks[idx];yb=ys[idx]
            logits=xb@self.w;logits=np.where(mb,logits,-1e8)
            logits-=logits.max(axis=1,keepdims=True)
            p=np.exp(logits)*mb;p/=p.sum(axis=1,keepdims=True)
            loss.append(float(-np.log(np.maximum(p[np.arange(len(idx)),yb],1e-30)).mean()))
            p[np.arange(len(idx)),yb]-=1
            grad=np.einsum('baf,ba->f',xb,p)/len(idx)
            self.updates+=1
            self.m=.9*self.m+.1*grad;self.v=.999*self.v+.001*grad*grad
            self.w-=.035*(self.m/(1-.9**self.updates))/(np.sqrt(self.v/(1-.999**self.updates))+1e-8)
        return {'updates':updates,'first_loss':round(loss[0],6),'last_loss':round(statistics.mean(loss[-min(len(loss),10):]),6)}
    def save(self,path):
        np.savez_compressed(path,w=self.w,seed=self.seed,updates=self.updates)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            m=cls(int(a['seed']));m.w=a['w'].copy();m.updates=int(a['updates'])
            if m.w.shape!=(F,) or not np.isfinite(m.w).all():raise ValueError('invalid weights')
        return m

def dataset(seed,worlds,steps):
    if steps not in HORIZONS or worlds not in (1,2):raise ValueError('resource cap')
    xs={a:[] for a in ARMS};ms=[];ys=[];receipts=[]
    for spec in specs('train',worlds):
        for style in SCHEDULES:
            w=World.create(spec)
            for t in range(steps):
                if w.stopped:break
                active=current_style(style,t)
                menu=w.menu(MAX_RECOVERIES);label=teacher(w,active)
                mask=np.zeros(7,dtype=bool);mask[:len(menu)]=True
                ms.append(mask);ys.append(menu.index(label))
                for arm in ARMS:
                    offered,x=features(w,active,arm)
                    if offered!=menu:raise ValueError('menu mutation')
                    padded=np.zeros((7,F));padded[:len(menu)]=x
                    xs[arm].append(padded)
                actual,_=intervene(w,label,t,seed,True)
                w.apply(actual,MAX_RECOVERIES)
            if replay(spec,w.decisions,MAX_RECOVERIES).state_sha()!=w.state_sha():
                raise ValueError('train replay fail')
            receipts.append((spec.ident,style,w.base_sha(),digest(w.decisions)))
    return {a:np.stack(v) for a,v in xs.items()},np.stack(ms),np.asarray(ys),digest(receipts)

def episode(spec,style,arm,model,horizon,forced):
    if spec.split!='development' or horizon not in (128,160):raise ValueError('DEV only')
    w=World.create(spec);scores=[];forced_log=[];match=[];window=-1;lat=[]
    for t in range(horizon):
        if w.stopped:break
        start=time.perf_counter_ns()
        active=current_style(style,t)
        menu,x=features(w,active,arm);probs=model.distribution(x)
        lat.append((time.perf_counter_ns()-start)/1e6)
        selection=menu[int(probs.argmax())];ref=teacher(w,active)
        actual,injection=intervene(w,selection,t,model.seed,forced)
        if injection:forced_log.append([t,injection])
        if t<=window and injection is None and any(a in MOVE for a in menu):
            match.append(selection==ref)
        scores.append({'t':t,'active_style':active,'choice':selection,'teacher':ref,
                       'prob':{a:round(float(p),8) for a,p in zip(menu,probs)}})
        w.apply(actual,MAX_RECOVERIES)
        if injection and injection['changed'] and actual in ('BACKTRACK','RELOCATE_START'):
            window=t+8
    if replay(spec,w.decisions,MAX_RECOVERIES).state_sha()!=w.state_sha():raise ValueError('rollout replay fail')
    return {'version':VERSION,'split':'development','world_id':spec.ident,
            'style':style,'arm':arm,'horizon':horizon,'forced':forced,
            'base_sha':w.base_sha(),'events':w.decisions,'scores':scores,
            'forced_log':forced_log,'recovery_n':len(match),
            'recovery_match':round(sum(match)/len(match),6) if match else None,
            'metrics':w.metrics(horizon),'final_sha':w.state_sha(),
            'ink':[list(p) for p in sorted(w.ink)],'trajectory_sha':digest(w.decisions),
            'action_distribution':dict(Counter(x['chosen'] for x in w.decisions)),
            'mean_latency_ms':round(statistics.mean(lat),6),'native_receipts':0}

def verify_episode(e,m):
    if e['version']!=VERSION or e['split']!='development' or e['native_receipts']!=0 or e['arm'] not in ARMS:
        raise ValueError('provenance error')
    candidates=[s for s in specs('development',1) if s.ident==e['world_id']]
    if len(candidates)!=1:raise ValueError('unapproved fixture')
    w=World.create(candidates[0]);logs=[];match=[];window=-1
    if e['base_sha']!=w.base_sha() or e['trajectory_sha']!=digest(e['events']):raise ValueError('source error')
    if len(e['events'])!=len(e['scores']):raise ValueError('truncated actions')
    for t,(ev,s) in enumerate(zip(e['events'],e['scores'])):
        active=current_style(e['style'],t)
        menu,x=features(w,active,e['arm']);p=m.distribution(x)
        selected=menu[int(p.argmax())];ref=teacher(w,active)
        if (s['t']!=t or s['active_style']!=active or s['choice']!=selected or s['teacher']!=ref or
            set(s['prob'])!=set(menu) or any(abs(s['prob'][a]-float(v))>2e-8 for a,v in zip(menu,p))):
            raise ValueError('forged probability')
        action,inject=intervene(w,selected,t,m.seed,e['forced'])
        if inject:logs.append([t,inject])
        if t<=window and inject is None and any(a in MOVE for a in menu):match.append(selected==ref)
        if action!=ev['chosen'] or w.apply(action,MAX_RECOVERIES)!=ev:raise ValueError('forged transition')
        if inject and inject['changed'] and action in ('BACKTRACK','RELOCATE_START'):window=t+8
    observed=round(sum(match)/len(match),6) if match else None
    if (logs!=e['forced_log'] or observed!=e['recovery_match'] or len(match)!=e['recovery_n']
        or w.metrics(e['horizon'])!=e['metrics'] or w.state_sha()!=e['final_sha']
        or [list(p) for p in sorted(w.ink)]!=e['ink']
        or dict(Counter(x['chosen'] for x in w.decisions))!=e['action_distribution']):
        raise ValueError('geometry or recovery tampering')
    return len(w.decisions)

def run(output,seed=581,worlds=1,updates=30):
    output=Path(output)
    if output.exists():raise FileExistsError('no overwrite')
    if not 0<=seed<100000 or worlds not in (1,2) or not 1<=updates<=80:raise ValueError('caps')
    output.mkdir(parents=True);started=time.perf_counter()
    models={a:Adapter(seed) for a in ARMS};logs=[]
    for stage,horizon in enumerate(HORIZONS):
        xs,ms,ys,pin=dataset(seed,worlds,horizon)
        for arm,m in models.items():
            result=m.train(xs[arm],ms,ys,updates,seed+stage*1009)
            logs.append({'arm':arm,'horizon':horizon,'corpus_sha':pin,
                         'examples':len(ys),**result})
    checkpoints={}
    for arm,m in models.items():
        m.save(output/(arm+'.npz'))
        checkpoints[arm]={'sha256':sha_file(output/(arm+'.npz')),'params':F,'updates':m.updates}
    episode_list=[];turns=0
    for arm,m in models.items():
        for spec in specs('development',1):
            for style in SCHEDULES:
                for horizon in (128,160):
                    for forced in (False,True):
                        e=episode(spec,style,arm,m,horizon,forced)
                        turns+=verify_episode(e,m)
                        (output/f"{arm}-{spec.ident}-{style}-{horizon}-f{int(forced)}.json").write_text(json.dumps(e,sort_keys=True))
                        if spec==specs('development',1)[0] and horizon==160:
                            _png(output/f'{arm}-{style}-force{int(forced)}.png',replay(spec,e['events'],MAX_RECOVERIES))
                        episode_list.append(e)
    summary={}
    for arm in ARMS:
        for horizon in (128,160):
            for forced in (False,True):
                group=[e for e in episode_list if (e['arm'],e['horizon'],e['forced'])==(arm,horizon,forced)]
                rec=[e['recovery_match'] for e in group if e['recovery_match'] is not None]
                summary[f'{arm}/{horizon}/forced{int(forced)}']={
                    'episodes':len(group),
                    'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in group),3),
                    'early_stops':sum(e['metrics']['stopped_early'] for e in group),
                    'post_recovery_teacher_match':round(statistics.mean(rec),4) if rec else None,
                    'action_distribution':dict(sum((Counter(e['action_distribution']) for e in group),Counter())),
                    'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in group),5)}
    source={**source_pins(),'dual_clock_lab.py':sha_file(Path(__file__))}
    report={'version':VERSION,'seed':seed,'source_sha256':source,'training':logs,
            'checkpoints':checkpoints,'episodes':len(episode_list),'verified_decisions':turns,
            'summary':summary,'sealed_heldout_unread':True,'native_receipts':0,
            'native_logical_move_ceiling':64,'elapsed_s':round(time.perf_counter()-started,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    (output/'SHA256SUMS.json').write_text(json.dumps({p.name:sha_file(p) for p in output.iterdir()},indent=2,sort_keys=True))
    return report

def verify(path):
    path=Path(path);r=json.loads((path/'report.json').read_text())
    if r['version']!=VERSION or not r['sealed_heldout_unread'] or r['native_receipts']!=0 or r['native_logical_move_ceiling']!=64:raise ValueError('provenance')
    expected={**source_pins(),'dual_clock_lab.py':sha_file(Path(__file__))}
    if r['source_sha256']!=expected:raise ValueError('source drift')
    manifest=json.loads((path/'SHA256SUMS.json').read_text())
    actual={p.name:sha_file(p) for p in path.iterdir() if p.name!='SHA256SUMS.json'}
    if manifest!=actual:raise ValueError('artifact edit')
    models={}
    for arm in ARMS:
        m=Adapter.load(path/(arm+'.npz'))
        if sha_file(path/(arm+'.npz'))!=r['checkpoints'][arm]['sha256'] or m.updates!=r['checkpoints'][arm]['updates']:
            raise ValueError('checkpoint edit')
        models[arm]=m
    episodes=turns=0
    for file in path.glob('*.json'):
        if file.name in ('report.json','SHA256SUMS.json'):continue
        e=json.loads(file.read_text());turns+=verify_episode(e,models[e['arm']]);episodes+=1
    if (episodes,turns)!=(r['episodes'],r['verified_decisions']):raise ValueError('missing episodes')
    return {'episodes':episodes,'decisions':turns,'files':len(actual)}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path);parser.add_argument('--verify',type=Path)
    parser.add_argument('--seed',type=int,default=581);parser.add_argument('--worlds',type=int,default=1)
    parser.add_argument('--updates',type=int,default=30)
    a=parser.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.output,a.seed,a.worlds,a.updates),sort_keys=True))
