"""GAL-011: observation-bounded branch-memory imitation on the external material host.

Research-only. Policy sees current host-legal IDs/paths, local radius-4 material
samples, and an episodic ledger derived ONLY from those samples and accepted
host receipts. The reference teacher uses precisely that ledger, never the
synthetic world's hidden corridor/goal. Learned linear softmax weights are
trained by online imitation and DAgger-style roll-ins. No game modifications.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from lab import START, INK, DIRS, DECISION_CAP, sha, png
from material_corridors import (CorridorWorld, ARMS, KINDS, TEST_BASE,
                                visible_material, vector, summarise, neighbors, digest)

SCHEMA='gal-011-memory-frontier/1'
TRAIN_BASE=1430000
EVAL_BASE=1530000
ARMS_TEST=('lift',)
FEATURES=('paint','paint_len','lift','retrace','anchor','jog','stop',
          'distance_progress','route_alignment','remaining_path_fraction',
          'target_now','target_after','target_exists','path_new_ink',
          'unknown_path','past_visit','facing_change','near_open',
          'near_ink','near_block','length','known_open_count','terminal_known')

class Ledger:
    """No full-world reads; invisible cells cannot enter this persistent memory."""
    def __init__(self):
        self.known={};self.visited={START:1};self.recorded_paints=set()
        self.observations=0
    def observe(self,w):
        data=visible_material(w,'material')
        self.known.update(data)
        self.observations+=1
        return data
    def accept(self,action,path,receipt):
        kind=action.split('_')[0]
        for p in path:
            self.visited[p]=self.visited.get(p,0)+1
            if kind in ('INK','JOG'):
                self.known[p]=INK;self.recorded_paints.add(p)
        assert receipt['path']==path and receipt['action_id']==action
    def targets(self):
        return {p for p,material in self.known.items() if material==0 and p!=START}
    def traversable(self):
        return {p for p,material in self.known.items() if material in (0,INK)}
    def distance_map(self,target):
        allowed=self.traversable() | {target}
        seen={target:0};q=deque([target])
        while q:
            p=q.popleft()
            for nxt in neighbors(p):
                if nxt in allowed and nxt not in seen:
                    seen[nxt]=seen[p]+1;q.append(nxt)
        return seen
    def route(self,start):
        """Shortest visible route to an observed, not-yet-painted cell."""
        targets=self.targets()
        allowed=self.traversable() | {start}
        q=deque([start]);parents={start:None}
        while q:
            p=q.popleft()
            if p in targets:
                path=[]
                while p!=start:
                    path.append(p);p=parents[p]
                return tuple(reversed(path))
            for nxt in neighbors(p):
                if nxt in allowed and nxt not in parents:
                    parents[nxt]=p;q.append(nxt)
        return ()

def choose_teacher(w,ledger,menu):
    """Bounded, observation-derived oracle; does not inspect corridor or goal."""
    # Prefer continuing to deposit new ink; ties favor longer strokes.
    paint=[]
    for name,path in menu.items():
        if name.startswith(('INK_','JOG_')) and START not in path:
            # Legal catalog may reveal a cell outside crop, but cannot reveal
            # its material; its being host-legal is part of the contract.
            new=sum(p not in ledger.recorded_paints for p in path)
            novelty=sum(p not in ledger.visited for p in path)
            paint.append((new,novelty,len(path),name))
    if paint:return max(paint)[3]
    route=ledger.route(w.cursor)
    if not route:return 'STOP'
    # Follow known shortest route. Only the host may authorize any segment.
    ranked=[]
    for name,path in menu.items():
        if not name.startswith(('LIFT_','RETRACE_')) or not path:continue
        alignment=0
        for a,b in zip(path,route):
            if a!=b:break
            alignment+=1
        if alignment and alignment==len(path) and len(path)<len(route):
            ranked.append((alignment,len(path),name))
    if ranked:return max(ranked)[2]
    return 'STOP'

def features(w,ledger,name,path,route):
    kind=name.split('_')[0]
    ink=kind in ('INK','JOG')
    moving=kind in ('INK','JOG','LIFT','RETRACE')
    end=path[-1] if path else w.cursor
    targets=ledger.targets()
    d0=len(route) if route else 0
    next_route=ledger.route(end) if moving else route
    d1=len(next_route) if next_route else 0
    # For a known frontier path, compare candidate path alignment to a single
    # fixed target route. Do not use hidden global material/goal fields.
    aligned=0
    for a,b in zip(path,route):
        if a!=b:break
        aligned+=1
    painted=sum(p not in ledger.recorded_paints and p!=START for p in path) if ink else 0
    crop=visible_material(w,'material')
    near=[crop[q] for q in neighbors(end) if q in crop]
    direction=name.split('_')[1] if kind in ('INK','LIFT','JOG') else w.heading
    return (
        float(ink),painted/4,float(kind=='LIFT'),float(kind=='RETRACE'),
        float(kind=='SAVE'),float(kind=='JOG'),float(kind=='STOP'),
        max(-1.,min(1.,(d0-d1)/4)) if route else 0.,
        aligned/4,
        (d0-aligned)/max(1,d0) if d0 else 0.,
        float(end in targets),float(bool(next_route)),float(bool(route)),
        painted/4,float(sum(p not in ledger.known for p in path))/4,
        min(4,sum(ledger.visited.get(p,0) for p in path))/4,
        float(direction!=w.heading),near.count(0)/4,near.count(INK)/4,
        sum(v in (3,5) for v in near)/4,len(path)/4,
        min(len(targets),16)/16,float(not targets))

MEMORY_INDICES=(7,8,9,10,11,12,13,15,21,22)
def mask_memory_features(v):
    return tuple(0. if k in MEMORY_INDICES else x for k,x in enumerate(v))

class Scorer:
    def __init__(self):self.weights=[0.0]*len(FEATURES)
    def probs(self,vs):
        scores=[sum(a*b for a,b in zip(v,self.weights)) for v in vs]
        m=max(scores);a=[math.exp(max(-40.,s-m)) for s in scores];z=sum(a)
        return [v/z for v in a]
    def decide(self,w,ledger,rng,mask_memory=False):
        menu,lease=w.offer();names=list(menu);route=ledger.route(w.cursor)
        vs=[features(w,ledger,n,menu[n],route) for n in names]
        if mask_memory:vs=[mask_memory_features(v) for v in vs]
        ps=self.probs(vs);x=rng.random();s=0.;chosen=names[-1]
        for n,p in zip(names,ps):
            s+=p
            if x<=s:chosen=n;break
        return chosen,menu,lease,names,vs,ps

def one_episode(model,arm,kind,seed,mode='trained',mix=0,lr=.06,train=False):
    if mode not in ('trained','knockout','blind_trained','teacher'):
        raise ValueError('unknown policy')
    lease=f'gal011-eval:{arm}:{mode}:{kind}:{seed}' if not train else f'gal011-train:{arm}:{kind}:{seed}'
    w=CorridorWorld(seed,kind,arm,lease);ledger=Ledger()
    rng=random.Random(f'gal011-choice:{arm}:{kind}:{seed}')
    mixrng=random.Random(f'gal011-mix:{arm}:{kind}:{seed}')
    updates=0;expert_followed=0;route_chosen=0
    while not w.stopped and w.decisions<DECISION_CAP and w.goal-w.own:
        ledger.observe(w)
        options,token=w.offer()
        target=choose_teacher(w,ledger,options)
        if mode=='teacher':action=target
        else:
            choice,menu,token,names,vecs,probs=model.decide(
                w,ledger,rng,mask_memory=mode in ('knockout','blind_trained'))
            if train:
                target_i=names.index(target)
                expect=[sum(p*v[k] for p,v in zip(probs,vecs)) for k in range(len(FEATURES))]
                for k in range(len(FEATURES)):
                    model.weights[k]=max(-10.,min(10.,model.weights[k]+lr*(vecs[target_i][k]-expect[k])))
                updates+=1
            action=target if train and mixrng.random()<mix else choice
        path=options[action]
        receipt=w.apply(action,token,original_response=f'gal011:{mode}:{action}')
        ledger.accept(action,path,receipt)
        expert_followed+=action==target
        route_chosen+=action.startswith(('LIFT_','RETRACE_'))
    good=len(w.own&w.goal)
    row={'arm':arm,'mode':mode,'kind':kind,'seed':seed,'complete':good==32,
         'correct':good,'coverage':round(good/32,4),'ink':len(w.own),
         'off_target_ink':len(w.own)-good,'decisions':w.decisions,
         'ink_per_decision':round(len(w.own)/max(1,w.decisions),4),
         'lift_decisions':route_chosen,'stop_early':w.stopped and good<32,
         'material_sha256':sha(w.material),'receipts_sha256':sha(w.trace),
         'observed_cells':len(ledger.known),'known_open_at_end':len(ledger.targets()),
         'native_receipts':0,'resident_slms':0}
    if w.own&w.protected or any(w.material[w.idx(p)]!=w.original[w.idx(p)] for p in w.protected):
        raise AssertionError('protected material modified')
    return w,row,{'updates':updates,'expert_followed':expert_followed}

def train(arm,worlds=48,blind=False):
    if arm not in ARMS_TEST or not 4<=worlds<=160:raise ValueError('bounded training required')
    m=Scorer();stats={'updates':0,'expert_followed':0,'worlds':worlds,'blind':blind}
    # A separate no-memory policy uses the same DAgger roll-ins and zeroes the
    # memory inputs *during optimizer updates*, never merely at inference.
    for i in range(worlds):
        seed=TRAIN_BASE+i;kind=KINDS[i%2]
        if not blind:
            _,_,s=one_episode(m,arm,kind,seed,train=True,mix=.90 if i<worlds//3 else .50)
        else:
            w=CorridorWorld(seed,kind,arm,f'gal011-blind-train:{arm}:{kind}:{seed}')
            led=Ledger();rng=random.Random(f'gal011-blind-choice:{arm}:{kind}:{seed}')
            mixrng=random.Random(f'gal011-blind-mix:{arm}:{kind}:{seed}')
            s={'updates':0,'expert_followed':0}
            while not w.stopped and w.decisions<DECISION_CAP and w.goal-w.own:
                led.observe(w);opts,token=w.offer();expert=choose_teacher(w,led,opts)
                names=list(opts);route=led.route(w.cursor)
                vecs=[mask_memory_features(features(w,led,n,opts[n],route)) for n in names]
                probs=m.probs(vecs)
                x=rng.random();acc=0.;choice=names[-1]
                for n,p in zip(names,probs):
                    acc+=p
                    if x<=acc:choice=n;break
                ti=names.index(expert)
                exp=[sum(p*v[k] for p,v in zip(probs,vecs)) for k in range(len(FEATURES))]
                for k in range(len(FEATURES)):
                    m.weights[k]=max(-10.,min(10.,m.weights[k]+.06*(vecs[ti][k]-exp[k])))
                action=expert if mixrng.random()<(.90 if i<worlds//3 else .50) else choice
                receipt=w.apply(action,token,original_response=f'gal011-blind-train:{action}')
                led.accept(action,opts[action],receipt)
                s['updates']+=1;s['expert_followed']+=action==expert
        stats['updates']+=s['updates'];stats['expert_followed']+=s['expert_followed']
    return m,stats

def run(out,train_worlds=48,eval_worlds=32,gallery=2):
    if not 4<=eval_worlds<=96 or not 0<=gallery<=4:raise ValueError('bounded eval')
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    (root/'gallery').mkdir(exist_ok=True)
    models={};training={};rows={};receipts={};hashes={}
    for arm in ARMS_TEST:
        trained,ts=train(arm,train_worlds)
        blind,bs=train(arm,train_worlds,blind=True)
        for mode,model,stats in (('trained',trained,ts),('blind_trained',blind,bs)):
            key=f'{arm}:{mode}'
            path=root/'checkpoints'/f'{arm}_{mode}.json'
            path.write_text(json.dumps({'schema':SCHEMA,'arm':arm,'mode':mode,'features':FEATURES,
                  'weights':model.weights,'training':stats,'teacher_full_map':False},indent=2,sort_keys=True)+'\n')
            hashes[key]=digest(path);training[key]=stats
        for mode in ('trained','knockout','blind_trained','teacher'):
            model=trained if mode!='blind_trained' else blind
            for kind in KINDS:
                panel=f'{arm}:{mode}:{kind}';rows[panel]=[];receipts[panel]=[]
                for j in range(eval_worlds):
                    seed=EVAL_BASE+j
                    w,row,_=one_episode(model,arm,kind,seed,mode)
                    rows[panel].append(row);receipts[panel].append(w.trace)
                    if j<gallery and kind=='tree':png(root/'gallery'/f'{arm}-{mode}-{seed}.png',w,scale=5)
    (root/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (root/'receipts.json').write_text(json.dumps(receipts,sort_keys=True,separators=(',',':'))+'\n')
    metrics={'schema':SCHEMA,'train_seeds':[TRAIN_BASE,TRAIN_BASE+train_worlds-1],
        'eval_seeds':[EVAL_BASE,EVAL_BASE+eval_worlds-1],
        'source_sha256':{p:digest(Path(__file__).with_name(p)) for p in ('lab.py','material_corridors.py','frontier_memory.py')},
        'checkpoints_sha256':hashes,'training':training,
        'episodes_sha256':digest(root/'episodes.json'),'receipts_sha256':digest(root/'receipts.json'),
        'summary':{p:summarise(items) for p,items in rows.items()},
        'native_receipts':0,'resident_slms':0,'game_code_edited':False,
        'observations':'radius-four material crop + accepted action receipt memory; no hidden goal'}
    (root/'metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(root)}\n'
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return metrics

def audit(root):
    root=Path(root);m=json.loads((root/'metrics.json').read_text())
    assert m['schema']==SCHEMA
    for path,hashval in m['source_sha256'].items():
        assert digest(Path(__file__).with_name(path))==hashval
    manifest={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,p=line.split('  ',1)
        if p in manifest or '..' in Path(p).parts or digest(root/p)!=h:raise ValueError('evidence corrupted')
        manifest[p]=h
    assert set(manifest)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    assert digest(root/'episodes.json')==m['episodes_sha256']
    assert digest(root/'receipts.json')==m['receipts_sha256']
    rows=json.loads((root/'episodes.json').read_text());traces=json.loads((root/'receipts.json').read_text())
    replays=receipts_checked=0
    models={}
    ntrain=m['train_seeds'][1]-TRAIN_BASE+1;neval=m['eval_seeds'][1]-EVAL_BASE+1
    for arm in ARMS_TEST:
        for mode in ('trained','blind_trained'):
            model,stats=train(arm,ntrain,blind=mode=='blind_trained')
            checkpoint=root/'checkpoints'/f'{arm}_{mode}.json'
            cp={'schema':SCHEMA,'arm':arm,'mode':mode,'features':FEATURES,
                'weights':model.weights,'training':stats,'teacher_full_map':False}
            if checkpoint.read_text()!=json.dumps(cp,indent=2,sort_keys=True)+'\n' or digest(checkpoint)!=m['checkpoints_sha256'][f'{arm}:{mode}']:
                raise ValueError('optimizer drift')
            models[arm,mode]=model
        for mode in ('trained','knockout','blind_trained','teacher'):
            model=models[arm,'blind_trained' if mode=='blind_trained' else 'trained']
            for kind in KINDS:
                key=f'{arm}:{mode}:{kind}';assert len(rows[key])==len(traces[key])==neval
                for j in range(neval):
                    seed=EVAL_BASE+j
                    w,row,_=one_episode(model,arm,kind,seed,mode)
                    if row!=rows[key][j] or sha(w.trace)!=sha(traces[key][j]):raise ValueError('model replay drift')
                    # Independent, action-by-action host replay, including stale action
                    # rejection, protected-cell preservation, original response tokens.
                    world=CorridorWorld(seed,kind,arm,f'gal011-eval:{arm}:{mode}:{kind}:{seed}')
                    for t in traces[key][j]:
                        offer,token=world.offer()
                        if t['action_id'] not in offer or token!=t['offered_sha256']:
                            raise ValueError('illegal replay action')
                        result=world.apply(t['action_id'],token,original_response=t['original_response'])
                        if sha(result)!=sha(t):raise ValueError('receipt mismatch')
                        receipts_checked+=1
                    if sha(world.material)!=row['material_sha256']:raise ValueError('material mismatch')
                    replays+=1
                if m['summary'][key]!=summarise(rows[key]):raise ValueError('summary mismatch')
    return {'checkpoints_retrained':len(models),'heldout_episodes_replayed':replays,
            'host_receipts_replayed':receipts_checked,'native_receipts':0}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True);p.add_argument('--train',type=int,default=48)
    p.add_argument('--eval',type=int,default=32);p.add_argument('--gallery',type=int,default=2)
    p.add_argument('--audit',action='store_true')
    args=p.parse_args()
    if args.audit:print(json.dumps(audit(args.output),indent=2))
    else:print(json.dumps(run(args.output,args.train,args.eval,args.gallery)['summary'],indent=2))
