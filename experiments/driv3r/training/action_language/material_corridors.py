"""GAL-010: material-raster corridor embodiment, no synthetic goal mask at inference.

External 48x48 CyberSand-like host. Local crop is actual material {air, inert,
protected, own ink}; target is ALL originally open corridor cells except START.
Only the training teacher and scorer see the full map. Candidate actions are
supplied, leased, and executed by the unchanged synthetic host authority.
No changes to game/runtime/provider, no native or resident SLM claims.
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
from lab import World, START, DIRS, W, H, INK, DECISION_CAP, sha, png

SCHEMA='gal-010-material-corridors/1'
TRAIN_BASE=1030000
TEST_BASE=1130000
ARMS=('cell','sweep','lift','retrace','anchor','jog')
VIEWS=('material','occluded')
KINDS=('path','tree')
CORRIDOR_SIZE=33
INK_BUDGET=40
FEATURES=('ink','lift','jog','retrace','anchor','stop','span','straight',
 'left','right','reverse','frontier_end','ink_end','solid_end','frontier_delta',
 'air_local','own_ink_local','solid_local','path_ink','path_lift','end_visible')
SRC=('lab.py','material_corridors.py','test_material_corridors.py')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def neighbors(p):return ((p[0]+dx,p[1]+dy) for dx,dy in DIRS.values())
def inside(p):return 2<=p[0]<W-2 and 2<=p[1]<H-2

def make_corridor(seed,kind):
    """Generate deterministic induced paths/branched trees without hidden targets."""
    if kind not in KINDS:raise ValueError('unknown corridor kind')
    for attempt in range(160):
        rng=random.Random(f'gal010:{seed}:{kind}:{attempt}')
        cells={START};current=START
        while len(cells)<CORRIDOR_SIZE:
            roots=[current] if kind=='path' else sorted(cells)
            frontier=[]
            for a in roots:
                for b in neighbors(a):
                    if inside(b) and b not in cells and sum(c in cells for c in neighbors(b))==1:
                        frontier.append((a,b))
            if not frontier:break
            if kind=='path':
                direction=(current[0]-(sorted(cells)[0][0] if len(cells)==1 else prev[0]),
                           current[1]-(sorted(cells)[0][1] if len(cells)==1 else prev[1]))
                preferred=[q for q in frontier if (q[1][0]-q[0][0],q[1][1]-q[0][1])==direction]
                a,b=rng.choice(preferred if preferred and rng.random()<.70 else frontier)
                prev=current;current=b
            else:
                # A mild bias toward already-growing limbs without requiring a tip.
                a,b=rng.choice(frontier)
            cells.add(b)
        degrees=[sum(n in cells for n in neighbors(p)) for p in cells]
        if len(cells)==CORRIDOR_SIZE and (kind=='path' or sum(x>=3 for x in degrees)>=2):
            if kind=='path' and max(degrees)>2:raise AssertionError('path self-cross')
            return frozenset(cells),attempt
    raise RuntimeError(f'corridor generation exhausted: {seed}, {kind}')

class CorridorWorld(World):
    def __init__(self,seed,kind,arm,lease):
        super().__init__(seed,'empty',arm,lease=lease)
        self.kind=kind
        self.corridor,self.attempt=make_corridor(seed,kind)
        self.goal=self.corridor-{START}
        self.material=[3]*(W*H)
        self.protected=set()
        for y in range(H):
            for x in range(W):
                if x in (0,W-1) or y in (0,H-1):
                    self.material[self.idx((x,y))]=5
                    self.protected.add((x,y))
        for p in self.corridor:self.material[self.idx(p)]=0
        self.original=self.material.copy()
        assert self.cursor==START and len(self.goal)==32
    def legal(self):
        offered=super().legal()
        remaining=INK_BUDGET-len(self.own)
        return {n:p for n,p in offered.items()
                if not n.startswith(('INK_','JOG_')) or len(p)<=remaining}

def visible_material(world,view):
    if view not in VIEWS:raise ValueError('bad view')
    if view=='occluded':return {}
    x,y=world.cursor
    return {(i,j):world.material[world.idx((i,j))]
            for i in range(max(0,x-4),min(W,x+5))
            for j in range(max(0,y-4),min(H,y+5))
            if abs(i-x)+abs(j-y)<=4}

def vector(world,name,path,view):
    """All material values restricted to *current-cursor* Manhattan radius 4.

    The action ID and host-legal path are allowed, not remote material peeks.
    Candidate endpoint neighbors outside crop are UNKNOWN, never read.
    """
    crop=visible_material(world,view)
    end=path[-1] if path else world.cursor
    kind=name.split('_')[0]
    heading=name.split('_')[1] if kind in ('INK','LIFT','JOG') else world.heading
    d=DIRS[heading];old=DIRS[world.heading]
    cross=old[0]*d[1]-old[1]*d[0]
    dot=old[0]*d[0]+old[1]*d[1]
    def context(p):
        values=[crop[q] for q in neighbors(p) if q in crop]
        return (values.count(0)/4,values.count(INK)/4,
                sum(v in (3,5) for v in values)/4)
    f1,i1,s1=context(end); f0,_,_=context(world.cursor)
    values=list(crop.values())
    return (float(kind=='INK'),float(kind=='LIFT'),float(kind=='JOG'),
       float(kind=='RETRACE'),float(kind=='SAVE'),float(kind=='STOP'),
       len(path)/4,float(dot==1),float(cross<0),float(cross>0),float(dot==-1),
       f1,i1,s1,f1-f0,values.count(0)/41,values.count(INK)/41,
       sum(v in (3,5) for v in values)/41,
       len(path)/4 if kind in ('INK','JOG') else 0,
       len(path)/4 if kind in ('LIFT','RETRACE') else 0,
       float(end in crop))

class Model:
    def __init__(self):self.weights=[0.0]*len(FEATURES)
    def probabilities(self,vectors):
        scores=[sum(v*w for v,w in zip(vec,self.weights)) for vec in vectors]
        m=max(scores);z=[math.exp(max(-40.,s-m)) for s in scores];t=sum(z)
        return [a/t for a in z]
    def choose(self,world,view,rng):
        menu,token=world.offer();names=list(menu)
        vecs=[vector(world,n,menu[n],view) for n in names]
        probs=self.probabilities(vecs)
        u=rng.random();acc=0;index=len(names)-1
        for i,p in enumerate(probs):
            acc+=p
            if u<=acc:index=i;break
        return names[index],token,names,vecs,probs

def distance_to_open(world,point,excluded=()):
    # Privileged graph planner used only by training teacher; never model input.
    todo=set(world.goal-world.own)-set(excluded)
    if not todo:return 0
    q=deque([(point,0)]);seen={point}
    while q:
        at,d=q.popleft()
        if at in todo:return d
        for nxt in neighbors(at):
            if nxt in world.corridor and nxt not in seen:
                q.append((nxt,d+1));seen.add(nxt)
    raise AssertionError('host corridor disconnected')

def teacher(world,menu):
    if not world.goal-world.own:return 'STOP'
    scored=[]
    for name,path in menu.items():
        kind=name.split('_')[0]
        if kind=='STOP' or kind=='SAVE':continue
        end=path[-1] if path else world.cursor
        gain=len(path) if kind in ('INK','JOG') else 0
        dist=distance_to_open(world,end,path if gain else ())
        score=10*gain-0.85*dist+(len(path)*.05 if gain else len(path)*.15)
        scored.append((score,name))
    return max(scored)[1] if scored else 'STOP'

def train(arm,view,worlds=24):
    if arm not in ARMS or view not in VIEWS or type(worlds)!=int or not 4<=worlds<=80:
        raise ValueError('invalid bounded campaign')
    model=Model();updates=expert_used=0
    for i in range(worlds):
        seed=TRAIN_BASE+i;kind=KINDS[i%2]
        w=CorridorWorld(seed,kind,arm,f'gal010-train:{arm}:{view}:{i}')
        rng=random.Random(f'gal010-train-choice:{arm}:{seed}')
        mix=random.Random(f'gal010-train-mix:{arm}:{seed}')
        while not w.stopped and w.decisions<DECISION_CAP and w.goal-w.own:
            action,token,names,vecs,probs=model.choose(w,view,rng)
            expert=teacher(w,w.legal());target=names.index(expert)
            expected=[sum(p*v[k] for p,v in zip(probs,vecs)) for k in range(len(FEATURES))]
            for k in range(len(FEATURES)):
                model.weights[k]=max(-8.,min(8.,model.weights[k]+.045*(vecs[target][k]-expected[k])))
            picked=expert if mix.random() < (.85 if i<worlds//2 else .4) else action
            w.apply(picked,token,original_response=f'gal010-train:{picked}')
            updates+=1;expert_used+=picked==expert
    return model,{'worlds':worlds,'updates':updates,'expert_followed':expert_used}

def episode(model,arm,view,kind,seed):
    w=CorridorWorld(seed,kind,arm,f'gal010-eval:{arm}:{view}:{kind}:{seed}')
    rng=random.Random(f'gal010-shared:{arm}:{kind}:{seed}')
    while not w.stopped and w.decisions<DECISION_CAP and w.goal-w.own:
        action,token,*_=model.choose(w,view,rng)
        w.apply(action,token,original_response=f'gal010-eval:{action}')
    good=len(w.own&w.goal)
    row={'arm':arm,'view':view,'kind':kind,'seed':seed,'corridor_attempt':w.attempt,
        'world_sha256':sha(w.original),'material_sha256':sha(w.material),
        'receipts_sha256':sha(w.trace),'correct':good,'coverage':round(good/32,4),
        'complete':good==32,'ink':len(w.own),'off_target_ink':len(w.own)-good,'decisions':w.decisions,
        'ink_per_decision':round(len(w.own)/max(1,w.decisions),4),
        'lift_decisions':sum(t['action_id'].startswith(('LIFT_','RETRACE_')) for t in w.trace),
        'stop_early':w.stopped and good<32,'native_receipts':0,'resident_slms':0}
    if w.own&w.protected or any(w.material[w.idx(p)]!=w.original[w.idx(p)] for p in w.protected):
        raise AssertionError('protected/non-goal write')
    return w,row

def summarise(rows):
    return {'episodes':len(rows),'completions':sum(r['complete'] for r in rows),
        'mean_coverage':round(statistics.mean(r['coverage'] for r in rows),4),
        'mean_decisions':round(statistics.mean(r['decisions'] for r in rows),3),
        'ink_per_decision':round(sum(r['ink'] for r in rows)/max(1,sum(r['decisions'] for r in rows)),4),
        'early_STOP':sum(r['stop_early'] for r in rows)}

def run(out,train_worlds=24,eval_worlds=16,gallery=2):
    if type(eval_worlds)!=int or not 4<=eval_worlds<=48 or not 0<=gallery<=3:
        raise ValueError('bounded evaluation')
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    rows={};traces={};training={};checkpoints={}
    for arm in ARMS:
        for view in VIEWS:
            model,stats=train(arm,view,train_worlds)
            key=f'{arm}:{view}';training[key]=stats
            cp={'schema':'gal010-imitation-linear/1','arm':arm,'view':view,
                'features':FEATURES,'weights':model.weights,'training':stats,
                'teacher_only_during_training':True,'native_execution':False}
            path=root/'checkpoints'/f'{arm}_{view}.json'
            path.write_text(json.dumps(cp,indent=2,sort_keys=True)+'\n')
            checkpoints[key]=digest(path)
            for kind in KINDS:
                panel=f'{arm}:{view}:{kind}';rows[panel]=[];traces[panel]=[]
                for i in range(eval_worlds):
                    seed=TEST_BASE+i
                    w,row=episode(model,arm,view,kind,seed)
                    rows[panel].append(row);traces[panel].append(w.trace)
                    if i<gallery and kind=='tree':
                        d=root/'gallery';d.mkdir(exist_ok=True)
                        png(d/f'{arm}-{view}-{seed}.png',w,scale=5)
    (root/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (root/'receipts.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    summary={k:summarise(v) for k,v in rows.items()}
    metrics={'schema':SCHEMA,'source_sha256':{p:digest(Path(__file__).with_name(p)) for p in SRC},
         'checkpoints_sha256':checkpoints,'training':training,
         'training_seed_range':[TRAIN_BASE,TRAIN_BASE+train_worlds-1],
         'heldout_seed_range':[TEST_BASE,TEST_BASE+eval_worlds-1],
         'source_of_authority':'synthetic material host','crop_manhattan_radius':4,
         'goal_definition':'all originally open cells except start; NOT read at inference',
         'STOP_unmasked':True,'native_receipts':0,'resident_slms':0,
         'episodes_sha256':digest(root/'episodes.json'),'receipts_sha256':digest(root/'receipts.json'),
         'summary':summary}
    (root/'metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n'
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return metrics

def audit(root):
    root=Path(root);m=json.loads((root/'metrics.json').read_text())
    if m['schema']!=SCHEMA or m['source_sha256']!={p:digest(Path(__file__).with_name(p)) for p in SRC}:
        raise ValueError('source/schema mismatch')
    manifest={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,p=line.split('  ',1)
        if p in manifest or '..' in Path(p).parts or Path(p).is_absolute() or not (root/p).is_file() or digest(root/p)!=h:
            raise ValueError('SHA manifest violation')
        manifest[p]=h
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    if set(manifest)!=actual:raise ValueError('missing/unmanifested files')
    rows=json.loads((root/'episodes.json').read_text())
    traces=json.loads((root/'receipts.json').read_text())
    if digest(root/'episodes.json')!=m['episodes_sha256'] or digest(root/'receipts.json')!=m['receipts_sha256']:
        raise ValueError('evidence tamper')
    verified=0
    for key,items in rows.items():
        arm,view,kind=key.split(':')
        for row,trace in zip(items,traces[key]):
            seed=row['seed'];w=CorridorWorld(seed,kind,arm,f'gal010-eval:{arm}:{view}:{kind}:{seed}')
            for t in trace:
                offered,token=w.offer()
                if t['action_id'] not in offered or t['offered_sha256']!=token:raise ValueError('illegal action')
                result=w.apply(t['action_id'],token,original_response=t['original_response'])
                if sha(result)!=sha(t):raise ValueError('receipt mismatch')
                verified+=1
            if sha(w.material)!=row['material_sha256'] or sha(w.trace)!=row['receipts_sha256']:
                raise ValueError('final material mismatch')
    worlds=m['training_seed_range'][1]-TRAIN_BASE+1
    heldout=m['heldout_seed_range'][1]-TEST_BASE+1
    replayed=0
    for arm in ARMS:
        for view in VIEWS:
            model,stats=train(arm,view,worlds);key=f'{arm}:{view}'
            cp={'schema':'gal010-imitation-linear/1','arm':arm,'view':view,
                'features':FEATURES,'weights':model.weights,'training':stats,
                'teacher_only_during_training':True,'native_execution':False}
            checkpoint=root/'checkpoints'/f'{arm}_{view}.json'
            if checkpoint.read_text()!=json.dumps(cp,indent=2,sort_keys=True)+'\n' or digest(checkpoint)!=m['checkpoints_sha256'][key]:
                raise ValueError('checkpoint drift')
            for kind in KINDS:
                panel=f'{arm}:{view}:{kind}'
                if len(rows[panel])!=heldout or len(traces[panel])!=heldout:raise ValueError('episode count drift')
                for j in range(heldout):
                    w,row=episode(model,arm,view,kind,TEST_BASE+j)
                    if row!=rows[panel][j] or sha(w.trace)!=sha(traces[panel][j]):
                        raise ValueError('model/episode mismatch')
                    replayed+=1
                if summarise(rows[panel])!=m['summary'][panel]:raise ValueError('summary mismatch')
    return {'policies_retrained':len(ARMS)*len(VIEWS),'episodes_replayed':replayed,
            'host_receipts_verified':verified,'native_receipts':0,'resident_slms':0}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True);p.add_argument('--train',type=int,default=24)
    p.add_argument('--eval',type=int,default=16);p.add_argument('--gallery',type=int,default=2)
    args=p.parse_args()
    print(json.dumps(run(args.output,args.train,args.eval,args.gallery)['summary'],indent=2))
