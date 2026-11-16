"""GAL-009: privileged-goal vs local/no-goal observation under identical legal menus.

External 48x48 synthetic CyberSand-like host. Training may use a privileged
geometry teacher; runtime policies receive only requested model observations.
The host owns offers, leases, STOP, protected cells, and material writes.
No native execution, production provider modification, or game code changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from lab import START, DIRS, DECISION_CAP, FEATURES, World, png, sha
from candidate_features import CandidateWorld, FAMILIES
from interaction_ablation import feature_map

SCHEMA = 'gal-009-hard-observation/1'
GOAL_SIZE, INK_BUDGET = 32, 40
TRAIN_BASE, TEST_BASE = 820000, 920000
ARMS = ('cell', 'lift', 'jog')
KINDS = ('path', 'tree')
TRAIN_VIEWS = ('full', 'local')
EVAL_VIEWS = ('full', 'local', 'blind')
OBS_FEATURES = ('hit', 'waste', 'distance', 'progress', 'endpoint', 'nearby', 'done', 'travel')
FEATURE_NAMES = FEATURES + ('near_x_span', 'clear_x_span', 'near_x_turn', 'clear_x_turn',
    'near_x_clear', 'near_x_visited', 'clear_x_persist', 'near_x_straight') + OBS_FEATURES
SRC = ('lab.py', 'candidate_features.py', 'interaction_ablation.py', 'hard_observation.py', 'test_hard_observation.py')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build_goal(seed, family, kind):
    if kind not in KINDS: raise ValueError('unknown morphology')
    base = World(seed, family, 'cell')
    for attempt in range(24):
        rng = random.Random(f'gal009-goal:{seed}:{family}:{kind}:{attempt}')
        current, occupied, selected, preferred = START, {START}, [], None
        for _ in range(GOAL_SIZE):
            if kind == 'path':
                choices = [(current[0]+dx, current[1]+dy) for dx,dy in DIRS.values()
                           if base.clear((current[0]+dx, current[1]+dy))
                           and (current[0]+dx, current[1]+dy) not in occupied]
                if not choices: break
                nxt = preferred if preferred in choices and rng.random() < .72 else rng.choice(choices)
                preferred = (nxt[0]*2-current[0], nxt[1]*2-current[1])
            else:
                frontier = [(p, (p[0]+dx,p[1]+dy)) for p in sorted(occupied)
                            for dx,dy in DIRS.values()
                            if base.clear((p[0]+dx,p[1]+dy))
                            and (p[0]+dx,p[1]+dy) not in occupied]
                if not frontier: break
                _, nxt = rng.choice(frontier)
            selected.append(nxt); occupied.add(nxt); current=nxt
        if len(selected)==GOAL_SIZE:
            return frozenset(selected),attempt
    raise ValueError('goal construction exhausted deterministic attempts')

class GoalWorld(CandidateWorld):
    def __init__(self, seed, family, arm, kind, lease):
        super().__init__(seed, family, arm, 'candidate', lease=lease)
        self.goal, self.attempt = build_goal(seed, family, kind)
        self.kind=kind
        assert not self.goal & self.protected
        assert all(self.original[self.idx(p)] == 0 for p in self.goal)

    def legal(self):
        menus=super().legal()
        left=INK_BUDGET-len(self.own)
        return {n:p for n,p in menus.items()
                if not n.startswith(('INK_', 'JOG_')) or len(p)<=left}

def observed(world, view):
    if view=='full': return world.goal-world.own
    if view=='blind': return frozenset()
    if view=='local':
        x,y=world.cursor
        return frozenset(p for p in world.goal-world.own if abs(x-p[0])+abs(y-p[1])<=4)
    raise ValueError('unknown observation view')

def vector(world, name, path, view):
    # Observability is restricted around the CURRENT cursor. The candidate
    # endpoint must not reveal target cells outside the fixed sensor region.
    targets=observed(world,view)
    stroke=name.startswith(('INK_', 'JOG_'))
    hit=sum(p in targets for p in path) if stroke else 0
    waste=(len(path)-hit) if stroke and targets else 0
    end=path[-1] if path else world.cursor
    dist=lambda at: min((abs(at[0]-p[0])+abs(at[1]-p[1]) for p in targets),default=0)
    d0,d1=dist(world.cursor),dist(end)
    near=sum(abs(end[0]-p[0])+abs(end[1]-p[1])<=3 for p in targets)
    g=(hit/4,waste/4,d1/48,(d0-d1)/4,float(end in targets),
       near/16,float(view=='full' and not targets),
       float(name.startswith(('LIFT_', 'RETRACE_'))))
    return feature_map(world,name,path,nonlinear=True)+g

class Model:
    def __init__(self):self.weights=[0.0]*len(FEATURE_NAMES)
    def distribution(self,vectors):
        scores=[sum(a*b for a,b in zip(v,self.weights)) for v in vectors]
        top=max(scores)
        exps=[math.exp(max(-40.,x-top)) for x in scores]
        total=sum(exps)
        return [x/total for x in exps]
    def select(self,world,view,rng):
        menu,token=world.offer()
        names=list(menu)
        vectors=[vector(world,n,menu[n],view) for n in names]
        prob=self.distribution(vectors)
        threshold=rng.random(); accum=0.;index=len(names)-1
        for i,p in enumerate(prob):
            accum+=p
            if threshold<=accum:index=i;break
        return names[index],token,names,vectors,prob

def teacher(world, menu):
    # Privileged teacher supervises training; these features never bypass
    # host menus or grant a learner write permission.
    if not world.goal-world.own:return 'STOP'
    options=[]
    for name,path in menu.items():
        if name=='STOP':continue
        g=vector(world,name,path,'full')[-len(OBS_FEATURES):]
        hit,waste,after,delta,_,near,_,_=g
        score=8*hit-15*waste+delta-1.2*after+.3*near
        if name.startswith(('LIFT_', 'RETRACE_')):score-=.09*len(path)
        if name=='SAVE_ANCHOR':score-=3
        options.append((score,hit,len(path),name))
    return max(options)[-1] if options else 'STOP'

def train(arm,view,worlds=32):
    if arm not in ARMS or view not in TRAIN_VIEWS or type(worlds)!=int or not 4<=worlds<=96:
        raise ValueError('bounded training required')
    model=Model();updates=0;expert_followed=0
    for i in range(worlds):
        seed=TRAIN_BASE+i;family=FAMILIES[i%len(FAMILIES)];kind=KINDS[i%2]
        w=GoalWorld(seed,family,arm,kind,f'gal009-train:{arm}:{view}:{i}')
        rng=random.Random(f'gal009-policy:{arm}:{view}:{seed}')
        mixing=random.Random(f'gal009-mix:{arm}:{view}:{seed}')
        while not w.stopped and w.decisions<DECISION_CAP and len(w.own)<INK_BUDGET and w.goal-w.own:
            action,token,names,vecs,probs=model.select(w,view,rng)
            expert=teacher(w,w.legal());ix=names.index(expert)
            expected=[sum(p*v[k] for p,v in zip(probs,vecs)) for k in range(len(model.weights))]
            for k in range(len(model.weights)):
                model.weights[k]=max(-8.,min(8.,model.weights[k]+.045*(vecs[ix][k]-expected[k])))
            chosen=expert if mixing.random()<(.8 if i<worlds//2 else .35) else action
            w.apply(chosen,token,original_response=f'gal009-train:{chosen}')
            expert_followed+=chosen==expert;updates+=1
    return model,{'worlds':worlds,'updates':updates,'expert_followed':expert_followed}

def eval_episode(model,arm,train_view,view,seed,kind):
    fam=FAMILIES[(seed-TEST_BASE)%len(FAMILIES)]
    w=GoalWorld(seed,fam,arm,kind,f'gal009-eval:{arm}:{train_view}:{view}:{kind}:{seed}')
    rng=random.Random(f'gal009-matched:{arm}:{seed}:{kind}')
    while not w.stopped and w.decisions<DECISION_CAP and len(w.own)<INK_BUDGET and w.goal-w.own:
        action,token,*_=model.select(w,view,rng)
        w.apply(action,token,original_response=f'gal009-eval:{action}')
    if len(w.own)>INK_BUDGET or any(w.material[w.idx(p)]!=w.original[w.idx(p)] for p in w.protected):
        raise AssertionError('host authority violation')
    correct=len(w.goal&w.own)
    row={'arm':arm,'train_view':train_view,'view':view,'kind':kind,'seed':seed,'family':fam,
         'world_sha256':sha(w.original),'goal_sha256':sha(sorted(w.goal)),
         'material_sha256':sha(w.material),'receipts_sha256':sha(w.trace),
         'goal_retries':w.attempt,'correct':correct,'waste':len(w.own)-correct,
         'complete':correct==GOAL_SIZE,'ink':len(w.own),'decisions':w.decisions,
         'travel_decisions':sum(not t['new_ink'] and not t['STOP'] for t in w.trace),
         'retrace_decisions':sum(t['action_id']=='RETRACE_1' for t in w.trace),
         'stop_early':w.stopped and correct<GOAL_SIZE,
         'native_receipts':0,'resident_slms':0,'evidence':'external_synthetic_only'}
    return w,row

def svg_goal(path,world):
    """Original synthetic objective/material overlay; NEVER a Godot capture."""
    from lab import W,H,INK
    square=9
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W*square}" height="{H*square}" viewBox="0 0 {W*square} {H*square}">',
           f'<rect width="100%" height="100%" fill="#151a23"/>']
    goal=set(world.goal)
    for y in range(H):
        for x in range(W):
            pos=(x,y);material=world.material[world.idx(pos)]
            if pos in goal:
                lines.append(f'<rect x="{x*square+1}" y="{y*square+1}" width="7" height="7" fill="#151a23" stroke="#51b9dd" stroke-width="1.5"/>')
            if material:
                color=('#74cf85' if pos in goal else '#f0a14c') if material==INK else '#b74339' if material==5 else '#505a6d'
                lines.append(f'<rect x="{x*square}" y="{y*square}" width="9" height="9" fill="{color}"/>')
    x,y=world.cursor
    lines.append(f'<circle cx="{(x+.5)*square}" cy="{(y+.5)*square}" r="3.4" fill="none" stroke="#ffffff" stroke-width="1.6"/>')
    lines.append('</svg>')
    Path(path).write_text(''.join(lines)+'\n')

def aggregate(rows):
    return {'n':len(rows),'complete':sum(r['complete'] for r in rows),
            'mean_coverage':round(statistics.mean(r['correct']/GOAL_SIZE for r in rows),4),
            'mean_waste':round(statistics.mean(r['waste'] for r in rows),3),
            'mean_decisions':round(statistics.mean(r['decisions'] for r in rows),3),
            'mean_travel':round(statistics.mean(r['travel_decisions'] for r in rows),3),
            'mean_retrace':round(statistics.mean(r['retrace_decisions'] for r in rows),3),
            'early_STOP':sum(r['stop_early'] for r in rows)}

def paired(full,limited):
    assert len(full)==len(limited)
    diffs=[];f_only=l_only=0
    for a,b in zip(full,limited):
        if (a['seed'],a['world_sha256'],a['goal_sha256']) != (b['seed'],b['world_sha256'],b['goal_sha256']):
            raise ValueError('unmatched world/goal custody')
        diffs.append(a['correct']-b['correct'])
        f_only+=int(a['complete'] and not b['complete'])
        l_only+=int(b['complete'] and not a['complete'])
    return {'mean_ink_coverage_advantage_full':round(statistics.mean(diffs)/GOAL_SIZE,4),
            'full_only_completions':f_only,'limited_only_completions':l_only}

def run(output,train_worlds=32,heldout=16,gallery=2):
    if type(heldout)!=int or not 4<=heldout<=60 or not 0<=gallery<=3:
        raise ValueError('bounded evaluation required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    (root/'checkpoints').mkdir(exist_ok=True)
    rows={};traces={};checkpoints={};training={}
    for arm in ARMS:
        for train_view in TRAIN_VIEWS:
            model,record=train(arm,train_view,train_worlds)
            key=f'{arm}:{train_view}'
            training[key]=record
            cp={'schema':'gal009-linear-distill/1','arm':arm,'train_view':train_view,
                'features':FEATURE_NAMES,'weights':model.weights,'training':record,
                'privileged_teacher_at_training_only':True,'external_synthetic_only':True}
            path=root/'checkpoints'/f'{arm}_{train_view}.json'
            path.write_text(json.dumps(cp,indent=2,sort_keys=True)+'\n')
            checkpoints[key]=digest(path)
            for kind in KINDS:
                for view in EVAL_VIEWS:
                    name=f'{arm}:{train_view}:{kind}:{view}'
                    rows[name]=[];traces[name]=[]
                    for j in range(heldout):
                        seed=TEST_BASE+j
                        w,r=eval_episode(model,arm,train_view,view,seed,kind)
                        rows[name].append(r);traces[name].append(w.trace)
                        if j<gallery and kind=='tree' and view in ('full','local'):
                            path=root/'gallery';path.mkdir(exist_ok=True)
                            png(path/f'{arm}-{train_view}-{view}-{seed}.png',w)
                            svg_goal(path/f'{arm}-{train_view}-{view}-{seed}.svg',w)
    (root/'episodes.json').write_text(json.dumps(rows,separators=(',',':'),sort_keys=True)+'\n')
    (root/'receipts.json').write_text(json.dumps(traces,separators=(',',':'),sort_keys=True)+'\n')
    summaries={k:aggregate(v) for k,v in rows.items()}
    comparisons={f'{arm}:{tr}:{kind}:{view}':paired(rows[f'{arm}:{tr}:{kind}:full'],rows[f'{arm}:{tr}:{kind}:{view}'])
                 for arm in ARMS for tr in TRAIN_VIEWS for kind in KINDS for view in ('local','blind')}
    result={'schema':SCHEMA,'source_sha256':{p:digest(Path(__file__).with_name(p)) for p in SRC},
            'checkpoint_sha256':checkpoints,'training':training,'train_seed_range':[TRAIN_BASE,TRAIN_BASE+train_worlds-1],
            'eval_seed_range':[TEST_BASE,TEST_BASE+heldout-1],
            'goal_size':GOAL_SIZE,'ink_cap':INK_BUDGET,'STOP_unmasked':True,
            'source_of_authority':'external_synthetic_host','native_receipts':0,'resident_slms':0,
            'episodes_sha256':digest(root/'episodes.json'),'receipts_sha256':digest(root/'receipts.json'),
            'summary':summaries,'comparisons':comparisons}
    (root/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (root/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n'
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return result

def audit(root):
    root=Path(root);metrics=json.loads((root/'metrics.json').read_text())
    if metrics['schema']!=SCHEMA or metrics['source_sha256']!={p:digest(Path(__file__).with_name(p)) for p in SRC}:
        raise ValueError('schema/source drift')
    manifest={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,p=line.split('  ',1)
        if p in manifest or Path(p).is_absolute() or '..' in Path(p).parts or not (root/p).is_file() or digest(root/p)!=h:
            raise ValueError('manifest drift')
        manifest[p]=h
    files={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    if files != set(manifest):raise ValueError('unmanifested artifacts')
    if digest(root/'episodes.json')!=metrics['episodes_sha256'] or digest(root/'receipts.json')!=metrics['receipts_sha256']:
        raise ValueError('evidence drift')
    episodes=json.loads((root/'episodes.json').read_text())
    receipts=json.loads((root/'receipts.json').read_text())
    train_worlds=metrics['train_seed_range'][1]-TRAIN_BASE+1
    heldout=metrics['eval_seed_range'][1]-TEST_BASE+1
    if not 4<=train_worlds<=96 or not 4<=heldout<=60:raise ValueError('unbounded or shifted seeds')
    # Distinct authority-path replay: consume *recorded* action IDs and their
    # original outputs without running any policy or teacher. World offers and
    # material transitions must independently reproduce every receipt.
    host_receipts=0
    for key,episode_rows in episodes.items():
        arm,tr,kind,view=key.split(':')
        for row,raw_trace in zip(episode_rows,receipts[key]):
            seed=row['seed'];fam=row['family']
            lease=f'gal009-eval:{arm}:{tr}:{view}:{kind}:{seed}'
            world=GoalWorld(seed,fam,arm,kind,lease)
            for item in raw_trace:
                menu,token=world.offer()
                if item['action_id'] not in menu or token!=item['offered_sha256']:
                    raise ValueError('illegal original action or stale offer')
                actual=world.apply(item['action_id'],token,original_response=item['original_response'])
                if sha(actual)!=sha(item):
                    raise ValueError('receipt/material mismatch')
                host_receipts+=1
            if sha(world.trace)!=row['receipts_sha256'] or sha(world.material)!=row['material_sha256']:
                raise ValueError('final material mismatch')
            if any(world.material[world.idx(p)]!=world.original[world.idx(p)] for p in world.protected):
                raise ValueError('protected material changed')
    count=decisions=0;seen=set()
    for arm in ARMS:
        for tr in TRAIN_VIEWS:
            model,record=train(arm,tr,train_worlds)
            k=f'{arm}:{tr}'
            cp={'schema':'gal009-linear-distill/1','arm':arm,'train_view':tr,
                'features':FEATURE_NAMES,'weights':model.weights,'training':record,
                'privileged_teacher_at_training_only':True,'external_synthetic_only':True}
            path=root/'checkpoints'/f'{arm}_{tr}.json'
            if path.read_text()!=json.dumps(cp,indent=2,sort_keys=True)+'\n' or digest(path)!=metrics['checkpoint_sha256'][k]:
                raise ValueError('checkpoint mismatch')
            for kind in KINDS:
                for view in EVAL_VIEWS:
                    key=f'{arm}:{tr}:{kind}:{view}';seen.add(key)
                    if len(episodes.get(key,[]))!=heldout or len(receipts.get(key,[]))!=heldout:raise ValueError('episode count')
                    for j in range(heldout):
                        world,row=eval_episode(model,arm,tr,view,TEST_BASE+j,kind)
                        if row!=episodes[key][j] or sha(world.trace)!=sha(receipts[key][j]):
                            raise ValueError('material/model replay mismatch')
                        count+=1;decisions+=world.decisions
                    if aggregate(episodes[key])!=metrics['summary'][key]:raise ValueError('summary mismatch')
    if set(episodes)!=seen or set(receipts)!=seen:raise ValueError('unknown panel')
    return {'checkpoints_retrained':len(metrics['checkpoint_sha256']),
            'episodes_replayed':count,'decisions_replayed':decisions,'host_receipts_verified':host_receipts,
            'native_receipts':0,'resident_slms':0}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--train',type=int,default=32)
    parser.add_argument('--eval',type=int,default=16)
    parser.add_argument('--gallery',type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.train,args.eval,args.gallery)['summary'],indent=2))
