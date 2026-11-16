"""GAL-008: goal-conditioned, host-legal geometric intent versus action vocabulary.

External 48x48 synthetic model-side experiment ONLY. Host offers, leases, material
writes and protected-cell decisions are delegated unchanged to GAL-001 World.
Goal masks are constructed solely from initially host-clear pixels and are
observation inputs, not authorisation to paint. No game code modifications.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from lab import DECISION_CAP, DIRS, START, World, FEATURES, png, sha
from candidate_features import CandidateWorld, FAMILIES
from interaction_ablation import feature_map
from shape_metrics import morphology

SCHEMA='gal-008-spatial-goal/1'
TRAIN_BASE=610000
TEST_BASE=710000
GOAL_SIZE=32
INK_BUDGET=40
ARMS=('cell','sweep','lift','retrace','anchor','jog')
KINDS=('path','tree')
GOAL_FEATURES=('goal_hit','off_goal','goal_after','goal_delta','goal_endpoint',
               'goal_near','goal_done','travel')
MODEL_FEATURES=FEATURES+('near_x_span', 'clear_x_span', 'near_x_turn', 'clear_x_turn',
    'near_x_clear', 'near_x_visited', 'clear_x_persist', 'near_x_straight')+GOAL_FEATURES
SRC=('lab.py','candidate_features.py','interaction_ablation.py','shape_metrics.py','goal_intent.py','test_goal_intent.py')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def goals(seed,family,kind,attempt=0):
    """32 initially clear, connected drawable pixels, with no label leakage.

    The sampled path/tree is created before any policy action. An unreachable
    start or inability to grow the full mask is an explicit refusal.
    """
    if kind not in KINDS:raise ValueError('unknown goal morphology')
    base=World(seed,family,'cell')
    rng=random.Random(f'gal008-goal:{seed}:{family}:{kind}:{attempt}')
    occupied={START}; points=[]; current=START; previous=None
    for i in range(GOAL_SIZE):
        if kind=='path':
            candidates=[(current[0]+dx,current[1]+dy) for dx,dy in DIRS.values()
                        if base.clear((current[0]+dx,current[1]+dy)) and
                        (current[0]+dx,current[1]+dy) not in occupied]
            if not candidates:
                # Bounded reproducible restart from an earlier end (not a claimed oracle success).
                return None
            if previous is not None and previous in candidates and rng.random()<.72:
                nxt=previous
            else:nxt=rng.choice(candidates)
            previous=(nxt[0]+(nxt[0]-current[0]),nxt[1]+(nxt[1]-current[1]))
            current=nxt
        else:
            frontier=[]
            for p in occupied:
                for dx,dy in DIRS.values():
                    q=(p[0]+dx,p[1]+dy)
                    if base.clear(q) and q not in occupied:
                        frontier.append((p,q))
            if not frontier:return None
            # Prefer a broad, branched tree while keeping a path to every pixel.
            parent,current=rng.choice(frontier)
        occupied.add(current);points.append(current)
    assert len(points)==GOAL_SIZE and not set(points)&base.protected
    return frozenset(points)

def goal_for(seed,family,kind):
    # Preserve seed isolation and reproducible goal families; don't skip failures silently.
    for attempt in range(16):
        out=goals(seed,family,kind,attempt)
        if out is not None:return out,attempt
    raise ValueError('could not construct legal goal')

class GoalWorld(CandidateWorld):
    def __init__(self,seed,family,arm,kind,*,lease):
        super().__init__(seed,family,arm,'candidate',lease=lease)
        self.goal,self.goal_attempt=goal_for(seed,family,kind)
        self.kind=kind
        self.goal_sha256=sha(sorted(self.goal))

    def legal(self):
        actions=super().legal()
        remaining=INK_BUDGET-len(self.own)
        return {k:path for k,path in actions.items()
                if not k.startswith(('INK_','JOG_')) or len(path)<=remaining}


def candidate(world,name,path):
    before=world.goal-world.own
    pset=set(path)
    hit=len(pset & before) if name.startswith(('INK_','JOG_')) else 0
    off=len(pset - world.goal) if name.startswith(('INK_','JOG_')) else 0
    endpoint=path[-1] if path else world.cursor
    def dist(loc):
        return min((abs(loc[0]-x)+abs(loc[1]-y) for x,y in before),default=0)
    d0=dist(world.cursor);d1=dist(endpoint)
    nearby=sum(abs(endpoint[0]-x)+abs(endpoint[1]-y)<=3 for x,y in before)
    vals=(hit/4,off/4,d1/48,(d0-d1)/4,
          float(endpoint in before),nearby/16,float(not before),
          float(name.startswith(('LIFT_','RETRACE_'))))
    return feature_map(world,name,path,nonlinear=True)+vals

class IntentPolicy:
    def __init__(self):self.weights=[0.0]*len(MODEL_FEATURES)
    def probabilities(self,vectors):
        scores=[sum(w*x for w,x in zip(self.weights,v)) for v in vectors]
        m=max(scores); ex=[math.exp(max(-40.0,s-m)) for s in scores]
        z=sum(ex);return [p/z for p in ex]

    def choose(self,world,rng):
        menu,token=world.offer();names=list(menu)
        vectors=[candidate(world,a,menu[a]) for a in names]
        probs=self.probabilities(vectors)
        u=rng.random();s=0.0;pick=len(names)-1
        for i,p in enumerate(probs):
            s+=p
            if u<=s:pick=i;break
        return names[pick],token,names,vectors,probs,pick


def oracle(world,menu):
    if not (world.goal-world.own):return 'STOP'
    candidates=[]
    for name,path in menu.items():
        if name=='STOP':continue
        vals=candidate(world,name,path)
        hit,off,after,delta,endpoint,near,done,travel=vals[-len(GOAL_FEATURES):]
        # Favor target-only painting, and nonpainting travel when stranded;
        # prohibit the teacher from treating goals as legal-action authority.
        score=8*hit-15*off+1.0*delta-1.2*after+.3*near
        if name.startswith(('LIFT_','RETRACE_')):score-=.09*len(path)
        if name=='SAVE_ANCHOR':score-=3
        candidates.append((score,hit,len(path),name))
    if not candidates:return 'STOP'
    # Avoid painting arbitrary off-goal paths when entirely unable to progress.
    return max(candidates)[-1]


def train(arm,episodes=60):
    if arm not in ARMS or type(episodes)!=int or not 4<=episodes<=120:
        raise ValueError('bounded training only')
    model=IntentPolicy();steps=teacher_steps=0
    for j in range(episodes):
        seed=TRAIN_BASE+j;family=FAMILIES[j%len(FAMILIES)];kind=KINDS[j%2]
        w=GoalWorld(seed,family,arm,kind,lease=f'gal008-train:{arm}:{j}')
        rng=random.Random(f'gal008-train-model:{arm}:{seed}')
        teacher_rng=random.Random(f'gal008-train-mix:{arm}:{seed}')
        while not w.stopped and w.decisions<DECISION_CAP and len(w.own)<INK_BUDGET and w.goal-w.own:
            action,token,names,vectors,probs,pick=model.choose(w,rng)
            expert=oracle(w,w.legal()); expert_index=names.index(expert)
            # Online legal-candidate cross-entropy imitation. Gradient is
            # measured on the exact host menu, before applying any action.
            expected=[sum(p*v[k] for p,v in zip(probs,vectors)) for k in range(len(model.weights))]
            for k in range(len(model.weights)):
                model.weights[k]=max(-8.,min(8.,model.weights[k]+.045*(vectors[expert_index][k]-expected[k])))
            # Dataset aggregation: late epochs use more learner trajectories.
            selected=expert if teacher_rng.random() < (.8 if j<episodes//2 else .35) else action
            w.apply(selected,token,original_response=f'gal008-training-output:{selected}')
            steps+=1; teacher_steps+=selected==expert
    return model,{'optimizer_updates':steps,'teacher_followed':teacher_steps,'train_seed_range':[TRAIN_BASE,TRAIN_BASE+episodes-1]}


def eval_episode(model,arm,seed,kind,variant):
    family=FAMILIES[(seed-TEST_BASE)%len(FAMILIES)]
    w=GoalWorld(seed,family,arm,kind,lease=f'gal008-eval:{arm}:{kind}:{seed}:{variant}')
    rng=random.Random(f'gal008-eval-paired:{arm}:{seed}:{kind}')
    while not w.stopped and w.decisions<DECISION_CAP and len(w.own)<INK_BUDGET and (w.goal-w.own):
        menu,token=w.offer()
        if variant=='oracle':action=oracle(w,menu)
        elif variant=='untrained':action,_,_,_,_,_=IntentPolicy().choose(w,rng)
        else:action,_,_,_,_,_=model.choose(w,rng)
        w.apply(action,token,original_response=f'gal008-{variant}-original:{action}')
    if len(w.own)>INK_BUDGET or any(w.material[w.idx(p)]!=w.original[w.idx(p)] for p in w.protected):
        raise AssertionError('host authority violated')
    correct=len(w.own & w.goal)
    row={'arm':arm,'variant':variant,'kind':kind,'seed':seed,'family':family,
         'initial_sha256':sha(w.original),'goal_sha256':w.goal_sha256,
         'goal_retries':w.goal_attempt,'material_sha256':sha(w.material),
         'receipts_sha256':sha(w.trace),'correct':correct,'goal_size':GOAL_SIZE,
         'ink':len(w.own),'waste':len(w.own-w.goal),'decisions':w.decisions,
         'logical_moves':w.logical_moves,'complete':correct==GOAL_SIZE,
         'selected_stop':w.stopped,'shape':morphology(w.own,w.trace),
         'native_receipts':0,'resident_slms':0,'evidence':'synthetic_only'}
    return w,row


def summary(rows):
    return {'episodes':len(rows),'complete':sum(x['complete'] for x in rows),
            'mean_target_coverage':round(statistics.mean(x['correct']/GOAL_SIZE for x in rows),4),
            'mean_correct_ink':round(statistics.mean(x['correct'] for x in rows),3),
            'mean_waste':round(statistics.mean(x['waste'] for x in rows),3),
            'mean_decisions':round(statistics.mean(x['decisions'] for x in rows),3),
            'target_ink_per_decision':round(sum(x['correct'] for x in rows)/max(1,sum(x['decisions'] for x in rows)),4),
            'early_stop':sum(x['selected_stop'] and not x['complete'] for x in rows)}


def run(out,train_worlds=60,eval_worlds=20,gallery_worlds=2):
    if type(eval_worlds)!=int or not 4<=eval_worlds<=80 or not 0<=gallery_worlds<=3:
        raise ValueError('bounded evaluation only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    rows={};traces={};checks={}
    for arm in ARMS:
        model,training=train(arm,train_worlds)
        check={'schema':'gal008-hostlegal-goal-policy/1','arm':arm,'features':MODEL_FEATURES,
               'weights':model.weights,'training':training,'optimizer':'host-menu cross-entropy imitation',
               'not_native_or_SLM':True}
        p=out/'checkpoints'/f'{arm}.json'
        p.write_text(json.dumps(check,indent=2,sort_keys=True)+'\n');checks[arm]=digest(p)
        for kind in KINDS:
            for variant in ('trained','untrained','oracle'):
                key=f'{arm}:{kind}:{variant}';rows[key]=[];traces[key]=[]
                for j in range(eval_worlds):
                    seed=TEST_BASE+j
                    w,r=eval_episode(model,arm,seed,kind,variant)
                    rows[key].append(r);traces[key].append(w.trace)
                    if j<gallery_worlds and variant in ('trained','oracle'):
                        p=out/'gallery';p.mkdir(exist_ok=True)
                        png(p/f'{arm}-{kind}-{variant}-{seed}.png',w)
    (out/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'receipts.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    results={'schema':SCHEMA,'source_sha256':{f:digest(Path(__file__).with_name(f)) for f in SRC},
             'checkpoint_sha256':checks,'train_worlds_per_arm':train_worlds,
             'test_seed_range':[TEST_BASE,TEST_BASE+eval_worlds-1],
             'goal_size':GOAL_SIZE,'host_ink_cap':INK_BUDGET,'STOP_unmasked':True,
             'model_observation_only':True,'external_synthetic_only':True,
             'summary':{k:summary(v) for k,v in rows.items()},
             'episodes_sha256':digest(out/'episodes.json'),'receipts_sha256':digest(out/'receipts.json')}
    (out/'metrics.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(out).as_posix()}\n'
       for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return results

class AuditRefusal(ValueError):pass


def audit(output):
    """Re-train all six checkpoints and independently replay EVERY model action.

    This is an external synthetic audit and must not be represented as native
    PAINT, host process W0 or a resident model admission receipt.
    """
    root=Path(output)
    metrics=json.loads((root/'metrics.json').read_text())
    if metrics.get('schema')!=SCHEMA or metrics.get('source_sha256')!={
            f:digest(Path(__file__).with_name(f)) for f in SRC}:
        raise AuditRefusal('schema or source custody drift')
    manifest={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,path=line.split('  ',1)
        pp=Path(path)
        if pp.is_absolute() or '..' in pp.parts or path in manifest:
            raise AuditRefusal('unsafe/duplicate artifact path')
        manifest[path]=h
        if not (root/pp).is_file() or digest(root/pp)!=h:
            raise AuditRefusal('artifact hash mismatch: '+path)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*')
            if p.is_file() and p.name!='SHA256SUMS'}
    if set(manifest)!=actual:raise AuditRefusal('missing or undeclared files')
    if digest(root/'episodes.json')!=metrics['episodes_sha256'] or digest(root/'receipts.json')!=metrics['receipts_sha256']:
        raise AuditRefusal('evidence digest mismatch')
    rows=json.loads((root/'episodes.json').read_text())
    traces=json.loads((root/'receipts.json').read_text())
    n=metrics['train_worlds_per_arm']
    count=metrics['test_seed_range'][1]-TEST_BASE+1
    if type(n)!=int or not 4<=n<=120 or type(count)!=int or not 4<=count<=80:
        raise AuditRefusal('unbounded or changed seed budgets')
    visited=set();episodes=decisions=0;models={}
    for arm in ARMS:
        model,training=train(arm,n)
        cp={'schema':'gal008-hostlegal-goal-policy/1','arm':arm,'features':MODEL_FEATURES,
            'weights':model.weights,'training':training,
            'optimizer':'host-menu cross-entropy imitation','not_native_or_SLM':True}
        frozen=json.dumps(cp,indent=2,sort_keys=True)+'\n'
        path=root/'checkpoints'/f'{arm}.json'
        if path.read_text()!=frozen or digest(path)!=metrics['checkpoint_sha256'][arm]:
            raise AuditRefusal('model checkpoint changed: '+arm)
        models[arm]=model
        for kind in KINDS:
            for variant in ('trained','untrained','oracle'):
                key=f'{arm}:{kind}:{variant}';visited.add(key)
                if len(rows.get(key,[]))!=count or len(traces.get(key,[]))!=count:
                    raise AuditRefusal('episode count mismatch: '+key)
                for j in range(count):
                    w,r=eval_episode(model,arm,TEST_BASE+j,kind,variant)
                    if r!=rows[key][j] or sha(w.trace)!=sha(traces[key][j]):
                        raise AuditRefusal('model or material replay mismatch: '+key+' #'+str(j))
                    episodes+=1;decisions+=w.decisions
                if summary(rows[key])!=metrics['summary'][key]:
                    raise AuditRefusal('summary changed: '+key)
    if set(rows)!=visited or set(traces)!=visited or set(metrics['summary'])!=visited:
        raise AuditRefusal('undeclared or missing evaluation panel')
    return {'retrained_checkpoints':len(models),'replayed_episodes':episodes,
            'replayed_model_decisions':decisions,'native_receipts':0,
            'w0_executions':0,'resident_slms':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--train',type=int,default=60);p.add_argument('--eval',type=int,default=20)
    p.add_argument('--gallery',type=int,default=2)
    args=p.parse_args();print(json.dumps(run(args.output,args.train,args.eval,args.gallery)['summary'],indent=2))
