"""GAL-014: task-conditional starting-cell semantics in a host-legal material corridor.

External synthetic experiment. The model sees a public one-bit objective descriptor,
leased host legal actions, radius-four material observations and accepted receipts;
it does not receive the hidden corridor. No game/runtime writes or native claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter
from lab import START, World, INK, DECISION_CAP, sha, png, HostRefusal
from material_corridors import CorridorWorld, visible_material
from frontier_memory import Ledger, Scorer, features, choose_teacher

SCHEMA='gal-014-start-semantics/1'
FEATURES=('hit_start','hit_required_start','hit_forbidden_start')
TRAIN_BASE=2030000
EVAL_BASE=2130000

class SemanticsWorld(CorridorWorld):
    """Host changes only in the *external harness*: exact ink budget matches objective."""
    def __init__(self,seed,kind,start_required,lease):
        super().__init__(seed,kind,'lift',lease)
        self.start_required=bool(start_required)
        self.goal=self.corridor if self.start_required else self.corridor-{START}
        self.target=len(self.goal)
    def legal(self):
        offered=World.legal(self)
        budget=self.target-len(self.own)
        return {name:path for name,path in offered.items()
                if not name.startswith(('INK_','JOG_')) or len(path)<=budget}

class GoalLedger(Ledger):
    def __init__(self,start_required):
        super().__init__();self.start_required=bool(start_required)
    def targets(self):
        return {p for p,m in self.known.items()
                if m==0 and (p!=START or self.start_required)}

class Adapted(Scorer):
    def __init__(self,base_weights):
        super().__init__()
        assert len(base_weights)==len(self.weights)
        self.weights=list(base_weights)
        self.adapter=[0.]*len(FEATURES)
    def vector(self,w,ledger,name,path,route):
        hit=bool(START in path and name.startswith(('INK_','JOG_')))
        return features(w,ledger,name,path,route), (
            float(hit),float(hit and w.start_required),float(hit and not w.start_required))
    def logits(self,vectors):
        return [sum(a*b for a,b in zip(base,self.weights))+
                sum(a*b for a,b in zip(extra,self.adapter)) for base,extra in vectors]
    def probability(self,vectors):
        scores=self.logits(vectors)
        mx=max(scores);exp=[math.exp(max(-40,s-mx)) for s in scores];z=sum(exp)
        return [v/z for v in exp]

def policy_action(w,ledger,model,decoder='greedy',guard='none',rng=None):
    """Task-aware suppression is model-side; offers remain host-authored and intact."""
    if guard not in ('none','always','conditional'):raise ValueError('guard')
    menu,token=w.offer()
    eligible=[name for name,path in menu.items()
              if not (guard=='always' or (guard=='conditional' and not w.start_required))
              or not (name.startswith(('INK_','JOG_')) and START in path)]
    if not eligible:raise AssertionError('STOP must remain available')
    route=ledger.route(w.cursor)
    vectors=[model.vector(w,ledger,n,menu[n],route) for n in eligible]
    scores=model.logits(vectors)
    if decoder=='greedy':
        # Stable lexical tie break and no hidden host-state access.
        chosen=eligible[max(range(len(eligible)),key=lambda i:(scores[i],-i))]
    elif decoder=='sample':
        if rng is None:raise ValueError('rng required')
        ps=model.probability(vectors);u=rng.random();acc=0.;chosen=eligible[-1]
        for name,p in zip(eligible,ps):
            acc+=p
            if u<=acc:chosen=name;break
    else:raise ValueError('decoder')
    return chosen,menu,token

def teacher_action(w,ledger,menu):
    """Observation-led teacher: never inspect w.goal or w.corridor."""
    paint=[]
    for name,path in menu.items():
        if name.startswith(('INK_','JOG_')) and (w.start_required or START not in path):
            new=sum(p not in ledger.recorded_paints for p in path)
            novelty=sum(p not in ledger.visited for p in path)
            paint.append((new,novelty,len(path),name))
    if paint:return max(paint)[-1]
    route=ledger.route(w.cursor)
    if not route:return 'STOP'
    ranked=[]
    for name,path in menu.items():
        if not name.startswith(('LIFT_','RETRACE_')) or not path:continue
        align=0
        for a,b in zip(path,route):
            if a!=b:break
            align+=1
        if align and align==len(path) and len(path)<len(route):
            ranked.append((align,len(path),name))
    return max(ranked)[-1] if ranked else 'STOP'

def episode(seed,kind,start_required,model,policy,train=False,mix=0,lr=.04,save_trace=True):
    if policy not in ('raw','always','conditional','adapted','teacher','adapted_sample'):
        raise ValueError('policy')
    lease=f'gal014:{"train" if train else "eval"}:{kind}:{seed}:{int(start_required)}:{policy}'
    w=SemanticsWorld(seed,kind,start_required,lease)
    ledger=GoalLedger(start_required)
    decoder='sample' if policy=='adapted_sample' else 'greedy'
    guard='always' if policy=='always' else 'conditional' if policy in ('conditional','adapted','adapted_sample') else 'none'
    rng=random.Random(f'gal014-choice:{seed}:{kind}:{int(start_required)}')
    mixrng=random.Random(f'gal014-rollin:{seed}:{kind}:{int(start_required)}')
    updates=0
    while not w.stopped and w.decisions<DECISION_CAP and w.goal-w.own:
        ledger.observe(w)
        menu,token=w.offer()
        expert=teacher_action(w,ledger,menu)
        if policy=='teacher':choice=expert
        else:
            choice,_,_=policy_action(w,ledger,model,decoder,guard,rng)
            if train:
                names=list(menu);route=ledger.route(w.cursor)
                vectors=[model.vector(w,ledger,n,menu[n],route) for n in names]
                probs=model.probability(vectors)
                j=names.index(expert)
                expected=[sum(p*vec[1][i] for p,vec in zip(probs,vectors))
                          for i in range(len(FEATURES))]
                for i in range(len(FEATURES)):
                    model.adapter[i]=max(-10.,min(10.,model.adapter[i]+lr*(vectors[j][1][i]-expected[i])))
                updates+=1
                if mixrng.random()<mix:choice=expert
        receipt=w.apply(choice,token,original_response=f'gal014:{policy}:{choice}')
        ledger.accept(choice,menu[choice],receipt)
    correct=len(w.goal&w.own)
    if w.protected&w.own:raise AssertionError('protected write')
    if any(w.material[w.idx(p)]!=w.original[w.idx(p)] for p in w.protected):
        raise AssertionError('protected raster change')
    row=dict(seed=seed,kind=kind,start_required=start_required,policy=policy,
             complete=(correct==w.target),correct=correct,target=w.target,
             coverage=correct/w.target,off_target=len(w.own-w.goal),
             ink=len(w.own),decisions=w.decisions,
             ink_per_decision=len(w.own)/max(1,w.decisions),
             start_painted=START in w.own,updates=updates,
             receipts_sha256=sha(w.trace),material_sha256=sha(w.material),
             original_sha256=sha(w.original))
    return w,row

def train(base,worlds=80):
    assert worlds>=4 and worlds<=240 and worlds%4==0
    m=Adapted(base);total=0
    for j in range(worlds):
        kind=('tree','path')[j%2]
        required=bool((j//2)%2)
        seed=TRAIN_BASE+j
        _,r=episode(seed,kind,required,m,'adapted',train=True,mix=.9 if j<worlds//2 else .55)
        total+=r['updates']
    return m,{'worlds':worlds,'optimizer_updates':total,'adapter_weights':m.adapter}

def audit_row(row,trace):
    w=SemanticsWorld(row['seed'],row['kind'],row['start_required'],
              f'gal014:eval:{row["kind"]}:{row["seed"]}:{int(row["start_required"])}:{row["policy"]}')
    for receipt in trace:
        menu,token=w.offer();name=receipt['action_id']
        assert name in menu and token==receipt['offered_sha256']
        again=w.apply(name,token,original_response=receipt['original_response'])
        assert again==receipt
    assert sha(w.trace)==row['receipts_sha256']
    assert sha(w.material)==row['material_sha256']
    assert len(w.own-w.goal)==row['off_target']
    return len(trace)

def run(output,ckpt,train_worlds=80,eval_seeds=24):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    cp=json.loads(Path(ckpt).read_text());base=cp['weights']
    model,stats=train(base,train_worlds)
    frozen=Adapted(base)
    (root/'checkpoint.json').write_text(json.dumps(dict(schema=SCHEMA,base_sha256=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
         adapter=model.adapter,training=stats),indent=2)+'\n')
    rows=[];receipts=[];replayed=0
    policies=('raw','always','conditional','adapted','adapted_sample','teacher')
    for j in range(eval_seeds):
        seed=EVAL_BASE+j;kind=('tree','path')[j%2]
        for required in (False,True):
            for policy in policies:
                candidate=frozen if policy in ('raw','always','conditional') else model
                w,r=episode(seed,kind,required,candidate,policy)
                replayed+=audit_row(r,w.trace)
                rows.append(r)
                receipts.append(dict(seed=seed,kind=kind,required=required,policy=policy,actions=w.trace))
    (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'receipts.json').write_text(json.dumps(receipts,separators=(',',':'))+'\n')
    summary={p:{'start_not_goal':summarise([r for r in rows if r['policy']==p and not r['start_required']]),
                'start_is_goal':summarise([r for r in rows if r['policy']==p and r['start_required']])}
             for p in policies}
    meta=dict(schema=SCHEMA,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_checkpoint_sha256=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
        adapter_checkpoint_sha256=hashlib.sha256((root/'checkpoint.json').read_bytes()).hexdigest(),
        train=stats,heldout_seeds=[EVAL_BASE,EVAL_BASE+eval_seeds-1],
        heldout_episodes=len(rows),audited_receipts=replayed,summary=summary,
        native_receipts=0,game_code_edited=False)
    (root/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    return meta

def summarise(rows):
    return dict(n=len(rows),completed=sum(r['complete'] for r in rows),
                mean_coverage=round(sum(r['coverage'] for r in rows)/max(1,len(rows)),4),
                off_target=sum(r['off_target'] for r in rows),
                mean_decisions=round(sum(r['decisions'] for r in rows)/max(1,len(rows)),2),
                start_painted=sum(r['start_painted'] for r in rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--train-worlds',type=int,default=80)
    parser.add_argument('--eval-seeds',type=int,default=24)
    a=parser.parse_args();run(a.out,a.checkpoint,a.train_worlds,a.eval_seeds)
