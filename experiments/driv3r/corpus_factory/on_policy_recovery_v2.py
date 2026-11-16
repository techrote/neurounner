#!/usr/bin/env python3
"""Source-pinned, synthetic TRAIN-only on-policy correction + sealed transfer evaluation.

Uses a CPU student to enter its own failure states, then a privileged geometric
teacher to label those observed states. No W0/native records or game writes.
Both world generator variants are external, synthetic, and research-only.
"""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import zipfile

from recovery_curriculum import (ACTIONS, BUDGET, DIRS, FAMILIES, apply,
                                 augmented_features, distance_field, fit,
                                 legal_state, make_world, observation, oracle,
                                 png, rollout, sha, teacher_data)

SCHEMA='cybersand/synthetic-on-policy-recovery-v2'
SOURCE_TRAIN=tuple(range(1000,1080))
CORRECTION_TRAIN=tuple(range(2200,2232))
EVAL_SEEDS=tuple(range(4000,4024))
CROSS_GENERATOR_SEEDS=tuple(range(6000,6024))
SOURCE_PINS={
 'recovery_curriculum.py':'fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4',
 'closed_loop_transfer_probe.py':'928537a3f02d227c51e36ba61815315d4a2c4ff7',
 'learnability_probe_scoped.py':'78d8c899e73d692ffe0540f6ac24dd0e173c38e0',
}


def verify_sources(directory):
    import subprocess
    for name,expected in SOURCE_PINS.items():
        actual=subprocess.run(['git','hash-object',str(directory/name)],check=True,capture_output=True,text=True).stdout.strip()
        if actual!=expected:raise ValueError(f'source baseline changed: {name} {actual}')


def make_cross_world(seed):
    """Independent held-out obstacle geometry, same 96x96 legal/observation contract.

    Deterministic alternating horizontal barriers with narrow staggered gates,
    sealed protected cells, and additional block islands; no fitting on this.
    """
    w=make_world(seed)
    rng=random.Random(864221+seed)
    blocked=set();protected=set()
    for k,y in enumerate((26,38,50,62,74)):
        lo,hi=17,89
        gate=18 + ((seed*7+k*23)%62)
        for x in range(lo,hi):
            if abs(x-gate)>2:blocked.add((x,y))
    for _ in range(8):
        x=rng.randrange(24,86);y=rng.randrange(11,86)
        for ix in range(x,x+rng.randint(2,5)):
            for iy in range(y,y+rng.randint(2,5)):
                if ix<96 and iy<96:blocked.add((ix,iy))
    for x in range(31,83,7):
        for y in range(17,81,5):
            if (x+y+seed)%3==0:protected.add((x,y))
    # Reserve start approach, deliberately do not reuse prior obstacle layout.
    safe={(x,y) for x in range(0,15) for y in range(42,55)}
    blocked-=safe;protected-=safe|blocked
    start=(7,48)
    reachable={start:0};q=deque([start]);walls=blocked|protected
    while q:
        x,y=q.popleft()
        for dx,dy in DIRS:
            p=(x+dx,y+dy)
            if 0<=p[0]<96 and 0<=p[1]<96 and p not in walls and p not in reachable:
                reachable[p]=reachable[(x,y)]+1;q.append(p)
    far=[p for p,d in reachable.items() if p[0]>65 and d>=55]
    if not far:
        # Deterministic resampling would bias evaluation; fail closed instead.
        raise ValueError(f'cross-generator seed {seed} has no reachable distal goal')
    goal=min(far,key=lambda p:(abs(p[0]-78)+abs(p[1]-48),p))
    materials={(x,y):('stone' if (x,y) in blocked else 'metal' if (x,y) in protected
            else 'sand' if (x//7+y//9+seed)%6<=2 else 'water' if (x+y+seed)%11<2 else 'empty')
            for x in range(96) for y in range(96)}
    return {'seed':seed,'generator':'cross_barrier_v2','goal':goal,'start':start,
            'obstacles':blocked,'protected':protected,'materials':materials,
            'shortest_goal':reachable[goal]}


_MODEL_CACHE={}
def decide_model(obs,vec,model):
    key=(id(vec),id(model))
    if key not in _MODEL_CACHE:
        weights={f:tuple(float(model.coef_[j,i]) for j in range(len(model.classes_)))
                 for f,i in vec.vocabulary_.items()}
        _MODEL_CACHE[key]=(weights,tuple(model.classes_),[float(x) for x in model.intercept_])
    weights,classes,intercept=_MODEL_CACHE[key]
    scores=intercept.copy()
    for feature,value in augmented_features(obs).items():
        terms=weights.get(feature)
        if terms is not None:
            for j,w in enumerate(terms):scores[j]+=w*value
    return max((a for a in obs['legal'] if a in classes),
               key=lambda a:(scores[classes.index(a)],-ACTIONS.index(a)))



def collect_corrections(seeds,vec,model,budget=160):
    """Student's own host-applied states, teacher label without teacher in prompt."""
    rows=[];stats=Counter();worlds=[]
    for seed in seeds:
        w=make_world(seed);dist=distance_field(w)
        fingerprint=sha(json.dumps(sorted(map(list,w['obstacles']|w['protected']))).encode())
        worlds.append({'seed':seed,'split':'TRAIN','generator':'cross-generator-v1',
                       'world_blocked_sha256':fingerprint})
        for family in FAMILIES:
            pos=w['start'];heading=0;visited={pos};history=[];past=[];counts=Counter()
            for tick in range(budget):
                moves,fresh,legal=legal_state(w,pos,heading,visited,history)
                obs=observation(w,pos,heading,visited,past,legal,tick,budget,family)
                teacher,negs=oracle(w,pos,heading,visited,history,dist,legal)
                student=decide_model(obs,vec,model)
                normalized=json.dumps({k:v for k,v in obs.items() if k!='family'},sort_keys=True,separators=(',',':'))
                state_fingerprint=sha(normalized.encode())
                counts[(pos,heading)]+=1
                state_repeated=counts[(pos,heading)]>1
                mismatch=student!=teacher
                # Important: retain all actions, not only losses, so legitimate
                # RETRACE choices and STOP-at-goal remain representable.
                rows.append({'observation':obs,'label':teacher,'seed':seed,
                             'world_group_id':f'onpolicy:{seed}',
                             'observable_sha256':state_fingerprint,
                             'student_action':student,'was_student_teacher_disagreement':mismatch,
                             'repeated_oriented_cell':state_repeated,
                             'negative_actions':negs,
                             'evidence_class':'synthetic-train-host-applied-on-policy-correction'})
                stats['observed']+=1
                stats['disagreements']+=int(mismatch)
                stats['repeated_states']+=int(state_repeated)
                stats['host_legal']+=int(student in legal)
                if student=='STOP':stats['early_stop']+=int(pos!=w['goal']);break
                pos,heading=apply(w,pos,heading,visited,history,student)
                visited.add(pos);past.append(student)
                if pos==w['goal']:stats['student_goal_arrivals']+=1
            stats['episodes']+=1
    return rows,worlds,dict(stats)


def dedup_corrections(rows,source_rows):
    keys=set();out=[];rejected=Counter()
    for obs,_,_,_ in source_rows:
        normalized=json.dumps({k:v for k,v in obs.items() if k!='family'},sort_keys=True,separators=(',',':'))
        keys.add(sha(normalized.encode()))
    for r in rows:
        key=r['observable_sha256']
        if key in keys:
            rejected['same_state_as_source_or_prior_correction']+=1
            continue
        keys.add(key);out.append(r)
    return out,dict(rejected)


def fit_student(rows,corrections,epochs=24):
    # Bounded single-epoch weight scaling; no evaluation-label mining.
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDClassifier
    combined=[(r[0],r[1]) for r in rows]+[(r['observation'],r['label']) for r in corrections]*2
    vec=DictVectorizer(sort=True);X=vec.fit_transform(augmented_features(o) for o,_ in combined)
    clf=SGDClassifier(loss='log_loss',alpha=.0001,max_iter=epochs,tol=None,
                      random_state=17,shuffle=True,learning_rate='optimal')
    clf.fit(X,[y for _,y in combined]);return vec,clf,{'epochs':int(clf.n_iter_),
            'samples_seen':int(clf.t_),'original_train_rows':len(rows),
            'deduplicated_on_policy_rows':len(corrections),'weighted_train_rows':len(combined),
            'feature_count':X.shape[1]}


def goal_terminal_rollout(w,family,vec,model,budget=BUDGET):
    """Host-applied, hard-stop at goal. Pre-goal cycle rates only."""
    pos=w['start']; heading=0; visited={pos}; history=[];past=[];path=[pos]
    counters=Counter();first_goal=None
    for tick in range(budget):
        moves,fresh,legal=legal_state(w,pos,heading,visited,history)
        obs=observation(w,pos,heading,visited,past,legal,tick,budget,family)
        action=decide_model(obs,vec,model)
        if action not in legal:raise AssertionError('unsafe student action')
        counters[action]+=1
        if action=='STOP':break
        pos,heading=apply(w,pos,heading,visited,history,action)
        if pos in w['protected'] or pos in w['obstacles']:raise AssertionError('protected or blocked')
        visited.add(pos);past.append(action);path.append(pos)
        if pos==w['goal']:
            first_goal=tick+1;break
    return {'goal_tick':first_goal,'turns':len(past),'unique_cells':len(visited),
            'retrace_steps':counters['RETRACE'],'actions':dict(sorted(counters.items())),
            'path':path}


def evaluate(seeds,make,policies,budget=BUDGET):
    records=[];worlds=[];examples=[]
    for seed in seeds:
        w=make(seed)
        worlds.append({'seed':seed,'generator':w.get('generator','cross-generator-v1'),
                       'goal':w['goal'],'shortest_goal':w['shortest_goal'],
                       'blocked_sha256':sha(json.dumps(sorted(map(list,w['obstacles']|w['protected']))).encode())})
        for family in FAMILIES:
            for name,(vec,model) in policies.items():
                roll=goal_terminal_rollout(w,family,vec,model,budget)
                roll.update({'policy':name,'seed':seed,'family':family})
                path=roll.pop('path')
                states=Counter(path)
                roll['revisited_positions']=sum(n-1 for n in states.values())
                roll['repeated_position_max']=max(states.values())
                roll['near_cycle_score']=sum(path[i]==path[i-2] for i in range(2,len(path)))
                records.append(roll)
                if seed in seeds[:2] and family=='greebles':
                    examples.append((f'{w.get("generator","original")}-{seed}-{name}.png',png(w,{'path':path})))
    result={}
    for name in policies:
        subset=[x for x in records if x['policy']==name]
        result[name]={'worlds':len(seeds),'family_rollouts':len(subset),
              'goals':sum(x['goal_tick'] is not None for x in subset),
              'independent_world_goal_successes':sum(any(x['goal_tick'] is not None for x in subset if x['seed']==s) for s in seeds),
              'mean_unique_cells':round(sum(x['unique_cells'] for x in subset)/len(subset),3),
              'mean_two_step_returns_before_goal':round(sum(x['near_cycle_score'] for x in subset)/len(subset),3),
              'mean_retrace_before_goal':round(sum(x['retrace_steps'] for x in subset)/len(subset),3),
              'mean_steps_to_goal_or_stop':round(sum(x['turns'] for x in subset)/len(subset),3)}
    return result,worlds,records,examples


def export_corrections(rows,worlds,path):
    if path.exists():raise FileExistsError(path)
    members={};lines=[];prov=[];negs=[]
    system=('Select one host-legal action for a synthetic 96x96 CyberSand-style worm. '
            'Goal-directed movement, immutable protected cells; answer one action token.')
    allowed_seeds={w['seed'] for w in worlds}
    for i,r in enumerate(rows):
        obs=r['observation'];label=r['label']
        if label not in obs['legal'] or r['student_action'] not in obs['legal']:raise ValueError('unsafe label')
        if r['seed'] not in allowed_seeds:raise ValueError('non-TRAIN seed')
        lines.append(json.dumps({'messages':[{'role':'system','content':system},
            {'role':'user','content':json.dumps(obs,sort_keys=True,separators=(',',':'))},
            {'role':'assistant','content':label}]},sort_keys=True,separators=(',',':')))
        prov.append(json.dumps({'index':i,'split':'TRAIN','world_group_id':r['world_group_id'],
            'observable_sha256':r['observable_sha256'],'teacher_label':label,
            'student_action':r['student_action'],'student_teacher_disagreement':r['was_student_teacher_disagreement'],
            'evidence_class':r['evidence_class'],'native':False,'model_generated_labels':False},sort_keys=True,separators=(',',':')))
        negs.append(json.dumps({'index':i,'teacher_action':label,'student_action':r['student_action'],
           'negative_actions':r['negative_actions']},sort_keys=True,separators=(',',':')))
    for name,content in (('train.messages.jsonl',lines),('train.provenance.jsonl',prov),
                         ('train.hard_negatives.jsonl',negs)):
        members[name]=('\n'.join(content)+'\n').encode()
    manifest={'schema':'cybersand/on-policy-recovery-corrections-v2',
       'data_class':'synthetic-host-applied-TRAIN-only','eligible_for_model_training':True,
       'count':len(rows),'worlds':worlds,'members_sha256':{n:sha(data) for n,data in members.items()},
       'teacher':'privileged synthetic geodesic; labels only; not in prompts',
       'forbidden':['not official DEV/heldout','not native evidence','not production default']}
    members['manifest.json']=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(members.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise ValueError('bad ZIP CRC')
        for n,d in manifest['members_sha256'].items():
            if sha(z.read(n))!=d:raise ValueError('bad member digest')
    return {'sha256':sha(path.read_bytes()),'rows':len(rows),'worlds':len(worlds)}


def run(out,source_seeds=SOURCE_TRAIN,correction_seeds=CORRECTION_TRAIN,
        evaluation_seeds=EVAL_SEEDS,cross_seeds=CROSS_GENERATOR_SEEDS,budget=BUDGET):
    if out.exists():raise FileExistsError('immutable experiment output already exists')
    if not 32<=budget<=256:raise ValueError('unsafe budget')
    groups=[set(x) for x in (source_seeds,correction_seeds,evaluation_seeds,cross_seeds)]
    if any(groups[i]&groups[j] for i in range(4) for j in range(i+1,4)):raise ValueError('world group leakage')
    verify_sources(Path(__file__).parent)
    rows,source_worlds,hards,labels,teacher_counts=teacher_data(source_seeds,budget)
    bvec,bmodel,base_fit=fit(rows)
    raw,correction_worlds,correction_stats=collect_corrections(correction_seeds,bvec,bmodel,min(160,budget))
    corrected,rejected=dedup_corrections(raw,rows)
    dvec,dmodel,dagger_fit=fit_student(rows,corrected)
    policies={'baseline':(bvec,bmodel),'on_policy_corrections':(dvec,dmodel)}
    summary1,w1,roll1,png1=evaluate(evaluation_seeds,make_world,policies,budget)
    summary2,w2,roll2,png2=evaluate(cross_seeds,make_cross_world,policies,budget)
    out.mkdir(parents=True);(out/'png').mkdir()
    export=export_corrections(corrected,correction_worlds,out/'train_only_corrections.zip')
    inventory=[]
    for name,data in png1+png2:
        (out/'png'/name).write_bytes(data);inventory.append({'name':name,'sha256':sha(data)})
    report={'schema':SCHEMA,'source_blob_pins':SOURCE_PINS,
       'data_class':'synthetic-model-on-policy-TRAIN / independent cross-generator synthetic EVAL',
       'note':'No W0/native evidence, no official DEV/heldout. Privileged geodesics label only. Synthetic same-generator and independent barrier-generator tests.',
       'group_partitions':{'source_train':list(source_seeds),'correction_train':list(correction_seeds),
                'evaluation_original_generator':list(evaluation_seeds),'evaluation_cross_generator':list(cross_seeds)},
       'source_teacher':{'rows':len(rows),'metadata':teacher_counts,'fit':base_fit},
       'on_policy':{'raw_rows':len(raw),'counts':correction_stats,'dedup_rejections':rejected,
                    'fit':dagger_fit,'export':export},
       'evaluation':{'same_generator':{'summary':summary1,'worlds':w1,'rollouts':roll1},
            'cross_generator':{'summary':summary2,'worlds':w2,'rollouts':roll2}},
       'original_pngs':inventory,'native_events':0,'checkpoint_promotions':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.out)
    print(json.dumps({k:v['summary'] for k,v in r['evaluation'].items()},indent=2))
