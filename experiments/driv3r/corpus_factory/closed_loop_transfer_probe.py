#!/usr/bin/env python3
"""Real CPU classifier -> host-applied external 96x96 worm rollouts; research only.

The generator below is NEW and fully synthetic. Its worlds are NOT official v1
DEV, actual CyberSand worlds, or native/W0 receipts. Model fitting consumes
only pinned TRAIN-only ZIPs. It never saves model weights or calls game APIs.
"""
from collections import Counter, deque
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import struct
import zipfile
import zlib

from learnability_probe_scoped import load, features

ACTIONS = ('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
DIRS = ((1,0),(0,1),(-1,0),(0,-1))
FAMILIES = ('greebles','pipework','architecture')
SOURCE_SHA = {'v1':'80b11c731e29306dd04c0b09539ab8fc7796592933276a13e7e44e83f17f6dbc',
              'probe':'ae3bcb2c9d06fca4ab69f427dba5e859da4638882850a26fe14a4103b97c2d2e'}


def sha(b): return hashlib.sha256(b).hexdigest()


def make_world(seed):
    """New cross-generator test worlds; reachability checked by BFS."""
    rng=random.Random(70219+seed)
    size=96; start=(7,48)
    obstacles=set(); protected=set()
    for i in range(10):
        x=rng.randrange(14,85); y=rng.randrange(6,82)
        w=rng.randrange(2,8); h=rng.randrange(2,10)
        obstacles.update((a,b) for a in range(x,min(96,x+w)) for b in range(y,min(96,y+h)))
    # Deterministic rails, blocked passages and nontraversable protected cells.
    for x in range(29,44):
        obstacles.add((x,43)); obstacles.add((x,53))
    for x in range(43,60):
        if (x+seed)%5==0:
            protected.update((x,y) for y in range(34,63,3))
    safe={(x,y) for x in range(3,14) for y in range(42,55)}
    obstacles-=safe; protected-=safe|obstacles
    q=deque([start]); reachable={start:0}
    while q:
        p=q.popleft()
        for dx,dy in DIRS:
            v=(p[0]+dx,p[1]+dy)
            if 0<=v[0]<96 and 0<=v[1]<96 and v not in obstacles|protected and v not in reachable:
                reachable[v]=reachable[p]+1;q.append(v)
    candidates=[p for p in reachable if reachable[p]>=38 and p[0]>48]
    if not candidates:raise ValueError('unreachable synthetic goal')
    goal=sorted(candidates,key=lambda p:(abs(p[0]-72)+abs(p[1]-48),p))[seed%min(50,len(candidates))]
    materials={}
    for y in range(96):
        for x in range(96):
            p=(x,y)
            materials[p]='stone' if p in obstacles else 'metal' if p in protected else 'sand' if (x//8+y//11+seed)%7<3 else 'water' if (x//8+y//11+seed)%7==3 else 'empty'
    return {'seed':seed,'goal':goal,'start':start,'obstacles':obstacles,'protected':protected,'materials':materials,'shortest_goal':reachable[goal]}


def legal_state(w,pos,heading,visited,history):
    moves={}
    for a,turn in (('STRAIGHT',0),('LEFT',-1),('RIGHT',1)):
        h=(heading+turn)%4;dx,dy=DIRS[h]; p=(pos[0]+dx,pos[1]+dy)
        if 0<=p[0]<96 and 0<=p[1]<96 and p not in w['obstacles'] and p not in w['protected']:
            moves[a]=(p,h)
    fresh=[a for a in moves if moves[a][0] not in visited]
    legal=list(moves)
    if len(fresh)>=2 and len(history)<2048:legal.append('BRANCH')
    if history:legal.append('RETRACE')
    legal.append('STOP')
    return moves,fresh,legal


def observation(w,pos,heading,visited,past,legal,tick,budget,family):
    symbol=[]; material=[]; protected=[]
    for dy in range(-2,3):
        s='';m=[];p=''
        for dx in range(-2,3):
            cell=(pos[0]+dx,pos[1]+dy); outside=not(0<=cell[0]<96 and 0<=cell[1]<96)
            s+='@' if (dx,dy)==(0,0) else '#' if outside or cell in w['obstacles'] or cell in w['protected'] else '+' if cell in visited else '.'
            m.append('boundary' if outside else w['materials'][cell]);p+='1' if cell in w['protected'] else '0'
        symbol.append(s);material.append(m);protected.append(p)
    return {'x':pos[0],'y':pos[1],'heading':heading,'local_5x5':symbol,
            'local_material_5x5':material,'local_protected_5x5':protected,
            'legal':legal,'last_actions':past[-4:],'family':family,'tick':tick,
            'budget_remaining':budget-tick,
            'goal_delta':[w['goal'][0]-pos[0],w['goal'][1]-pos[1]],'goal_reached':pos==w['goal']}


def train_model(path,kind,epochs=24):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDClassifier
    if sha(path.read_bytes())!=SOURCE_SHA[kind]:raise ValueError('TRAIN source SHA pin mismatch')
    rows=load(path,kind)
    if not rows or len({r[2] for r in rows})<30:raise ValueError('insufficient TRAIN worlds')
    vec=DictVectorizer(sparse=True,sort=True)
    X=vec.fit_transform(features(obs,True) for obs,_,_ in rows)
    model=SGDClassifier(loss='log_loss',alpha=.0001,max_iter=epochs,tol=None,
                        random_state=17,shuffle=True,learning_rate='optimal')
    model.fit(X,[label for _,label,_ in rows])
    return vec,model,{'train_rows':len(rows),'train_worlds':len({r[2] for r in rows}),
                      'optimizer_epochs':int(model.n_iter_),'optimizer_samples_seen':int(model.t_),
                      'feature_count':X.shape[1]}


def predict(policy,obs,vec=None,model=None,rng=None):
    if policy=='random':return rng.choice([a for a in obs['legal'] if a!='STOP'] or ['STOP'])
    scores=model.decision_function(vec.transform([features(obs,True)]))[0]
    classes=list(model.classes_)
    return max((a for a in obs['legal'] if a in classes),key=lambda a:(scores[classes.index(a)],-ACTIONS.index(a)))


def rollout(w,family,policy,vec=None,model=None,budget=128):
    pos=w['start'];heading=0;visited={pos};history=[];past=[];path=[pos];actions=Counter()
    rng=random.Random(92031+w['seed']*111+sum(map(ord,family)))
    first_goal=None; illegal=0; novel=[]; stopped=False
    for tick in range(budget):
        moves,fresh,legal=legal_state(w,pos,heading,visited,history)
        obs=observation(w,pos,heading,visited,past,legal,tick,budget,family)
        action=predict(policy,obs,vec,model,rng)
        if action not in legal:illegal+=1;raise AssertionError('model escaped host legal menu')
        actions[action]+=1
        if action=='STOP': stopped=True;novel.append(0);past.append(action);break
        prior=pos
        if action=='RETRACE':pos,heading=history.pop()
        else:
            arm=next(a for a in ('RIGHT','LEFT','STRAIGHT') if a in fresh) if action=='BRANCH' else action
            if action=='BRANCH' and not fresh:raise AssertionError('illegal branch application')
            history.append((pos,heading));pos,heading=moves[arm]
        novel.append(int(pos not in visited));visited.add(pos)
        if pos in w['protected'] or pos in w['obstacles']:raise AssertionError('protected-cell write')
        if pos==w['goal'] and first_goal is None:first_goal=tick+1
        past.append(action);path.append(pos)
    return {'seed':w['seed'],'family':family,'policy':policy,'steps':len(past),'visited':len(visited),
            'novel_by_32':[sum(novel[i:i+32]) for i in range(0,budget,32)],
            'goal_tick':first_goal,'stopped':stopped,'illegal':illegal,
            'actions':dict(sorted(actions.items())),'path':path}


def png(w,roll,scale=3):
    """Original host-applied PNG, no AI image claim; protected cells respected."""
    width=height=96
    grid=bytearray(width*height*3)
    for y in range(96):
        for x in range(96):
            p=(x,y);m=w['materials'][p]
            c=(21,23,30) if m=='empty' else (47,43,38) if m=='stone' else (33,40,53) if m=='metal' else (43,36,29) if m=='sand' else (28,39,47)
            i=3*(y*96+x);grid[i:i+3]=bytes(c)
    for x,y in roll['path']:
        i=3*(y*96+x);grid[i:i+3]=b'\xe8\xb0\x59'
    x,y=w['goal'];i=3*(y*96+x);grid[i:i+3]=b'\x7b\xf0\x9a'
    out=bytearray()
    for y in range(96):
        row=b''.join(bytes(grid[3*(y*96+x):3*(y*96+x)+3])*scale for x in range(96))
        for _ in range(scale):out.extend(b'\x00'+row)
    def chunk(label,data):return struct.pack('>I',len(data))+label+data+struct.pack('>I',zlib.crc32(label+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',96*scale,96*scale,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(bytes(out),9))+chunk(b'IEND',b'')


def run(v1,probe,out,seeds=(3,7,11,13,19,23),budget=128):
    if out.exists():raise ValueError('refuse overwrite of output directory')
    if not 32<=budget<=256 or not seeds:raise ValueError('bounded research budget/seeds required')
    out.mkdir(parents=True); (out/'png').mkdir()
    fitted={}
    training={}
    for kind,path in (('v1',v1),('probe',probe)):
        vec,model,meta=train_model(path,kind)
        fitted[kind]=(vec,model);training[kind]=meta
    records=[];manifest=[]
    for seed in seeds:
        w=make_world(seed)
        for family in FAMILIES:
            for policy in ('v1','probe','random'):
                vec,model=fitted.get(policy,(None,None))
                result=rollout(w,family,policy,vec,model,budget)
                name=f'world{seed:03d}-{family}-{policy}.png'
                pixels=png(w,result);(out/'png'/name).write_bytes(pixels)
                manifest.append({'file':name,'sha256':sha(pixels),'evidence_class':'cross-generator-synthetic-host-applied'})
                records.append(result)
    summary={}
    for policy in ('v1','probe','random'):
        a=[r for r in records if r['policy']==policy]
        summary[policy]={'runs':len(a),'mean_unique_visited':round(sum(r['visited'] for r in a)/len(a),3),
                         'mean_steps':round(sum(r['steps'] for r in a)/len(a),3),
                         'total_goal_arrivals':sum(r['goal_tick'] is not None for r in a),
                         'total_early_stops':sum(r['stopped'] for r in a),
                         'total_illegal':sum(r['illegal'] for r in a),
                         'mean_novel_by_32':[round(sum(r['novel_by_32'][i] if len(r['novel_by_32'])>i else 0 for r in a)/len(a),3) for i in range((budget+31)//32)],
                         'actions':dict(sorted(sum((Counter(r['actions']) for r in a),Counter()).items()))}
        for r in a:r.pop('path')
    report={'schema':'cybersand-corpus-cross-generator-host-rollout/1',
            'warning':'Real local CPU fitted classifiers and externally HOST-APPLIED synthetic steps. NOT primary Sprocket/LFM, NOT native, NOT official DEV/heldout, and no model/checkpoint promotion.',
            'research_evidence_class':'external-synthetic-cross-generator-inference',
            'sources':{k:{'sha256':SOURCE_SHA[k],**v} for k,v in training.items()},
            'world_generator':'cross-generator-v1 independent synthetic procedural/96x96',
            'world_seeds':list(seeds),'turn_budget':budget,'summary':summary,'rollouts':records,'png_inventory':manifest,
            'checkpoint_artifacts':0,'native_events':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,default=128)
    a=p.parse_args();r=run(a.v1,a.probe,a.output,budget=a.budget)
    print(json.dumps(r['summary'],sort_keys=True,indent=2))
