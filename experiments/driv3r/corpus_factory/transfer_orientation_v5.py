#!/usr/bin/env python3
"""TRAIN-only matched orientation curriculum; unseen slalom transfer.

Synthetic host actions only. Native/Godot assets are never touched. Family
replicas within a world are *not* independent statistical replicates.
"""
from collections import Counter
from collections import deque
from pathlib import Path
import hashlib
import json
import random
import subprocess

from recovery_curriculum import (DIRS, FAMILIES, apply, distance_field, fit,
                                 legal_state, make_world, observation, oracle,
                                 teacher_data, sha)
from domain_recovery_v3 import (collect_on_worlds, make_slalom_world,
                                validate_partitions, world_digest)
from on_policy_recovery_v2 import (make_cross_world, dedup_corrections,
                                   evaluate, fit_student, export_corrections,
                                   verify_sources)
from topology_control_v4 import sample_world_balanced

SCHEMA='cybersand/synthetic-orientation-transfer-v5'
SOURCE=tuple(range(1000,1006))
SHARED_ORIGINAL=tuple(range(2500,2502))
HORIZONTAL=tuple(range(7500,7502))
VERTICAL=tuple(range(8500,8502))
EVAL_ORIGINAL=tuple(range(4500,4504))
EVAL_HORIZONTAL=tuple(range(6500,6504))
EVAL_SLALOM=tuple(range(9500,9508))
BUDGET=192
SOURCE_PINS={
 'recovery_curriculum.py':'fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4',
 'on_policy_recovery_v2.py':'3804065af7432b57a52b524de4f4140c54f49e92',
 'domain_recovery_v3.py':'c284f09de420f05f95746b141e5d990dddc22cc1',
 'topology_control_v4.py':'283da5b5d33e9861bfa2e8d8475b469696e932af',
}


def make_vertical_comb_world(seed):
    """Independent TRAIN-only vertical obstacles, not the held-out slalom maker.

    Four full-height walls each have one 9-cell gate. Their gate heights use
    nonlinear seed mixing; extra protected cells and patches are source-unique.
    """
    rng=random.Random(947277+seed*181)
    obstacles=set();protected=set();start=(7,48);goal=(89,48)
    for k,x in enumerate((27,43,59,75)):
        gate=32+((seed*79 + k*37 + k*k*19)%34)
        for y in range(96):
            if abs(y-gate)>4:obstacles.add((x,y))
    for i in range(24):
        x=rng.randrange(17,89);y=rng.randrange(11,87)
        if (x,y) not in obstacles:protected.add((x,y))
    for i in range(10):
        x=rng.randrange(18,87);y=rng.randrange(8,88)
        for xx in range(x,min(x+2,96)):
            for yy in range(y,min(y+2,96)):
                if (xx,yy) not in obstacles:protected.add((xx,yy))
    keep={(x,y) for x in range(3,15) for y in range(42,55)}|{goal}
    obstacles-=keep;protected-=keep|obstacles
    dist={start:0};q=deque([start]);blocked=obstacles|protected
    while q:
        p=q.popleft()
        for dx,dy in DIRS:
            v=(p[0]+dx,p[1]+dy)
            if 0<=v[0]<96 and 0<=v[1]<96 and v not in blocked and v not in dist:
                dist[v]=dist[p]+1;q.append(v)
    if goal not in dist:raise ValueError(f'no path for vertical-comb seed {seed}')
    materials={(x,y):('stone' if (x,y) in obstacles else
                'metal' if (x,y) in protected else
                'sand' if (x//7+3*y//11+seed)%9<4 else
                'water' if (x*2+y+seed)%13<3 else 'empty')
              for x in range(96) for y in range(96)}
    return {'seed':seed,'generator':'vertical_comb_v5_train',
            'start':start,'goal':goal,'obstacles':obstacles,'protected':protected,
            'materials':materials,'shortest_goal':dist[goal]}


def validate(source,orig,hori,vert,eorig,ehori,eslalom,budget):
    if not 136<=budget<=256:raise ValueError('budget must accommodate slalom route')
    if not (len(orig)==len(hori)==len(vert)) or not orig:
        raise ValueError('size-matched correction groups required')
    parts={'source_train':(source,make_world),'shared_train':(orig,make_world),
           'horizontal_train':(hori,make_cross_world),
           'vertical_train':(vert,make_vertical_comb_world),
           'eval_original':(eorig,make_world),
           'eval_horizontal':(ehori,make_cross_world),
           'eval_slalom':(eslalom,make_slalom_world)}
    seen=set();worlds={}
    for name,(seeds,maker) in parts.items():
        if len(seeds)!=len(set(seeds)) or set(seeds)&seen:
            raise ValueError('seed overlap across TRAIN/EVAL or repeats')
        seen.update(seeds)
        worlds[name]=[maker(s) for s in seeds]
    counts=validate_partitions(worlds)
    for name in ('eval_original','eval_horizontal','eval_slalom'):
        for w in worlds[name]:
            minimum=distance_field(w).get((*w['start'],0))
            if minimum is None or minimum>budget:
                raise ValueError(f'impossible eval under budget: {name}:{w["seed"]} {minimum}')
    return counts,worlds


def oracle_goal_rollout(w,budget):
    dist=distance_field(w);pos=w['start'];heading=0;visited={pos};history=[];past=[]
    for tick in range(budget):
        _,_,legal=legal_state(w,pos,heading,visited,history)
        action,_=oracle(w,pos,heading,visited,history,dist,legal)
        if action not in legal:raise AssertionError('oracle illegal')
        if action=='STOP':break
        pos,heading=apply(w,pos,heading,visited,history,action)
        visited.add(pos);past.append(action)
        if pos==w['goal']:return tick+1
    return None


def run(out,source=SOURCE,orig=SHARED_ORIGINAL,hori=HORIZONTAL,vert=VERTICAL,
        eorig=EVAL_ORIGINAL,ehori=EVAL_HORIZONTAL,eslalom=EVAL_SLALOM,budget=BUDGET):
    out=Path(out)
    if out.exists():raise FileExistsError('immutable output already exists')
    for name,blob in SOURCE_PINS.items():
        actual=subprocess.run(['git','hash-object',str(Path(__file__).parent/name)],
                  text=True,capture_output=True,check=True).stdout.strip()
        if actual!=blob:raise ValueError(f'parent source pin mismatch: {name}')
    verify_sources(Path(__file__).parent)
    counts,worlds=validate(source,orig,hori,vert,eorig,ehori,eslalom,budget)
    source_rows,source_worlds,hard,labels,teacher=teacher_data(source,budget)
    bvec,bmodel,bfit=fit(source_rows)
    common,common_worlds,common_stats=collect_on_worlds(orig,make_world,'original_v1',bvec,bmodel,budget)
    horizontal,horizontal_worlds,hstats=collect_on_worlds(hori,make_cross_world,'barrier_v2',bvec,bmodel,budget)
    vertical,vertical_worlds,vstats=collect_on_worlds(vert,make_vertical_comb_world,'vertical_comb_v5_train',bvec,bmodel,budget)
    hclean,hdrops=dedup_corrections(common+horizontal,source_rows)
    vclean,vdrops=dedup_corrections(common+vertical,source_rows)
    n=min(len(hclean),len(vclean))
    if n<2*(len(orig)+len(hori)):raise ValueError('insufficient corrective examples')
    hselected=sample_world_balanced(hclean,n)
    vselected=sample_world_balanced(vclean,n)
    hvec,hmodel,hfit=fit_student(source_rows,hselected)
    vvec,vmodel,vfit=fit_student(source_rows,vselected)
    policies={'horizontal_corrections':(hvec,hmodel),'vertical_corrections':(vvec,vmodel)}
    evals={};images=[]
    for domain,seeds,maker in (('original',eorig,make_world),
                               ('horizontal',ehori,make_cross_world),
                               ('unseen_slalom',eslalom,make_slalom_world)):
        summary,worldmeta,rollouts,original_png=evaluate(seeds,maker,policies,budget)
        oracle_steps=[{'seed':s,'steps':oracle_goal_rollout(maker(s),budget),
                       'oriented_shortest':distance_field(maker(s)).get((*maker(s)['start'],0)),
                       'sha256':world_digest(maker(s))} for s in seeds]
        evals[domain]={'summary':summary,'worlds':worldmeta,'trajectories':rollouts,
                       'oracle_upper_bound':oracle_steps}
        images.extend((domain+'--'+name,data) for name,data in original_png)
    out.mkdir(parents=True);(out/'png').mkdir()
    exports={'horizontal':export_corrections(hselected,common_worlds+horizontal_worlds,out/'train_horizontal.zip'),
             'vertical':export_corrections(vselected,common_worlds+vertical_worlds,out/'train_vertical.zip')}
    inventory=[]
    for name,data in images:
        (out/'png'/name).write_bytes(data);inventory.append({'name':name,'sha256':sha(data)})
    report={'schema':SCHEMA,'evidence_class':'external-synthetic-CPU-training-and-host-applied-world-disjoint-eval',
        'warning':'Experimental TRAIN-only corrections. Evaluation is not official DEV/heldout, native CyberSand, or SLM inference. The three family replicas are correlated.',
        'source_pins':SOURCE_PINS,'groups':counts,
        'seed_partitions':{'source':list(source),'shared_original':list(orig),'horizontal_correction':list(hori),
             'vertical_correction':list(vert),'eval_original':list(eorig),'eval_horizontal':list(ehori),'eval_slalom':list(eslalom)},
        'train_source':{'rows':len(source_rows),'hard_negatives':hard,'labels':labels,
                        'baseline_fit':bfit,'episode_stats':teacher},
        'correction':{'horizontal_pool':len(hclean),'vertical_pool':len(vclean),
             'horizontal_dedup_rejected':hdrops,'vertical_dedup_rejected':vdrops,
             'matched_rows_per_arm':n,'common_stats':common_stats,
             'horizontal_stats':hstats,'vertical_stats':vstats,
             'horizontal_fit':hfit,'vertical_fit':vfit,'exports':exports},
        'evaluation':evals,'pngs':inventory,'native_events':0,'production_promotions':0,'paid_spend':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();r=run(args.out)
    print(json.dumps({k:v['summary'] for k,v in r['evaluation'].items()},sort_keys=True,indent=2))
