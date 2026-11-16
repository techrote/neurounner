#!/usr/bin/env python3
"""TRAIN-only multi-topology on-policy corrections and independent synthetic transfer.

This does not edit CyberSand game code. Synthetic host legally applies every
student action; privileged whole-world geodesics are teachers, never prompts.
The v2 cross-barrier generator becomes a TRAIN domain; a new vertical-gate
slalom stays evaluation-only and is never imported into optimizer examples.
"""
from collections import Counter, deque
import json
from pathlib import Path
import random

from recovery_curriculum import (ACTIONS, BUDGET, DIRS, FAMILIES, apply,
    distance_field, fit, legal_state, make_world, observation, oracle, sha,
    teacher_data)
from on_policy_recovery_v2 import (decide_model,
    dedup_corrections, evaluate, export_corrections, fit_student,
    make_cross_world, verify_sources)

SCHEMA = 'cybersand/synthetic-multi-topology-recovery-v3'
SOURCE_TRAIN = tuple(range(1000, 1048))
ORIGINAL_CORRECTIONS = tuple(range(2300, 2312))
BARRIER_CORRECTIONS = tuple(range(7300, 7312))
ORIGINAL_EVAL = tuple(range(4100, 4112))
BARRIER_EVAL = tuple(range(6100, 6112))
SLALOM_EVAL = tuple(range(9100, 9112))
SOURCE_BLOBS = {
    'on_policy_recovery_v2.py':'3804065af7432b57a52b524de4f4140c54f49e92',
    'recovery_curriculum.py':'fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4',
    'closed_loop_transfer_probe.py':'928537a3f02d227c51e36ba61815315d4a2c4ff7',
    'learnability_probe_scoped.py':'78d8c899e73d692ffe0540f6ac24dd0e173c38e0',
}


def make_slalom_world(seed):
    """Independent vertical-gate generator. Every world/goal BFS-verified.

    Five high walls with independently jittered narrow holes, optional
    protected pockets and local material texture. No training on this maker.
    """
    rng=random.Random(762117+seed)
    blocked=set(); protected=set(); start=(7,48)
    for k,x in enumerate((25,38,51,64,77)):
        gate=27+((seed*11+k*17)%43)
        for y in range(23,75):
            if abs(y-gate)>2:blocked.add((x,y))
    for k in range(9):
        x=rng.randrange(16,86); y=rng.randrange(13,81)
        for a in range(x,min(96,x+rng.randint(2,5))):
            for b in range(y,min(96,y+rng.randint(2,5))):
                blocked.add((a,b))
    for x in range(18,88,9):
        for y in range(15,85,7):
            if (x+3*y+seed)%5==0: protected.add((x,y))
    safe={(x,y) for x in range(0,15) for y in range(42,55)}
    blocked-=safe
    protected-=safe|blocked
    reachable={start:0};q=deque([start]);walls=blocked|protected
    while q:
        x,y=q.popleft()
        for dx,dy in DIRS:
            p=(x+dx,y+dy)
            if 0<=p[0]<96 and 0<=p[1]<96 and p not in walls and p not in reachable:
                reachable[p]=reachable[(x,y)]+1;q.append(p)
    candidates=[p for p in reachable if p[0]>83 and reachable[p]>=76]
    if not candidates:raise ValueError(f'slalom seed {seed} has no distal goal')
    goal=min(candidates,key=lambda p:(abs(p[0]-89)+abs(p[1]-48),p))
    materials={(x,y):('stone' if (x,y) in blocked else
                'metal' if (x,y) in protected else
                'sand' if (x//5+2*(y//8)+seed)%7<3 else
                'water' if (2*x+y+seed)%17<3 else 'empty')
               for y in range(96) for x in range(96)}
    return {'seed':seed,'generator':'vertical_slalom_v3',
            'start':start,'goal':goal,'shortest_goal':reachable[goal],
            'obstacles':blocked,'protected':protected,'materials':materials}


def world_digest(w):
    canonical=json.dumps({'start':w['start'],'goal':w['goal'],
        'blocked':sorted(map(list,w['obstacles'])),
        'protected':sorted(map(list,w['protected']))},
        sort_keys=True,separators=(',',':'))
    return sha(canonical.encode())


def validate_partitions(groups):
    """Require unique generator/seed ID *and* cross-split world geometry."""
    keys={}
    for partition,worlds in groups.items():
        for w in worlds:
            key=(w.get('generator','original'),w['seed'])
            digest=world_digest(w)
            if key in keys or digest in (v[1] for v in keys.values()):
                raise ValueError(f'world identity/geometry reuse at {partition}:{key}')
            keys[key]=(partition,digest)
    return {k:len(v) for k,v in groups.items()}


def collect_on_worlds(seeds, maker, domain, vec, model, budget=160):
    """Student-controlled host trajectories, teacher at the visited state.

    World/domain group IDs are retained, not inferred from a seed alone.
    """
    output=[]; worlds=[];stats=Counter()
    for seed in seeds:
        w=maker(seed)
        worlds.append({'seed':seed,'split':'TRAIN','generator':domain,
                       'world_group_id':f'{domain}:{seed}',
                       'world_sha256':world_digest(w)})
        dist=distance_field(w)
        for family in FAMILIES:
            pos=w['start'];heading=0;visited={pos};history=[];past=[];seen=Counter()
            for tick in range(budget):
                moves,fresh,legal=legal_state(w,pos,heading,visited,history)
                obs=observation(w,pos,heading,visited,past,legal,tick,budget,family)
                teacher,negs=oracle(w,pos,heading,visited,history,dist,legal)
                student=decide_model(obs,vec,model)
                if student not in legal or teacher not in legal:
                    raise AssertionError('non-host-legal policy/teacher action')
                normalized=json.dumps({k:v for k,v in obs.items() if k!='family'},
                                      sort_keys=True,separators=(',',':'))
                seen[(pos,heading)]+=1
                output.append({'observation':obs,'label':teacher,'seed':seed,
                  'world_group_id':f'{domain}:{seed}',
                  'observable_sha256':sha(normalized.encode()),
                  'student_action':student,
                  'was_student_teacher_disagreement':student!=teacher,
                  'repeated_oriented_cell':seen[(pos,heading)]>1,
                  'negative_actions':negs,
                  'evidence_class':'synthetic-train-host-applied-multi-topology-correction'})
                stats['observed']+=1;stats['disagreements']+=int(student!=teacher)
                stats['repeated_states']+=int(seen[(pos,heading)]>1)
                if student=='STOP':
                    stats['early_stops']+=int(pos!=w['goal'])
                    break
                pos,heading=apply(w,pos,heading,visited,history,student)
                if pos in w['protected'] or pos in w['obstacles']:
                    raise AssertionError('protected or blocked penetration')
                visited.add(pos);past.append(student)
                if pos==w['goal']:
                    stats['student_goal_arrivals']+=1
                    break
            stats['episodes']+=1
    return output,worlds,dict(stats)


def run(out, source_seeds=SOURCE_TRAIN, original_train=ORIGINAL_CORRECTIONS,
        barrier_train=BARRIER_CORRECTIONS, original_eval=ORIGINAL_EVAL,
        barrier_eval=BARRIER_EVAL, slalom_eval=SLALOM_EVAL, budget=160):
    if out.exists():raise FileExistsError('immutable experiment directory already exists')
    if not 32<=budget<=192:raise ValueError('bounded decision budget required')
    seed_groups=(source_seeds,original_train,barrier_train,original_eval,barrier_eval,slalom_eval)
    if any(set(seed_groups[i])&set(seed_groups[j]) for i in range(6) for j in range(i+1,6)):
        raise ValueError('split seed overlap')
    verify_sources(Path(__file__).parent)
    import subprocess
    for name,expected in SOURCE_BLOBS.items():
        actual=subprocess.run(['git','hash-object',str(Path(__file__).parent/name)],
                              capture_output=True,text=True,check=True).stdout.strip()
        if actual!=expected:raise ValueError(f'source blob pin mismatch {name}')
    world_groups={
        'source_train':[make_world(s) for s in source_seeds],
        'original_correction_train':[make_world(s) for s in original_train],
        'barrier_correction_train':[make_cross_world(s) for s in barrier_train],
        'original_eval':[make_world(s) for s in original_eval],
        'barrier_eval':[make_cross_world(s) for s in barrier_eval],
        'slalom_eval':[make_slalom_world(s) for s in slalom_eval],
    }
    partition_counts=validate_partitions(world_groups)
    rows,_,hard,label_counts,source_stats=teacher_data(source_seeds,budget)
    bvec,bmodel,base_fit=fit(rows)
    arows,aworlds,astats=collect_on_worlds(original_train,make_world,
                                           'original_v1',bvec,bmodel,budget)
    brows,bworlds,bstats=collect_on_worlds(barrier_train,make_cross_world,
                                           'barrier_v2',bvec,bmodel,budget)
    original_dedup,original_drops=dedup_corrections(arows,rows)
    mixed_dedup,mixed_drops=dedup_corrections(arows+brows,rows)
    ovec,omodel,ofit=fit_student(rows,original_dedup)
    mvec,mmodel,mfit=fit_student(rows,mixed_dedup)
    policies={'original_only':(ovec,omodel),'mixed_topology':(mvec,mmodel)}
    evaluations={};pngs=[]
    for name,seeds,maker in (('original_generator',original_eval,make_world),
                              ('barrier_generator',barrier_eval,make_cross_world),
                              ('untrained_slalom_generator',slalom_eval,make_slalom_world)):
        summary,worlds,trajectories,images=evaluate(seeds,maker,policies,budget)
        feasible=[]
        for w in world_groups[{'original_generator':'original_eval',
              'barrier_generator':'barrier_eval',
              'untrained_slalom_generator':'slalom_eval'}[name]]:
            feasible.append({'seed':w['seed'],
              'oriented_teacher_steps':distance_field(w).get((*w['start'],0)),
              'world_sha256':world_digest(w)})
        evaluations[name]={'summary':summary,'worlds':worlds,
                           'oriented_teacher_steps':feasible,'trajectories':trajectories}
        pngs.extend((name,fn,content) for fn,content in images)
    out.mkdir(parents=True);(out/'png').mkdir()
    exported=export_corrections(mixed_dedup,aworlds+bworlds,out/'train_only_mixed_corrections.zip')
    inventory=[]
    for domain,filename,payload in pngs:
        unique=domain+'--'+filename
        (out/'png'/unique).write_bytes(payload)
        inventory.append({'name':unique,'sha256':sha(payload),'domain':domain})
    report={'schema':SCHEMA,
      'evidence_class':'external-synthetic-real-CPU-SGD-and-host-applied-inference',
      'warning':'Not native CyberSand, not official DEV/heldout, not Sprocket/LFM; family replicas correlated. Never feed evaluation rows to TRAIN.',
      'source_blob_pins':SOURCE_BLOBS,
      'source_split':{'worlds':len(source_seeds),'rows':len(rows),'hard_negatives':hard,
                      'teacher_labels':label_counts,'student_fit':base_fit,'stats':source_stats},
      'partition_world_counts':partition_counts,
      'partition_world_sha256':{name:[{'seed':w['seed'],'sha256':world_digest(w)} for w in worlds]
                                 for name,worlds in world_groups.items()},
      'corrections':{'original_raw':len(arows),'barrier_raw':len(brows),
                     'original_stats':astats,'barrier_stats':bstats,
                     'original_deduplicated':len(original_dedup),
                     'mixed_deduplicated':len(mixed_dedup),
                     'original_duplicates':original_drops,'mixed_duplicates':mixed_drops,
                     'original_only_model':ofit,'mixed_model':mfit,'export':exported},
      'evaluation':evaluations,'png_inventory':inventory,
      'native_events':0,'checkpoint_promotions':0,'paid_spend':0}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.out)
    print(json.dumps({k:v['summary'] for k,v in r['evaluation'].items()},sort_keys=True,indent=2))
