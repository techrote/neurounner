"""DEV-only paired off-policy recovery stress: real tiny optimizers and host-legal original paths.

No final holdout, native, d1/LFM/Sprocket, model downloads or provider changes.
The synthetic editor is not a native interface; every forced action remains host-legal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import random
import resource
import statistics
import time
from pathlib import Path

import numpy as np

from .policy import Ranker, menu_matrix, teacher, teacher_scores, random_legal_persistent
from .train import HORIZONS, MAX_RECOVERIES, _png, sha_file, source_pins, make_corpus
from .world import MOVE, World, digest, replay, specs

VERSION = 'driv3r-long-horizon-forced-divergence/1'
TRAIN_ARMS = {'ordinary_offpolicy_013':.13, 'stress_offpolicy_040':.40}
PROBE_TURNS = (12,35,71)  # attempt on these zero-based decisions; no posthoc picking


def train_arm(name: str, seed: int, worlds: int, updates: int, output: Path):
    if name not in TRAIN_ARMS or not (1<=worlds<=9 and 1<=updates<=180):
        raise ValueError('Training budget or arm invalid')
    frac=TRAIN_ARMS[name]
    ranker=Ranker(seed,hidden=24)
    stages=[]
    for index,horizon in enumerate(HORIZONS):
        X, mask, y, prov=make_corpus(horizon,seed,worlds,adversarial_fraction=frac)
        result=ranker.optimize(X,mask,y,steps=updates,batch=64,
                               lr=.007 if index<2 else .004,seed=seed+index*917)
        stages.append({'horizon':horizon,'examples':len(y),
                       'teacher_label_counts':prov['labels'],
                       'off_policy_interventions':prov['off_policy_interventions'],
                       'world_list_sha256':prov['original_worlds_sha256'],
                       'optimizer':result})
    filename=f'{name}-seed{seed}.npz'
    ranker.checkpoint(output/filename)
    restored=Ranker.restore(output/filename)
    if restored.updates!=3*updates or any(not np.array_equal(restored.params[k],ranker.params[k])
                                             for k in ranker.params):
        raise AssertionError('Restored checkpoint differs')
    return restored, {'arm':name,'checkpoint':filename,
                      'sha256':sha_file(output/filename),
                      'updates':restored.updates,'parameters':restored.parameter_count,
                      'stages':stages}


def forced_choice(world: World, preferred: str, horizon:int):
    """One predeclared poor-ranked MOVE among legal alternative ink moves; no STOP force.

    If only a single MOVE is legal (or none), record a skipped intervention.
    The teacher score ranks candidates at the *current* state only; it never
    observes any future material changes or hidden optimal routes.
    """
    menu, X=menu_matrix(world,horizon)
    other=[(float(score), action) for action,score in
           zip(menu,teacher_scores(world,menu,X)) if action in MOVE and action!=preferred]
    return min(other,key=lambda t:(t[0],t[1]))[1] if other else None


def run_episode(spec, ranker:Ranker|None, arm:str, horizon:int,
                seed:int, perturbed:bool, out:Path):
    if horizon not in HORIZONS or arm not in (*TRAIN_ARMS,'teacher_mixture','random_legal_persistent'):
        raise ValueError('Unapproved research arm/horizon')
    if (ranker is None) != (arm in ('teacher_mixture','random_legal_persistent')):
        raise ValueError('Learned checkpoint identity is mandatory')
    if spec.split!='development':
        raise ValueError('Only DEVELOPMENT scenarios permitted; held-out is sealed')
    w=World.create(spec)
    rng=random.Random(seed*137+spec.seed*17+horizon)
    lat=[];dist=Counter();divergences=[];planned=[];probs=[]
    start=time.perf_counter()
    for t in range(horizon):
        if w.stopped:break
        t0=time.perf_counter_ns()
        if arm=='teacher_mixture':
            preferred=teacher(w,horizon,rng)[0];scores=None
        elif arm=='random_legal_persistent':
            preferred=random_legal_persistent(w,rng);scores=None
        else:
            preferred,scores=ranker.choose(w,horizon)
        lat.append(time.perf_counter_ns()-t0)
        chosen=preferred
        if perturbed and t in PROBE_TURNS:
            alt=forced_choice(w,preferred,horizon)
            planned.append({'decision':t,'preferred':preferred,'selected_alternative':alt,
                            'world_pre_sha256':w.state_sha(),
                            'reason':'no_legal_alternative' if alt is None else 'lowest_local_teacher_score_other_legal_move'})
            if alt is not None:
                chosen=alt
                divergences.append({'decision':t,'preferred':preferred,'applied':chosen,
                                    'pre_sha256':w.state_sha(),'ink_before':len(w.ink)})
        event=w.apply(chosen,MAX_RECOVERIES)
        dist[chosen]+=1
        if scores is not None:
            probs.append({'decision':t,'probabilities':scores,'chosen':chosen})
    bare=[{k:v for k,v in e.items() if k!='policy_probabilities'} for e in w.decisions]
    replayed=replay(spec,bare,MAX_RECOVERIES)
    if replayed.state_sha()!=w.state_sha() or replayed.ink!=w.ink:
        raise AssertionError('State mismatch from original host-action replay')
    for row in divergences:
        i=row['decision']
        # Actual newly painted cells in next 16 decision records, not a success oracle.
        row['new_ink_next_16_decisions'] = sum(len(e['painted']) for e in bare[i+1:i+17])
        row['cursor_moves_next_16_decisions'] = sum(e['logical_moves'] -
                (bare[j-1]['logical_moves'] if j else 0)
                for j,e in enumerate(bare[i+1:i+17],start=i+1))
    out.mkdir(parents=True,exist_ok=True)
    prefix=f'{spec.ident}-{horizon}-{arm}-{int(perturbed)}'
    _png(out/(prefix+'.png'),w)
    result={'version':VERSION,'evidence':'original_synthetic_host_legal_replay',
            'world_id':spec.ident,'split':spec.split,'family':spec.family,
            'mode':spec.mode,'challenge':spec.challenge,'seed':spec.seed,
            'base_sha256':w.base_sha(),'horizon':horizon,'arm':arm,
            'policy_seed':seed,'perturbed':perturbed,
            'forced_schedule':list(PROBE_TURNS),'attempts':planned,'actual_interventions':divergences,
            'action_distribution':dict(sorted(dist.items())),'metrics':w.metrics(horizon),
            'decision_probabilities':probs,
            'mean_policy_latency_ms':round(statistics.mean(lat)/1e6,6),
            'p95_policy_latency_ms':round(sorted(lat)[int(.95*(len(lat)-1))]/1e6,6),
            'wall_seconds':round(time.perf_counter()-start,6),
            'trajectory_sha256':digest(bare),'final_state_sha256':w.state_sha(),
            'native_executed':False,'native_receipts':0,'events':bare}
    (out/(prefix+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def summarize(runs):
    counts=Counter()
    for x in runs:counts.update(x['action_distribution'])
    return {'episodes':len(runs),'mean_ink':round(statistics.mean(x['metrics']['unique_painted_cells'] for x in runs),4),
            'mean_turns':round(statistics.mean(x['metrics']['turns'] for x in runs),4),
            'mean_branches':round(statistics.mean(x['metrics']['branch_cells'] for x in runs),4),
            'mean_recoveries':round(statistics.mean(x['metrics']['recoveries'] for x in runs),4),
            'early_stops':sum(x['metrics']['stopped_early'] for x in runs),
            'mean_logical_moves':round(statistics.mean(x['metrics']['logical_cursor_moves'] for x in runs),4),
            'native_64_ceiling_breaches':sum(x['metrics']['logical_cursor_moves']>64 for x in runs),
            'mean_policy_latency_ms':round(statistics.mean(x['mean_policy_latency_ms'] for x in runs),6),
            'attempted_forces':sum(len(x['attempts']) for x in runs),
            'actual_forces':sum(len(x['actual_interventions']) for x in runs),
            'mean_post_force_ink_16':round(statistics.mean(
                i['new_ink_next_16_decisions'] for x in runs for i in x['actual_interventions']),4)
                if any(x['actual_interventions'] for x in runs) else None,
            'action_distribution':dict(sorted(counts.items())),
            'hard_negative_cases':[{'world':x['world_id'],'family':x['family'],'challenge':x['challenge'],
                                    'ink':x['metrics']['unique_painted_cells'],
                                    'early_stop':x['metrics']['stopped_early'],
                                    'trajectory_sha256':x['trajectory_sha256']}
                for x in runs if x['metrics']['stopped_early'] or
                  x['metrics']['unique_painted_cells']<x['horizon']*.6]}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',default=572,type=int)
    p.add_argument('--train-worlds-per-family',default=4,type=int)
    p.add_argument('--dev-worlds-per-family',default=1,type=int)
    p.add_argument('--updates-per-stage',default=100,type=int)
    a=p.parse_args(argv)
    if (not 0<=a.seed<=100000 or not 1<=a.train_worlds_per_family<=9
        or not 1<=a.dev_worlds_per_family<=2 or not 1<=a.updates_per_stage<=180):
        raise ValueError('Resource caps')
    if a.output.exists():raise FileExistsError('Refuse to overwrite original checkpoint/trajectory')
    a.output.mkdir(parents=True)
    t0=time.perf_counter()
    models={};training=[]
    for arm in TRAIN_ARMS:
        model,record=train_arm(arm,a.seed,a.train_worlds_per_family,a.updates_per_stage,a.output)
        models[arm]=model;training.append(record)
    dev=specs('development',a.dev_worlds_per_family)
    dev.extend(replace(s,challenge='trap_corridor') for s in list(dev) if s.seed%2==0)
    results={};all_runs=[]
    for horizon in HORIZONS:
        results[str(horizon)]={}
        for arm in (*TRAIN_ARMS,'teacher_mixture','random_legal_persistent'):
            branches={}
            for perturb in (False,True):
                runs=[run_episode(s,models.get(arm),arm,horizon,a.seed,perturb,
                                  a.output/'trajectories'/str(horizon)/arm/('forced' if perturb else 'intact'))
                      for s in dev]
                all_runs.extend(runs)
                branches['forced' if perturb else 'intact']=summarize(runs)
            results[str(horizon)][arm]=branches
    report={'version':VERSION,'train_seed':a.seed,'source_sha256':{
                **source_pins(),'intervention_lab.py':sha_file(Path(__file__))},
            'checkpoint_records':training,'research_scope':'development only',
            'heldout_read':False,'train_worlds_per_family':a.train_worlds_per_family,
            'dev_worlds_per_family':a.dev_worlds_per_family,'updates_per_stage':a.updates_per_stage,
            'worlds':[{'id':s.ident,'split':s.split,'family':s.family,'challenge':s.challenge,
                       'base_sha256':World.create(s).base_sha()} for s in dev],
            'results':results,'episode_count':len(all_runs),
            'wall_seconds':round(time.perf_counter()-t0,4),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'native_executed':False,'native_64_logical_move_ceiling':'unchanged',
            'slm_models_loaded':[],'paid_gpu_or_api':False}
    (a.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    hashed={str(f.relative_to(a.output)):sha_file(f)
            for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'SHA256SUMS.json').write_text(json.dumps(hashed,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'wall_seconds':report['wall_seconds'],
        'rss_kib':report['peak_rss_kib'],'episode_count':report['episode_count'],
        'checkpoints':[{'arm':r['arm'],'sha256':r['sha256']} for r in training],
        'dev_128':results['128']},sort_keys=True))
    return report

if __name__=='__main__':main()
