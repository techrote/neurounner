"""Bounded DEVELOPMENT-only on-policy corrective imitation (DAgger-like) ablation.

Two identical 36 -> 72 -> 128 pretrained NumPy policy copies receive an equal
extra optimizer budget: original offline teacher corpus vs policy-state teacher
corrections after deliberately legal diversions. Neither receives hindsight,
held-out worlds or native receipts. No reward optimisation or native execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time

import numpy as np

from .intervention_lab import PROBE_TURNS, forced_choice
from .policy import ACTION_MAX, Ranker, menu_matrix, teacher
from .train import HORIZONS, MAX_RECOVERIES, _png, make_corpus, sha_file, source_pins
from .world import MOVE, Spec, World, digest, replay, specs

VERSION = 'driv3r-long-horizon-on-policy-correction/1'
ARMS = ('extra_offline_teacher', 'on_policy_correction')
MAX_WORLDS = 6
MAX_UPDATES = 180


def development_specs(n: int) -> list[Spec]:
    if not 1 <= n <= 2:
        raise ValueError('DEVELOPMENT world cap exceeded')
    source = specs('development', n)
    return source + [replace(s, challenge='trap_corridor') for s in source if s.seed%2==0]


def on_policy_corpus(policy: Ranker, seed: int, worlds: int, horizon: int=128):
    """Query a deterministic teacher on states actually visited by frozen policy.

    Force at predetermined turns only. Teacher supplies *label* at each state,
    never a magic recovery action or a future reward. Every applied action is
    legal. Training uses no DEVELOPMENT/HELDOUT seed or family.
    """
    if horizon not in HORIZONS or not 1 <= worlds <= MAX_WORLDS:
        raise ValueError('Training world/horizon cap exceeded')
    Xs=[]; masks=[]; labels=[]; world_records=[]; forced=0; counts=Counter()
    training = specs('train', worlds)
    training += [replace(s,challenge='trap_corridor') for s in training if s.seed%2 == 0]
    for spec in training:
        w=World.create(spec)
        rng=random.Random(seed*70001+spec.seed*83+horizon)
        for t in range(horizon):
            if w.stopped:
                break
            menu, X = menu_matrix(w,horizon)
            teacher_choice,_,_=teacher(w,horizon,rng)
            masks.append(np.arange(ACTION_MAX)<len(menu))
            Xs.append(X)
            labels.append(menu.index(teacher_choice))
            counts[teacher_choice]+=1
            policy_choice,_=policy.choose(w,horizon)
            chosen=policy_choice
            if t in PROBE_TURNS:
                alt=forced_choice(w,policy_choice,horizon)
                if alt is not None:
                    chosen=alt
                    forced+=1
            w.apply(chosen,MAX_RECOVERIES)
        replayed=replay(spec,w.decisions,MAX_RECOVERIES)
        if replayed.state_sha()!=w.state_sha():
            raise AssertionError('Original on-policy training path replay failed')
        world_records.append({'world_id':spec.ident,'base_sha256':w.base_sha(),
                              'original_trajectory_sha256':digest(w.decisions),
                              'family':spec.family,'challenge':spec.challenge,
                              'decisions':w.steps, 'ink':len(w.ink)})
    return np.stack(Xs),np.stack(masks),np.asarray(labels,dtype=np.int64), {
        'source':'TRAIN_only_frozen_policy_states_plus_local_teacher_labels',
        'forced_moves':forced,'worlds':world_records,
        'worlds_sha256':digest(world_records),
        'label_distribution':dict(sorted(counts.items())),
        'examples':len(labels)}


def train_matched(seed: int, worlds: int, updates: int, out: Path):
    if not 1 <= worlds <= MAX_WORLDS or not 1 <= updates <= MAX_UPDATES:
        raise ValueError('Research optimizer cap exceeded')
    common=Ranker(seed,hidden=24)
    pretrained=[]
    for stage,horizon in enumerate(HORIZONS):
        X,M,y,p=make_corpus(horizon,seed,worlds,adversarial_fraction=.13)
        result=common.optimize(X,M,y,steps=updates,batch=64,
                               lr=.007 if stage<2 else .004,seed=seed+stage*917)
        pretrained.append({'horizon':horizon,'training_worlds_sha256':p['original_worlds_sha256'],
                           'examples':len(y),'optimizer':result})
    # Deepcopy includes Adam moments: both arms continue from IDENTICAL optimizer
    # state, not only identical checkpoint weights.
    extra=copy.deepcopy(common)
    corrected=copy.deepcopy(common)
    aX,aM,ay,ap=make_corpus(128,seed,worlds,adversarial_fraction=.13)
    bX,bM,by,bp=on_policy_corpus(common,seed,worlds)
    ares=extra.optimize(aX,aM,ay,steps=updates,batch=64,lr=.003,seed=seed+9911)
    bres=corrected.optimize(bX,bM,by,steps=updates,batch=64,lr=.003,seed=seed+9911)
    result={};models={}
    for arm,model,prov,optim in ((ARMS[0],extra,ap,ares),(ARMS[1],corrected,bp,bres)):
        cp=out/f'{arm}-seed{seed}.npz'
        model.checkpoint(cp)
        restored=Ranker.restore(cp)
        if (restored.updates!=4*updates or
            any(not np.array_equal(restored.params[k],model.params[k]) for k in model.params)):
            raise AssertionError('Original trained weights not restorable')
        result[arm]={'checkpoint':cp.name,'checkpoint_sha256':sha_file(cp),
                     'parameter_count':restored.parameter_count,'optimizer_updates':restored.updates,
                     'extra_corpus_sha256':digest({'labels':prov.get('label_distribution',prov.get('labels')),
                                                   'world_sha256':prov.get('worlds_sha256',prov.get('original_worlds_sha256'))}),
                     'optimizer':optim}
        models[arm]=restored
    return models,{'pretrained':pretrained,'extra':result,
                   'train_policy_state_provenance':bp,
                   'no_heldout_or_development_in_optimizer':True}


def evaluate(spec: Spec, model: Ranker, arm: str, horizon: int, force: bool, out: Path):
    if spec.split!='development' or arm not in ARMS or horizon not in HORIZONS:
        raise ValueError('Evaluation DEVELOPMENT identity/horizon/arm invalid')
    w=World.create(spec); times=[]; counts=Counter();probs=[];forces=[]
    begin=time.perf_counter_ns()
    for t in range(horizon):
        if w.stopped: break
        stamp=time.perf_counter_ns()
        preferred,p=model.choose(w,horizon)
        times.append(time.perf_counter_ns()-stamp)
        action=preferred
        if force and t in PROBE_TURNS:
            alternative=forced_choice(w,preferred,horizon)
            forces.append({'decision':t,'preferred':preferred,'applied':alternative,
                           'state_sha256':w.state_sha()})
            if alternative is not None:action=alternative
        probs.append({'decision':t,'preferred':preferred,'probabilities':p})
        w.apply(action,MAX_RECOVERIES)
        counts[action]+=1
    reproduced=replay(spec,w.decisions,MAX_RECOVERIES)
    if reproduced.state_sha()!=w.state_sha():
        raise AssertionError('Full exact-state host-like replay mismatch')
    out.mkdir(parents=True,exist_ok=True)
    pfx=f'{spec.ident}-{horizon}-{arm}-{int(force)}'
    _png(out/(pfx+'.png'),w)
    events=w.decisions
    data={'version':VERSION,'world_id':spec.ident,'split':spec.split,'family':spec.family,
          'challenge':spec.challenge,'mode':spec.mode,'seed':spec.seed,
          'base_sha256':w.base_sha(),'arm':arm,'horizon':horizon,'force':force,
          'forced_schedule':list(PROBE_TURNS),'forces':forces,
          'checkpoint_sha256':None,'source_replay':'original_deterministic_synthetic_host_like',
          'probabilities':probs,'events':events,'action_distribution':dict(sorted(counts.items())),
          'metrics':w.metrics(horizon),'trajectory_sha256':digest(events),
          'final_state_sha256':w.state_sha(),
          'mean_policy_latency_ms':round(statistics.mean(times)/1e6,6),
          'p95_policy_latency_ms':round(sorted(times)[int(.95*(len(times)-1))]/1e6,6),
          'episode_wall_seconds':round((time.perf_counter_ns()-begin)/1e9,6),
          'native_executed':False,'native_receipts':0}
    return out/(pfx+'.json'),data


def verify_episode(obj: dict, model: Ranker):
    if obj['split']!='development' or obj['arm'] not in ARMS or obj['horizon'] not in HORIZONS:
        raise ValueError('Unapproved source identity')
    match=[s for s in development_specs(2) if s.ident==obj['world_id']]
    if len(match)!=1:
        raise ValueError('Unrecognized DEV world identity')
    spec=match[0]
    if (spec.family,spec.seed,spec.mode,spec.challenge)!=(obj['family'],obj['seed'],obj['mode'],obj['challenge']):
        raise ValueError('Tampered world attributes')
    w=World.create(spec)
    if w.base_sha()!=obj['base_sha256'] or digest(obj['events'])!=obj['trajectory_sha256']:
        raise ValueError('World or event hash mismatch')
    if obj['native_executed'] or obj['native_receipts']:
        raise ValueError('Synthetic trajectory falsely labeled native')
    forces=[];counts=Counter()
    if len(obj['probabilities'])!=len(obj['events']):
        raise ValueError('Missing or surplus policy probability event')
    for t,event in enumerate(obj['events']):
        preferred,p=model.choose(w,obj['horizon'])
        entry=obj['probabilities'][t]
        if entry['decision']!=t or entry['preferred']!=preferred or set(entry['probabilities'])!=set(p):
            raise ValueError('Policy decision identity mismatch')
        if (abs(sum(entry['probabilities'].values())-1)>1e-6 or
            any(abs(entry['probabilities'][k]-p[k])>2.1e-8 for k in p)):
            raise ValueError('Policy probability mismatch')
        choice=preferred
        if obj['force'] and t in PROBE_TURNS:
            alt=forced_choice(w,preferred,obj['horizon'])
            forces.append({'decision':t,'preferred':preferred,'applied':alt,'state_sha256':w.state_sha()})
            if alt is not None: choice=alt
        if event['chosen']!=choice or w.apply(choice,MAX_RECOVERIES)!=event:
            raise ValueError('Original decision / material state forged')
        counts[choice]+=1
    if (forces!=obj['forces'] or obj['forced_schedule']!=list(PROBE_TURNS) or
        w.state_sha()!=obj['final_state_sha256'] or w.metrics(obj['horizon'])!=obj['metrics'] or
        dict(sorted(counts.items()))!=obj['action_distribution']):
        raise ValueError('Forged forced schedule, metrics or final state')
    if w.ink.intersection(w.materials) or w.ink.intersection(w.protected):
        raise ValueError('Illegal painted cell')
    return w


def summarize(episodes):
    c=Counter()
    for e in episodes:c.update(e['action_distribution'])
    return {'episodes':len(episodes),
            'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in episodes),4),
            'mean_turns':round(statistics.mean(e['metrics']['turns'] for e in episodes),4),
            'mean_branches':round(statistics.mean(e['metrics']['branch_cells'] for e in episodes),4),
            'mean_recoveries':round(statistics.mean(e['metrics']['recoveries'] for e in episodes),4),
            'early_stops':sum(e['metrics']['stopped_early'] for e in episodes),
            'logical_move_gt_64':sum(e['metrics']['logical_cursor_moves']>64 for e in episodes),
            'mean_latency_ms':round(statistics.mean(e['mean_policy_latency_ms'] for e in episodes),6),
            'action_distribution':dict(sorted(c.items())),
            'negative_cases':[{'world_id':e['world_id'],'horizon':e['horizon'],'force':e['force'],
                               'ink':e['metrics']['unique_painted_cells'],
                               'trajectory_sha256':e['trajectory_sha256']}
                              for e in episodes if e['metrics']['stopped_early'] or
                                     e['metrics']['unique_painted_cells'] < e['horizon']*.6]}


def main(argv=None):
    arg=argparse.ArgumentParser()
    arg.add_argument('--output',type=Path,required=True)
    arg.add_argument('--seed',type=int,default=573)
    arg.add_argument('--train-worlds-per-family',type=int,default=4)
    arg.add_argument('--dev-worlds-per-family',type=int,default=1)
    arg.add_argument('--updates-per-stage',type=int,default=100)
    a=arg.parse_args(argv)
    if (not 0<=a.seed<=100000 or not 1<=a.train_worlds_per_family<=MAX_WORLDS or
        not 1<=a.dev_worlds_per_family<=2 or not 1<=a.updates_per_stage<=MAX_UPDATES):
        raise ValueError('Resource budget exceeded')
    if a.output.exists():
        raise FileExistsError('Original result path exists; refuse overwrite')
    a.output.mkdir(parents=True)
    start=time.perf_counter()
    models,train_prov=train_matched(a.seed,a.train_worlds_per_family,a.updates_per_stage,a.output)
    runs={};all_episodes=[]
    for h in HORIZONS:
        for arm in ARMS:
            for force in (False,True):
                label=f'{h}/{arm}/{int(force)}';runs[label]=[]
                for s in development_specs(a.dev_worlds_per_family):
                    path,data=evaluate(s,models[arm],arm,h,force,a.output/'trajectories'/str(h)/arm/str(int(force)))
                    data['checkpoint_sha256']=sha_file(a.output/train_prov['extra'][arm]['checkpoint'])
                    path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
                    verify_episode(data,models[arm])
                    runs[label].append(data);all_episodes.append(data)
    metrics={label:summarize(vals) for label,vals in runs.items()}
    report={'version':VERSION,'seed':a.seed,'train_worlds_per_family':a.train_worlds_per_family,
            'dev_worlds_per_family':a.dev_worlds_per_family,
            'updates_per_stage':a.updates_per_stage,'training':train_prov,
            'comparisons':metrics,'episode_count':len(all_episodes),
            'source_sha256':source_pins(),
            'additional_source_sha256':{Path(__file__).name:sha_file(Path(__file__))},
            'elapsed_seconds':round(time.perf_counter()-start,5),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'heldout_read':False,'native_executed':False,'slm_loaded':False,'paid_external_gpu_or_api':False,
            'native_64_logical_move_ceiling':'unchanged'}
    (a.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    manifest={str(p.relative_to(a.output)):sha_file(p) for p in sorted(a.output.rglob('*'))
              if p.is_file() and p.name!='SHA256SUMS.json'}
    (a.output/'SHA256SUMS.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'episode_count':len(all_episodes),'elapsed_seconds':report['elapsed_seconds'],
                      'peak_rss_kib':report['peak_rss_kib'],
                      'comparison_128':{k:v for k,v in metrics.items() if k.startswith('128/')}},sort_keys=True))
    return report


if __name__=='__main__':
    main()
