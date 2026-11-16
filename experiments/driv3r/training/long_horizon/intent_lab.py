"""Paired DEV-only intent controllability ablation, real 36/72/128 tiny optimizers.

All mode commands are model-side temporary overlays: World materials, its source
identity and host legality remain immutable. No held-out, native or paid calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import itertools
import random
import json
from pathlib import Path
import resource
import statistics
import time

import numpy as np

from .policy import Ranker, menu_matrix, teacher_scores, teacher, ACTION_MAX
from .train import HORIZONS, MAX_RECOVERIES, _png, make_corpus, sha_file, source_pins
from .world import MODES, MOVE, World, digest, replay, specs

VERSION = 'driv3r-long-horizon-intent-ablation/1'
ARMS = ('intent_aware', 'intent_blind', 'intent_aware_counterfactual', 'intent_aware_cross_features')
MODE_SLICE = slice(20,23)


def cross_features(X):
    """Expose action-dependent interaction basis; no host permission or oracle."""
    X = X.copy()
    wall,route,motif = [X[...,i].copy() for i in (20,21,22)]
    X[...,20] = wall * (X[...,12]+.4*X[...,13])
    X[...,21] = route * X[...,14]
    X[...,22] = motif * (X[...,5]+X[...,6]+.15*X[...,15]*(X[...,5]-X[...,6]))
    return X


def choose(world: World, model: Ranker, horizon: int, requested: str, blind: bool, crossed: bool=False):
    if requested not in MODES: raise ValueError('Unrecognised model-only command')
    original = world.spec
    try:
        world.spec = replace(original, mode=requested)
        menu, X = menu_matrix(world, horizon)
        teacher_label = menu[int(teacher_scores(world,menu,X).argmax())]
        if blind: X[:,MODE_SLICE] = 0.
        if crossed: X = cross_features(X)
        probs = model.probs(X,len(menu))
        index = int(probs.argmax())
        return menu[index], {a:round(float(probs[i]),8) for i,a in enumerate(menu)}, teacher_label
    finally:
        world.spec = original


def paired_counterfactual_corpus(horizon: int, seed: int, n_per_family: int):
    """One shared host state, three alternative *model-only* intent labels.

    The host executes its own original teacher instruction, not these three
    proposed counterfactual actions. No alternate-state material edits occur.
    """
    from dataclasses import replace
    if horizon not in HORIZONS or not 1 <= n_per_family <= 9: raise ValueError('Budget')
    data, masks, labels=[] ,[], []
    teacher_counts=Counter(); discordant=0; total_states=0
    train_worlds=specs('train',n_per_family)
    train_worlds.extend(replace(s,challenge='trap_corridor') for s in list(train_worlds) if s.seed%2==0)
    train_ids=[]
    for spec in train_worlds:
        w=World.create(spec);train_ids.append([spec.ident,w.base_sha()])
        rng=random.Random(seed*100000+horizon*193+spec.seed)
        for _ in range(horizon):
            if w.stopped:break
            original=w.spec; choices=[]
            for mode in MODES:
                try:
                    w.spec=replace(original,mode=mode)
                    menu,X=menu_matrix(w,horizon)
                    expected=menu[int(teacher_scores(w,menu,X).argmax())]
                    choices.append(expected)
                    mask=np.zeros(ACTION_MAX,dtype=np.bool_);mask[:len(menu)]=True
                    data.append(X);masks.append(mask);labels.append(menu.index(expected))
                    teacher_counts[f'{mode}:{expected}']+=1
                finally:
                    w.spec=original
            total_states+=1;discordant+=int(len(set(choices))>1)
            selected,*_=teacher(w,horizon,rng)
            alternatives=[a for a in w.menu() if a in MOVE and a!=selected]
            applied=rng.choice(alternatives) if alternatives and rng.random()<.13 else selected
            w.apply(applied,MAX_RECOVERIES)
    return (np.stack(data),np.stack(masks),np.asarray(labels,dtype=np.int64),
            {'original_worlds_sha256':digest(train_ids),'state_count':total_states,
             'counterfactual_rows':len(labels),'teacher_mode_action_counts':dict(sorted(teacher_counts.items())),
             'teacher_discordant_state_fraction':round(discordant/max(1,total_states),6)})

def train(seed: int, worlds: int, updates: int, output: Path):
    models, records = {}, []
    for arm in ARMS:
        model = Ranker(seed=seed, hidden=24)
        stages=[]
        for stage,horizon in enumerate(HORIZONS):
            X, mask, labels, prov = (paired_counterfactual_corpus(horizon,seed,worlds)
                  if arm in ('intent_aware_counterfactual','intent_aware_cross_features') else make_corpus(horizon,seed,worlds))
            if arm == 'intent_blind': X[:,:,MODE_SLICE] = 0.
            if arm == 'intent_aware_cross_features': X = cross_features(X)
            opt = model.optimize(X,mask,labels,steps=updates,batch=64,
                                 lr=.007 if stage < 2 else .004,seed=seed+stage*917)
            stages.append({'horizon':horizon,'training_examples':len(labels),
                           'source_train_worlds_sha256':prov['original_worlds_sha256'],
                           'label_distribution':prov.get('labels',prov.get('teacher_mode_action_counts')),
                           'counterfactual_discordant_fraction':prov.get('teacher_discordant_state_fraction'),
                           'optimizer':opt})
        checkpoint=output/f'{arm}-seed{seed}.npz'
        model.checkpoint(checkpoint)
        restored = Ranker.restore(checkpoint)
        assert restored.updates == 3*updates
        assert all(np.array_equal(model.params[k],restored.params[k]) for k in model.params)
        models[arm]=restored
        records.append({'arm':arm,'seed':seed,'sha256':sha_file(checkpoint),
                        'filename':checkpoint.name,'parameters':restored.parameter_count,
                        'updates':restored.updates,'stages':stages})
    return models,records


def run_episode(spec, horizon: int, arm: str, model: Ranker, seed: int,
                initial_mode: str, next_mode: str|None, switch: int|None, output: Path):
    if spec.split != 'development' or horizon not in HORIZONS or arm not in ARMS:
        raise ValueError('Only development, approved horizons and trained arms')
    if initial_mode not in MODES or (next_mode is not None and next_mode not in MODES):
        raise ValueError('Invalid intent')
    if (switch is None) != (next_mode is None) or (switch is not None and not 0 < switch < horizon):
        raise ValueError('Invalid switch specification')
    w=World.create(spec)
    base=w.base_sha()
    dist=Counter(); lat=[]; commands=[]
    start=time.perf_counter()
    for turn in range(horizon):
        if w.stopped: break
        mode = initial_mode if switch is None or turn < switch else next_mode
        t0=time.perf_counter_ns()
        selected, probabilities, teacher_choice = choose(w,model,horizon,mode,arm=='intent_blind',arm=='intent_aware_cross_features')
        lat.append(time.perf_counter_ns()-t0)
        event=w.apply(selected,MAX_RECOVERIES)
        dist[selected]+=1
        commands.append({'turn':turn,'model_only_requested_mode':mode,
                         'host_legal_menu':event['legal'], 'chosen':selected,
                         'probabilities':probabilities,'teacher_argmax':teacher_choice,
                         'pre_sha256':event['pre_sha256']})
    assert w.base_sha()==base, 'A model-only intent cannot mutate host source identity'
    reproduced=replay(spec,w.decisions,MAX_RECOVERIES)
    if reproduced.state_sha()!=w.state_sha(): raise AssertionError('Replay failed')
    ink=set(w.ink)
    move_events=[e for e in w.decisions if e['chosen'] in MOVE]
    near_material=0
    for e in move_events:
        x,y=e['cursor_after']
        near_material+=int(any((x+dx,y+dy) in w.materials or (x+dx,y+dy) in w.protected
                            for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy))
    turns=sum(a['chosen']!=b['chosen'] for a,b in zip(move_events,move_events[1:]))
    result={'version':VERSION,'spec_id':spec.ident,'family':spec.family,'seed':spec.seed,
            'split':spec.split,'source_base_sha256':base,'horizon':horizon,
            'arm':arm,'model_seed':seed,'initial_model_intent':initial_mode,
            'post_switch_intent':next_mode,'intent_switch_turn':switch,
            'native_executed':False,'native_receipts':0,
            'command_source':'temporary model features only, not host state',
            'action_distribution':dict(sorted(dist.items())),'metrics':w.metrics(horizon),
            'route_east_displacement':w.cursor[0]-48,
            'wall_neighbour_fraction':round(near_material/max(1,len(move_events)),6),
            'motif_turn_fraction':round(turns/max(1,len(move_events)-1),6),
            'mean_decision_latency_ms':round(statistics.mean(lat)/1e6,6),
            'p95_decision_latency_ms':round(sorted(lat)[int(.95*(len(lat)-1))]/1e6,6),
            'wall_seconds':round(time.perf_counter()-start,6),
            'teacher_argmax_matches':sum(c['teacher_argmax']==c['chosen'] for c in commands),
            'trajectory_sha256':digest(w.decisions),'final_state_sha256':w.state_sha(),
            'ink_cells':sorted([*p] for p in ink),'commands':commands,'events':w.decisions}
    output.mkdir(parents=True, exist_ok=True)
    key=f'{spec.ident}-{horizon}-{arm}-{initial_mode}-{next_mode or "fixed"}-seed{seed}'
    (output/f'{key}.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    _png(output/f'{key}.png',w)
    return result


def pair_stats(traces):
    # Three independent requested intents over exactly the same obstacle/protection map.
    by_mode={r['initial_model_intent']:r for r in traces}
    sets=[set(map(tuple,by_mode[m]['ink_cells'])) for m in MODES]
    jac=[]
    for left,right in itertools.combinations(sets,2):
        jac.append(len(left&right)/max(1,len(left|right)))
    return {'mean_cross_intent_ink_jaccard':round(statistics.mean(jac),6),
            'different_final_geometries':len({tuple(sorted(s)) for s in sets}),
            'route_east_displacement':by_mode['route']['route_east_displacement'],
            'motif_turn_fraction':by_mode['motif']['motif_turn_fraction'],
            'wall_neighbour_fraction':by_mode['wall']['wall_neighbour_fraction']}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed-start',type=int,default=573)
    p.add_argument('--seed-count',type=int,default=2)
    p.add_argument('--train-worlds-per-family',type=int,default=4)
    p.add_argument('--updates-per-stage',type=int,default=100)
    p.add_argument('--dev-worlds-per-family',type=int,default=2)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Immutable output exists')
    if (not 573<=a.seed_start<=575 or not 1<=a.seed_count<=2
        or a.seed_start+a.seed_count>576 or not 1<=a.train_worlds_per_family<=9
        or not 1<=a.updates_per_stage<=180 or not 1<=a.dev_worlds_per_family<=2):
        raise ValueError('Bounded CPU budget exceeded')
    a.output.mkdir(parents=True)
    start=time.perf_counter()
    # Only development specs. Never construct or inspect structurally held-out worlds.
    geo=specs('development',a.dev_worlds_per_family)
    if len(geo)!=4*a.dev_worlds_per_family:raise AssertionError('Unexpected split')
    all_models=[]; all_runs=[]; per_arm={}
    for seed in range(a.seed_start,a.seed_start+a.seed_count):
        models,record=train(seed,a.train_worlds_per_family,a.updates_per_stage,a.output)
        all_models.extend(record)
        for horizon in HORIZONS:
            for arm in ARMS:
                pairs=[]
                for spec in geo:
                    matched=[]
                    for mode in MODES:
                        matched.append(run_episode(spec,horizon,arm,models[arm],seed,mode,None,None,
                                                   a.output/'traces'/str(seed)/str(horizon)/arm))
                    # The host base SHA changes with Spec.mode, which is FIXED per spec;
                    # requested temporary model intent never changes this SHA.
                    assert len({r['source_base_sha256'] for r in matched})==1
                    pairs.append(pair_stats(matched))
                    all_runs.extend(matched)
                    if horizon==128:
                        # Two predeclared command changes, no model retraining.
                        for before,after in (('route','wall'),('motif','route')):
                            all_runs.append(run_episode(spec,horizon,arm,models[arm],seed,
                                  before,after,36 if before=='route' else 72,
                                  a.output/'traces'/str(seed)/'128'/arm/'switched'))
                per_arm[f'{seed}-{horizon}-{arm}']={
                    'matched_world_groups':len(pairs),
                    'mean_cross_intent_ink_jaccard':round(statistics.mean(
                        r['mean_cross_intent_ink_jaccard'] for r in pairs),6),
                    'mean_distinct_geometry_count':round(statistics.mean(
                        r['different_final_geometries'] for r in pairs),4),
                    'route_east_displacement':round(statistics.mean(
                        r['route_east_displacement'] for r in pairs),4),
                    'motif_turn_fraction':round(statistics.mean(
                        r['motif_turn_fraction'] for r in pairs),6),
                    'wall_neighbour_fraction':round(statistics.mean(
                        r['wall_neighbour_fraction'] for r in pairs),6)}
    assert len(all_runs)==a.seed_count*len(geo)*len(HORIZONS)*len(ARMS)*3 + a.seed_count*len(geo)*len(ARMS)*2
    by_arm={}
    for arm in ARMS:
        rows=[r for r in all_runs if r['arm']==arm]
        counts=Counter()
        for r in rows:counts.update(r['action_distribution'])
        by_arm[arm]={'episodes':len(rows),'action_distribution':dict(sorted(counts.items())),
                     'teacher_argmax_agreement':round(sum(r['teacher_argmax_matches'] for r in rows)/max(1,sum(r['metrics']['decisions'] for r in rows)),6),
                     'early_stops':sum(r['metrics']['stopped_early'] for r in rows),
                     'mean_policy_latency_ms':round(statistics.mean(r['mean_decision_latency_ms'] for r in rows),6),
                     'native_64_move_exceedances':sum(r['metrics']['logical_cursor_moves']>64 for r in rows)}
    report={'version':VERSION,'source_sha256':{**source_pins(),'intent_lab.py':sha_file(Path(__file__))},
            'seed_start':a.seed_start,'seed_count':a.seed_count,'worlds_per_family_train':a.train_worlds_per_family,
            'dev_worlds_per_family':a.dev_worlds_per_family,'updates_per_stage':a.updates_per_stage,
            'checkpoint_records':all_models,'paired_metrics':per_arm,'by_arm':by_arm,
            'episodes':len(all_runs),'dev_base_sha256':[World.create(s).base_sha() for s in geo],
            'source_faithfulness':'synthetic host-legal one-cell sweeps and hashed material replay',
            'heldout_read':False,'native_executed':False,'native_receipts':0,
            'paid_calls':0,'model_families_loaded':['tiny_numpy_mlp_625_parameters'],
            'wall_seconds':round(time.perf_counter()-start,4),
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (a.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    file_hashes={str(p.relative_to(a.output)):sha_file(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    (a.output/'SHA256SUMS.json').write_text(json.dumps(file_hashes,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'checkpoint_records':[{'arm':r['arm'],'seed':r['seed'],'sha256':r['sha256']} for r in all_models],
          'episodes':len(all_runs),'128_metrics':{k:v for k,v in per_arm.items() if '-128-' in k},
          'by_arm':by_arm,'wall_seconds':report['wall_seconds'],'rss_kib':report['peak_process_rss_kib']},sort_keys=True))
    return report

if __name__=='__main__':main()
