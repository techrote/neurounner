"""P14 goal-reversal relational memory: CPU-trained *synthetic* game policies.

This is a falsification experiment, not a CyberSand game change or native
playtest. The frozen P0 legal-action World supplies obstacles, protection and
painting. Matched histories and matched goal profiles isolate the conjunction
of an earlier ordering cue and a newly switched spatial goal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from population_lab import FEATURES, NICHES, draw_svg, stable_seed
from population_p10 import ACTIONS, require_torch
from population_p12 import GRAMMARS, HISTORY, encode_history, split_seeds, tensorize, world
from population_p13 import build_model as p13_model

MODES = ('pairwise_goal', 'pairwise_goal_blind', 'recurrent_gru', 'shuffled_order', 'state_only')
PHASES = ('first_branch', 'retrace', 'other_branch', 'stop')
GOAL_PROFILES = {
    'train_low': {'E': (20, 5), 'W': (1, 5)},
    'train_high': {'E': (20, 15), 'W': (1, 15)},
    'heldout_far': {'E': (20, 19), 'W': (1, 19)},
}
GOAL_FIELDS = tuple(FEATURES.index(f) for f in ('goal_progress', 'guidance'))


def state(w, niche, target, meta):
    if target not in w.legal():
        raise AssertionError(f'illegal teacher {target}, legal={w.legal()}')
    x = np.zeros((len(ACTIONS), len(FEATURES)), np.float32)
    m = np.zeros(len(ACTIONS), np.bool_)
    for a in w.legal():
        idx = ACTIONS.index(a)
        x[idx], m[idx] = w.feature(a), True
    history = encode_history([h['action'] for h in w.history])
    return dict(xs=x, mask=m, history=history, niche=NICHES.index(niche),
                target=ACTIONS.index(target), **meta)


def mission_pair(seed, card, depth, grammar, niche, goal_profile, goal_side):
    if grammar not in GRAMMARS or goal_profile not in GOAL_PROFILES or goal_side not in ('E', 'W'):
        raise ValueError('bad grammar/goal choice')
    if niche not in NICHES: raise ValueError('unknown niche')
    sequences = []
    for order_id, survey in enumerate(GRAMMARS[grammar]):
        w = world(seed, card, depth)
        w.second_goal = GOAL_PROFILES[goal_profile][goal_side]
        precursor = []
        for a in survey: precursor.extend((a, 'RETRACE'))
        precursor.extend(['S'] * depth + ['RETRACE'] * (depth - 1))
        for a in precursor:
            if a not in w.legal(): raise AssertionError(f'illegal survey {a}')
            w.step(a)
        if not w.switched or w.goal != w.second_goal: raise AssertionError('switch did not occur')
        # XOR of historical E/W order, current goal reversal, niche parity.
        east_first = survey.index('E') < survey.index('W')
        invert = (goal_side == 'W') ^ (NICHES.index(niche) % 2 == 1)
        first = 'E' if east_first ^ invert else 'W'
        desired = (first, 'RETRACE', 'W' if first == 'E' else 'E', 'STOP')
        meta = dict(seed=seed, card=card, depth=depth, grammar=grammar,
                    niche_name=niche, goal_profile=goal_profile,
                    goal_side=goal_side, order_id=order_id, phase='')
        initial = deepcopy(w)
        steps = []
        for phase, a in zip(PHASES, desired):
            meta['phase'] = phase
            steps.append(state(w, niche, a, meta))
            w.step(a)
        if not w.stopped or w.painted & (w.protected | w.obstacles):
            raise AssertionError('invalid mission painting')
        sequences.append(dict(states=steps, initial=initial, final=w,
                              desired=desired, precursor=precursor))
    a,b = sequences
    for name in ('cursor', 'painted', 'parent', 'goal', 'second_goal', 'steps', 'heading'):
        if getattr(a['initial'], name) != getattr(b['initial'], name):
            raise AssertionError(f'paired worlds differ in {name}')
    if not np.array_equal(a['states'][0]['xs'], b['states'][0]['xs']):
        raise AssertionError('earlier ordering leaked into present features')
    if not np.array_equal(a['states'][0]['mask'], b['states'][0]['mask']):
        raise AssertionError('earlier ordering leaked into legal menu')
    if sorted(a['precursor']) != sorted(b['precursor']) or a['desired'][0] == b['desired'][0]:
        raise AssertionError('not a balanced counterfactual pair')
    return a,b


def dataset(seeds, grammars, depths, goal_profiles, cards=(0,1), niches=NICHES):
    return [s for seed in seeds for card in cards for grammar in grammars
            for depth in depths for niche in niches for profile in goal_profiles
            for goal_side in ('E','W') for e in mission_pair(seed,card,depth,grammar,niche,profile,goal_side)
            for s in e['states']]


def build_model(mode, width=32):
    if mode not in MODES: raise ValueError(mode)
    return p13_model('pairwise_order' if mode.startswith('pairwise') else
                     'state_only' if mode == 'state_only' else 'recurrent_gru', width)


def inputs(X,H,mode,sham=None):
    if mode=='pairwise_goal_blind':
        X=X.clone(); X[:,:,list(GOAL_FIELDS)] = 0
    if mode=='state_only': H=H*0
    elif mode=='shuffled_order':
        if sham is None: raise ValueError('shuffle needs seeded generator')
        # Preserve token counts per episode, destroy temporal order.
        from population_p10 import shuffled
        H=shuffled(H,sham)
    return X,H


def fit(model, mode, train, val, seed, epochs=10, batch=128, lr=.004):
    torch,_=require_torch()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed)
    sham=torch.Generator().manual_seed(stable_seed('p14', seed, mode, 'sham'))
    curves=[]; updates=0
    for epoch in range(epochs):
        model.train(); losses=[]
        for ids in torch.randperm(len(train[0]),generator=rng).split(batch):
            X,M,H,N,Y=(z[ids] for z in train)
            X,H=inputs(X,H,mode,sham)
            raw=torch.nn.functional.cross_entropy(model(X,M,H,N),Y,reduction='none')
            directional=(Y==ACTIONS.index('E')) | (Y==ACTIONS.index('W'))
            weights=torch.where(directional,5.,1.)
            loss=(raw*weights).sum()/weights.sum()
            if not torch.isfinite(loss):raise AssertionError('nonfinite optimizer loss')
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step();updates+=1; losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            X,M,H,N,Y=val
            X,H=inputs(X,H,mode,sham)
            v=float(torch.nn.functional.cross_entropy(model(X,M,H,N),Y))
        curves.append(dict(epoch=epoch,train_ce=round(float(np.mean(losses)),6),val_ce=round(v,6)))
    return dict(updates=updates,epochs=epochs,parameters=sum(p.numel() for p in model.parameters()),curve=curves)


def predict(model, mode, rows, seed=9):
    torch,_=require_torch()
    X,M,H,N,Y=tensorize(rows)
    sham=torch.Generator().manual_seed(stable_seed('p14',seed,mode,'eval'))
    X,H=inputs(X,H,mode,sham)
    model.eval()
    with torch.no_grad(): pred=model(X,M,H,N).argmax(-1).numpy()
    legal=M.numpy()[np.arange(len(rows)),pred]
    if not bool(legal.all()):raise AssertionError('host-illegal predicted action')
    return pred


def evaluate(model,mode,rows):
    pred=predict(model,mode,rows)
    right=pred==np.array([r['target'] for r in rows])
    first=np.array([i for i,r in enumerate(rows) if r['phase']=='first_branch'])
    first_right=right[first]
    # Order: each paired history has four rows, two histories are adjacent.
    both=first_right.reshape(-1,2).all(axis=1)
    by_goal={d:round(float(np.mean([right[i] for i,r in enumerate(rows)
                                   if r['phase']=='first_branch' and r['goal_side']==d])),6)
             for d in ('E','W')}
    return dict(accuracy=round(float(right.mean()),6), first=round(float(first_right.mean()),6),
                paired_both=round(float(both.mean()),6),by_goal=by_goal,
                legal_rate=1.0, decisions=len(rows),pairs=len(both))


def choose(model,mode,w,niche,history,sham):
    torch,_=require_torch()
    meta=dict(phase='rollout',seed=w.seed,card=0,depth=0,grammar='rollout',
              niche_name=niche,goal_profile='rollout',goal_side='E',order_id=0)
    row=state(w,niche,'STOP',meta)
    X,M,H,N,_=tensorize([row])
    X,H=inputs(X,H,mode,sham)
    model.eval()
    with torch.no_grad(): j=int(model(X,M,H,N).argmax(-1))
    a=ACTIONS[j]
    if a not in w.legal():raise AssertionError('host-illegal rollout')
    return a


def rollout(model,mode,seeds,grammars,depths,profiles, gallery=None, gallery_limit=10):
    torch,_=require_torch()
    sham=torch.Generator().manual_seed(stable_seed('p14',mode,'rollout'))
    total=0;won=0;actions=0;by_niche={n:[0,0] for n in NICHES}
    gallery_path=Path(gallery) if gallery else None
    for seed in seeds:
      for card in (0,1):
       for grammar in grammars:
        for depth in depths:
         for niche in NICHES:
          for profile in profiles:
           for side in ('E','W'):
            for episode in mission_pair(seed,card,depth,grammar,niche,profile,side):
                w=deepcopy(episode['initial']); actual=[]; desired=list(episode['desired'])
                for tick in range(4):
                    if w.stopped:break
                    a=choose(model,mode,w,niche,list(w.history),sham)
                    actual.append(a); w.step(a);actions+=1
                good=(actual==desired)
                total+=1;won+=int(good); by_niche[niche][0]+=int(good);by_niche[niche][1]+=1
                if gallery_path and total<=gallery_limit:
                    gallery_path.mkdir(parents=True,exist_ok=True)
                    rec=dict(seed=seed,card=card,grammar=grammar,depth=depth,niche=niche,
                             goal_profile=profile,goal_side=side,order_id=episode['states'][0]['order_id'],
                             desired=desired,actual=actual,success=good,history=w.history,
                             painted=sorted(w.painted),legal=True)
                    stem=f'episode{total:03d}'
                    (gallery_path/(stem+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
                    (gallery_path/(stem+'.svg')).write_text(draw_svg(w))
    return dict(episodes=total,mission_success=round(won/total,6),actions=actions,
                legal_rate=1.,by_niche={k:round(v[0]/v[1],6) for k,v in by_niche.items()})


def run(output,seed=654,train_worlds=8,validation_worlds=4,heldout_worlds=4,epochs=10,width=32):
    if not (4<=train_worlds<=60 and 2<=validation_worlds<=30 and
            2<=heldout_worlds<=30 and 3<=epochs<=40 and 16<=width<=96):
        raise ValueError('outside bounded synthetic CPU research envelope')
    torch,_=require_torch();torch.set_num_threads(1)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir(); (out/'gallery').mkdir()
    splits=split_seeds(seed,train_worlds,validation_worlds,heldout_worlds)
    t=dataset(splits['train'],('train_a','train_b'),(5,7),('train_low','train_high'))
    v=dataset(splits['validation'],('train_a','train_b'),(5,),('train_low','train_high'))
    cases={'seen':dataset(splits['holdout'],('train_a','train_b'),(7,),('train_low',)),
           'unseen_grammar':dataset(splits['holdout'],('heldout_c',),(7,),('train_high',)),
           'unseen_goal_and_gap':dataset(splits['holdout'],('heldout_c',),(9,),('heldout_far',))}
    T,V=tensorize(t),tensorize(v)
    metrics={};updates=0
    for mode in MODES:
        torch.manual_seed(stable_seed('p14',seed,'init',mode.startswith('pairwise')))
        model=build_model(mode,width)
        old={k:v.clone() for k,v in model.state_dict().items()}
        info=fit(model,mode,T,V,stable_seed('p14',seed,'train'),epochs)
        if not any(not torch.equal(old[k],v) for k,v in model.state_dict().items()):
            raise AssertionError('model did not train')
        ck=out/'models'/f'{mode}.pt'
        torch.save(dict(format='population-p14',mode=mode,seed=seed,width=width,
                        source='synthetic-P0',weights=model.state_dict(),actions=ACTIONS,features=FEATURES),ck)
        info['sha256']=hashlib.sha256(ck.read_bytes()).hexdigest();updates+=info['updates']
        tests={name:evaluate(model,mode,rows) for name,rows in cases.items()}
        mission=rollout(model,mode,splits['holdout'],('heldout_c',),(9,),('heldout_far',),
                        out/'gallery'/mode)
        metrics[mode]=dict(training=info,tests=tests,closed_loop=mission)
    report=dict(format='population-p14-goal-relational-mission',native=False,pretrained_slm=False,
                trained=True,seed=seed,torch=torch.__version__,numpy=np.__version__,
                splits=splits,train_rows=len(t),validation_rows=len(v),
                total_updates=updates,protocol=dict(history=HISTORY,
                train_grammars=['train_a','train_b'],unseen_grammar='heldout_c',
                train_goal_profiles=['train_low','train_high'],unseen_goal_profile='heldout_far',
                heldout_depth=9,phases=PHASES),models=metrics,
                limitations=['Synthetic P0 host only; no native runtime correspondence proven',
                             'Fixed four-step scripted teacher, not genuine self-play',
                             'Pairwise temporal bias is explicitly engineered',
                             'Six niche labels not validated artistic personas',
                             'Goal signal deliberately supplied through world candidate features'])
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(('mode','case','first','paired_both','mission','legal'))
        for mode,d in metrics.items():
            for case,m in d['tests'].items():
                w.writerow((mode,case,m['first'],m['paired_both'],
                            d['closed_loop']['mission_success'] if case=='unseen_goal_and_gap' else '',m['legal_rate']))
    (out/'manifest.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(out).as_posix()+'\n'
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.sha256'))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=654)
    p.add_argument('--train-worlds',type=int,default=8)
    p.add_argument('--validation-worlds',type=int,default=4)
    p.add_argument('--heldout-worlds',type=int,default=4)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--width',type=int,default=32)
    r=run(**vars(p.parse_args()))
    print(json.dumps(dict(seed=r['seed'],updates=r['total_updates'],
          first={mode:d['tests']['unseen_goal_and_gap']['first'] for mode,d in r['models'].items()},
          mission={mode:d['closed_loop']['mission_success'] for mode,d in r['models'].items()}),sort_keys=True))

if __name__ == '__main__': main()
