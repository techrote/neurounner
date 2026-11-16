"""P12: held-out cue grammar and long-gap temporal-memory falsification.

External synthetic research only; imports the frozen P0 legal World contract.
Counterfactual paired worlds have identical current observable state, painted
cells, action histogram and final 8 tokens but opposite order-dependent targets.
This is NOT native CyberSand, SLM pretraining, real-game integration or self-play.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from population_lab import FEATURES, NICHES, World, draw_svg, stable_seed
from population_p10 import ACTIONS, require_torch, shuffled
from population_p11 import build_model as p11_build_model

MODES = ('full_history', 'last8_history', 'shuffled_history', 'state_only')
GRAMMARS = {
    'train_a': (('E','W','N'), ('W','E','N')),
    'train_b': (('E','N','W'), ('W','N','E')),
    'heldout_c': (('N','E','W'), ('N','W','E')),
}
HISTORY = 32
ROOTS = ((3,9),(4,10),(5,9),(15,9),(16,10),(17,9))


def split_seeds(seed, train_worlds=48, validation_worlds=16, heldout_worlds=24):
    counts = dict(train=train_worlds, validation=validation_worlds, holdout=heldout_worlds)
    d = {part: [stable_seed('p12',seed,part,i) for i in range(n)] for part,n in counts.items()}
    ids = sum(d.values(), [])
    if len(ids) != len(set(ids)): raise AssertionError('split-seed collision')
    return d


def world(seed, card, depth):
    if card not in (0,1) or depth not in (5,7,9): raise ValueError('bad world card/depth')
    w = World.generate(seed, max_steps=28)
    root = ROOTS[stable_seed('p12',seed,card,'root')%len(ROOTS)]
    w.cursor = root; w.painted={root}; w.parent={}; w.history=[]; w.steps=0
    w.heading='E'; w.switched=False; w.stopped=False
    w.goal=(18,11) if card==0 else (3,3)
    w.second_goal=(4,18) if card==0 else (18,18)
    # The smallest cleared pocket supporting the matched cue, a dead-end
    # retrace chain, and legal east/west probe options. Keep every other
    # obstacle/protected/material hazard exactly as generated.
    cx,cy=root
    needed={(cx,cy-1),(cx+1,cy),(cx-1,cy),(cx+1,cy+1),(cx-1,cy+1)}
    needed.update((cx,cy+k) for k in range(depth+1))
    assert all(1<=x<21 and 1<=y<21 for x,y in needed)
    if needed & w.protected: raise AssertionError('probe intersects protected materials')
    w.obstacles.difference_update(needed)
    return w


def encode_history(actions, width=HISTORY):
    if len(actions)>width: raise ValueError('history capacity exceeded')
    h=np.zeros(width,dtype=np.int64)
    h[-len(actions):]=[ACTIONS.index(a)+1 for a in actions]
    return h


def pair(seed, card, depth, grammar, niche):
    if grammar not in GRAMMARS or niche not in NICHES: raise ValueError('bad task identifier')
    rows=[]
    for order_id,permutation in enumerate(GRAMMARS[grammar]):
        w=world(seed,card,depth)
        precursor=[]
        for branch in permutation:
            precursor.extend((branch,'RETRACE'))
        # Long, common, host-legal distractor chain ends one cell below root.
        precursor.extend(['S']*depth + ['RETRACE']*(depth-1))
        for action in precursor:
            if action not in w.legal(): raise AssertionError(f'illegal precursor {action}')
            w.step(action)
        if not w.switched or not {'E','W'}.issubset(w.legal()):
            raise AssertionError('probe does not admit two post-switch painting options')
        order_e_before_w=permutation.index('E')<permutation.index('W')
        choose_e=order_e_before_w ^ (NICHES.index(niche)%2==1)
        target='E' if choose_e else 'W'
        x=np.zeros((len(ACTIONS),len(FEATURES)),np.float32)
        mask=np.zeros(len(ACTIONS),np.bool_)
        for a in w.legal():
            ix=ACTIONS.index(a); x[ix]=w.feature(a); mask[ix]=True
        hist=encode_history(precursor)
        rows.append(dict(xs=x,mask=mask,history=hist,niche=NICHES.index(niche),
                         target=ACTIONS.index(target),seed=seed,card=card,
                         depth=depth,grammar=grammar,order_id=order_id,world=w,
                         precursor=precursor,first_e=order_e_before_w))
    a,b=rows
    if not (np.array_equal(a['xs'],b['xs']) and np.array_equal(a['mask'],b['mask'])):
        raise AssertionError('order accidentally leaked through present candidate features')
    for field in ('painted','parent','cursor','goal','heading','steps'):
        if getattr(a['world'],field)!=getattr(b['world'],field):
            raise AssertionError(f'counterfactuals differ in {field}')
    if sorted(a['history'])!=sorted(b['history']): raise AssertionError('different action histograms')
    if not np.array_equal(a['history'][-8:],b['history'][-8:]):
        raise AssertionError('last-eight baseline is not information-theoretically blind')
    if a['target']==b['target']: raise AssertionError('unbalanced paired labels')
    if a['world'].painted & (a['world'].protected|a['world'].obstacles):
        raise AssertionError('painted into an obstacle or protected cell')
    return a,b


def dataset(seeds, grammars, depths, cards=(0,1), niches=NICHES):
    return [row for seed in seeds for card in cards for depth in depths
            for grammar in grammars for niche in niches for row in pair(seed,card,depth,grammar,niche)]


def tensorize(rows):
    torch,_=require_torch()
    if not rows: raise ValueError('empty data')
    return (torch.as_tensor(np.stack([r['xs'] for r in rows]),dtype=torch.float32),
            torch.as_tensor(np.stack([r['mask'] for r in rows]),dtype=torch.bool),
            torch.as_tensor(np.stack([r['history'] for r in rows]),dtype=torch.long),
            torch.as_tensor([r['niche'] for r in rows],dtype=torch.long),
            torch.as_tensor([r['target'] for r in rows],dtype=torch.long))


def transform(hist, mode, sham_gen=None):
    if mode=='state_only': return hist*0
    if mode=='last8_history':
        h=hist.clone();h[:,:-8]=0;return h
    if mode=='shuffled_history':
        if sham_gen is None: raise ValueError('sham needs generator')
        return shuffled(hist,sham_gen)
    return hist


def model_for(mode,width):
    return p11_build_model('state_only' if mode=='state_only' else 'token_gru',width)


def fit(model,mode,training,validation,seed,epochs=16,batch_size=128,lr=.004):
    torch,_=require_torch()
    rng=torch.Generator().manual_seed(seed)
    sham=torch.Generator().manual_seed(stable_seed('p12',seed,'sham'))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    updates=0; curves=[]
    for epoch in range(epochs):
        model.train();losses=[]
        for ix in torch.randperm(len(training[0]),generator=rng).split(batch_size):
            X,M,H,N,Y=(v[ix] for v in training)
            logits=model(X,M,transform(H,mode,sham),N)
            loss=torch.nn.functional.cross_entropy(logits,Y)
            if not bool(torch.isfinite(loss)): raise AssertionError('nonfinite training')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2)
            opt.step();updates+=1;losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            X,M,H,N,Y=validation
            # Don't let stochastic sham validation hide the actual value of order.
            v=float(torch.nn.functional.cross_entropy(model(X,M,transform(H,mode,sham),N),Y))
        curves.append(dict(epoch=epoch,train_ce=round(float(np.mean(losses)),5),val_ce=round(v,5)))
    return dict(updates=updates,epochs=epochs,curve=curves)


def assess(model,mode,rows,save=None,max_pairs=8):
    torch,_=require_torch()
    model.eval();X,M,H,N,Y=tensorize(rows)
    fixed=torch.Generator().manual_seed(stable_seed('p12',mode,'eval'))
    with torch.no_grad():
        pred=model(X,M,transform(H,mode,fixed),N).argmax(-1).numpy()
    legal=M.numpy()[np.arange(len(pred)),pred]
    if not bool(legal.all()): raise AssertionError('illegal predicted action')
    right=pred==Y.numpy()
    paired=right.reshape(-1,2).all(-1)
    metrics=dict(accuracy=round(float(right.mean()),6),
                 paired_both_correct=round(float(paired.mean()),6),
                 pairs=len(rows)//2,decisions=len(rows),legal_action_rate=float(legal.mean()),
                 by_niche={n:(round(float(right[[i for i,r in enumerate(rows) if r['niche']==j]].mean()),6)
                           if any(r['niche']==j for r in rows) else None) for j,n in enumerate(NICHES)})
    if save is not None:
        save=Path(save);save.mkdir(parents=True,exist_ok=True)
        ids=np.linspace(0,len(rows)//2-1,min(max_pairs,len(rows)//2),dtype=int)
        for p in ids:
            for order_id in (0,1):
                row=rows[2*int(p)+order_id];w=deepcopy(row['world'])
                action=ACTIONS[int(pred[2*int(p)+order_id])]
                w.step(action)
                if w.painted & (w.protected|w.obstacles): raise AssertionError('unsafe painted trace')
                summary=dict(native=False,source='P0 synthetic game host',seed=row['seed'],
                             depth=row['depth'],grammar=row['grammar'],
                             niche=NICHES[row['niche']],order_id=order_id,
                             precursor=row['precursor'],desired=ACTIONS[row['target']],
                             predicted=action,correct=bool(right[2*int(p)+order_id]),
                             trace=w.history,painted=sorted(w.painted))
                stem=f'pair{int(p):04d}-order{order_id}'
                (save/(stem+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
                (save/(stem+'.svg')).write_text(draw_svg(w))
    return metrics


def run(output,seed=649,train_worlds=48,validation_worlds=16,
        heldout_worlds=24,epochs=16,width=40):
    if not(12<=train_worlds<=120 and 8<=validation_worlds<=50 and
           8<=heldout_worlds<=70 and 3<=epochs<=30 and 16<=width<=96):
        raise ValueError('campaign exceeds bounded research limits')
    torch,_=require_torch();torch.set_num_threads(1)
    splits=split_seeds(seed,train_worlds,validation_worlds,heldout_worlds)
    train_rows=dataset(splits['train'],('train_a','train_b'),(5,7))
    val_rows=dataset(splits['validation'],('train_a','train_b'),(5,7))
    tests={
        'seen_grammar_gap7':dataset(splits['holdout'],('train_a','train_b'),(7,)),
        'unseen_grammar_gap7':dataset(splits['holdout'],('heldout_c',),(7,)),
        'seen_grammar_gap9':dataset(splits['holdout'],('train_a','train_b'),(9,)),
        'unseen_grammar_gap9':dataset(splits['holdout'],('heldout_c',),(9,)),
    }
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    (folder/'models').mkdir();(folder/'gallery').mkdir()
    tensors=tensorize(train_rows);validation=tensorize(val_rows)
    results={};total=0
    for mode in MODES:
        torch.manual_seed(stable_seed('p12',seed,'matched_init'))
        m=model_for(mode,width)
        initial={k:v.detach().clone() for k,v in m.state_dict().items()}
        learning=fit(m,mode,tensors,validation,stable_seed('p12',seed,'matched_batch'),epochs)
        if not any(not torch.equal(initial[k],v) for k,v in m.state_dict().items()):
            raise AssertionError('no weight training')
        path=folder/'models'/f'{mode}.pt'
        torch.save(dict(format='population-research-p12',seed=seed,
                        mode=mode,width=width,history=HISTORY,actions=ACTIONS,
                        niches=NICHES,state_dict=m.state_dict()),path)
        learning['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        learning['parameters']=sum(p.numel() for p in m.parameters())
        total+=learning['updates']
        cases={k:assess(m,mode,rows,folder/'gallery'/mode/k if k=='unseen_grammar_gap9' else None)
               for k,rows in tests.items()}
        results[mode]=dict(training=learning,heldout=cases)
    audit=dict(format='population-research-p12',native=False,pretrained_slm=False,
               full_episode_selfplay=False,training='real torch CPU AdamW',
               seed=seed,torch_version=torch.__version__,numpy_version=np.__version__,
               split_seeds=splits,train_examples=len(train_rows),
               validation_examples=len(val_rows),optimizer_updates=total,
               models=results,protocol=dict(train_grammars=['train_a','train_b'],
                    withheld_grammar='heldout_c',train_depths=[5,7],
                    withheld_depth=9,full_history=HISTORY,truncated_history=8),
               limitations=['Synthetic P0 World, not verified current native Workbench contract',
                   'Controlled single-decision memory probe, not autonomous mission completion',
                   'Same three branch symbols and fixed goal-rule across cue grammars',
                   'Static world snapshots and no learned material identity semantics',
                   'No auxiliary label or history-order supervision; target next action only'])
    (folder/'results.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (folder/'metrics.csv').open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(['seed','mode','case','accuracy','both_correct','legal','pairs'])
        for mode,m in results.items():
            for case,v in m['heldout'].items():
                writer.writerow([seed,mode,case,v['accuracy'],v['paired_both_correct'],v['legal_action_rate'],v['pairs']])
    (folder/'manifest.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+
        '  '+p.relative_to(folder).as_posix()+'\n' for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name!='manifest.sha256'))
    return audit


def main():
    p=argparse.ArgumentParser(description='P12 held-out grammar and memory-gap experiment')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=649)
    p.add_argument('--train-worlds',type=int,default=48)
    p.add_argument('--validation-worlds',type=int,default=16)
    p.add_argument('--heldout-worlds',type=int,default=24)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--width',type=int,default=40)
    a=p.parse_args();r=run(**vars(a))
    print(json.dumps(dict(seed=r['seed'],optimizer_updates=r['optimizer_updates'],
        heldout={m:{k:v['accuracy'] for k,v in z['heldout'].items()}
                 for m,z in r['models'].items()}),sort_keys=True))

if __name__=='__main__':main()