"""P11: paired, host-legal, order-dependent goal-switch memory benchmark.

External synthetic CyberSand-like 22x22 worlds only. This is a falsification
experiment for P10's weak memory claim, not a native game runner, pretrained
SLM, or self-play. In paired episodes, the same observable world and action
multiset require opposite choices because the initial inspection order differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from copy import deepcopy

import numpy as np

from population_lab import DIRECTIONS, FEATURES, NICHES, World, stable_seed
from population_p10 import ACTIONS, HISTORY, observe, require_torch, shuffled

MODES = ('state_only', 'token_gru', 'history_sham')
BRANCH_PAIRS = (('E','N'), ('W','N'), ('E','W'))
ROOTS = {
    'train': ((3,11),(4,6),(15,7),(15,15)),
    'validation': ((4,15),(15,5)),
    'holdout': ((5,16),(16,6),(16,16)),
}



def build_model(mode, width=40):
    """Order/niche interaction, unlike P10's additive context-only student."""
    torch,nn=require_torch()
    if mode not in MODES:raise ValueError(mode)
    class MemoryStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode=mode
            self.width=width
            self.features=nn.Linear(len(FEATURES),width)
            self.niche=nn.Embedding(len(NICHES),width)
            self.actions=nn.Embedding(len(ACTIONS)+1,width)
            self.gru=nn.GRU(width,width,batch_first=True)
            self.fusion=nn.Sequential(nn.Linear(width*2,width),nn.Tanh(),
                                      nn.Linear(width,width),nn.Tanh())
            self.base=nn.Linear(width,1)
            self.order_head=nn.Linear(width,2)
        def forward(self,xs,mask,hist,niche,return_aux=False):
            z=torch.tanh(self.features(xs))
            n=self.niche(niche)
            if self.mode!='state_only':
                _,h=self.gru(self.actions(hist));ctx=h[-1]
            else:
                ctx=torch.zeros_like(n)
            fused=self.fusion(torch.cat([n,ctx],dim=-1))
            scores=(z*fused.unsqueeze(1)).sum(-1)/(self.width**.5)+self.base(z).squeeze(-1)
            masked=scores.masked_fill(~mask,-1e9)
            if return_aux:
                return masked,self.order_head(ctx)
            return masked
    return MemoryStudent()

def splits(seed, train_worlds=80, validation_worlds=24, heldout_worlds=36):
    sizes = {'train': train_worlds, 'validation': validation_worlds, 'holdout': heldout_worlds}
    out = {k: [stable_seed('p11',seed,k,i) for i in range(count)] for k,count in sizes.items()}
    if len(set(sum(out.values(), []))) != sum(map(len,out.values())):
        raise AssertionError('world split collision')
    return out


def start_world(seed, split='train', card=0):
    """Preserve obstacles/protection elsewhere; clear only a tiny legal pocket.

    The root spatial distribution is split-specific to test heldout geometry.
    Goal changes at step five, immediately before the order-dependent probe.
    """
    if split not in ROOTS or card not in (0,1):
        raise ValueError('unknown split/card')
    w=World.generate(seed, max_steps=10)
    choices=ROOTS[split]
    root=choices[stable_seed(seed,'p11_root',card)%len(choices)]
    w.cursor=root
    w.painted={root}
    w.parent={}
    w.history=[]
    w.steps=0
    w.switched=False
    w.heading='E'
    w.goal=(18,11) if card==0 else (3,3)
    w.second_goal=(4,18) if card==0 else (18,18)
    needed={root,(root[0]+1,root[1]),(root[0]-1,root[1]),
            (root[0],root[1]-1),(root[0],root[1]+1),
            (root[0]+1,root[1]+1),(root[0]-1,root[1]+1)}
    assert all(1<=x<w.width-1 and 1<=y<w.height-1 for x,y in needed)
    w.obstacles.difference_update(needed)
    if needed & w.protected:
        raise AssertionError('selected root intersects protected region')
    return w


def pair(seed, split, card, branches, niche):
    """Two true legal world rollouts, counterfactually paired at an identical state.

    E,RETRACE,N,RETRACE,S versus N,RETRACE,E,RETRACE,S (or variants).
    Both have identical painting/parent maps, cursor, goal and heading. The
    first surveyed branch acts as the hidden earlier mission cue; the goal
    switches after the final S move. The correct next paint direction E/W
    depends only on order (and niche's fixed orientation convention).
    """
    if branches not in BRANCH_PAIRS or niche not in NICHES:
        raise ValueError('unrecognised branch pair or niche')
    rows=[]
    for orientation in (0,1):
        w=start_world(seed,split,card)
        order=branches if orientation==0 else branches[::-1]
        for a in (order[0],'RETRACE',order[1],'RETRACE','S'):
            if a not in w.legal():
                raise AssertionError(f'precursor not host-legal: {a}')
            w.step(a)
        if not w.switched or not {'E','W'}.issubset(w.legal()):
            raise AssertionError('probe missing post-switch legal E/W choices')
        hist=[v['action'] for v in w.history]
        X,M,H=observe(w,hist)
        assert len(hist)==5 and H.tolist()[-5:]==[ACTIONS.index(a)+1 for a in hist]
        # Niche-conditioned mission direction, balanced within every pair.
        chosen_east = bool(orientation==0) ^ bool(NICHES.index(niche)%2)
        target='E' if chosen_east else 'W'
        rows.append(dict(xs=X,mask=M,history=H,niche=NICHES.index(niche),
                         target=ACTIONS.index(target),world=w,order=hist,
                         orientation=orientation,card=card,seed=seed,
                         branches=branches,mission='remember first survey order at goal switch'))
    a,b=rows
    if not np.array_equal(a['xs'],b['xs']) or not np.array_equal(a['mask'],b['mask']):
        raise AssertionError('paired present observations leaked historical cue')
    if (a['world'].painted!=b['world'].painted or a['world'].parent!=b['world'].parent or
        a['world'].cursor!=b['world'].cursor or a['world'].heading!=b['world'].heading or
        a['world'].goal!=b['world'].goal or a['world'].steps!=b['world'].steps):
        raise AssertionError('counterfactual pair not state-matched')
    if sorted(a['history'].tolist())!=sorted(b['history'].tolist()):
        raise AssertionError('action histogram mismatch')
    if a['target']==b['target']:
        raise AssertionError('paired goals not opposite')
    return a,b


def dataset(seeds, split, cards=(0,1)):
    examples=[]
    for seed in seeds:
        for card in cards:
            for branches in BRANCH_PAIRS:
                for niche in NICHES:
                    examples.extend(pair(seed,split,card,branches,niche))
    return examples


def to_tensors(examples):
    torch,_=require_torch()
    return (torch.as_tensor(np.stack([r['xs'] for r in examples]),dtype=torch.float32),
            torch.as_tensor(np.stack([r['mask'] for r in examples]),dtype=torch.bool),
            torch.as_tensor(np.stack([r['history'] for r in examples]),dtype=torch.long),
            torch.as_tensor([r['niche'] for r in examples],dtype=torch.long),
            torch.as_tensor([r['target'] for r in examples],dtype=torch.long))


def train(model, train_data, validation, seed, epochs=12, batch_size=128, lr=.006):
    torch,_=require_torch()
    gen=torch.Generator().manual_seed(seed)
    sham_gen=torch.Generator().manual_seed(stable_seed('p11','shuffle',seed))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    history=[];updates=0
    for epoch in range(epochs):
        model.train(); losses=[]
        for ix in torch.randperm(train_data[0].shape[0],generator=gen).split(batch_size):
            X,M,H,N,Y=(a[ix] for a in train_data)
            if model.mode=='history_sham':H=shuffled(H,sham_gen)
            logits,aux=model(X,M,H,N,return_aux=True)
            orientation=((Y==ACTIONS.index('E')) ^ (N%2!=0)).long()
            loss=torch.nn.functional.cross_entropy(logits,Y)+0.5*torch.nn.functional.cross_entropy(aux,orientation)
            if not torch.isfinite(loss):raise AssertionError('nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step();updates+=1;losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            X,M,H,N,Y=validation
            pred,aux=model(X,M,H,N,return_aux=True)
            val=float(torch.nn.functional.cross_entropy(pred,Y))
            orientation=((Y==ACTIONS.index('E')) ^ (N%2!=0)).long()
            aux_ce=float(torch.nn.functional.cross_entropy(aux,orientation))
        history.append(dict(epoch=epoch,train_ce=round(float(np.mean(losses)),6),
                            val_ce=round(val,6),val_order_ce=round(aux_ce,6)))
    return dict(updates=updates,epochs=epochs,history=history,
                final_val_ce=history[-1]['val_ce'])


def evaluate(model, examples, gallery_dir=None, max_gallery=36):
    torch,_=require_torch()
    model.eval();X,M,H,N,Y=to_tensors(examples)
    with torch.no_grad():
        logits=model(X,M,H,N)
        pred=logits.argmax(-1).numpy()
        ce=float(torch.nn.functional.cross_entropy(logits,Y))
        missing=H.clone().zero_()
        blind=model(X,M,missing,N).argmax(-1).numpy()
        swapped=model(X,M,H.reshape(-1,2,HISTORY).flip(1).reshape(-1,HISTORY),N).argmax(-1).numpy()
    target=Y.numpy()
    correct=(pred==target)
    if np.any(~M.numpy()[np.arange(len(pred)),pred]):
        raise AssertionError('illegal action issued')
    # Every two rows have identical current state and opposite reward target.
    paired=correct.reshape(-1,2).all(axis=1)
    flips=(pred.reshape(-1,2)[:,0]!=pred.reshape(-1,2)[:,1])
    results=dict(accuracy=round(float(correct.mean()),6),
                 pair_both_correct_rate=round(float(paired.mean()),6),
                 prediction_flip_rate=round(float(flips.mean()),6),
                 no_history_accuracy=round(float((blind==target).mean()),6),
                 swapped_history_accuracy=round(float((swapped==target).mean()),6),
                 cross_entropy=round(ce,6),
                 legal_action_rate=1., decisions=len(examples),
                 pair_count=len(examples)//2,
                 by_niche={n:round(float(correct[[i for i,r in enumerate(examples) if r['niche']==j]].mean()),6)
                           for j,n in enumerate(NICHES)})
    if gallery_dir is not None:
        gallery_dir.mkdir(parents=True,exist_ok=True)
        # Include successful and failed pairs, rather than cherry-picking.
        pick=np.linspace(0,len(examples)//2-1,min(max_gallery//2,len(examples)//2),dtype=int)
        for pair_i in pick:
            for direction in (0,1):
                idx=int(2*pair_i+direction)
                row=examples[idx]
                w=deepcopy(row['world'])
                action=ACTIONS[int(pred[idx])]
                before=w.cursor
                if action not in w.legal():raise AssertionError('invalid gallery action')
                w.step(action)
                if w.painted & (w.obstacles|w.protected):
                    raise AssertionError('painting inside protected/obstacle region')
                record=dict(synthetic=True,native=False,pretrained_slm=False,
                            seed=row['seed'],card=row['card'],niche=NICHES[row['niche']],
                            branches=row['branches'],orientation=row['orientation'],
                            mission=row['mission'],precursor=row['order'],
                            probe_state={'cursor':before,'legal':sorted([a for a in ACTIONS if row['mask'][ACTIONS.index(a)]]),
                                         'switched_goal':True},
                            desired=ACTIONS[row['target']],chosen=action,
                            correct=bool(correct[idx]),trace=w.history,
                            painted=sorted(w.painted),obstacles=sorted(w.obstacles),
                            protected=sorted(w.protected))
                (gallery_dir/f'pair{pair_i:03d}-order{direction}.json').write_text(json.dumps(record,indent=2)+'\n')
                # Deliberately retain original geometry and branch/paint history.
                from population_lab import draw_svg
                (gallery_dir/f'pair{pair_i:03d}-order{direction}.svg').write_text(draw_svg(w))
    return results


def run(output, seed=646, train_worlds=80, validation_worlds=24,
        heldout_worlds=36, epochs=24, width=40):
    if not (12<=train_worlds<=300 and 8<=validation_worlds<=100 and
            8<=heldout_worlds<=150 and 3<=epochs<=40 and 16<=width<=128):
        raise ValueError('experiment size outside bounded limits')
    torch,_=require_torch()
    torch.set_num_threads(1)
    split=splits(seed,train_worlds,validation_worlds,heldout_worlds)
    train_ex=dataset(split['train'],'train')
    val_ex=dataset(split['validation'],'validation')
    test_ex=dataset(split['holdout'],'holdout')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir();(output/'gallery').mkdir()
    ttrain=to_tensors(train_ex);tval=to_tensors(val_ex)
    model_info={}; heldout={};update_count=0
    for mode in MODES:
        torch.manual_seed(stable_seed('p11',seed,'matched_init'))
        model=build_model(mode,width)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        learning=train(model,ttrain,tval,stable_seed('p11',seed,'matched_optimizer'),epochs)
        if not any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):
            raise AssertionError('optimizer never modified network')
        checkpoint=output/'models'/f'{mode}.pt'
        torch.save(dict(format='population-research-p11-student',seed=seed,mode=mode,
                        width=width,feature_vocab=FEATURES,action_vocab=ACTIONS,
                        history=HISTORY,niches=NICHES,
                        state_dict=model.state_dict()),checkpoint)
        learning['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        learning['parameters']=sum(p.numel() for p in model.parameters())
        model_info[mode]=learning;update_count+=learning['updates']
        heldout[mode]=evaluate(model,test_ex,output/'gallery'/mode)
    # Dataset audit: the labels differ in every state-matched pair; target
    # and split seeds are never read from the model-side observation.
    result=dict(format='population-research-p11',seed=seed,kind='PAIRED_ORDER_DEPENDENT_GOAL_SWITCH',
                native=False,pretrained_slm=False,real_optimizer=True,model_side_only=True,
                source='CyberSand-like World.generate and host-legal action menus',
                cohorts=1,niches=list(NICHES),architectures=list(MODES),
                splits=split,split_sha256=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest(),
                train_examples=len(train_ex),validation_examples=len(val_ex),
                heldout_examples=len(test_ex),heldout_pairs=len(test_ex)//2,
                epochs=epochs,width=width,optimizer_updates=update_count,
                metrics=model_info,heldout=heldout,
                limitations=[
                    'Deliberately constructed short-horizon order-memory oracle, not emergent gameplay competence',
                    'The earlier mission cue is action-order information; no native CyberSand material sensor exists',
                    'Cleared seven-cell pockets ensure host-legal matched trajectories; rest of world retains obstacles and protected regions',
                    'From-scratch GRU sequence student, not a pretrained SLM, adapter, or policy-gradient/self-play agent',
                    'Distinct seeds and map roots are synthetic splits, not independent native gameplay acceptance',
                    'Torch checkpoint binary hashes may differ across Python/torch environments'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'manifest.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix()+'\n'
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.sha256'))
    return result


def main():
    parser=argparse.ArgumentParser(description='P11 CPU synthetic pairwise memory falsification')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=646)
    parser.add_argument('--train-worlds',type=int,default=80)
    parser.add_argument('--validation-worlds',type=int,default=24)
    parser.add_argument('--heldout-worlds',type=int,default=36)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--width',type=int,default=40)
    x=parser.parse_args()
    r=run(x.output,x.seed,x.train_worlds,x.validation_worlds,x.heldout_worlds,x.epochs,x.width)
    print(json.dumps(dict(seed=r['seed'],train_examples=r['train_examples'],
        optimizer_updates=r['optimizer_updates'],heldout=r['heldout']),sort_keys=True))

if __name__=='__main__':main()
