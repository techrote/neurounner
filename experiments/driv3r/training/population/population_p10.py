"""P10: teacher-to-small-sequence-model distillation on legal synthetic game menus.

CPU-only, external research. A *from-scratch* recurrent token model is trained
on action histories plus per-candidate host features. It is NOT a pretrained
SLM, production policy, policy-gradient method, or native CyberSand runtime.
Compare it with state-only and order-destroyed history controls; retain both
successful and failed checkpoints. torch is optional for P0..P9, needed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from population_lab import DIRECTIONS, FEATURES, NICHES, World, draw_svg, softmax, stable_seed, utility
from population_p1 import load_policy

ACTIONS = tuple(DIRECTIONS) + ('RETRACE', 'STOP')
HISTORY = 8
ARCHITECTURES = ('state_only', 'token_gru', 'history_sham')


def require_torch():
    import torch
    from torch import nn
    return torch, nn


def scenario(seed, card=0, horizon=36):
    """Read-only-compatible P0 grid dynamics; withheld cards alter geometry."""
    if card not in (0, 1, 2):
        raise ValueError('unknown scenario')
    w = World.generate(seed, horizon)
    if card == 1:  # TRAIN: deterministic retrace/recovery cul-de-sac
        w.obstacles.update({(5,11), (4,10), (4,12)})
        w.obstacles.discard((4,11))
    if card == 2:  # HELDOUT-only unseen staggered wall morphology
        w.obstacles.update((x,6) for x in range(2,20) if x not in (4,14))
        w.obstacles.update((x,16) for x in range(2,20) if x not in (9,18))
        w.goal=(19,4)
        w.second_goal=(4,18)
    w.obstacles.difference_update(w.painted | w.protected | {w.goal,w.second_goal})
    w.protected.difference_update(w.painted | {w.goal,w.second_goal})
    return w


def observe(w, history):
    """Fixed action vocabulary, but host decides the legal menu at every step."""
    legal=w.legal()
    xs=np.zeros((len(ACTIONS),len(FEATURES)),dtype=np.float32)
    mask=np.zeros((len(ACTIONS),),dtype=np.bool_)
    for a in legal:
        i=ACTIONS.index(a)
        xs[i]=w.feature(a)
        mask[i]=True
    hist=np.zeros((HISTORY,),dtype=np.int64)
    seq=[ACTIONS.index(a)+1 for a in history[-HISTORY:]]
    if seq:
        hist[-len(seq):]=seq
    assert mask.any() and mask[ACTIONS.index('STOP')]
    return xs, mask, hist


def source_splits(seed, train_worlds, validation_worlds, heldout_worlds):
    splits={part:[stable_seed('p10',seed,part,i) for i in range(count)]
            for part,count in [('train',train_worlds),('validation',validation_worlds),('holdout',heldout_worlds)]}
    flat=sum(splits.values(),[])
    if len(flat)!=len(set(flat)):
        raise AssertionError('split seed collision')
    return splits


def samples(teachers, seeds, teacher_variants=2, horizon=36):
    """Explore teacher rollouts with legal random/off-policy transitions.

    TRAIN-only teacher perturbations form real niche-specialised distillation
    examples. The teacher target is a probability distribution, not a forged
    'correct' action or final score.
    """
    examples=[]
    for ni,n in enumerate(NICHES):
        for vi in range(teacher_variants):
            base=teachers[n]
            teacher=base if vi==0 else base.mutate(stable_seed('p10',n,vi,'teacher'))
            for si,seed in enumerate(seeds):
                card=(si+vi)%2  # no heldout morphology enters TRAIN/validation
                w=scenario(seed,card,horizon)
                rng=np.random.default_rng(stable_seed('p10',seed,n,vi,'roll'))
                hist=[]
                while w.steps<w.max_steps and not w.stopped:
                    X,mask,tokens=observe(w,hist)
                    available=np.flatnonzero(mask)
                    teacher_p=teacher.predict(X[available].astype(np.float64))
                    target=np.zeros(len(ACTIONS),np.float32)
                    target[available]=teacher_p
                    examples.append((X,mask,tokens,ni,target))
                    if rng.random()<.20:
                        eligible=[i for i in available if ACTIONS[i]!='STOP']
                        chosen=int(rng.choice(eligible or available.tolist()))
                    else:
                        chosen=int(rng.choice(available,p=teacher_p))
                    action=ACTIONS[chosen]
                    assert action in w.legal()
                    w.step(action);hist.append(action)
    return examples


def tensorize(examples, device='cpu'):
    torch,_=require_torch()
    if not examples:
        raise ValueError('empty training sample')
    arr=[]
    for i,dtype in enumerate((torch.float32,torch.bool,torch.long,torch.long,torch.float32)):
        arr.append(torch.as_tensor(np.stack([row[i] for row in examples]),dtype=dtype,device=device))
    return tuple(arr)


def build_model(mode, width=40):
    torch,nn=require_torch()
    if mode not in ARCHITECTURES:
        raise ValueError(mode)
    class CandidateSequenceStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.features=nn.Linear(len(FEATURES),width)
            self.niche=nn.Embedding(len(NICHES),width)
            self.actions=nn.Embedding(len(ACTIONS)+1,width)
            self.gru=nn.GRU(width,width,batch_first=True)
            self.base=nn.Linear(width,1)
            self.mode=mode
            self.width=width
        def forward(self, xs, mask, hist, niche):
            z=torch.tanh(self.features(xs))
            ctx=self.niche(niche)
            if self.mode!='state_only':
                emb=self.actions(hist)
                _,h=self.gru(emb)
                ctx=ctx+h[-1]
            score=(z*ctx.unsqueeze(1)).sum(-1)/(self.width**.5)+self.base(z).squeeze(-1)
            return score.masked_fill(~mask,-1e9)
    return CandidateSequenceStudent()


def shuffled(history,rng):
    """Order control preserves every row's action-token multiset and padding."""
    out=history.clone()
    torch,_=require_torch()
    for row in out:
        valid=(row!=0).nonzero(as_tuple=False).view(-1)
        if valid.numel()>1:
            order=torch.randperm(valid.numel(),generator=rng)
            row[valid]=row[valid[order]]
    return out


def fit(model, train, validation, seed, epochs=5, batch_size=128, lr=.004):
    torch,_=require_torch()
    torch.manual_seed(seed)
    gen=torch.Generator().manual_seed(seed)
    sham_gen=torch.Generator().manual_seed(stable_seed('p10',seed,'sham_order'))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    updates=0; training=[]
    for epoch in range(epochs):
        model.train(); indices=torch.randperm(train[0].shape[0],generator=gen)
        epoch_loss=[]
        for ids in indices.split(batch_size):
            X,M,H,N,Y=(a[ids] for a in train)
            if model.mode=='history_sham':
                H=shuffled(H,sham_gen)
            logits=model(X,M,H,N)
            loss=-(Y*torch.log_softmax(logits,-1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise AssertionError('nonfinite student training')
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step();updates+=1;epoch_loss.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            X,M,H,N,Y=validation
            valid=float((-(Y*torch.log_softmax(model(X,M,H,N),-1)).sum(-1)).mean())
        training.append(dict(epoch=epoch,train_ce=float(np.mean(epoch_loss)),val_ce=valid))
    return {'updates':updates,'history':training,'best_val_ce':min(x['val_ce'] for x in training),
            'final_val_ce':training[-1]['val_ce']}


def play_student(model, niche, seed, card=0, horizon=36, teacher=None):
    torch,_=require_torch()
    model.eval()
    w=scenario(seed,card,horizon)
    prior=[]; reward=0.;agree=0; matched=0
    for _ in range(horizon):
        X,M,H=observe(w,prior)
        t=torch.as_tensor(X[None]); mask=torch.as_tensor(M[None]); history=torch.as_tensor(H[None])
        ni=torch.tensor([NICHES.index(niche)])
        with torch.no_grad():
            logits=model(t,mask,history,ni)[0]
        chosen=int(logits.argmax())
        action=ACTIONS[chosen]
        if action not in w.legal():
            raise AssertionError('illegal student action')
        if teacher is not None:
            legal=np.flatnonzero(M)
            t_choice=legal[int(np.argmax(teacher.logits(X[legal].astype(np.float64))))]
            agree+=int(chosen==t_choice);matched+=1
        reward+=utility(niche,X[chosen],w)
        w.step(action);prior.append(action)
        if w.painted & (w.obstacles | w.protected):
            raise AssertionError('illegal painting')
        if w.stopped:break
    return dict(reward=reward,world=w,agreement=agree/max(1,matched),
                retraces=prior.count('RETRACE'),decisions=len(prior),
                painted=len(w.painted)-1,early_stop=int(w.stopped and w.steps<horizon))



def play_teacher(policy, niche, seed, card=0, horizon=36):
    w=scenario(seed,card,horizon)
    reward=0.
    while w.steps<w.max_steps and not w.stopped:
        legal=w.legal()
        X=np.stack([w.feature(a) for a in legal])
        choice=policy.choice(X)
        reward+=utility(niche,X[choice],w)
        w.step(legal[choice])
    return dict(reward=reward, early_stop=int(w.stopped and w.steps<horizon),
                painted=len(w.painted)-1, retraces=sum(h['action']=='RETRACE' for h in w.history))


def run(p0_output, output, seed=643, train_worlds=16, validation_worlds=5,
        heldout_worlds=6, epochs=5, teacher_variants=2, width=40, horizon=36):
    if not (5<=train_worlds<=100 and 3<=validation_worlds<=60 and
            3<=heldout_worlds<=60 and 2<=epochs<=50 and 1<=teacher_variants<=5
            and 16<=width<=128 and 16<=horizon<=100):
        raise ValueError('invalid bounded study')
    torch,_=require_torch()
    torch.set_num_threads(1)
    torch.manual_seed(stable_seed('p10',seed,'torch'))
    np.random.seed(stable_seed('p10',seed,'numpy'))
    p0_output=Path(p0_output);output=Path(output)
    source=json.loads((p0_output/'results.json').read_text())
    if source.get('format')!='population-research-v1' or source.get('language_model') is not False:
        raise ValueError('requires proven synthetic P0 parent')
    teachers={n:load_policy(p0_output/'models'/f'{n}.json') for n in NICHES}
    splits=source_splits(seed,train_worlds,validation_worlds,heldout_worlds)
    p0seeds=[stable_seed(source['seed'],name,i)
             for name,count in (('train',source['train_seed_count']),
                                ('selection',source['selection_seed_count']),
                                ('test',source['heldout_seed_count'])) for i in range(count)]
    if set(sum(splits.values(),[])) & set(p0seeds):
        raise AssertionError('P0/SLM split leakage')
    train_ex=samples(teachers,splits['train'],teacher_variants,horizon)
    val_ex=samples(teachers,splits['validation'],teacher_variants,horizon)
    train=tensorize(train_ex);val=tensorize(val_ex)
    output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir();(output/'gallery').mkdir()
    metrics={};rollouts={}
    for mode in ARCHITECTURES:
        init_seed=stable_seed('p10',seed,'matched_init')
        torch.manual_seed(init_seed)
        model=build_model(mode,width)
        init={k:v.detach().clone() for k,v in model.state_dict().items()}
        trace=fit(model,train,val,stable_seed('p10',seed,'matched_train_order'),epochs=epochs)
        changed=any(not torch.equal(v,init[k]) for k,v in model.state_dict().items())
        if not changed:raise AssertionError('optimizer never updated student')
        path=output/'models'/f'{mode}.pt'
        torch.save({'architecture':mode,'state_dict':model.state_dict(),'width':width,
                    'seed':seed,'action_vocab':ACTIONS,'feature_vocab':FEATURES,
                    'history':HISTORY,'niches':NICHES},path)
        trace['parameter_count']=sum(p.numel() for p in model.parameters())
        trace['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        summary={}
        for card in (0,2):
            per_niche={}
            for niche in NICHES:
                trials=[play_student(model,niche,s,card,horizon,teachers[niche])
                        for s in splits['holdout']]
                per_niche[niche]={k:round(float(np.mean([t[k] for t in trials])),5)
                                  for k in ('reward','agreement','retraces','decisions','painted','early_stop')}
                world=trials[0]['world']
                label=f'{mode}-{niche}-card{card}'
                (output/'gallery'/f'{label}.svg').write_text(draw_svg(world))
                (output/'gallery'/f'{label}.json').write_text(json.dumps(dict(
                    synthetic=True,native=False,pretrained_slm=False,mode=mode,
                    niche=niche,seed=splits['holdout'][0],card=card,
                    trace=world.history,painted=sorted(world.painted),
                    obstacles=sorted(world.obstacles),protected=sorted(world.protected)),indent=2)+'\n')
            summary[str(card)]={k:round(float(np.mean([per_niche[n][k] for n in NICHES])),5)
                                for k in ('reward','agreement','retraces','decisions','painted','early_stop')}
            summary[str(card)]['by_niche']=per_niche
        metrics[mode]=trace
        rollouts[mode]=summary
    teacher_control={str(card):{k:round(float(np.mean([
        play_teacher(teachers[n],n,s,card,horizon)[k]
        for n in NICHES for s in splits['holdout']])),5)
        for k in ('reward','early_stop','painted','retraces')}
        for card in (0,2)}
    result=dict(format='population-research-p10',seed=seed,
                kind='FROM_SCRATCH_SMALL_RECURRENT_ACTION_LM_DISTILLATION',
                native=False,pretrained_slm=False,game_code_modified=False,
                real_optimizer=True,model_side_only=True,
                teacher='P0 Adam-trained synthetic niche MLPs + mutations',
                architectures=list(ARCHITECTURES),niches=list(NICHES),
                splits=splits,split_sha256=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
                training_examples=len(train_ex),validation_examples=len(val_ex),
                heldout_worlds=heldout_worlds,teacher_variants=teacher_variants,
                horizon=horizon,epochs=epochs,width=width,
                metrics=metrics,heldout=rollouts,teacher_control=teacher_control,limitations=[
                    'From-scratch tiny GRU action-token sequence model; NOT a pretrained language model or adapter',
                    'Teacher-supervised distillation, not reinforcement learning or real multi-agent self-play',
                    'Synthetic 22x22 legal-action host; not CyberSand native-game deployment or gameplay quality',
                    'P0 teacher and handcrafted synthetic utility have training-set inductive bias',
                    'History-sham is permuted during training but evaluated with intact history; not a perfect architecture control',
                    'Single cohort cannot establish behavioural generalisation superiority'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'manifest.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(output))+'\n'
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.sha256'))
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--p0-output',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--seed',type=int,default=643)
    a.add_argument('--train-worlds',type=int,default=16)
    a.add_argument('--validation-worlds',type=int,default=5)
    a.add_argument('--heldout-worlds',type=int,default=6)
    a.add_argument('--epochs',type=int,default=5)
    a.add_argument('--teacher-variants',type=int,default=2)
    a.add_argument('--width',type=int,default=40)
    a.add_argument('--horizon',type=int,default=36)
    x=a.parse_args()
    result=run(x.p0_output,x.output,x.seed,x.train_worlds,x.validation_worlds,
               x.heldout_worlds,x.epochs,x.teacher_variants,x.width,x.horizon)
    print(json.dumps({'seed':result['seed'],'examples':result['training_examples'],
                      'metrics':result['heldout']},sort_keys=True))

if __name__=='__main__':main()
