"""DG010: transfer frozen independently trained donor-neural activations into a game-law receiver.

Synthetic CyberSand-facing observations; no actual game changes or KV transplant.
Donor families: wet-seeking, branch/retrace, painting. Each family has +/-
independently trained neural policies, and train/evaluation donor SEEDS are disjoint.
The receiving model never sees law signs, authored donor targets, or query labels.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ('LEFT', 'RIGHT', 'UP', 'DOWN', 'PAINT', 'RETRACE')
N, D, PROBES = 6, 10, 10
HELDOUT = (1, 1, 1)
ALLELES = [x for x in itertools.product((-1, 1), repeat=3) if x != HELDOUT]
SIG = PROBES * N + 48  # frozen output logits plus true penultimate hidden-state statistics


def sha(x):
    return hashlib.sha256(x).hexdigest()


def world(rng):
    # goal, wet material, danger, novelty, direction inertia, branch potential,
    # paintability, history, solid proximity, clearance
    x = rng.uniform(-1, 1, size=(N,D)).astype('float32')
    for j in (1,2,6,8):
        x[:,j] = (rng.random(N) < (.42 if j==1 else .3 if j==2 else .68 if j==6 else .4)).astype('float32')
    for j in (3,5,7,9):
        x[:,j] = rng.random(N).astype('float32')
    legal = rng.random(N) > np.array([.14]*4+[.2,.3])
    legal[4] &= bool(x[4,6]); legal[:4] &= ~(x[:4,2] > .9)
    if not legal.any(): legal[int(rng.integers(N))] = True
    return x, legal.astype('bool')


def scores(x, legal, laws=(0,0,0), extra=False):
    # Baseline utility plus *distinct* signed atomic game laws.
    z=np.zeros(x.shape[:-1],dtype='float32')
    z[..., :4] = .7*x[..., :4, 0]+.4*x[..., :4, 3]+.3*x[..., :4,4]-.4*x[..., :4,2]
    z[...,4] = .1+.65*x[...,4,6]+.1*x[...,4,1]
    z[...,5] = -.1+.65*x[...,5,7]
    wet,branch,paint = laws
    z[..., :4] += wet*2.2*(2*x[..., :4,1]-1)
    z[..., :4] += branch*1.9*(2*x[..., :4,5]-1)
    z[..., 5] -= branch*1.1
    z[..., 4] += paint*2.9*(.25+.75*x[...,4,1])
    # Pairwise coupling is identifiable across the seven visible triples.
    z[...,4] += .65*wet*paint*(2*x[...,4,1]-1)
    z[...,5] += .45*wet*branch*x[...,5,7]
    # This truly withheld three-way interaction is NOT identifiable from training.
    if extra and tuple(laws)==HELDOUT:
        z[...,5] += 7.*(x[...,5,7]+.25)
    z=np.where(legal,z,-1e4).astype('float32')
    return z


def with_action_id(x):
    eye=torch.eye(N, dtype=x.dtype, device=x.device)[None,:,:].expand(x.shape[0],-1,-1)
    return torch.cat((x,eye),-1)


class Donor(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Linear(D+N,48)
        self.b=nn.Linear(48,24)
        self.c=nn.Linear(24,1)
    def forward(self,x,return_hidden=False):
        h=F.gelu(self.b(F.gelu(self.a(with_action_id(x)))))
        z=self.c(h).squeeze(-1)
        return (z,h) if return_hidden else z


def train_donor(family, sign, seed, steps=230):
    # Authored scoring laws are *only teacher targets* for independently trained
    # neural donor policies. The receiver sees neither labels nor this function.
    rng=np.random.default_rng(seed)
    # Generate paired observation/legality worlds.
    pairs=[world(rng) for _ in range(1700)]
    x=torch.from_numpy(np.stack([a for a,_ in pairs])); legal=torch.from_numpy(np.stack([b for _,b in pairs]))
    law=[0,0,0];law[family]=sign
    target=torch.from_numpy(scores(x.numpy(),legal.numpy(),law))
    torch.manual_seed(seed)
    m=Donor();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.0003)
    for t in range(steps):
        ix=torch.as_tensor(rng.integers(len(x),size=112),dtype=torch.long)
        z=m(x[ix]); y=target[ix]; mask=legal[ix]
        # Regression recovers relative utility under each donor's signed law.
        loss=((z-y).square()*mask).sum()/mask.sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
    with torch.no_grad():
        valid=[world(rng) for _ in range(300)]
        v=torch.from_numpy(np.stack([a for a,_ in valid]));mask=torch.from_numpy(np.stack([b for _,b in valid]))
        oracle=torch.from_numpy(scores(v.numpy(),mask.numpy(),law))
        predicted=m(v).masked_fill(~mask,-1e4)
        acc=float((predicted.argmax(-1)==oracle.argmax(-1)).float().mean())
    m.eval()
    return m,acc


def get_signature(m, probe):
    with torch.no_grad():
        z,h=m(probe,True)
        # Learned-model-derived outputs AND authentic hidden activations;
        # no oracle law bits or authored action labels in the transfer.
        z=(z-z.mean(-1,keepdim=True))/2.5
        sig=torch.cat((z.flatten(),h.mean((0,1)),h.std((0,1))),0)
        assert sig.numel()==SIG and torch.isfinite(sig).all()
        return sig


def donor_bank(seed, steps, probes):
    bank={}; report={}
    for family in range(3):
        for sign in (-1,1):
            donor_seed=seed+family*1000+(sign+1)*73
            m,acc=train_donor(family,sign,donor_seed,steps)
            bank[(family,sign)]=(m,get_signature(m,probes))
            report[f'{family}:{sign}']=dict(seed=donor_seed,accuracy=round(acc,5))
    return bank,report


def samples(seed,count,split):
    rng=np.random.default_rng(seed)
    pool=[HELDOUT] if split=='sealed' else ALLELES
    xs,ls,ys,law,keys=[],[],[],[],[]
    for _ in range(count):
        f,l=world(rng);a=pool[int(rng.integers(len(pool)))]
        xs.append(f);ls.append(l);law.append(a)
        ys.append(int(scores(f,l,a).argmax()))
        keys.append(sha(f.tobytes()+l.tobytes()))
    assert len(set(keys))==count
    return dict(x=torch.from_numpy(np.stack(xs)),legal=torch.from_numpy(np.stack(ls)),
                y=torch.as_tensor(ys),law=torch.as_tensor(law),keys=keys)


def signatures(laws,bank):
    return torch.stack([torch.stack([bank[(j,int(l[j]))][1] for j in range(3)])
                        for l in laws.tolist()])


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(SIG,64),nn.GELU(),nn.Linear(64,16),nn.Tanh())
        self.policy=nn.Sequential(nn.Linear(D+N+48,96),nn.GELU(),nn.Linear(96,64),
                                  nn.GELU(),nn.Linear(64,1))
    def forward(self,x,legal,source):
        n=x.shape[0]
        ctx=self.context(source).reshape(n,48)
        inp=torch.cat((with_action_id(x),ctx[:,None,:].expand(-1,N,-1)),-1)
        return self.policy(inp).squeeze(-1).masked_fill(~legal,-1e4)


class Blind(nn.Module):
    """Exact receiver architecture/capacity, trained with an absent source."""
    def __init__(self):
        super().__init__()
        self.net=Receiver()
    def forward(self,x,legal):
        null=x.new_zeros((x.shape[0],3,SIG))
        return self.net(x,legal,null)


def accuracy(logits,c):
    assert c['legal'].gather(1,logits.argmax(-1)[:,None]).all()
    return round(float((logits.argmax(-1)==c['y']).float().mean()),5)


def channel(s,mode):
    # Identical receiver capacity in all arms; suppress the actual model
    # outputs or true penultimate activation statistics before encoding.
    if mode=='all': return s
    q=s.clone()
    if mode=='hidden':q[...,:PROBES*N]=0
    elif mode=='logits':q[...,PROBES*N:]=0
    else:raise ValueError(mode)
    return q

def eval_set(r,b,c,bank,other,seed,mode='all'):
    r.eval();b.eval();s=channel(signatures(c['law'],bank),mode)
    with torch.no_grad():
        real=r(c['x'],c['legal'],s)
        blind=b(c['x'],c['legal'])
        absent=r(c['x'],c['legal'],s*0)
        swapped=s.clone()
        swapped[:,0]=channel(signatures(c['law']*torch.as_tensor([-1,1,1]),bank),mode)[:,0]
        wrong=r(c['x'],c['legal'],swapped)
        foreign=r(c['x'],c['legal'],channel(signatures(c['law'],other),mode))
        # Context carries no query identity: each source signature is frozen.
        observed=[real,blind,absent,wrong,foreign]
        for pred in observed:assert c['legal'].gather(1,pred.argmax(-1)[:,None]).all()
        target_wrong=[]
        for i in range(len(c['y'])):
            a=c['law'][i].tolist();a[0]=-a[0]
            target_wrong.append(int(scores(c['x'][i].numpy(),c['legal'][i].numpy(),a).argmax()))
        cf_y=torch.as_tensor(target_wrong)
        output=dict(correct_source=accuracy(real,c),source_absent=accuracy(absent,c),
            wrong_wet_donor=accuracy(wrong,c),query_only=accuracy(blind,c),
            other_seed_donors=accuracy(foreign,c),
            switched_donor_matches_counterfactual=round(float((wrong.argmax(-1)==cf_y).float().mean()),5),
            actual_counterfactual_change_rate=round(float((cf_y!=c['y']).float().mean()),5),
            predicted_counterfactual_change_rate=round(float((wrong.argmax(-1)!=real.argmax(-1)).float().mean()),5))
        changed=(cf_y!=c['y']);predchange=(wrong.argmax(-1)!=real.argmax(-1))
        output['cf_change_recall']=round(float((changed&predchange).sum()/changed.sum().clamp_min(1)),5)
        # Completely unseen three-body law: deliberately unidentifiable.
        if torch.all(c['law']==torch.as_tensor(HELDOUT)):
            extra=torch.as_tensor([int(scores(c['x'][i].numpy(),c['legal'][i].numpy(),HELDOUT,True).argmax())
                                   for i in range(len(c['y']))])
            output['unseen_third_order_oracle_action_match']=round(float((real.argmax(-1)==extra).float().mean()),5)
            output['unseen_third_order_gold_action_changes']=round(float((extra!=c['y']).float().mean()),5)
    return output


def selftest():
    c=samples(123,100,'train'); d=samples(124,100,'sealed')
    assert not set(c['keys']).intersection(d['keys'])
    assert not torch.any(torch.all(c['law']==torch.as_tensor(HELDOUT),-1))
    assert torch.all(d['law']==torch.as_tensor(HELDOUT))
    assert torch.all(c['legal'].gather(1,c['y'][:,None]))
    m=Donor(); probes=torch.randn(PROBES,N,D)
    assert get_signature(m,probes).shape==(SIG,)
    s=torch.randn(5,3,SIG);x=c['x'][:5];legal=c['legal'][:5]
    r=Receiver();assert r(x,legal,s).shape==(5,N)
    assert legal.gather(1,r(x,legal,s).argmax(-1)[:,None]).all()
    a,l=world(np.random.default_rng(17))
    assert not np.array_equal(scores(a,l,(1,0,0)),scores(a,l,(-1,0,0)))
    assert not np.array_equal(scores(a,l,HELDOUT),scores(a,l,HELDOUT,True))
    print('DG010 DISJOINT/LEGAL/THIRD-ORDER/SOURCE-STATE PASS')


def run(out,seed,donor_steps,receiver_steps,train_count,sealed_count):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
    t0=time.monotonic(); path=Path(out);path.mkdir(parents=True,exist_ok=True)
    prng=np.random.default_rng(1901)
    probes=torch.from_numpy(np.stack([world(prng)[0] for _ in range(PROBES)]))
    assert sha(probes.numpy().tobytes())
    train_bank, train_donor_metrics=donor_bank(seed+10000,donor_steps,probes)
    eval_bank, eval_donor_metrics=donor_bank(seed+30000,donor_steps,probes)
    training=samples(seed+1,train_count,'train')
    validation=samples(seed+2,500,'val')
    sealed=samples(seed+3,sealed_count,'sealed')
    assert not (set(training['keys'])&set(validation['keys']) or
                set(training['keys'])&set(sealed['keys']) or
                set(validation['keys'])&set(sealed['keys']))
    source_train=signatures(training['law'],train_bank)
    models={mode:Receiver() for mode in ('all','logits','hidden')}
    opts={mode:torch.optim.AdamW(model.parameters(),lr=.0017,weight_decay=.0001) for mode,model in models.items()}
    b=Blind();bopt=torch.optim.AdamW(b.parameters(),lr=.0017,weight_decay=.0001)
    rng=np.random.default_rng(seed+900); history=[]
    for step in range(receiver_steps):
        idx=torch.from_numpy(rng.integers(train_count,size=96))
        x,legal,y=training['x'][idx],training['legal'][idx],training['y'][idx]
        losses={}
        for mode,r in models.items():
            r.train()
            logits=r(x,legal,channel(source_train[idx],mode))
            loss=F.cross_entropy(logits,y)
            opts[mode].zero_grad();loss.backward();nn.utils.clip_grad_norm_(r.parameters(),3);opts[mode].step()
            losses[mode]=round(float(loss.detach()),5)
        b.train();pred=b(x,legal);bl=F.cross_entropy(pred,y)
        bopt.zero_grad();bl.backward();bopt.step()
        if step==0 or (step+1)%150==0 or step+1==receiver_steps:
            stats={mode:eval_set(r,b,validation,train_bank,eval_bank,seed+step,mode) for mode,r in models.items()}
            row=dict(step=step+1,losses=losses,blind_loss=round(float(bl.detach()),5),validation=stats)
            print(json.dumps(row),flush=True);history.append(row)
    heldout={mode:eval_set(r,b,sealed,eval_bank,train_bank,seed+777,mode) for mode,r in models.items()}
    # Additional causal check: same unseen triple worlds with untouched donor states
    # versus a source bank from opposite law. Keep BOTH query and legal mask fixed.
    with torch.no_grad():
        sig=signatures(sealed['law'],eval_bank)
        pred=models['all'](sealed['x'],sealed['legal'],sig).argmax(-1)
    decisions=[]
    for i in range(min(64,sealed_count)):
        decisions.append(dict(query_sha=sealed['keys'][i],legal=[ACTIONS[j] for j,v in enumerate(sealed['legal'][i]) if v],
                              chosen=ACTIONS[int(pred[i])],expert=ACTIONS[int(sealed['y'][i])]))
    for bank,name in [(train_bank,'train'),(eval_bank,'sealed')]:
        for (family,sign),(m,signature) in bank.items():
            torch.save(m.state_dict(),path/f'donor-{name}-family{family}-sign{sign}.pt')
    for mode,r in models.items():torch.save(r.state_dict(),path/f'receiver-{mode}.pt')
    torch.save(b.state_dict(),path/'blind.pt')
    result=dict(experiment='DG010',seed=seed, donor_training_steps=donor_steps,
                receiver_training_steps=receiver_steps,train_count=train_count,sealed_count=sealed_count,
                train_donor_metrics=train_donor_metrics, eval_donor_metrics=eval_donor_metrics,
                donor_architecture='independent randomly initialized neural regressors, frozen model output+hidden activations',
                nonclaim='NOT native KV, general physics discovery, CyberSand native trajectory, or unsupervised donor learning',
                heldout=heldout,history=history,calibration_probe_sha256=sha(probes.numpy().tobytes()),
                seconds=round(time.monotonic()-t0,2))
    (path/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (path/'original_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    hashes={p.name:sha(p.read_bytes()) for p in sorted(path.iterdir()) if p.is_file()}
    (path/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('DG010 HELDOUT '+json.dumps(heldout),flush=True)
    print('DG010 ARTIFACT HASHES '+json.dumps(hashes),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--out',default='/tmp/dg010')
    p.add_argument('--seed',type=int,default=26011615)
    p.add_argument('--donor-steps',type=int,default=230)
    p.add_argument('--receiver-steps',type=int,default=700)
    p.add_argument('--train-cases',type=int,default=4500)
    p.add_argument('--sealed-cases',type=int,default=900)
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(a.out,a.seed,a.donor_steps,a.receiver_steps,a.train_cases,a.sealed_cases)
