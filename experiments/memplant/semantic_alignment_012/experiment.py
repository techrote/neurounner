"""DG012: learn game actions from semantically scrambled neural donor observations.

Research-only synthetic CyberSand-facing game-law transfer. Source specialists are
independently trained on private permutations/reflections of ten observation axes.
The receiver never sees those permutations, law labels, donor identity, donor
weights, authored utilities or internal activations: only paired calibration
(world features, frozen neural output) or a raw-logit control.

This is *not* native KV transfer, game-code integration or physics discovery.
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

ACTIONS = ('LEFT','RIGHT','UP','DOWN','PAINT','RETRACE')
N,D,P = 6,10,24
SIG = P*N
HELDOUT = (1,1,1)
ALLELES = [a for a in itertools.product((-1,1), repeat=3) if a != HELDOUT]


def sha(b): return hashlib.sha256(b).hexdigest()


def world(rng):
    """Local action-wise observations (not a native CyberSand snapshot)."""
    x=rng.uniform(-1,1,size=(N,D)).astype('float32')
    for j,pr in ((1,.42),(2,.30),(6,.68),(8,.40)):
        x[:,j]=(rng.random(N)<pr).astype('float32')
    for j in (3,5,7,9): x[:,j]=rng.random(N).astype('float32')
    legal=rng.random(N)>np.array([.14]*4+[.2,.3])
    legal[4] &= bool(x[4,6]);legal[:4] &= ~(x[:4,2]>.9)
    if not legal.any():legal[int(rng.integers(N))]=True
    return x,legal.astype('bool')


def scores(x,legal,laws=(0,0,0),extra=False):
    z=np.zeros(x.shape[:-1],dtype='float32')
    z[...,:4]=.7*x[...,:4,0]+.4*x[...,:4,3]+.3*x[...,:4,4]-.4*x[...,:4,2]
    z[...,4]=.1+.65*x[...,4,6]+.1*x[...,4,1]
    z[...,5]=-.1+.65*x[...,5,7]
    wet,branch,paint=laws
    z[...,:4] += wet*2.2*(2*x[...,:4,1]-1)
    z[...,:4] += branch*1.9*(2*x[...,:4,5]-1)
    z[...,5] -= branch*1.1
    z[...,4] += paint*2.9*(.25+.75*x[...,4,1])
    z[...,4] += .65*wet*paint*(2*x[...,4,1]-1)
    z[...,5] += .45*wet*branch*x[...,5,7]
    if extra and tuple(laws)==HELDOUT:z[...,5]+=7*(x[...,5,7]+.25)
    return np.where(legal,z,-1e4).astype('float32')


def dataset(seed,count,kind):
    rng=np.random.default_rng(seed);xs=[];ls=[];laws=[];ys=[];keys=[]
    for _ in range(count):
        x,l=world(rng)
        law=HELDOUT if kind=='sealed' else (tuple((-1,1)[int(q)] for q in rng.integers(0,2,size=3)) if kind=='mixed' else ALLELES[int(rng.integers(len(ALLELES)))])
        xs.append(x);ls.append(l);laws.append(law);ys.append(int(scores(x,l,law).argmax()))
        keys.append(sha(x.tobytes()+l.tobytes()))
    assert len(set(keys))==count
    return dict(x=torch.from_numpy(np.stack(xs)),legal=torch.from_numpy(np.stack(ls)),
                laws=torch.tensor(laws,dtype=torch.long),y=torch.tensor(ys),keys=keys)


def with_action_id(x):
    eye=torch.eye(N,dtype=x.dtype,device=x.device)[None,:,:].expand(x.shape[0],-1,-1)
    return torch.cat((x,eye),-1)


class Specialist(nn.Module):
    def __init__(self,kind,perm,reflect):
        super().__init__()
        self.kind=kind
        self.register_buffer('perm',torch.as_tensor(perm,dtype=torch.long))
        self.register_buffer('reflect',torch.as_tensor(reflect,dtype=torch.float32))
        dims=(D+N,32,16) if kind=='narrow' else (D+N,96,48) if kind=='wide' else (D+N,64,64)
        self.a=nn.Linear(dims[0],dims[1]);self.b=nn.Linear(dims[1],dims[2]);self.c=nn.Linear(dims[2],1)
        if kind=='gated':self.g=nn.Linear(dims[0],dims[2])
    def forward(self,x):
        # Private donor coordinate system: receiver gets no permutation/reflection.
        hidden=x.index_select(-1,self.perm)*self.reflect
        z=with_action_id(hidden)
        if self.kind=='gated':
            h=F.silu(self.b(F.silu(self.a(z))))*torch.sigmoid(self.g(z))
        else:h=F.gelu(self.b(F.gelu(self.a(z))))
        return self.c(h).squeeze(-1)


def train_specialist(kind,family,sign,seed,steps):
    rng=np.random.default_rng(seed)
    perm=rng.permutation(D);reflect=rng.choice(np.array([-1.,1.],dtype='float32'),size=D)
    w=[world(rng) for _ in range(1400)]
    x=torch.from_numpy(np.stack([a for a,_ in w]));legal=torch.from_numpy(np.stack([b for _,b in w]))
    laws=[0,0,0];laws[family]=sign
    target=torch.from_numpy(scores(x.numpy(),legal.numpy(),laws))
    torch.manual_seed(seed);m=Specialist(kind,perm,reflect)
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.0003)
    for _ in range(steps):
        ix=torch.as_tensor(rng.integers(len(x),size=112),dtype=torch.long)
        pred=m(x[ix]);mask=legal[ix];loss=((pred-target[ix]).square()*mask).sum()/mask.sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
    m.eval()
    w=[world(rng) for _ in range(250)]
    x=torch.from_numpy(np.stack([a for a,_ in w]));legal=torch.from_numpy(np.stack([b for _,b in w]))
    gold=torch.from_numpy(scores(x.numpy(),legal.numpy(),laws)).argmax(-1)
    with torch.no_grad():accuracy=float((m(x).masked_fill(~legal,-1e4).argmax(-1)==gold).float().mean())
    return m,round(accuracy,5)


def probe_worlds(seed):
    rng=np.random.default_rng(seed)
    return torch.from_numpy(np.stack([world(rng)[0] for _ in range(P)]))


def signature(m,probe,mode,mispaired=False,budget=P):
    assert budget in (8,16,24)
    probe=probe[:budget]
    with torch.no_grad():
        z=m(probe).float()
        z=z-z.mean(-1,keepdim=True)
        if mode=='raw':
            # Raw model outputs only, no contextual information about worlds.
            out=torch.zeros(SIG)
            out[:budget*N]=(z/2.5).flatten()
            return out
        if mode!='paired':raise ValueError(mode)
        # Identify effects from NATURALLY PAIRED (world, neural output)
        # observations. No coordinate-set interventions, law bits or feature
        # ordering maps. Fit separate regularized action-specific slopes.
        if mispaired:
            idx=torch.as_tensor(np.random.default_rng(113).permutation(budget)).long()
            z=z.index_select(0,idx)
        coef=[]
        for action in range(N):
            X=probe[:,action,:];Y=z[:,action]
            X=X-X.mean(0,keepdim=True);Y=Y-Y.mean()
            # Eight to 24 paired observations; ridge controls the underdetermined case.
            beta=torch.linalg.solve(X.T@X + torch.eye(D)*1.0,X.T@Y)
            coef.append(beta/3.0)
        sig=torch.stack(coef).flatten()
        assert sig.numel()==D*N
        out=torch.zeros(SIG)
        out[:D*N]=sig
        assert torch.isfinite(out).all()
        return out


def banks(seed,steps,probes):
    out={};quality={}
    for kind,offset in [('narrow',10000),('wide',20000),('gated',30000)]:
        bank={}
        for family in range(3):
            for sign in (-1,1):
                childseed=seed+offset+family*1001+(sign+1)*61
                m,acc=train_specialist(kind,family,sign,childseed,steps)
                sigs={(mode,k,budget):signature(m,p,mode,budget=budget)
                      for mode in ('raw','paired') for k,p in enumerate(probes) for budget in (8,16,24)}
                sigs.update({('mispaired',k,budget):signature(m,p,'paired',True,budget)
                             for k,p in enumerate(probes) for budget in (8,16,24)})
                bank[(family,sign)]=dict(model=m,sig=sigs,seed=childseed,acc=acc)
                quality[f'{kind}:{family}:{sign}']=dict(seed=childseed,accuracy=acc,
                    private_perm_sha256=sha(m.perm.numpy().tobytes()+m.reflect.numpy().tobytes()))
        out[kind]=bank
    return out,quality


def source(laws,bank,mode,probe_index,budget=P):
    return torch.stack([torch.stack([bank[(j,int(row[j]))]['sig'][(mode,probe_index,budget)] for j in range(3)])
                      for row in laws.tolist()])


class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.e=nn.Sequential(nn.Linear(SIG,72),nn.GELU(),nn.Linear(72,32),nn.GELU())
        self.c=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,48),nn.Tanh())
        self.p=nn.Sequential(nn.Linear(D+N+48,96),nn.GELU(),nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x,legal,src):
        s=self.c(self.e(src).mean(1))
        f=torch.cat((with_action_id(x),s[:,None].expand(-1,N,-1)),dim=-1)
        return self.p(f).squeeze(-1).masked_fill(~legal,-1e4)


def accuracy(pred,target):return round(float((pred.argmax(-1)==target).float().mean()),5)


def evaluate(models,c,bank,probe_index,budget=P):
    out={}; law=c['laws']; xx=c['x'];legal=c['legal']; y=c['y']
    flip=law.clone();flip[:,0]*=-1
    cf=torch.tensor([int(scores(xx[i].numpy(),legal[i].numpy(),tuple(flip[i].tolist())).argmax()) for i in range(len(y))])
    for method,m in models.items():
        src=source(law,bank,method,probe_index,budget) if method!='blind' else torch.zeros(len(y),3,SIG)
        with torch.no_grad():
            logits=m(xx,legal,src); pred=logits.argmax(-1)
            assert legal.gather(1,pred[:,None]).all()
            result={'accuracy':accuracy(logits,y)}
            if method!='blind':
                zero=m(xx,legal,src*0).argmax(-1)
                s=src.clone();s[:,0]=source(flip,bank,method,probe_index,budget)[:,0]
                swapped=m(xx,legal,s).argmax(-1)
                result.update(zero_accuracy=round(float((zero==y).float().mean()),5),
                              wrong_wet_original_accuracy=round(float((swapped==y).float().mean()),5),
                              wrong_wet_counterfactual_accuracy=round(float((swapped==cf).float().mean()),5),
                              cf_true_change_rate=round(float((cf!=y).float().mean()),5),
                              cf_swap_change_rate=round(float((swapped!=pred).float().mean()),5),
                              cf_change_recall=round(float((((swapped!=pred)&(cf!=y)).sum()/(cf!=y).sum().clamp_min(1))),5))
                if method=='paired':
                    miss=source(law,bank,'mispaired',probe_index,budget)
                    mispred=m(xx,legal,miss).argmax(-1)
                    result['mispaired_probe_accuracy']=round(float((mispred==y).float().mean()),5)
                    # Donor order is deliberately hidden; this is architectural invariance.
                    shuf=m(xx,legal,src[:,[2,0,1]]).argmax(-1)
                    result['slot_permutation_disagreements']=int((shuf!=pred).sum())
            out[method]=result
    with torch.no_grad():
        src=source(law,bank,'paired',probe_index,budget)
        pred=models['paired'](xx,legal,src).argmax(-1)
    if torch.all(law==torch.tensor(HELDOUT)):
        third=torch.tensor([int(scores(xx[i].numpy(),legal[i].numpy(),HELDOUT,True).argmax()) for i in range(len(y))])
        out['unseen_third_order']=dict(match=round(float((pred==third).float().mean()),5),
                                       oracle_change_rate=round(float((third!=y).float().mean()),5))
    return out


def selftest():
    m=Specialist('gated',np.arange(D)[::-1].copy(),np.ones(D))
    p=probe_worlds(13)
    assert signature(m,p,'paired').shape==(SIG,)
    assert signature(m,p,'raw').shape==(SIG,)
    assert not torch.allclose(signature(m,p,'paired'),signature(m,p,'paired',True))
    c=dataset(5,45,'train'); d=dataset(6,45,'sealed')
    assert not set(c['keys'])&set(d['keys'])
    assert not torch.all(torch.all(c['laws']==torch.as_tensor(HELDOUT),-1))
    assert torch.all(d['laws']==torch.as_tensor(HELDOUT))
    r=Receiver();src=torch.randn(5,3,SIG)
    a=r(c['x'][:5],c['legal'][:5],src)
    b=r(c['x'][:5],c['legal'][:5],src[:,[2,1,0]])
    assert torch.allclose(a,b,atol=1e-6)
    assert c['legal'][:5].gather(1,a.argmax(-1)[:,None]).all()
    print('DG012 SELFTEST: disjoint/world split, law holdout, scrambling, lawful outputs, slot invariance PASS')


def run(out,seed,donor_steps,receiver_steps,train_count,sealed_count):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    t0=time.monotonic();path=Path(out);path.mkdir(parents=True,exist_ok=True)
    probes=[probe_worlds(713+i*89) for i in range(5)]
    bank,quality=banks(seed,donor_steps,probes)
    train=dataset(seed+1,train_count,'train');val=dataset(seed+2,500,'train');sealed=dataset(seed+3,sealed_count,'sealed');mixed=dataset(seed+4,sealed_count,'mixed')
    assert not (set(train['keys'])&set(val['keys']) or set(train['keys'])&set(sealed['keys']) or set(val['keys'])&set(sealed['keys']) or set(mixed['keys'])&(set(train['keys'])|set(val['keys'])|set(sealed['keys'])))
    cache={(kind,method,p,budget):source(train['laws'],bank[kind],method,p,budget)
           for kind in ('narrow','wide') for method in ('paired','raw') for p in (0,1,2) for budget in (8,16,24)}
    models={m:Receiver() for m in ('paired','raw','blind')}
    optim={m:torch.optim.AdamW(model.parameters(),lr=.0017,weight_decay=.0001) for m,model in models.items()}
    rng=np.random.default_rng(seed+900);history=[]
    for step in range(receiver_steps):
        idx=torch.from_numpy(rng.integers(train_count,size=96));kind=('narrow','wide')[int(rng.integers(2))];p=int(rng.integers(3));budget=(8,16,24)[int(rng.integers(3))]
        x=train['x'][idx];l=train['legal'][idx];y=train['y'][idx]
        losses={}
        for mode,m in models.items():
            src=cache[(kind,mode,p,budget)][idx] if mode!='blind' else torch.zeros(len(idx),3,SIG)
            z=m(x,l,src);loss=F.cross_entropy(z,y)
            optim[mode].zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);optim[mode].step()
            losses[mode]=round(float(loss.detach()),5)
        if step==0 or (step+1)%150==0 or step+1==receiver_steps:
            stats=evaluate(models,val,bank['wide'],1)
            row=dict(step=step+1,losses=losses,validation=stats)
            history.append(row);print('DG012 TRAIN '+json.dumps(row),flush=True)
    sealed_metrics={};mixed_metrics={}
    for kind in ('narrow','wide','gated'):
        for p,name in ((0,'familiar'),(4,'unseen')):
            for budget in (8,16,24):
                sealed_metrics[f'{kind}:{name}:{budget}']=evaluate(models,sealed,bank[kind],p,budget)
                mixed_metrics[f'{kind}:{name}:{budget}']=evaluate(models,mixed,bank[kind],p,budget)
    for kind,b in bank.items():
        for (family,sign),entry in b.items():
            torch.save(entry['model'].state_dict(),path/f'donor-{kind}-{family}-{sign}.pt')
    for mode,m in models.items():torch.save(m.state_dict(),path/f'receiver-{mode}.pt')
    src=source(sealed['laws'],bank['gated'],'paired',4,24)
    with torch.no_grad():chosen=models['paired'](sealed['x'],sealed['legal'],src).argmax(-1)
    decisions=[dict(world_sha=sealed['keys'][i],legal=[ACTIONS[j] for j,v in enumerate(sealed['legal'][i]) if v],
                    predicted=ACTIONS[int(chosen[i])],expert=ACTIONS[int(sealed['y'][i])]) for i in range(min(64,sealed_count))]
    (path/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    report=dict(experiment='DG012',seed=seed,train_cases=train_count,sealed_cases=sealed_count,
                donor_steps=donor_steps,receiver_steps=receiver_steps,
                source_observation='8/16/24 paired canonical world observations and frozen neural donor logits; no donor feature ordering maps exposed',
                training_source_architectures=['narrow','wide'],sealed_source_architecture='gated',
                source_private_permutation_and_reflection=True,withheld_combination=HELDOUT,
                probe_sha256=[sha(p.numpy().tobytes()) for p in probes],donor_quality=quality,
                history=history,sealed=sealed_metrics,mixed_law_sealed=mixed_metrics,
                scientific_scope='supervised synthetic donor functional behaviour, NOT native KV transfer or discovered physics',
                duration_seconds=round(time.monotonic()-t0,2))
    (path/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    hashes={p.name:sha(p.read_bytes()) for p in sorted(path.iterdir()) if p.is_file()}
    (path/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('DG012 SEALED '+json.dumps(sealed_metrics),flush=True)
    print('DG012 MIXED '+json.dumps(mixed_metrics),flush=True)
    print('DG012 HASHES '+json.dumps(hashes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true');p.add_argument('--seed',type=int,default=26011620)
    p.add_argument('--out',default='/tmp/dg012');p.add_argument('--donor-steps',type=int,default=220)
    p.add_argument('--receiver-steps',type=int,default=900);p.add_argument('--train-cases',type=int,default=5200)
    p.add_argument('--sealed-cases',type=int,default=900)
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(a.out,a.seed,a.donor_steps,a.receiver_steps,a.train_cases,a.sealed_cases)
