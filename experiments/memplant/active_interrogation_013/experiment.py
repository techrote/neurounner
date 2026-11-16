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



# DG013: active, budget-controlled black-box donor interrogation.
# Experimental extension of DG012, retaining all frozen-parent architecture and
# source-faithful synthetic game-law definitions verbatim for provenance.

class ActiveProbeRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(30,64),nn.GELU(),nn.Linear(64,32),
                                  nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.layers(x).squeeze(-1)


def probe_features(x):
    return torch.cat([x.mean(1),x.std(1,unbiased=False),x.amax(1)-x.amin(1)],dim=-1)


def candidate_pool(seed,n=64):
    rng=np.random.default_rng(seed)
    return torch.from_numpy(np.stack([world(rng)[0] for _ in range(n)]))


def demeaned_logits(m,x):
    with torch.no_grad():
        y=m(x).float()
        return y-y.mean(-1,keepdim=True)


def selector_targets(models,x):
    with torch.no_grad():
        a=torch.stack([demeaned_logits(m,x) for m in models])
        a=a/a.square().mean((1,2),keepdim=True).sqrt().clamp_min(.1)
        return a.var(0).mean(-1)


def train_probe_ranker(bank,seed,steps):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    models=[bank[k][(f,s)]['model'] for k in ('narrow','wide')
            for f in range(3) for s in (-1,1)]
    x=candidate_pool(seed+701,2300)
    v=candidate_pool(seed+29001,500)
    y=selector_targets(models,x);vy=selector_targets(models,v)
    f=probe_features(x);vf=probe_features(v)
    ranker=ActiveProbeRanker();opt=torch.optim.AdamW(ranker.parameters(),lr=.002,weight_decay=.001)
    losses=[]
    for step in range(steps):
        ix=torch.as_tensor(rng.integers(len(x),size=128))
        pred=ranker(f[ix]);loss=F.mse_loss(pred,y[ix])
        opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0 or step+1==steps:losses.append(round(float(loss.detach()),6))
    with torch.no_grad():
        pred=ranker(vf)
        rmse=float(F.mse_loss(pred,vy).sqrt())
        corr=float(np.corrcoef(pred.numpy(),vy.numpy())[0,1])
    ranker.eval()
    return ranker,dict(updates=steps,source_models=len(models),
        independent_validation_rmse=round(rmse,6),
        independent_validation_correlation=round(corr,6),loss_samples=losses)


def select_probe_indices(pool,budget,mode,ranker,real_donor,bank,random_seed):
    assert budget in (8,16,24) and len(pool)==64
    if mode=='random':
        return [int(q) for q in np.random.default_rng(random_seed).choice(64,budget,replace=False)], None
    if mode=='geometry':
        x=probe_features(pool).numpy()
        x=(x-x.mean(0))/(x.std(0)+.25)
        inv=np.eye(x.shape[1])*4.
        out=[]
        for _ in range(budget):
            scores=np.array([-1e9 if j in out else x[j]@inv@x[j] for j in range(64)])
            j=int(scores.argmax());out.append(j)
            v=inv@x[j];inv-=np.outer(v,v)/(1+x[j]@v)
        return out, None
    with torch.no_grad():
        rank=ranker(probe_features(pool)).numpy()
    if mode=='learned':
        return [int(q) for q in np.argsort(-rank)[:budget]], None
    if mode!='adaptive':raise ValueError(mode)
    # Surrogate bank is restricted to previously trained narrow/wide models.
    # The held-out real gated donor is queried ONCE per selected probe; no peeks.
    sur_models=[bank[k][(f,s)]['model'] for k in ('narrow','wide')
                for f in range(3) for s in (-1,1)]
    sur=np.stack([demeaned_logits(m,pool).numpy() for m in sur_models])
    selected=[];observed=[];weights=np.ones(len(sur_models))/len(sur_models)
    for step in range(budget):
        if step<2:score=rank.copy()
        else:
            mean=(sur*weights[:,None,None]).sum(0)
            variance=((sur-mean[None])**2*weights[:,None,None]).sum((0,2))
            a=(rank-rank.mean())/(rank.std()+1e-5)
            b=(variance-variance.mean())/(variance.std()+1e-5)
            score=.45*a+.55*b
        score[selected]=-np.inf
        chosen=int(score.argmax());selected.append(chosen)
        observed.append(demeaned_logits(real_donor,pool[chosen:chosen+1]).numpy()[0])
        discrepancy=((sur[:,selected]-np.stack(observed)[None])**2).mean((1,2))
        logp=-discrepancy/.65;logp-=logp.max()
        weights=np.exp(logp);weights/=weights.sum()
    return selected, np.stack(observed)


def paired_signature_cached(probe,centered_output):
    # Exact DG012 ridge-calibrated signature, using already observed donor logits.
    # No second call to the held-out gated donor.
    z=torch.from_numpy(np.asarray(centered_output,dtype='float32'))
    coefficients=[]
    for action in range(N):
        X=probe[:,action,:];Y=z[:,action]
        X=X-X.mean(0,keepdim=True);Y=Y-Y.mean()
        beta=torch.linalg.solve(X.T@X + torch.eye(D), X.T@Y)
        coefficients.append(beta/3.)
    out=torch.zeros(SIG)
    out[:D*N]=torch.stack(coefficients).flatten()
    return out


def selected_source_bank(bank,ranker,pool,budget,mode,seed):
    fingerprints={};selection={}
    for family in range(3):
        for sign in (-1,1):
            real=bank['gated'][(family,sign)]['model']
            inds,already_observed=select_probe_indices(pool,budget,mode,ranker,real,bank,
                                      seed+family*101+sign*13)
            selection[f'{family}:{sign}']=inds
            fingerprints[(family,sign)]=(signature(real,pool[inds],'paired',budget=budget)
                 if already_observed is None else paired_signature_cached(pool[inds],already_observed))
    return fingerprints,sha(json.dumps(selection,sort_keys=True).encode())


def assembled_source(laws,fingerprints):
    return torch.stack([torch.stack([fingerprints[j,int(row[j])] for j in range(3)])
                        for row in laws.tolist()])


def eval_active_probes(models,bank,ranker,seed,cases):
    mixed=dataset(seed+11000,cases,'mixed')
    sealed=dataset(seed+12000,cases,'sealed')
    assert not set(mixed['keys'])&set(sealed['keys'])
    received=models['paired'];blind=models['blind']
    flip=mixed['laws'].clone();flip[:,0]*=-1
    cf_gold=torch.tensor([int(scores(mixed['x'][i].numpy(),mixed['legal'][i].numpy(),
                          tuple(flip[i].tolist())).argmax()) for i in range(cases)])
    true_change_rate=round(float((cf_gold!=mixed['y']).float().mean()),5)
    results={};selection_hashes={};trajectory_source={}
    with torch.no_grad():
        blind_logits=blind(mixed['x'],mixed['legal'],torch.zeros(cases,3,SIG))
        results['blind']=dict(mixed_accuracy=accuracy(blind_logits,mixed['y']))
    for budget in (8,16,24):
        for mode in ('random','geometry','learned','adaptive'):
            observed=[]
            for pool_index in range(3):
                pool=candidate_pool(seed+14000+pool_index*791)
                fingerprints,key=selected_source_bank(bank,ranker,pool,budget,mode,
                                                       seed+pool_index*200)
                selection_hashes[f'{budget}:{mode}:{pool_index}']=key
                if budget==16 and pool_index==0:trajectory_source[mode]=fingerprints
                with torch.no_grad():
                    mp=received(mixed['x'],mixed['legal'],assembled_source(mixed['laws'],fingerprints))
                    sp=received(sealed['x'],sealed['legal'],assembled_source(sealed['laws'],fingerprints))
                    ma=accuracy(mp,mixed['y']);sa=accuracy(sp,sealed['y'])
                    wet_cf=received(mixed['x'],mixed['legal'],assembled_source(flip,fingerprints))
                    cf_accuracy=accuracy(wet_cf,cf_gold)
                    assert mixed['legal'].gather(1,mp.argmax(-1)[:,None]).all()
                    assert sealed['legal'].gather(1,sp.argmax(-1)[:,None]).all()
                    assert mixed['legal'].gather(1,wet_cf.argmax(-1)[:,None]).all()
                observed.append((ma,sa,cf_accuracy))
            results[f'{budget}:{mode}']=dict(
                mixed_mean=round(float(np.mean([q[0] for q in observed])),5),
                mixed_by_probe_bank=[q[0] for q in observed],
                withheld_mean=round(float(np.mean([q[1] for q in observed])),5),
                withheld_by_probe_bank=[q[1] for q in observed],
                counterfactual_wet_mean=round(float(np.mean([q[2] for q in observed])),5),
                counterfactual_wet_by_probe_bank=[q[2] for q in observed],
                counterfactual_true_action_change_rate=true_change_rate,
                gated_donor_queries_per_bank=6*budget)
    return results,selection_hashes,trajectory_source


from collections import deque

class SyntheticGameRollout:
    """Stateful grid: blocked cells, wetness, protected cells, history and paint."""
    def __init__(self,seed,side=13):
        self.rng=np.random.default_rng(seed);self.side=side
        self.walls=self.rng.random((side,side))<.17
        self.wet=self.rng.random((side,side))<.35
        self.protected=self.rng.random((side,side))<.15
        self.x=side//2;self.y=side//2;self.walls[self.y,self.x]=False
        self.goal=(int(self.rng.integers(side)),int(self.rng.integers(side)))
        self.walls[self.goal[1],self.goal[0]]=False
        self.painted=set();self.path=[(self.x,self.y)]
        self.visits={(self.x,self.y):1}
        self.retrace=0;self.hits=0;self.travel=0;self.loops=0

    def observe(self):
        x,_=world(self.rng);legal=np.ones(N,dtype='bool')
        directions=((-1,0),(1,0),(0,-1),(0,1))
        old_dist=abs(self.x-self.goal[0])+abs(self.y-self.goal[1])
        for j,(dx,dy) in enumerate(directions):
            xx,yy=self.x+dx,self.y+dy
            allowed=0<=xx<self.side and 0<=yy<self.side and not self.walls[yy,xx]
            legal[j]=allowed
            if allowed:
                x[j,0]=np.clip((old_dist-abs(xx-self.goal[0])-abs(yy-self.goal[1]))/2+
                               .4*self.rng.uniform(-1,1),-1,1)
                x[j,1]=self.wet[yy,xx]
                x[j,2]=float(self.protected[yy,xx])
                x[j,5]=min(1.,.2*self.visits.get((xx,yy),0))
                x[j,3]=self.rng.random()
            else:x[j,2]=1
        x[4,1]=float(self.wet[self.y,self.x])
        x[4,6]=float(not self.protected[self.y,self.x] and
                       (self.x,self.y) not in self.painted)
        legal[4]=bool(x[4,6]);legal[5]=len(self.path)>1
        x[5,7]=min(1.,len(self.path)/10)
        if not legal.any():legal[5]=True
        return x,legal

    def step(self,a):
        if a<4:
            dx,dy=((-1,0),(1,0),(0,-1),(0,1))[a]
            self.x+=dx;self.y+=dy;self.path.append((self.x,self.y))
            self.travel+=1
            pos=(self.x,self.y);self.visits[pos]=self.visits.get(pos,0)+1
            if self.visits[pos]>1:self.loops+=1
        elif a==4:self.painted.add((self.x,self.y))
        else:
            self.retrace+=1
            if len(self.path)>1:
                self.path.pop();self.x,self.y=self.path[-1]
        if (self.x,self.y)==self.goal:self.hits+=1

    def planner_action(self):
        # Independent reachability ceiling: shortest legal route across static walls.
        start=(self.x,self.y)
        if start==self.goal:return None
        directions=((-1,0),(1,0),(0,-1),(0,1))
        queue=deque([(start,None)]);seen={start}
        while queue:
            at,first=queue.popleft()
            for action,(dx,dy) in enumerate(directions):
                nx,ny=at[0]+dx,at[1]+dy
                neighbor=(nx,ny)
                if not (0<=nx<self.side and 0<=ny<self.side) or self.walls[ny,nx] or neighbor in seen:continue
                first_action=action if first is None else first
                if neighbor==self.goal:return first_action
                seen.add(neighbor);queue.append((neighbor,first_action))
        return None

    def metrics(self):
        return dict(unique_visited=len(self.visits),painted=len(self.painted),
                    travel=self.travel,retrace=self.retrace,loops=self.loops,
                    reached=int(self.hits>0),illegal=0)


def evaluate_trajectories(receiver,sources,seed,steps=72,world_count=18):
    summary={};records=[]
    for mode,source in list(sources.items())+[('authored_expert',None),('goal_planner',None)]:
        values=[]
        for i in range(world_count):
            env=SyntheticGameRollout(seed+31000+i*73)
            laws=tuple(int(q) for q in np.random.default_rng(seed+5200+i).choice([-1,1],3))
            src=assembled_source(torch.tensor([laws]),source) if source is not None else None
            trace=[]
            for t in range(steps):
                x,legal=env.observe()
                with torch.no_grad():
                    action=env.planner_action() if mode=='goal_planner' else None
                    if action is None:
                        action=(int(scores(x,legal,laws).argmax()) if src is None else
                          int(receiver(torch.from_numpy(x[None]),
                            torch.from_numpy(legal[None]),src).argmax(-1)[0]))
                if not legal[action]:raise AssertionError('illegal game action emitted')
                env.step(action)
                if i<2:trace.append(dict(t=t,action=ACTIONS[action],position=[env.x,env.y]))
            values.append(env.metrics())
            if i<2:records.append(dict(method=mode,world_seed=seed+31000+i*73,
                                     laws=laws,trajectory=trace,metrics=values[-1]))
        summary[mode]={k:round(float(np.mean([r[k] for r in values])),5)
                        for k in values[0]}
        summary[mode].update(world_count=world_count,decisions=world_count*steps)
    return summary,records


def dg013_selftest():
    base=dataset(5,24,'train')
    assert not torch.all(base['laws']==torch.as_tensor(HELDOUT))
    q=candidate_pool(88)
    assert probe_features(q).shape==(64,30)
    assert ActiveProbeRanker()(probe_features(q)).shape==(64,)
    assert len(select_probe_indices(q,8,'geometry',ActiveProbeRanker(),None,None,9)[0])==8
    env=SyntheticGameRollout(77)
    for _ in range(10):
        x,legal=env.observe()
        assert legal.any()
        env.step(int(np.flatnonzero(legal)[0]))
    print('DG013 SELFTEST: legal stateful game, source holdout, query geometry, ranker PASS')


def run_dg013(out,seed,donor_steps,receiver_steps,train_cases,sealed_cases,
              ranker_steps,world_count,trajectory_steps):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    parent=path/'dg012-trained'
    # Rebuild all neural donors/receivers from pinned DG012 source. This costs
    # CPU but ensures CI artifacts are genuinely source- and checkpoint-matched.
    run(parent,seed,donor_steps,receiver_steps,train_cases,sealed_cases)
    bank={}
    for kind in ('narrow','wide','gated'):
        sub={}
        for family in range(3):
            for sign in (-1,1):
                m=Specialist(kind,np.arange(D),np.ones(D,dtype='float32'))
                sd=torch.load(parent/f'donor-{kind}-{family}-{sign}.pt',
                              map_location='cpu',weights_only=True)
                m.load_state_dict(sd);m.eval()
                sub[(family,sign)]=dict(model=m)
        bank[kind]=sub
    received={}
    for mode in ('paired','blind'):
        m=Receiver()
        m.load_state_dict(torch.load(parent/f'receiver-{mode}.pt',
                                     map_location='cpu',weights_only=True));m.eval()
        received[mode]=m
    ranker,training=train_probe_ranker(bank,seed+2,ranker_steps)
    result,probe_hashes,trajectory_source=eval_active_probes(
        received,bank,ranker,seed+2,sealed_cases)
    long_horizon,trace=evaluate_trajectories(received['paired'],
        trajectory_source,seed+2,trajectory_steps,world_count)
    torch.save(ranker.state_dict(),path/'trained-probe-ranker.pt')
    (path/'trajectories.json').write_text(json.dumps(trace,indent=2)+'\n')
    report=dict(experiment='DG013',seed=seed+2,
        parent_dg012_seed=seed,parent_architecture='retrained and frozen source-pinned DG012',
        selector_training=training,probe_result=result,probe_index_sha256=probe_hashes,
        trajectories=long_horizon,
        training_parent_steps=dict(donor=donor_steps,receiver=receiver_steps),
        sealed_cases=sealed_cases,trajectory_steps=trajectory_steps,
        trajectory_worlds=world_count,source_sha256=sha(Path(__file__).read_bytes()),
        scientific_scope='supervised synthetic specialist transfer; selector trained on narrow/wide; gated donor sealed; no native CyberSand gameplay',
        limitations='adaptive posterior uses trained surrogate calls on 64 candidates; only actual gated donor queries are budgeted; stateful synthetic rollout not native game; inherited teacher does not discover unobserved physics',
        duration_seconds=round(time.monotonic()-start,2))
    (path/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    hashes={p.name:sha(p.read_bytes()) for p in path.iterdir() if p.is_file()}
    (path/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('DG013 FINAL '+json.dumps(dict(report=report,hashes=hashes)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='DG013 budgeted donor interrogation + stateful policy rollouts')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--seed',type=int,default=26011620)
    p.add_argument('--out',default='/tmp/dg013')
    p.add_argument('--donor-steps',type=int,default=220)
    p.add_argument('--receiver-steps',type=int,default=900)
    p.add_argument('--train-cases',type=int,default=5200)
    p.add_argument('--sealed-cases',type=int,default=900)
    p.add_argument('--ranker-steps',type=int,default=450)
    p.add_argument('--trajectory-steps',type=int,default=72)
    p.add_argument('--worlds',type=int,default=18)
    a=p.parse_args()
    if a.selftest:dg013_selftest()
    else:run_dg013(a.out,a.seed,a.donor_steps,a.receiver_steps,
                   a.train_cases,a.sealed_cases,a.ranker_steps,a.worlds,
                   a.trajectory_steps)
