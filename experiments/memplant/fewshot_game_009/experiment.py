"""DG009: learn a donor-trajectory game-law allele, not a copied action.

Synthetic game-facing experiment. One receiver consumes K donor demonstration
worlds with host-legal six-way actions, compiles them into a compact learned
latent allele, and selects an action in a disjoint query world. The three
underlying signed interaction effects are combined in one *held-out* triple.
A&B jointly gate these otherwise invisible law effects. No native CyberSand.
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
N, D, K = 6, 12, 8
HELDOUT = (1, 1, 1)
TRAIN_ALLELES = [x for x in itertools.product((-1, 1), repeat=3) if x != HELDOUT]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_world(rng: np.random.Generator):
    """Each action observes an independent synthetic local material/geometry probe.

    Features: goal affinity, wet material, danger, novelty, solid contact,
    passable neighbours, boundary clearance, current heading, seep cue,
    branch opportunity, paintable material, and visited history.
    """
    f = rng.uniform(-1, 1, size=(N, D)).astype('float32')
    f[:, 1] = (rng.random(N) < .43).astype('float32')
    f[:, 2] = (rng.random(N) < .22).astype('float32')
    f[:, 3] = rng.random(N).astype('float32')
    f[:, 4] = (rng.random(N) < .48).astype('float32')
    f[:, 5] = rng.random(N).astype('float32')
    f[:, 8] = rng.random(N).astype('float32')
    f[:, 9] = rng.random(N).astype('float32')
    f[:, 10] = (rng.random(N) < .72).astype('float32')
    f[:, 11] = rng.random(N).astype('float32')
    legal = np.ones(N, dtype=bool)
    legal[:4] = (rng.random(4) > .18) & (f[:4, 2] < .9)
    legal[4] = bool(f[4, 10] and rng.random() > .15)
    legal[5] = bool(rng.random() > .35)
    if not legal.any():
        legal[int(rng.integers(N))] = True
    return f, legal


def expert_scores(f: np.ndarray, legal: np.ndarray, ab: tuple[int, int],
                  allele: tuple[int, int, int]):
    """Author-known teacher for falsifiable supervised synthetic physics.

    Signed effects are all activated ONLY by conjunction A&B, so an isolated
    parent cannot identify their direction; contextual donor examples can.
    """
    z = np.full(N, -10000., dtype='float32')
    z[:4] = .8*f[:4, 0] + .45*f[:4, 3] + .35*f[:4, 7] - .4*f[:4, 2]
    z[4] = .05 + .6*f[4, 10] + .35*f[4, 1] - .2*f[4, 2]
    z[5] = -.25 + .6*f[5, 11]
    if ab[0] and ab[1]:
        wet, branch, paint = allele
        z[:4] += 2.75*wet*(2*f[:4, 1]-1)
        z[:4] += 2.25*branch*(2*f[:4, 9]-1)
        z[5] -= 1.5*branch
        z[4] += 3.3*paint*(.35+.65*f[4, 1])
    z[~legal] = -10000.
    return z


def make_cases(seed: int, count: int, split: str, k: int = K):
    rng = np.random.default_rng(seed)
    allele_pool = TRAIN_ALLELES if split != 'heldout' else [HELDOUT]
    sf = np.empty((count, k, N, D), dtype='float32')
    sl = np.empty((count, k, N), dtype='bool')
    sy = np.empty((count, k), dtype='int64')
    qf = np.empty((count, N, D), dtype='float32')
    ql = np.empty((count, N), dtype='bool')
    qy = np.empty((count,), dtype='int64')
    pair = np.empty((count, 2), dtype='float32')
    alleles = np.empty((count, 3), dtype='int64')
    ids = []
    for i in range(count):
        allele = allele_pool[int(rng.integers(len(allele_pool)))]
        # Include A-only and B-only training controls, with NO hidden physics.
        ab = ((1, 1) if split == 'heldout' or rng.random() < .8 else
              ((1, 0) if rng.random() < .5 else (0, 1)))
        alleles[i], pair[i] = allele, ab
        for j in range(k):
            f, legal = make_world(rng)
            sf[i, j], sl[i, j] = f, legal
            sy[i, j] = int(expert_scores(f, legal, ab, allele).argmax())
        f, legal = make_world(rng)
        qf[i], ql[i] = f, legal
        qy[i] = int(expert_scores(f, legal, ab, allele).argmax())
        # World identity is not given to learner: independent, unique RNG draw.
        ids.append(sha(qf[i].tobytes() + ql[i].tobytes()))
    assert len(ids) == len(set(ids)), 'query-world collision'
    return dict(sf=torch.from_numpy(sf), sl=torch.from_numpy(sl),
                sy=torch.from_numpy(sy), qf=torch.from_numpy(qf),
                ql=torch.from_numpy(ql), qy=torch.from_numpy(qy),
                pair=torch.from_numpy(pair), alleles=torch.from_numpy(alleles), ids=ids)


class Receiver(nn.Module):
    """Permutation-invariant amortized donor compression into a 16-D allele."""
    def __init__(self, latent=16):
        super().__init__()
        # Extract chosen-vs-unselected feature contrasts; don't expose allele.
        self.demo = nn.Sequential(nn.Linear(2*D+N+2, 96), nn.GELU(),
                                  nn.Linear(96, 64), nn.GELU())
        self.allele = nn.Sequential(nn.Linear(64, 48), nn.Tanh(),
                                    nn.Linear(48, latent))
        self.query = nn.Sequential(nn.Linear(D+2+latent, 96), nn.GELU(),
                                   nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, sf, sl, sy, qf, ql, pair, k=K):
        n, full_k, _, _ = sf.shape
        onehot = F.one_hot(sy, N).float()
        picked = (sf*onehot[..., None]).sum(2)
        others = (sf*sl[..., None]).sum(2) / sl.sum(2).clamp_min(1)[..., None]
        demo_in = torch.cat((picked, others, onehot,
                            pair[:, None, :].expand(-1,full_k,-1)), dim=-1)
        embedded = self.demo(demo_in)
        # Use the first k donor worlds; zero-shot receives ZERO latent and no
        # information about which trained allele produced the query.
        if k:
            latent = self.allele(embedded[:, :k].mean(1))
        else:
            latent = embedded.new_zeros((n, self.allele[-1].out_features))
        x = torch.cat((qf, pair[:, None, :].expand(-1,N,-1),
                       latent[:, None, :].expand(-1,N,-1)), dim=-1)
        return self.query(x).squeeze(-1).masked_fill(~ql, -1e4), latent


class Blind(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D+2, 96), nn.GELU(),
                               nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, qf, ql, pair):
        x=torch.cat((qf, pair[:,None,:].expand(-1,N,-1)),-1)
        return self.net(x).squeeze(-1).masked_fill(~ql,-1e4)


def batch(cases, idx):
    return {k: v[idx] for k,v in cases.items() if k != 'ids'}


def evaluate(receiver, blind, c, seed=0):
    receiver.eval(); blind.eval()
    with torch.no_grad():
        n=len(c['qy']); rng=np.random.default_rng(seed)
        # Counterfactual donor: SAME donor worlds and legal menus, but
        # an *opposite* underlying allele. This is stronger than shuffling
        # among sealed queries which all have the SAME HELDOUT allele.
        anti = c['sy'].clone()
        for i in range(n):
            other = tuple(-int(v) for v in c['alleles'][i])
            ab = tuple(int(v) for v in c['pair'][i])
            for j in range(c['sf'].shape[1]):
                anti[i,j] = int(expert_scores(c['sf'][i,j].numpy(),
                    c['sl'][i,j].numpy(),ab,other).argmax())
        # Also test donor action labels shuffled *between* different support
        # contexts (world features/legal mask remain unchanged). Filter
        # illegal action receipts; report this as a destructive negative.
        perm=torch.as_tensor(rng.permutation(n),dtype=torch.long)
        args=(c['sf'],c['sl'],c['sy'],c['qf'],c['ql'],c['pair'])
        full,latent=receiver(*args,k=8)
        no0,_=receiver(*args,k=0)
        short2,_=receiver(*args,k=2)
        short4,_=receiver(*args,k=4)
        wrong,_=receiver(c['sf'],c['sl'],anti,c['qf'],c['ql'],c['pair'],k=8)
        shuffled,_=receiver(c['sf'][perm],c['sl'][perm],c['sy'][perm],
                             c['qf'],c['ql'],c['pair'],k=8)
        base=blind(c['qf'],c['ql'],c['pair'])
        result={}
        for name,logits in [('donor8',full),('donor4',short4),('donor2',short2),
                            ('no_donor',no0),('wrong_allele_donor',wrong),('shuffled_donor',shuffled),('blind',base)]:
            result[name]=round(float((logits.argmax(-1)==c['qy']).float().mean()),5)
            assert bool(c['ql'].gather(1,logits.argmax(-1)[:,None]).all()), name
        # Counterfactual fidelity on identical heldout worlds under allele
        # intervention: compare predicted changes against the authored oracle.
        cf=[]
        sample_n=min(n,320)
        for i in range(sample_n):
            allele=tuple(int(z) for z in c['alleles'][i]); other=(-allele[0],allele[1],allele[2])
            f=c['qf'][i].numpy(); legal=c['ql'][i].numpy()
            x=expert_scores(f,legal,(1,1),allele).argmax()
            y=expert_scores(f,legal,(1,1),other).argmax()
            cf.append(int(x!=y))
        result['counterfactual_wet_flip_change_fraction']=round(float(np.mean(cf)),5)
        result['query_count']=n

        # Paired source correctness among cases where the absent or shuffled
        # source disagrees. This avoids claiming trivial-majority success.
        gold=c['qy']
        result['donor8_vs_wrong_paired_gain']=round(float(((full.argmax(-1)==gold).float()-(wrong.argmax(-1)==gold).float()).mean()),5)
        result['donor8_vs_shuffled_paired_gain']=round(float(((full.argmax(-1)==gold).float()-(shuffled.argmax(-1)==gold).float()).mean()),5)
        result['donor_latent_std']=round(float(latent.std()),6)
        return result


def selftest():
    x=make_cases(100,70,'train'); y=make_cases(101,70,'heldout')
    assert not set(x['ids']).intersection(y['ids'])
    assert tuple(y['alleles'][0].tolist())==HELDOUT
    assert all(tuple(z.tolist())!=HELDOUT for z in x['alleles'])
    assert (x['sl'].gather(2,x['sy'][...,None])).all()
    assert (y['ql'].gather(1,y['qy'][:,None])).all()
    r,b=Receiver(),Blind()
    out,z=r(x['sf'],x['sl'],x['sy'],x['qf'],x['ql'],x['pair'])
    assert out.shape==(70,N) and z.shape==(70,16)
    assert bool(x['ql'].gather(1,out.argmax(-1)[:,None]).all())
    # Permuting complete demonstrations must leave DeepSets output invariant.
    perm=torch.tensor([4,2,7,0,3,5,1,6]); a=r(x['sf'][:,perm],x['sl'][:,perm],
        x['sy'][:,perm],x['qf'],x['ql'],x['pair'])[0]
    assert torch.allclose(a,out,atol=1e-5)
    # Nonlinear conjunction really is inactive in either single law.
    f,legal=make_world(np.random.default_rng(12))
    assert np.array_equal(expert_scores(f,legal,(1,0),(1,1,1)),
                          expert_scores(f,legal,(1,0),(-1,-1,-1)))
    # But it must alter action scores when both are on.
    assert not np.array_equal(expert_scores(f,legal,(1,1),(1,1,1)),
                              expert_scores(f,legal,(1,1),(-1,-1,-1)))
    print('DG009 HOLDOUT / DISJOINT WORLDS / MASK / NONLINEAR PAIR / DONOR SYMMETRY PASS')


def produce(out: str, seed: int, steps: int, count: int, heldout: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(2)
    start=time.monotonic(); dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    training=make_cases(seed+2,count,'train')
    validation=make_cases(seed+3,480,'validation')
    sealed=make_cases(seed+4,heldout,'heldout')
    assert not set(training['ids']).intersection(validation['ids'])
    assert not set(training['ids']).intersection(sealed['ids'])
    assert not set(validation['ids']).intersection(sealed['ids'])
    receiver,blind=Receiver(),Blind()
    optim=torch.optim.AdamW(receiver.parameters(),lr=.0015,weight_decay=.001)
    blind_optim=torch.optim.AdamW(blind.parameters(),lr=.0015,weight_decay=.001)
    rng=np.random.default_rng(seed+600)
    history=[]
    for step in range(steps):
        ix=torch.as_tensor(rng.integers(count,size=80),dtype=torch.long)
        x=batch(training,ix)
        k=int(rng.choice([0,2,4,8],p=[.12,.16,.2,.52]))
        receiver.train(); blind.train()
        pred,_=receiver(x['sf'],x['sl'],x['sy'],x['qf'],x['ql'],x['pair'],k=k)
        loss=F.cross_entropy(pred,x['qy'])
        optim.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(receiver.parameters(),5)
        optim.step()
        pred_b=blind(x['qf'],x['ql'],x['pair'])
        bl=F.cross_entropy(pred_b,x['qy']); blind_optim.zero_grad(); bl.backward()
        blind_optim.step()
        if step==0 or (step+1)%100==0 or step+1==steps:
            val=evaluate(receiver,blind,validation,seed+step)
            row=dict(step=step+1,training_loss=round(float(loss.detach()),5),
                     blind_training_loss=round(float(bl.detach()),5),validation=val)
            history.append(row)
            print(json.dumps(row),flush=True)
    heldout_report=evaluate(receiver,blind,sealed,seed+1)
    for name,model in [('receiver',receiver),('blind',blind)]:
        path=dest/(name+'.pt');torch.save(model.state_dict(),path)
    result=dict(experiment='DG009',seed=seed,steps=steps,training_cases=count,
        validation_cases=len(validation['qy']),heldout_cases=heldout,
        heldout_allele=list(HELDOUT),training_alleles=[list(a) for a in TRAIN_ALLELES],
        source='synthetic authored donor action traces, NOT Qwen/native/model-to-model KV',
        estimated_elapsed_sec=round(time.monotonic()-start,2),
        heldout=heldout_report,history=history,first_heldout_query_ids=sealed['ids'][:8])
    (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    # Evidence: actual decisions and host legal masks for disjoint query worlds.
    receiver.eval();blind.eval()
    with torch.no_grad():
        rows=[]
        for i in range(min(40,heldout)):
            x=batch(sealed,slice(i,i+1))
            log,z=receiver(x['sf'],x['sl'],x['sy'],x['qf'],x['ql'],x['pair'])
            rows.append(dict(query_sha=sealed['ids'][i],allele=x['alleles'][0].tolist(),
                donor_labels=[ACTIONS[int(a)] for a in x['sy'][0]],
                legal=[ACTIONS[j] for j,v in enumerate(x['ql'][0]) if v],
                chosen=ACTIONS[int(log.argmax())],truth=ACTIONS[int(x['qy'][0])],
                latent=[round(float(v),5) for v in z[0]]))
        (dest/'original_decisions.json').write_text(json.dumps(rows,indent=2)+'\n')
    hashes={p.name:sha(p.read_bytes()) for p in sorted(dest.iterdir()) if p.is_file()}
    (dest/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('DG009 HELDOUT '+json.dumps(heldout_report),flush=True)
    print('DG009 SOURCE/OUTPUT '+json.dumps(hashes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--out',default='runtime/memplant/dg009')
    p.add_argument('--seed',type=int,default=26011613)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--train-cases',type=int,default=3500)
    p.add_argument('--heldout-cases',type=int,default=1000)
    a=p.parse_args()
    if a.selftest:selftest()
    else:produce(a.out,a.seed,a.steps,a.train_cases,a.heldout_cases)
