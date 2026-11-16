"""DG011: heterogeneous neural donors, unlabeled permutation-invariant inheritance.

Uses only synthetic game observations and authored *donor training* targets from DG010.
The receiver sees donor output logits, not law bits, labels, or target actions.
Models are genuinely trained independently; no game or production code changes.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'neural_donors_010'))
import experiment as dg010

N, D, PROBES = dg010.N, dg010.D, dg010.PROBES
HELDOUT = dg010.HELDOUT
ALLELES = [a for a in itertools.product((-1, 1), repeat=3) if a != HELDOUT]
SIG = N * PROBES
ACTIONS = ('LEFT','RIGHT','UP','DOWN','PAINT','RETRACE')

def sha(b): return hashlib.sha256(b).hexdigest()

class Donor(nn.Module):
    """Three genuinely different network architectures with compatible action logits."""
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'narrow':
            self.a=nn.Linear(D+N,32); self.b=nn.Linear(32,16); self.c=nn.Linear(16,1)
        elif kind == 'wide':
            self.a=nn.Linear(D+N,96); self.b=nn.Linear(96,48); self.c=nn.Linear(48,1)
        elif kind == 'gated':
            self.a=nn.Linear(D+N,64); self.b=nn.Linear(64,64)
            self.g=nn.Linear(D+N,64); self.c=nn.Linear(64,1)
        else: raise ValueError(kind)
    def forward(self,x):
        z=dg010.with_action_id(x)
        if self.kind=='gated':
            h=F.silu(self.a(z));h=F.silu(self.b(h))*torch.sigmoid(self.g(z))
        else: h=F.gelu(self.b(F.gelu(self.a(z))))
        return self.c(h).squeeze(-1)

def train_donor(kind,family,sign,seed,steps):
    rng=np.random.default_rng(seed); law=[0,0,0];law[family]=sign
    pairs=[dg010.world(rng) for _ in range(1300)]
    x=torch.from_numpy(np.stack([a for a,_ in pairs])); legal=torch.from_numpy(np.stack([b for _,b in pairs]))
    target=torch.from_numpy(dg010.scores(x.numpy(),legal.numpy(),law))
    torch.manual_seed(seed); m=Donor(kind); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.0003)
    for _ in range(steps):
        ix=torch.as_tensor(rng.integers(len(x),size=112),dtype=torch.long)
        z=m(x[ix]);mask=legal[ix];loss=((z-target[ix]).square()*mask).sum()/mask.sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
    m.eval()
    with torch.no_grad():
        pairs=[dg010.world(rng) for _ in range(250)]
        xx=torch.from_numpy(np.stack([a for a,_ in pairs]));ll=torch.from_numpy(np.stack([b for _,b in pairs]))
        y=torch.from_numpy(dg010.scores(xx.numpy(),ll.numpy(),law)).argmax(-1)
        acc=float((m(xx).masked_fill(~ll,-1e4).argmax(-1)==y).float().mean())
    return m,round(acc,5)

def make_probes(seed):
    rng=np.random.default_rng(seed)
    return torch.from_numpy(np.stack([dg010.world(rng)[0] for _ in range(PROBES)]))

def signature(m,probe,mode="intervention"):
    if mode == "raw":
        with torch.no_grad():
            z=m(probe);z=(z-z.mean(-1,keepdim=True))/2.5
            return z.flatten()
    assert mode == "intervention", mode
    """Architecture-agnostic intervention fingerprint: true neural output changes.

    Each of ten feature columns is set to 0 vs 1 on the same probe worlds,
    and six per-action output deltas are averaged across the probe bank.
    This does NOT expose oracle labels or the donor's assigned law sign.
    """
    with torch.no_grad():
        response=[]
        for feature in range(D):
            lo=probe.clone();hi=probe.clone()
            lo[:,:,feature]=0.;hi[:,:,feature]=1.
            response.append((m(hi)-m(lo)).mean(0)/4.)
        z=torch.stack(response).flatten()
        assert z.numel()==SIG and torch.isfinite(z).all()
        return z

def make_bank(seed,kind,steps,probe_banks,mode):
    bank={};metrics={}
    for family in range(3):
        for sign in (-1,1):
            s=seed+family*1111+(sign+1)*61
            m,acc=train_donor(kind,family,sign,s,steps)
            bank[(family,sign)]=dict(model=m, signatures=[signature(m,p,mode) for p in probe_banks])
            metrics[f'{family}:{sign}']=dict(accuracy=acc,source_seed=s)
    return bank,metrics

def dataset(seed,n,split):
    rng=np.random.default_rng(seed);a=[];l=[];y=[];law=[];keys=[]
    for _ in range(n):
        xx,ll=dg010.world(rng)
        allele=HELDOUT if split=='sealed' else ALLELES[int(rng.integers(len(ALLELES)))]
        a.append(xx);l.append(ll);law.append(allele)
        y.append(int(dg010.scores(xx,ll,allele).argmax()))
        keys.append(sha(xx.tobytes()+ll.tobytes()))
    assert len(set(keys))==n
    return dict(x=torch.from_numpy(np.stack(a)),legal=torch.from_numpy(np.stack(l)),
                y=torch.tensor(y),law=torch.tensor(law),keys=keys)

def source_for(laws,bank,probe_index=0):
    return torch.stack([torch.stack([bank[(j,int(row[j]))]['signatures'][probe_index] for j in range(3)])
                        for row in laws.tolist()])

class SetReceiver(nn.Module):
    """Unlabeled DeepSets donor encoder; invariant to donor order by construction."""
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(SIG,64),nn.GELU(),nn.Linear(64,32),nn.GELU())
        self.combine=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,48),nn.Tanh())
        self.policy=nn.Sequential(nn.Linear(D+N+48,96),nn.GELU(),nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x,legal,source):
        context=self.combine(self.encode(source).mean(1))
        z=dg010.with_action_id(x)
        inputs=torch.cat([z,context[:,None,:].expand(-1,N,-1)],-1)
        return self.policy(inputs).squeeze(-1).masked_fill(~legal,-1e4)

class Blind(nn.Module):
    def __init__(self):
        super().__init__();self.net=SetReceiver()
    def forward(self,x,legal):
        return self.net(x,legal,torch.zeros((x.shape[0],3,SIG),dtype=x.dtype))

def acc(pred,y):return round(float((pred.argmax(-1)==y).float().mean()),5)

def evaluate(model,blind,c,bank,probe_index,seed):
    src=source_for(c['law'],bank,probe_index)
    g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        real=model(c['x'],c['legal'],src)
        blind_p=blind(c['x'],c['legal'])
        absent=model(c['x'],c['legal'],src*0)
        perm=torch.rand((len(src),3),generator=g).argsort(-1)
        shuffled_slots=model(c['x'],c['legal'],src.gather(1,perm[:,:,None].expand(-1,-1,SIG)))
        flipped=c['law'].clone();flipped[:,0]*=-1
        swapped=src.clone();swapped[:,0]=source_for(flipped,bank,probe_index)[:,0]
        wrong=model(c['x'],c['legal'],swapped)
        cf=torch.tensor([int(dg010.scores(c['x'][i].numpy(),c['legal'][i].numpy(),tuple(flipped[i].tolist())).argmax())
                         for i in range(len(c['y']))])
        result=dict(correct=acc(real,c['y']),blind=acc(blind_p,c['y']),zero=acc(absent,c['y']),
                    permuted_slots=acc(shuffled_slots,c['y']),permutation_disagree=int((real.argmax(-1)!=shuffled_slots.argmax(-1)).sum()),
                    wrong_wet_original=acc(wrong,c['y']),wrong_wet_counterfactual=acc(wrong,cf),
                    oracle_counterfactual_change=round(float((cf!=c['y']).float().mean()),5),
                    donor_counterfactual_recall=round(float((((wrong.argmax(-1)!=real.argmax(-1))&(cf!=c['y'])).sum()/(cf!=c['y']).sum().clamp_min(1))),5))
        for p in (real,blind_p,absent,shuffled_slots,wrong):
            assert c['legal'].gather(1,p.argmax(-1)[:,None]).all()
    return result

def selftest():
    probe=make_probes(4);m=Donor('gated');assert signature(m,probe).shape==(SIG,)
    c=dataset(3,25,'train');d=dataset(4,25,'sealed')
    assert not set(c['keys'])&set(d['keys'])
    assert not torch.any(torch.all(c['law']==torch.tensor(HELDOUT),-1))
    assert torch.all(d['law']==torch.tensor(HELDOUT))
    r=SetReceiver();x=c['x'][:5];legal=c['legal'][:5];src=torch.randn(5,3,SIG)
    a=r(x,legal,src);b=r(x,legal,src[:,[2,0,1]])
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.all(legal.gather(1,a.argmax(-1)[:,None]))
    print('DG011 SELFTEST: disjoint worlds / withheld triple / exact slot invariance / legal output PASS')

def run(out,seed,donor_steps,receiver_steps,train_cases,sealed_cases,signature_mode):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    start=time.monotonic();path=Path(out);path.mkdir(parents=True,exist_ok=True)
    probes=[make_probes(713+i*89) for i in range(5)]
    banks={};metrics={}
    for name,kind,offset in [('a','narrow',10000),('b','wide',20000),('c','gated',30000)]:
        banks[name],metrics[name]=make_bank(seed+offset,kind,donor_steps,probes,signature_mode)
    training=dataset(seed+1,train_cases,'train');validation=dataset(seed+2,500,'val');sealed=dataset(seed+3,sealed_cases,'sealed')
    assert not (set(training['keys'])&set(validation['keys']) or
                set(training['keys'])&set(sealed['keys']) or set(validation['keys'])&set(sealed['keys']))
    cache={(name,p):source_for(training['law'],banks[name],p) for name in ('a','b') for p in (0,1,2)}
    model=SetReceiver();blind=Blind()
    opt=torch.optim.AdamW(model.parameters(),lr=.0017,weight_decay=.0001)
    opt_blind=torch.optim.AdamW(blind.parameters(),lr=.0017,weight_decay=.0001)
    rng=np.random.default_rng(seed+900);history=[]
    for step in range(receiver_steps):
        idx=torch.from_numpy(rng.integers(train_cases,size=96))
        name=('a','b')[int(rng.integers(2))];p=int(rng.integers(3))
        src=cache[(name,p)][idx]; x=training['x'][idx]; legal=training['legal'][idx];y=training['y'][idx]
        z=model(x,legal,src);loss=F.cross_entropy(z,y)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        zb=blind(x,legal);lb=F.cross_entropy(zb,y)
        opt_blind.zero_grad();lb.backward();opt_blind.step()
        if step==0 or (step+1)%150==0 or step+1==receiver_steps:
            val=evaluate(model,blind,validation,banks['b'],1,seed+step)
            row=dict(step=step+1,train_loss=round(float(loss.detach()),5),validation=val)
            print(json.dumps(row),flush=True);history.append(row)
    sealed_results={}
    for kind in ('a','b','c'):
        for probe_idx,probe_name in [(0,'familiar_probes'),(4,'unseen_probes')]:
            sealed_results[f'{kind}:{probe_name}']=evaluate(model,blind,sealed,banks[kind],probe_idx,seed+101)
    with torch.no_grad():
        pred=model(sealed['x'],sealed['legal'],source_for(sealed['law'],banks['c'],4)).argmax(-1)
    third=torch.tensor([int(dg010.scores(sealed['x'][i].numpy(),sealed['legal'][i].numpy(),HELDOUT,extra=True).argmax())
                        for i in range(sealed_cases)])
    negative=dict(third_order_action_match=round(float((pred==third).float().mean()),5),
                  third_order_target_change=round(float((third!=sealed['y']).float().mean()),5))
    for bank_name,bank in banks.items():
        for (family,sign),entry in bank.items():
            torch.save(entry['model'].state_dict(),path/f'donor-{bank_name}-family{family}-sign{sign}.pt')
    torch.save(model.state_dict(),path/'set_receiver.pt');torch.save(blind.state_dict(),path/'blind.pt')
    decisions=[dict(world_sha=sealed['keys'][i],legal=[ACTIONS[j] for j,v in enumerate(sealed['legal'][i]) if v],
                    chosen=ACTIONS[int(pred[i])],gold=ACTIONS[int(sealed['y'][i])]) for i in range(min(64,sealed_cases))]
    (path/'original_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    result=dict(experiment='DG011',seed=seed,signature_mode=signature_mode,donor_steps=donor_steps,receiver_steps=receiver_steps,
                train_cases=train_cases,sealed_cases=sealed_cases,architecture_train=['narrow','wide'],
                architecture_sealed='gated',probe_sha256=[sha(p.numpy().tobytes()) for p in probes],
                donor_accuracy=metrics,history=history,sealed=sealed_results,unseen_nonidentifiable_law=negative,
                scientifically_valid_claim='supervised output-level neural donor transfer only; NOT native KV inheritance, true physics discovery or real CyberSand play',
                seconds=round(time.monotonic()-start,2))
    (path/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    hashes={f.name:sha(f.read_bytes()) for f in sorted(path.iterdir()) if f.is_file()}
    (path/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('DG011 SEALED '+json.dumps(sealed_results),flush=True)
    print('DG011 NEGATIVE '+json.dumps(negative),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    p.add_argument('--out',default='/tmp/dg011');p.add_argument('--seed',type=int,default=26011618)
    p.add_argument('--donor-steps',type=int,default=220);p.add_argument('--receiver-steps',type=int,default=750)
    p.add_argument('--train-cases',type=int,default=5200);p.add_argument('--sealed-cases',type=int,default=900)
    p.add_argument('--signature-mode',choices=['intervention','raw'],default='intervention')
    args=p.parse_args()
    if args.selftest:selftest()
    else:run(args.out,args.seed,args.donor_steps,args.receiver_steps,args.train_cases,args.sealed_cases,args.signature_mode)
