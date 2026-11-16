"""DG008: frozen specialised neural game-law donors -> compositional recipient.

Research-only, synthetic 9x9 CyberSand-like legal actions. Neither native KV
transplantation nor undiscovered physics. No CyberSand game files are touched.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_ACTIONS = 6
N_FEATURES = 15
N_FAMILIES = 6
MAX_LAWS = 4
ACTIONS = ('LEFT', 'RIGHT', 'UP', 'DOWN', 'PAINT', 'RETRACE')
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PAIRS = ((0,4), (1,3), (2,5), (0,5), (1,4))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world(seed):
    r = random.Random(seed)
    tiles = np.array(r.choices([0,1,2,3], weights=[35,28,26,11], k=81),dtype=np.int8).reshape(9,9)
    protected = np.array([r.random()<.09 for _ in range(81)],dtype=bool).reshape(9,9)
    free = [(x,y) for y in range(9) for x in range(9) if tiles[y,x]!=3 and not protected[y,x]]
    start = r.choice(free)
    goal = r.choice([p for p in free if p!=start])
    return tiles,protected,start,goal


def action_features(seed):
    r = random.Random(seed+103)
    tiles,protected,pos,goal = world(seed)
    history=[pos]
    for _ in range(r.randint(0,4)):
        x,y=history[-1]; dx,dy=r.choice(DIRS); n=(x+dx,y+dy)
        if 0<=n[0]<9 and 0<=n[1]<9 and tiles[n[1],n[0]]!=3 and not protected[n[1],n[0]]:
            history.append(n)
    pos=history[-1]; heading=r.randrange(4)
    marked = {r.choice(history)} if r.random()<.35 else set()
    return offer_state(tiles,protected,pos,goal,history,marked,heading)


def offer_state(tiles,protected,pos,goal,history,marked,heading):
    x,y=pos;gx,gy=goal;dist=abs(gx-x)+abs(gy-y)
    features=np.zeros((N_ACTIONS,N_FEATURES),dtype=np.float32)
    mask=np.zeros(N_ACTIONS,dtype=bool)
    for j in range(N_ACTIONS):
        nx,ny=(x+DIRS[j][0],y+DIRS[j][1]) if j<4 else (history[-2] if j==5 and len(history)>1 else pos)
        inside=0<=nx<9 and 0<=ny<9
        tile=int(tiles[ny,nx]) if inside else 3
        mask[j]=(inside and tile!=3 and not protected[ny,nx]) if j<4 else (not protected[y,x] if j==4 else len(history)>1)
        a=features[j]
        a[0]=(dist-(abs(gx-nx)+abs(gy-ny)))/2
        a[1]=(abs(gx-nx)+abs(gy-ny))/16
        a[2]=float((nx,ny) not in history)
        a[3]=float(j==heading)
        a[4:7]=[float(tile==t) for t in (0,1,2)]
        a[7]=float((nx,ny)==goal)
        a[8]=float(j==4)
        a[9]=float(j==5)
        a[10]=float(j<4)
        a[11]=float(pos in marked)
        a[12]=float(dist<=2)
        a[13]=float(len(history)>1 and history[-2]==(nx,ny))
        a[14]=1.
    assert mask.any() and mask[4]
    return features,mask


def base_scores(f):
    return (f[...,10]*(1.15*f[...,0]+.24*f[...,2]+.17*f[...,3]-.13*f[...,13])
            +f[...,8]*(.38+.55*f[...,12]+.25*f[...,7]-1.05*f[...,11])
            +f[...,9]*(-.3+.55*f[...,13]))


def effects(features,codes):
    """Authored synthetic atomic laws; codes 2*family + sign bit."""
    f=features
    feats=(1.7*f[...,5]*f[...,10], 1.7*f[...,6]*f[...,10],
           1.4*f[...,8]*(1-f[...,11]), 1.4*f[...,3]*f[...,10],
           1.35*(f[...,0]-.45*f[...,2])*f[...,10],
           1.65*f[...,9]-.53*f[...,13]*f[...,10])
    v=torch.zeros_like(f[...,0])
    for code in codes:
        if code<0:continue
        family,sign=divmod(int(code),2)
        v=v+feats[family]*(1 if sign else -1)
    return v


def pair_effects(features,codes):
    f=features
    active={int(c)//2:(1 if int(c)%2 else -1) for c in codes if int(c)>=0}
    terms={(0,4): 1.05*f[...,5]*f[...,0]*f[...,10],
           (1,3): 1.15*f[...,6]*f[...,3]*f[...,10],
           (2,5): .90*f[...,8]*f[...,11],
           (0,5): .75*f[...,9]*f[...,12],
           (1,4): 1.0*f[...,6]*f[...,0]*f[...,10]}
    v=torch.zeros_like(f[...,0])
    for pair,x in terms.items():
        if pair[0] in active and pair[1] in active:
            v=v+active[pair[0]]*active[pair[1]]*x
    return v


def one(seed,count,nlaws):
    r=random.Random(seed)
    f=[];m=[];codes=[];scores=[];labels=[]
    for j in range(count):
        wi=seed*100000+j
        feat,leg=action_features(wi)
        k=r.choice((0,1,1,2,2)) if nlaws==-1 else nlaws
        families=r.sample(range(N_FAMILIES),k)
        law=[2*i+r.randrange(2) for i in families]
        law=law+[-1]*(MAX_LAWS-len(law))
        ft=torch.from_numpy(feat)
        target=base_scores(ft)+effects(ft,law)+pair_effects(ft,law)
        f.append(feat);m.append(leg);codes.append(law);scores.append(target.numpy());
        labels.append(int(target.masked_fill(~torch.from_numpy(leg),-1e6).argmax()))
    return {'features':torch.from_numpy(np.stack(f)), 'legal':torch.from_numpy(np.stack(m)),
            'codes':torch.tensor(codes,dtype=torch.long),
            'scores':torch.from_numpy(np.stack(scores)), 'target':torch.tensor(labels)}


class Donor(nn.Module):
    """Specialised parent: families [0,1,2] or [3,4,5], six signs per parent."""
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(6,16)
        self.predict=nn.Sequential(nn.Linear(16+N_FEATURES,48),nn.Tanh(),nn.Linear(48,1))
    def forward(self,local_code,feat):
        emb=self.embedding(local_code)
        z=torch.cat((feat,emb[:,None,:].expand(-1,N_ACTIONS,-1)),-1)
        return self.predict(z).squeeze(-1)


def train_donor(seed,group,steps):
    torch.manual_seed(seed+group*17)
    model=Donor()
    data=one(seed+101+group,4200,-1)
    f=data['features'];rng=torch.Generator().manual_seed(seed+993+group)
    opt=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=0)
    curve=[]
    for it in range(1,steps+1):
        ix=torch.randint(0,len(f),(192,),generator=rng)
        local=torch.randint(0,6,(len(ix),),generator=rng)
        codes=[2*(group*3+j//2)+j%2 for j in local.tolist()]
        targets=torch.stack([effects(f[k],(c,)) for k,c in zip(ix.tolist(),codes)])
        pred=model(local,f[ix]);loss=F.mse_loss(pred,targets)
        opt.zero_grad();loss.backward();opt.step()
        if it%100==0 or it==steps:curve.append((it,round(float(loss.detach()),6)))
    model.eval()
    return model,curve


def tokens(donors,data,random_mode=False):
    B=len(data['codes']);f=data['features'];codes=data['codes']
    emb=torch.zeros(B,MAX_LAWS,16);delta=torch.zeros(B,MAX_LAWS,N_ACTIONS)
    with torch.no_grad():
        for col in range(MAX_LAWS):
            for group in (0,1):
                valid=(codes[:,col]>=0)&((codes[:,col]//6)==group)
                if not valid.any():continue
                inds=valid.nonzero(as_tuple=True)[0]
                local=codes[inds,col]-group*6
                model=donors[group]
                emb[inds,col]=model.embedding(local)
                delta[inds,col]=model(local,f[inds])
    return emb,delta


class Receiver(nn.Module):
    """Learn only pairwise residuals; frozen donor deltas supply atomic law effects."""
    def __init__(self):
        super().__init__()
        self.pair=nn.Sequential(nn.Linear(N_FEATURES+2*16+2,80),nn.Tanh(),nn.Linear(80,48),nn.Tanh(),nn.Linear(48,1))
    def forward(self,feat,emb,delta,codes,with_pairs=True):
        out=base_scores(feat)+delta.sum(1)
        if with_pairs:
            for a,b in itertools.combinations(range(MAX_LAWS),2):
                valid=((codes[:,a]>=0)&(codes[:,b]>=0)).float()[:,None]
                # Both orderings, averaged, to prevent input-column identity leakage.
                joined=[]
                for i,j in ((a,b),(b,a)):
                    ea=emb[:,i,None,:].expand(-1,N_ACTIONS,-1)
                    eb=emb[:,j,None,:].expand(-1,N_ACTIONS,-1)
                    x=torch.cat((feat,ea,eb,delta[:,i,:,None],delta[:,j,:,None]),-1)
                    joined.append(self.pair(x).squeeze(-1))
                out=out+.5*(joined[0]+joined[1])*valid
        return out


def accuracy(scores,data):
    prediction=scores.masked_fill(~data['legal'],-1e5).argmax(-1)
    assert bool(data['legal'].gather(1,prediction[:,None]).all())
    return float((prediction==data['target']).float().mean()),prediction


def train_receiver(seed,donors,steps=450):
    train=one(seed+200,6500,-1)
    valid=one(seed+300,1200,-1)
    et,dt=tokens(donors,train)
    ev,dv=tokens(donors,valid)
    torch.manual_seed(seed+543)
    model=Receiver();opt=torch.optim.AdamW(model.parameters(),lr=.003)
    gen=torch.Generator().manual_seed(seed+12)
    best=1e9; weights=None;curve=[]
    for step in range(1,steps+1):
        ix=torch.randint(0,len(train['target']),(160,),generator=gen)
        estimated=model(train['features'][ix],et[ix],dt[ix],train['codes'][ix])
        loss=F.mse_loss(estimated,train['scores'][ix])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():
                val=F.mse_loss(model(valid['features'],ev,dv,valid['codes']),valid['scores'])
                acc,_=accuracy(model(valid['features'],ev,dv,valid['codes']),valid)
            curve.append((step,round(float(loss.detach()),6),round(float(val),6),round(acc,6)))
            if float(val)<best:
                best=float(val);weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights);model.eval()
    return model,curve


def evaluate(model,donors,seed,nlaws,n=1600):
    data=one(seed+nlaws*11,n,nlaws)
    em,de=tokens(donors,data)
    with torch.no_grad():
        outputs={
            'grafted_pairwise':model(data['features'],em,de,data['codes']),
            'atomic_additive':model(data['features'],em,de,data['codes'],False),
            'no_donor':base_scores(data['features']),
        }
        perm=torch.randperm(len(em),generator=torch.Generator().manual_seed(seed+447))
        outputs['shuffled_donor']=model(data['features'],em[perm],de[perm],data['codes'])
        outputs['oracle_expert']=data['scores']
    oracle_additive = torch.stack([
        base_scores(ft)+effects(ft,codes) for ft,codes in zip(data['features'],data['codes'])
    ])
    oracle_add_prediction=oracle_additive.masked_fill(~data['legal'],-1e5).argmax(-1)
    challenge=(oracle_add_prediction!=data['target'])
    report={'_interaction_challenge': {'cases':int(challenge.sum()),'n':n,
        'fraction':round(float(challenge.float().mean()),6)}}
    for name,scores in outputs.items():
        hit,p=accuracy(scores,data)
        mse=float(F.mse_loss(scores,data['scores']))
        distribution={action:int((p==i).sum()) for i,action in enumerate(ACTIONS)}
        report[name]={'action_accuracy':round(hit,6),'score_mse':round(mse,6),'action_counts':distribution,'n':n,
                      'interaction_challenge_accuracy':(round(float((p[challenge]==data['target'][challenge]).float().mean()),6) if challenge.any() else None)}
    return report



def interaction_stress(model,donors,seed,nlaws=3,n=8000):
    """Selection-conditional evaluation: mine *sealed*, teacher-defined hard cases.

    This does NOT estimate the unconditioned population accuracy. Cases are
    selected solely because authored pair terms change the oracle legal action.
    """
    d=one(seed,n,nlaws)
    oracle_add=torch.stack([base_scores(f)+effects(f,law)
                            for f,law in zip(d['features'],d['codes'])])
    arg=oracle_add.masked_fill(~d['legal'],-1e5).argmax(-1)
    ix=(arg!=d['target']).nonzero(as_tuple=True)[0]
    chosen={key:(value[ix] if isinstance(value,torch.Tensor) else value)
            for key,value in d.items()}
    e,v=tokens(donors,chosen)
    with torch.no_grad():
        graft=model(chosen['features'],e,v,chosen['codes'])
        add=model(chosen['features'],e,v,chosen['codes'],False)
    a,_=accuracy(graft,chosen);b,_=accuracy(add,chosen)
    return {'examined':n,'interaction_changing_cases':len(ix),
            'grafted_action_accuracy':round(a,6),
            'additive_action_accuracy':round(b,6),
            'warning':'Outcome-conditioned evaluation, not natural-frequency accuracy'}

def causal_pairs(model,donors,seed,n=600):
    """Batch the SAME-world law flips to avoid serial tiny model inference."""
    rng=random.Random(seed+223)
    fs=[];masks=[];codes=[];truth=[]
    for j in range(n):
        f,mask=action_features(seed*200000+j)
        families=rng.sample(range(6),3)
        a=[2*x+rng.randrange(2) for x in families]
        b=a.copy();b[rng.randrange(3)]^=1
        for law in (a,b):
            law=law+[-1]*(MAX_LAWS-len(law))
            ft=torch.from_numpy(f)
            target=base_scores(ft)+effects(ft,law)+pair_effects(ft,law)
            fs.append(f);masks.append(mask);codes.append(law)
            truth.append(int(target.masked_fill(~torch.from_numpy(mask),-1e6).argmax()))
    d={'features':torch.from_numpy(np.stack(fs)),
       'legal':torch.from_numpy(np.stack(masks)),
       'codes':torch.tensor(codes)}
    em,de=tokens(donors,d)
    gold=torch.tensor(truth).reshape(n,2)
    expected=(gold[:,0]!=gold[:,1])
    with torch.no_grad():
        scores={'grafted_pairwise':model(d['features'],em,de,d['codes']),
