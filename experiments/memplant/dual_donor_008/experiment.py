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
