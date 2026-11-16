"""C18 research-only host-attested trail features and bounded policy residual.

The game/runtime must never consume this module. Full training and synthetic
host evaluation sources, pretrained prerequisites and safetensors are archived
separately; the repository contains the small reviewable inference contract.
"""
import numpy as np
import torch
from torch import nn


def host_trail_facts(history, goal, backtrack_legal, backtrack_target):
    found=[j for j,p in enumerate(history) if p==goal]
    found=found[-1] if found else None
    on_trail=float(found is not None)
    depth=float(len(history)-found)/max(1,len(history)) if found is not None else 0.
    next_is_goal=float(bool(backtrack_legal and history and backtrack_target==goal and
                            backtrack_target==history[-1]))
    return np.asarray([on_trail,depth,next_is_goal],dtype=np.float32)


def attach_host_facts(observation, facts):
    """Extend ten legal-action feature rows from 29 to 32, without rewriting them."""
    if observation.shape!=(10,29) or np.asarray(facts).shape!=(3,):
        raise ValueError('C18 expects 10x29 candidate menu and three host facts')
    return np.concatenate((observation,np.tile(np.asarray(facts,dtype=np.float32),(10,1))),axis=1)


class AttestedTrailResidual(nn.Module):
    """Only BACKTRACK (candidate 8) is altered; prior and legality stay frozen."""
    def __init__(self, prior, scale=2., advice_filter='any', width=16):
        super().__init__()
        if advice_filter not in ('any','backtrack'):
            raise ValueError('unknown advice filter')
        self.prior=prior
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        self.delta=nn.Sequential(nn.Linear(32,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.constant_(self.delta[-1].bias,0.)
        self.scale=float(scale)
        self.advice_filter=advice_filter

    def forward(self,x):
        if x.shape[-2:]!=(10,32):
            raise ValueError('C18 expects candidate menu with 10x32 features')
        old=self.prior(x[...,:29]);result=old.clone()
        b=x[...,8,:]
        active=b[...,2]*b[...,4]*(b[...,26].abs()>.5).to(b.dtype)*b[...,27]*b[...,29]
        if self.advice_filter=='backtrack':
            active=active*(b[...,26]>0).to(b.dtype)
        correction=active*self.scale*torch.sigmoid(self.delta(b).squeeze(-1))
        result[...,8]=old[...,8]+correction
        return result
