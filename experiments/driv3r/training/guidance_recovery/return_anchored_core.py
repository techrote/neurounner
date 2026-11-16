"""C19 model-only return-anchored BACKTRACK residual (synthetic research).

Frozen C12 receives the original 29-feature action menu. Additional three
features are host-attested executed history facts, not privileged geometry.
This does not grant legal permissions, alter materials or affect CyberSand.
The trainable head is bounded around the successful C18-A0 +1-logit gate.
"""
from __future__ import annotations
import torch
from torch import nn


class ReturnAnchoredGate(nn.Module):
    """Only BACKTRACK logit changes under live host guidance and trail proof.

    Model output is the ten *unmasked* candidate scores; callers MUST enforce
    the host legal action mask before selecting and executing any action.
    """
    def __init__(self,prior,width=12,scale=.8,variant='full'):
        super().__init__()
        self.prior=prior
        for p in prior.parameters():p.requires_grad_(False)
        self.scale=float(scale)
        self.variant=variant
        if variant not in ('full','local'):raise ValueError(variant)
        self.indices=(7,8,17,22,26,27,28,29,30,31) if variant=='local' else tuple(range(32))
        self.head=nn.Sequential(nn.Linear(len(self.indices),width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,x):
        if x.shape[-2:]!=(10,32):raise ValueError('expected 10x32 candidate features')
        old=self.prior(x[...,:29])
        out=old.clone()
        b=x[...,8,:]
        # action=BACKTRACK, legal, signed host advice, active confidence,
        # goal is attested on executed history. These are necessary conditions.
        active=b[...,2]*b[...,4]*(b[...,26].abs()>.5).to(b.dtype)*b[...,27]*b[...,29]
        correction=active*(1.+self.scale*torch.tanh(self.head(b[...,list(self.indices)]).squeeze(-1)))
        out[...,8]=old[...,8]+correction
        return out
