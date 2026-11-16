"""C12 anti-countermand synthetic-world tests; no native permissions."""
import copy
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from experiment import ARMS
from signed_feedback import HostAdvice, observe
from topology_transfer import build, ALL_FAMILIES
from selective_continue import SelectiveContinue, examples, optimize

class Tiny(nn.Module):
    def forward(self,x):
        return (x[...,2]*1.5+x[...,4]*.4+x[...,7]*.2)

class C12Tests(unittest.TestCase):
    def test_paired_correction_observations_and_labels(self):
        X,Y,r=examples(4)
        self.assertEqual(X.shape[1:],(10,29))
        self.assertEqual(len(X),len(Y))
        self.assertEqual(len(r),len(X))
        self.assertEqual(set(Y.tolist()),{0.,1.})
        self.assertEqual(set(z['arm'] for z in r),set(ARMS))
        self.assertTrue(all(z['protected_sha256'] and z['menu_sha256'] for z in r))

    def test_no_cue_and_positive_cue_invariant(self):
        old=Tiny();torch.manual_seed(12)
        model=SelectiveContinue(old)
        with torch.no_grad():model.delta[-1].bias.fill_(2.)
        for family in ALL_FAMILIES:
            w=build(12517,1,family,0,False,'reverse')
            for sign in ('none','backtrack','continue'):
                v=copy.deepcopy(w);ad=HostAdvice(v)
                if sign!='none':getattr(ad,'issue_'+sign)(v)
                cs,x=observe(v,1.,ad);t=torch.from_numpy(x)
                a=old(t);b=model(t)
                self.assertTrue(torch.equal(a[:8],b[:8]))
                self.assertEqual(a[9],b[9])
                if sign!='continue':self.assertTrue(torch.equal(a,b))
                else:self.assertNotEqual(a[8],b[8])

    def test_refused_action_remains_illegal_even_with_cue(self):
        w=build(7111,-1,'hook',0,False,'reverse')
        advice=HostAdvice(w);advice.issue_continue(w)
        cs,x=observe(w,1.,advice)
        bad=next(c for c in cs if c.action[0]=='INK' and not c.legal)
        protected=set(w.protected)
        self.assertFalse(w.apply(bad))
        self.assertEqual(w.protected,protected)

    def test_cancel_and_expired_return_to_frozen_behavior(self):
        w=build(7611,1,'zigzag',0,False,'reverse')
        advice=HostAdvice(w);advice.issue_continue(w,ttl=1)
        old=Tiny();model=SelectiveContinue(old)
        with torch.no_grad():model.delta[-1].bias.fill_(1.)
        advice.cancel(w);_,x=observe(w,1.,advice)
        self.assertTrue(torch.equal(old(torch.from_numpy(x)),model(torch.from_numpy(x))))
        advice.issue_continue(w,ttl=1)
        advice.advance(w);_,x=observe(w,1.,advice)
        self.assertTrue(torch.equal(old(torch.from_numpy(x)),model(torch.from_numpy(x))))

    def test_real_optimizer_updates_and_frozen_prior(self):
        X,Y,_=examples(6)
        model=SelectiveContinue(Tiny())
        before=[p.clone() for p in model.delta.parameters()]
        _,steps,losses=optimize(model,X,Y,epochs=2)
        self.assertGreater(steps,0)
        self.assertEqual(len(losses),2)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.delta.parameters())))

if __name__=='__main__':unittest.main()
