"""No-Git-binary, native-free C19 compatibility and permission invariants."""
import unittest
import torch
from torch import nn
from return_anchored_core import ReturnAnchoredGate

class ToyFrozenPolicy(nn.Module):
    def forward(self,x):
        return x[...,7]+.3*x[...,13]-.12*x[...,22]


def menu(advice=1.,trail=1.,legal=1.,confidence=1.):
    x=torch.zeros(10,32,dtype=torch.float32)
    x[:,4]=1
    x[8,2]=1;x[8,4]=legal;x[8,26]=advice;x[8,27]=confidence;x[8,29]=trail
    x[:,7]=torch.arange(10)/10.
    return x

class C19GateTests(unittest.TestCase):
    def setUp(self):
        self.prior=ToyFrozenPolicy()
        self.full=ReturnAnchoredGate(self.prior,variant='full')
        self.local=ReturnAnchoredGate(self.prior,variant='local')

    def test_cue_off_exact(self):
        for model in (self.full,self.local):
            x=menu(advice=0)
            self.assertTrue(torch.equal(model(x),self.prior(x[:,:29])))

    def test_trail_off_exact(self):
        for model in (self.full,self.local):
            x=menu(trail=0)
            self.assertTrue(torch.equal(model(x),self.prior(x[:,:29])))

    def test_expired_confidence_exact(self):
        for model in (self.full,self.local):
            x=menu(confidence=0)
            self.assertTrue(torch.equal(model(x),self.prior(x[:,:29])))

    def test_illegal_backtrack_no_edit(self):
        for model in (self.full,self.local):
            x=menu(legal=0)
            self.assertTrue(torch.equal(model(x),self.prior(x[:,:29])))
            self.assertEqual(float(x[8,4]),0.)

    def test_any_sign_enables_bounded_correction(self):
        for model in (self.full,self.local):
            for sign in (1.,-1.):
                x=menu(advice=sign)
                y=model(x);old=self.prior(x[:,:29])
                self.assertAlmostEqual(float(y[8]-old[8]),1.,places=6)
                self.assertTrue(torch.equal(y[:8],old[:8]))
                self.assertTrue(torch.equal(y[9:],old[9:]))

    def test_strict_inactive_other_action_invariance(self):
        for model in (self.full,self.local):
            for arg in (dict(advice=1),dict(advice=-1),dict(trail=0)):
                x=menu(**arg);old=self.prior(x[:,:29]);new=model(x)
                self.assertTrue(torch.equal(old[:8],new[:8]))
                self.assertTrue(torch.equal(old[9:],new[9:]))

    def test_trainable_head_only(self):
        for model in (self.full,self.local):
            self.assertFalse(any(p.requires_grad for p in model.prior.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):self.full(torch.zeros(10,29))

    def test_reject_unknown_variant(self):
        with self.assertRaises(ValueError):ReturnAnchoredGate(self.prior,variant='unsupported')

    def test_bounded_residual_after_optimizing(self):
        for model in (self.full,self.local):
            with torch.no_grad():model.head[-1].bias.fill_(8.)
            x=menu();old=self.prior(x[:,:29]);new=model(x)
            delta=float(new[8]-old[8]);self.assertGreaterEqual(delta,.2-1e-5)
            self.assertLessEqual(delta,1.8+1e-5)

if __name__=='__main__':unittest.main()
