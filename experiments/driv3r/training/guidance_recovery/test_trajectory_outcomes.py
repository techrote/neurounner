"""C14 synthetic-host and correction-only research regression suite."""
import unittest
import numpy as np
import torch
from experiment import ARMS,sha
from topology_transfer import build,ALL_FAMILIES
from signed_feedback import HostAdvice,observe
from trajectory_outcomes_core import suffix,outcome_reward,fit,RetentionGate

class TrajectoryOutcomeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_reward_success_has_priority(self):
        class Dummy:
            def __init__(self,d):self.d=d
            def distance(self):return self.d
        self.assertGreater(outcome_reward(Dummy(0),7,10,10),outcome_reward(Dummy(7),7,0,1))
    def test_worlds_materials_unchanged_under_directives(self):
        for fam in ALL_FAMILIES:
            for outcome in ('reverse','ahead'):
                w=build(1919,-1,fam,0,False,outcome)
                p=sha(sorted(w.protected));old=set(w.inked)
                a=HostAdvice(w);a.issue_backtrack(w)
                cs,x=observe(w,ARMS[0],a)
                self.assertEqual(x.shape,(10,29))
                self.assertTrue(cs[8].legal)
                self.assertTrue(w.apply(cs[8]))
                a.advance(w)
                self.assertEqual(sha(sorted(w.protected)),p)
                self.assertTrue(old<=w.inked)
    def test_weak_and_strong_preserve_host_menu(self):
        for fam in ALL_FAMILIES:
            w=build(19391,1,fam,0,False,'ahead')
            menus=[]
            for strength in (0.,.35,1.):
                a=HostAdvice(w)
                if strength:a.issue_continue(w,strength)
                cs,x=observe(w,0.,a)
                menus.append([(c.action,c.target,c.legal) for c in cs])
            self.assertEqual(menus[0],menus[1]);self.assertEqual(menus[1],menus[2])
    def test_frozen_policy_parameter_scope(self):
        prior=torch.nn.Linear(29,1)
        # Gate expects the prior to return 10 per-candidate values.
        gate=RetentionGate(prior)
        self.assertFalse(any(p.requires_grad for p in gate.prior.parameters()))
        self.assertTrue(all(p.requires_grad for p in gate.delta.parameters()))
    def test_advice_invariance_and_other_action_invariance(self):
        # Lightweight deterministic prior for architecture guard, no checkpoint needed.
        class Baseline(torch.nn.Module):
            def forward(self,x):return (x[...,0]+x[...,7]-x[...,8]*.2).contiguous()
        prior=Baseline();gate=RetentionGate(prior)
        with torch.no_grad():
            gate.delta[-1].bias.fill_(1.)
            for directive in ('none','expired','backtrack','continue'):
                w=build(6772,-1,'hook',1,True,'reverse');a=HostAdvice(w)
                if directive=='backtrack':a.issue_backtrack(w)
                if directive=='continue':a.issue_continue(w)
                if directive=='expired':a.issue_continue(w,ttl=1);a.advance(w)
                _,x=observe(w,.35,a);inp=torch.from_numpy(x)
                old=prior(inp);new=gate(inp)
                self.assertTrue(torch.equal(old[:8],new[:8]))
                self.assertTrue(torch.equal(old[9:],new[9:]))
                if directive in ('none','expired'):self.assertTrue(torch.equal(old,new))
                else:self.assertNotEqual(float(old[8]),float(new[8]))

if __name__=='__main__':unittest.main()
