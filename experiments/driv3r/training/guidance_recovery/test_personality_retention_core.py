"""C13 optimizer core guards: synthetic-only, frozen-base, host-legal action menus."""
import unittest
import torch
from torch import nn
from signed_feedback import HostAdvice, observe
from topology_transfer import ALL_FAMILIES, build
from personality_retention_core import RetentionGate, shortest_dist, teacher_backtrack, gather, fit

class Tiny(nn.Module):
    def forward(self,x):return x[...,2]*1.2+x[...,4]*.3+x[...,7]*.3

class CoreTests(unittest.TestCase):
    def test_teacher_never_labelled_unreachable_as_success(self):
        for family in ALL_FAMILIES:
            for outcome in ('reverse','ahead'):
                w=build(79271,1,family,0,False,outcome)
                cs,_=observe(w,1.,HostAdvice(w))
                self.assertEqual(shortest_dist(w,w.goal),0)
                self.assertEqual(shortest_dist(w,(-1,-1)),999)
                self.assertIn(teacher_backtrack(w,cs),(0.,1.,None))

    def test_frozen_and_nonbacktrack_logits(self):
        old=Tiny();model=RetentionGate(old)
        with torch.no_grad():model.delta[-1].bias.fill_(1.)
        for family in ALL_FAMILIES:
            for directive in ('none','backtrack','continue','expired'):
                w=build(79171,-1,family,0,False,'reverse')
                advice=HostAdvice(w)
                if directive=='backtrack':advice.issue_backtrack(w)
                if directive=='continue':advice.issue_continue(w)
                if directive=='expired':advice.issue_backtrack(w,ttl=1);advice.advance(w)
                _,x=observe(w,.35,advice);t=torch.from_numpy(x)
                with torch.no_grad():a=old(t);b=model(t)
                self.assertTrue(torch.equal(a[:8],b[:8]))
                self.assertEqual(a[9],b[9])
                if directive in ('none','expired'):self.assertTrue(torch.equal(a,b))
                else:self.assertNotEqual(a[8],b[8])

    def test_host_prohibitions_survive_advice(self):
        w=build(98271,1,'hook',1,True,'reverse')
        a=HostAdvice(w);a.issue_backtrack(w)
        candidates,_=observe(w,1.,a)
        illegal=next(c for c in candidates if c.action[0]=='INK' and not c.legal)
        protected=set(w.protected)
        self.assertFalse(w.apply(illegal))
        self.assertEqual(w.protected,protected)

    def test_actual_optimizer_updates_and_source_hashes(self):
        X,Y,W,receipts=gather(Tiny(),n=8,horizon=5)
        self.assertEqual(X.shape[1:],(10,29))
        self.assertEqual(len(X),len(receipts))
        self.assertEqual(set(Y.tolist()),{0.,1.})
        self.assertTrue(all(r['world_sha256'] and r['menu_sha256'] for r in receipts))
        model=RetentionGate(Tiny())
        before=[p.detach().clone() for p in model.delta.parameters()]
        model,steps,losses=fit(model,X,Y,W,epochs=2)
        self.assertGreater(steps,0)
        self.assertEqual(len(losses),2)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.delta.parameters())))

if __name__=='__main__':unittest.main()
