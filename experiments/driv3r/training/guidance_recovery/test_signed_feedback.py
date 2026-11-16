"""C11 host-signed and countermanded recovery safety, corpus, and optimizer checks."""
import unittest
import numpy as np
import torch
from topology_transfer import build
from signed_feedback import HostAdvice, observe, SignedGate, training_data, train, guard
from persistent_feedback import PersistentGate

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(25,1)
    def forward(self,x):return self.layer(x).squeeze(-1)

class SignedFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.world=build(540012,1,'hairpin',0,False,'ahead')

    def test_advisory_is_host_only_and_bounded(self):
        a=HostAdvice(self.world)
        with self.assertRaises(ValueError): a.issue_continue(self.world,1.2)
        with self.assertRaises(ValueError): a.issue_backtrack(self.world,ttl=100)
        a.issue_backtrack(self.world)
        self.assertEqual(a.active(self.world),1)
        a.issue_continue(self.world,.4,ttl=3)
        self.assertEqual(a.active(self.world),-1)
        self.assertEqual(self.world.recovery_cue_ttl,0)
        self.assertEqual(a.context,None)
        for _ in range(3): a.advance(self.world)
        self.assertEqual(a.active(self.world),0)

    def test_zero_strength_is_inactive(self):
        a=HostAdvice(self.world);a.issue_backtrack(self.world,0.)
        self.assertEqual(a.active(self.world),0)
        self.assertEqual(self.world.recovery_cue_ttl,0)
        a.issue_continue(self.world,0.)
        self.assertEqual(a.active(self.world),0)
        _,x=observe(self.world,.35,a)
        self.assertEqual(x[8,23],0.)

    def test_material_cannot_be_modified_by_countermand(self):
        a=HostAdvice(self.world)
        before=set(self.world.protected);ink=set(self.world.inked)
        a.issue_backtrack(self.world)
        a.issue_continue(self.world)
        a.cancel(self.world)
        self.assertEqual(before,self.world.protected)
        self.assertEqual(ink,self.world.inked)

    def test_observation_retains_25_features(self):
        a=HostAdvice(self.world);a.issue_continue(self.world)
        cs,x=observe(self.world,.35,a)
        self.assertEqual(x.shape,(10,29))
        self.assertTrue(np.array_equal(x[:,:25],np.asarray([c.features for c in cs],np.float32)))
        self.assertEqual(x[8,26],-1.)
        self.assertTrue(cs[8].legal)

    def test_inactive_and_nonbacktrack_scores_exact(self):
        old=PersistentGate(Tiny())
        # not a production scorer, just exact shape-compatible frozen input
        model=SignedGate(old)
        g=guard(old,model)
        self.assertGreater(g['exact_inactive_pairs'],0)
        self.assertGreater(g['revocations'],0)

    def test_recovery_cue_does_not_make_rock_legal(self):
        a=HostAdvice(self.world);a.issue_continue(self.world)
        cs,_=observe(self.world,1.0,a)
        self.assertTrue(any(not c.legal for c in cs))
        illegal=next(c for c in cs if c.action[0]=='INK' and not c.legal)
        before=set(self.world.protected)
        self.assertFalse(self.world.apply(illegal))
        self.assertEqual(before,self.world.protected)

    def test_supervision_includes_true_and_false_and_all_arms(self):
        X,Y,recs=training_data(n=4)
        self.assertEqual(X.shape[1:],(10,29))
        self.assertEqual(len(recs),len(X))
        self.assertEqual(set(Y.tolist()),{0.,1.})
        self.assertEqual(set(r['arm'] for r in recs),{0.,.35,1.})
        self.assertTrue(all(r['obs_sha256'] and r['menu_sha256'] for r in recs))

    def test_optimizer_updates_weights(self):
        torch.manual_seed(38)
        old=PersistentGate(Tiny())
        model=SignedGate(old)
        X,Y,_=training_data(n=4)
        before=[p.detach().clone() for p in model.delta.parameters()]
        model,steps,losses=train(model,X,Y,epochs=2)
        self.assertGreater(steps,0)
        self.assertEqual(len(losses),2)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.delta.parameters())))

if __name__=='__main__':unittest.main()
