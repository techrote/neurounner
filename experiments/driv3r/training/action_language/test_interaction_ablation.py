"""Adversarial safety, feature-isolation, source replay and optimizer tests."""
import json
import tempfile
import unittest
from pathlib import Path

from lab import World, FEATURES, HostRefusal, START, sha
from candidate_features import CandidateWorld
from interaction_ablation import (InkTargetWorld, feature_map, Policy, train,
                                  rollout, run, audit, AuditRefusal, TARGET, INTERACTIONS)

class InteractionTests(unittest.TestCase):
    def test_stop_not_masked(self):
        for arm in ('cell','sweep'):
            w=InkTargetWorld(310001,'pillars',arm,'candidate')
            self.assertIn('STOP',w.legal())
            self.assertEqual(len(feature_map(w,'STOP',())),len(FEATURES)+len(INTERACTIONS))
            self.assertTrue(all(x==0 for x in feature_map(w,'STOP',())[len(FEATURES):]))

    def test_legacy_prefix_identical(self):
        w=InkTargetWorld(310001,'gates','sweep','candidate')
        for action,path in w.legal().items():
            self.assertEqual(feature_map(w,action,path)[:len(FEATURES)], w.features(action,path))

    def test_pairwise_does_not_affect_offers(self):
        w=InkTargetWorld(310002,'empty','sweep','candidate')
        offer=w.offer()
        for name,path in offer[0].items():
            feature_map(w,name,path,nonlinear=True)
            feature_map(w,name,path,nonlinear=False)
        self.assertEqual(w.offer(),offer)
        self.assertEqual(len(w.own),0)

    def test_host_refuses_forged_action(self):
        w=InkTargetWorld(310003,'empty','sweep','candidate')
        _,token=w.offer()
        with self.assertRaises(HostRefusal):w.apply('INK_E_999',token)
        with self.assertRaises(HostRefusal):w.apply('INK_E_1','foreign-token')
        self.assertEqual(w.decisions,0)

    def test_foreign_lease_refused(self):
        x=InkTargetWorld(310001,'empty','sweep','candidate',lease='lease-x')
        y=InkTargetWorld(310001,'empty','sweep','candidate',lease='lease-y')
        _,token=x.offer()
        with self.assertRaises(HostRefusal):y.apply('INK_E_1',token)
        self.assertEqual(y.material,y.original)

    def test_exact_ink_budget_and_protected_preservation(self):
        for arm in ('cell','sweep'):
            w=InkTargetWorld(310001,'scatter',arm,'candidate')
            while len(w.own)<TARGET and w.decisions<96:
                menu,token=w.offer()
                actions=[a for a in menu if a.startswith('INK_')]
                if not actions:break
                w.apply(actions[0],token)
            self.assertLessEqual(len(w.own),TARGET)
            self.assertTrue(all(w.material[w.idx(q)]==5 for q in w.protected))
            self.assertIn('STOP',w.legal())

    def test_training_is_real_and_bounded(self):
        p,n=train('cell',True,3)
        self.assertGreater(n,0)
        self.assertTrue(any(abs(x)>1e-8 for x in p.weights))
        self.assertEqual(len(p.weights),len(FEATURES)+len(INTERACTIONS))
        with self.assertRaises(ValueError):train('sweep',True,121)

    def test_original_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            r=run(d,train_worlds=2,eval_worlds=5,gallery_worlds=0)
            self.assertEqual(audit(d)['replayed_episodes'],20)
            src=Path(d)/'receipts.json'
            data=json.loads(src.read_text())
            data['cell_linear'][0][0]['action_id']='INK_E_999'
            src.write_text(json.dumps(data))
            with self.assertRaises(AuditRefusal):audit(d)

if __name__=='__main__':unittest.main()
