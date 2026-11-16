"""GAL-008 local scope/host-protection and reproducibility tests."""
from __future__ import annotations
import json
import random
import tempfile
import unittest
from pathlib import Path
from lab import HostRefusal, START, World, DECISION_CAP
from candidate_features import FAMILIES
from goal_intent import (ARMS, KINDS, GOAL_SIZE, INK_BUDGET, GoalWorld,
                         candidate, goal_for, oracle, train, eval_episode,
                         MODEL_FEATURES, IntentPolicy, run, summary, sha, audit, AuditRefusal)

class GoalTests(unittest.TestCase):
    def test_goals_only_originally_host_clear(self):
        for family in FAMILIES:
            for kind in KINDS:
                for seed in range(710000,710040):
                    g,a=goal_for(seed,family,kind)
                    w=World(seed,family,'cell')
                    self.assertEqual(len(g),GOAL_SIZE)
                    self.assertTrue(all(w.clear(p) for p in g))
                    self.assertTrue(not g&w.protected)
                    self.assertEqual((g,a),goal_for(seed,family,kind))
    def test_goal_masks_do_not_grant_authority(self):
        for arm in ARMS:
            w=GoalWorld(710015,'scatter',arm,'tree',lease='test-custody')
            menu,token=w.offer()
            self.assertIn('STOP',menu)
            self.assertEqual(set(menu),set(w.legal()))
            self.assertTrue(all(not path or all(p not in w.protected for p in path) for path in menu.values()))
            with self.assertRaises(HostRefusal):w.apply('INK_E_999',token)
            with self.assertRaises(HostRefusal):w.apply('STOP','incorrect-token')
            offered_next,token2=w.offer()
            action=next((x for x in offered_next if x.startswith('INK_')),'STOP')
            w.apply(action,token2)
            with self.assertRaises(HostRefusal):w.apply('STOP',token2)
    def test_features_are_action_specific_and_no_future_label(self):
        w=GoalWorld(710005,'empty','sweep','path',lease='test-features')
        menu=w.legal()
        vecs=[candidate(w,a,p) for a,p in menu.items()]
        self.assertTrue(all(len(v)==len(MODEL_FEATURES) for v in vecs))
        self.assertNotEqual(len(set(vecs)),1)
        self.assertEqual(candidate(w,'STOP',())[len(MODEL_FEATURES)-8],0)
        self.assertEqual(oracle(w,menu) in menu,True)
        self.assertEqual(w.decisions,0)
    def test_training_is_deterministic_and_bounded(self):
        a,ta=train('sweep',4);b,tb=train('sweep',4)
        self.assertEqual(a.weights,b.weights)
        self.assertEqual(ta,tb)
        with self.assertRaises(ValueError):train('sweep',1000)
        with self.assertRaises(ValueError):train('unknown',4)
    def test_host_receipts_replayed_from_selected_actions(self):
        p,_=train('lift',4)
        w,row=eval_episode(p,'lift',710024,'tree','trained')
        reconstructed=GoalWorld(710024,w.family,'lift','tree',lease=w.lease)
        for receipt in w.trace:
            reconstructed.apply(receipt['action_id'],reconstructed.offer()[1],original_response=receipt['original_response'])
        self.assertEqual(w.trace,reconstructed.trace)
        self.assertEqual(w.material,reconstructed.material)
        self.assertEqual(row['goal_sha256'],sha(sorted(reconstructed.goal)))
        self.assertLessEqual(row['ink'],INK_BUDGET)
    def test_untrained_is_negative_control_not_claimed_as_policy(self):
        policy=IntentPolicy()
        world,row=eval_episode(policy,'cell',710001,'path','untrained')
        self.assertLessEqual(len(world.own),INK_BUDGET)
        self.assertEqual(row['native_receipts'],0)
        self.assertEqual(row['evidence'],'synthetic_only')
    def test_summaries_have_denominators(self):
        p,_=train('sweep',4)
        r=[eval_episode(p,'sweep',710000+i,'path','trained')[1] for i in range(4)]
        s=summary(r)
        self.assertEqual(s['episodes'],4)
        self.assertGreaterEqual(s['mean_target_coverage'],0)
        self.assertLessEqual(s['mean_target_coverage'],1)
    def test_audit_replays_and_rejects_tampered_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            run(td,train_worlds=4,eval_worlds=4,gallery_worlds=0)
            evidence=audit(td)
            self.assertEqual(evidence['retrained_checkpoints'],6)
            self.assertEqual(evidence['replayed_episodes'],6*2*3*4)
            receipts=Path(td,'receipts.json')
            receipts.write_text(receipts.read_text()+' ')
            with self.assertRaises(AuditRefusal):audit(td)

    def test_run_manifest_and_source_custody(self):
        with tempfile.TemporaryDirectory() as td:
            out=run(td,train_worlds=4,eval_worlds=4,gallery_worlds=0)
            self.assertEqual(len(out['checkpoint_sha256']),len(ARMS))
            self.assertEqual(len(out['summary']),len(ARMS)*len(KINDS)*3)
            for line in Path(td,'SHA256SUMS').read_text().splitlines():
                digest,path=line.split('  ',1)
                from goal_intent import digest as file_digest
                self.assertEqual(file_digest(Path(td,path)),digest)
            self.assertTrue(out['STOP_unmasked'])

if __name__=='__main__':unittest.main()
