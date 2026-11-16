"""MZ-003 source-only contract tests."""
import unittest
from .run import CODE_FILES,audit_source,safe_scores,PIN
from experiments.driv3r.model_zoo.mamba130.world import World,ACTIONS
from experiments.driv3r.model_zoo.mamba130.counterfactual import samples,paired_switch_metrics,raw_features
class RWKVTests(unittest.TestCase):
    def test_pinned(self):
        self.assertEqual(PIN,'16136bb51c0cf16356f410956d38b48f249c73e1')
    def test_custom_code_screen(self):
        src={f:'import torch\n' for f in CODE_FILES}
        self.assertEqual(len(audit_source(src)),3)
        src['modeling_rwkv7.py']='import subprocess\n'
        with self.assertRaisesRegex(RuntimeError,'prohibited'):audit_source(src)
        src['modeling_rwkv7.py']='exec("bad")\n'
        with self.assertRaisesRegex(RuntimeError,'dynamic'):audit_source(src)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):audit_source({'modeling_rwkv7.py':'pass'})
    def test_host_must_mask_illegal_action(self):
        w=World.make(12)
        wrong=next(a for a in ACTIONS if a not in w.legal())
        scores=[0.]*len(ACTIONS)
        scores[ACTIONS.index(wrong)]=100.
        safe=safe_scores(scores,w.legal())
        self.assertIn(safe,w.legal())
        self.assertNotEqual(safe,wrong)
        with self.assertRaisesRegex(ValueError,'illegal'):w.apply(wrong)
    def test_goal_pairs_and_disjoint_splits(self):
        train=samples(range(400,406),8)
        dev=samples(range(900,904),8)
        test=samples(range(1200,1204),8)
        self.assertEqual(train,samples(range(400,406),8))
        self.assertFalse({r[2] for r in train}&{r[2] for r in dev})
        self.assertFalse({r[2] for r in test}&{r[2] for r in dev})
        self.assertEqual(len(raw_features(train[0][0])),114)
        self.assertGreater(paired_switch_metrics(test,[r[1] for r in test])['changed_target_pairs'],20)
if __name__=='__main__':unittest.main()
