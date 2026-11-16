"""M16 synthetic-only contract and adversarial replay tests (no native dependencies)."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from .m16_recovery import World, MLP, POLICIES, decide, rollout, replay_receipts, TRAIN, DEV, EVAL

class M16Contract(unittest.TestCase):
    def setUp(self):torch.manual_seed(1601);self.heads={'untrained':MLP().eval(),'bc-mlp':MLP().eval(),'recovery-mlp':MLP().eval()}
    def test_split_rejection(self):
        with self.assertRaises(ValueError):World(EVAL[0],'Greebles','train')
        with self.assertRaises(ValueError):World(TRAIN[0],'Greebles','eval')
    def test_legal_only(self):
        for seed in EVAL[:3]:
            for fam in ('Greebles','Pipework','Architecture'):
                w=World(seed,fam,'eval')
                for i in range(32):
                    w.progress();obs=w.observe();a=decide(w,'untrained',self.heads,[obs]);self.assertIn(a,w.legal())
                    w.apply(a)
                    self.assertFalse(w.block[w.y,w.x]);self.assertFalse(w.protect[w.y,w.x])
                    if a==4:break
    def test_independent_receipts(self):
        record=rollout(EVAL[0],'Greebles','greedy',self.heads)
        replay_receipts(record)
        broken=copy.deepcopy(record);broken['events'][0]['obs_sha256']='0'*64
        with self.assertRaises(AssertionError):replay_receipts(broken)
        broken=copy.deepcopy(record);broken['events'][0]['legal']=['STOP']
        with self.assertRaises(AssertionError):replay_receipts(broken)
        broken=copy.deepcopy(record);broken['events'][0]['state_sha256']='f'*64
        with self.assertRaises(AssertionError):replay_receipts(broken)
    def test_model_is_deterministic_and_masked(self):
        w=World(EVAL[0],'Pipework','eval');w.progress()
        o=w.observe();a=decide(w,'untrained',self.heads,[o]);b=decide(w,'untrained',self.heads,[o]);self.assertEqual(a,b)
        self.assertIn(a,w.legal())

if __name__=='__main__':unittest.main()
