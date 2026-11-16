"""Bounded adverse cases for policy-state corrective imitation and original replay."""
import copy
from pathlib import Path
import tempfile
import unittest

from .on_policy_adaptation import (ARMS, development_specs, evaluate,
                                   on_policy_corpus, train_matched, verify_episode)
from .policy import Ranker
from .world import Spec


class OnPolicyAdaptationTests(unittest.TestCase):
    def test_dev_only_worlds(self):
        got=development_specs(1)
        self.assertTrue(got)
        self.assertTrue(all(s.split=='development' and 3000<=s.seed<4000 for s in got))
        with self.assertRaises(ValueError):development_specs(3)

    def test_train_only_corpus_legal(self):
        x,mask,y,prov=on_policy_corpus(Ranker(573),573,1,36)
        self.assertEqual(x.shape[0],len(y))
        self.assertTrue(all(mask[i,y[i]] for i in range(len(y))))
        self.assertGreater(len(prov['worlds']),0)
        self.assertTrue(all(row['decisions']<=36 for row in prov['worlds']))
        with self.assertRaises(ValueError):on_policy_corpus(Ranker(1),1,1,256)

    def test_equal_initialization_real_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            models,meta=train_matched(573,1,2,Path(td))
            self.assertEqual(set(models),set(ARMS))
            self.assertEqual({models[k].updates for k in ARMS},{8})
            self.assertEqual({models[k].parameter_count for k in ARMS},{625})
            self.assertTrue(meta['no_heldout_or_development_in_optimizer'])

    def test_original_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            m=Ranker(573)
            path,o=evaluate(development_specs(1)[0],m,ARMS[0],36,True,Path(td))
            self.assertIsNotNone(verify_episode(o,m))
            wrong=copy.deepcopy(o)
            wrong['probabilities'][0]['probabilities'][next(iter(wrong['probabilities'][0]['probabilities']))]=0
            with self.assertRaises(ValueError):verify_episode(wrong,m)
            wrong=copy.deepcopy(o);wrong['native_receipts']=1
            with self.assertRaises(ValueError):verify_episode(wrong,m)
            wrong=copy.deepcopy(o);wrong['events'][0]['chosen']='MOVE_BAD'
            with self.assertRaises(ValueError):verify_episode(wrong,m)
            wrong=copy.deepcopy(o);wrong['split']='heldout'
            with self.assertRaises(ValueError):verify_episode(wrong,m)

    def test_forced_no_excess_128_native_claim(self):
        with tempfile.TemporaryDirectory() as td:
            _,o=evaluate(development_specs(1)[0],Ranker(573),ARMS[1],128,True,Path(td))
            self.assertFalse(o['native_executed'])
            self.assertEqual(o['native_receipts'],0)
            self.assertEqual(o['forced_schedule'],[12,35,71])


if __name__=='__main__':unittest.main()
