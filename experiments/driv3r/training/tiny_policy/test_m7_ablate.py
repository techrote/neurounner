"""M7 model input ablation, replay and evidence adversarial checks."""
import copy,unittest
import numpy as np
from .core import ACTIONS,FAMILIES
from .m5_topology_train import EVAL_SEEDS,maze_world
from .m7_ablate import change_observation,evaluate,replay,load_heads

class M7Tests(unittest.TestCase):
    def test_only_declared_feature_channels_are_ablated(self):
        a=maze_world(EVAL_SEEDS[0],FAMILIES[0]).observe()
        self.assertEqual(a.shape,(162,))
        for m in ('live','frozen-input','no-history','no-obstacles','swap-family'):
            b=change_observation(a,m)
            self.assertEqual(b.shape,a.shape)
            self.assertFalse(np.shares_memory(a,b))
            self.assertTrue(np.array_equal(a[147:151],b[147:151]))
        self.assertTrue(np.array_equal(a[151:154],change_observation(a,'no-history')[151:154]))
        self.assertEqual(float(change_observation(a,'no-history')[49:98].sum()),0.)
        self.assertEqual(float(change_observation(a,'no-obstacles')[:49].sum()),0.)
        self.assertFalse(np.array_equal(a[151:154],change_observation(a,'swap-family')[151:154]))
        with self.assertRaises(ValueError):change_observation(a,'illicit')
    def test_host_owns_legality_and_stop(self):
        w=maze_world(EVAL_SEEDS[0],FAMILIES[0])
        for dx,dy in ((0,-1),(0,1),(-1,0),(1,0)):
            w.protected[w.y+dy,w.x+dx]=True
        self.assertEqual(w.legal(),(4,))
        self.assertEqual(w.apply(4)['action'],'STOP')
    def test_replay_rejects_tampering(self):
        receipt,_=evaluate(EVAL_SEEDS[0],FAMILIES[0],'novel',None,None)
        self.assertEqual(replay(receipt).state_sha256(),receipt['end_sha256'])
        for field,value in (('before_sha256','1'*64),('observation_sha256','2'*64),('legal',['STOP'])):
            bad=copy.deepcopy(receipt);bad['actions'][0][field]=value
            with self.assertRaises(ValueError):replay(bad)
    def test_split_historical_not_blind(self):
        self.assertEqual(len(EVAL_SEEDS),12)
        self.assertTrue(all(23000<=s<=23011 for s in EVAL_SEEDS))
if __name__=='__main__':unittest.main()
