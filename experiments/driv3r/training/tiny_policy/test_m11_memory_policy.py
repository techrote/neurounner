"""Adversarial interface and no-future-leakage checks for M11 research."""
import unittest
import numpy as np
import torch
from .core import DELTA, FAMILIES
from .m11_memory_policy import (world,nearest_frontier,novelty,MLP,GRU,TCN,
                                replay,rollout,TRAIN,DEV,EVAL)

class TestM11(unittest.TestCase):
    def test_disjoint_seed_allocations_and_heldout_topologies(self):
        self.assertFalse(set(TRAIN)&set(DEV) or set(TRAIN)&set(EVAL) or set(DEV)&set(EVAL))
        with self.assertRaises(ValueError):world(51000,FAMILIES[0],'eval')
        a=world(53000,FAMILIES[0],'eval');b=world(53000,FAMILIES[1],'eval')
        np.testing.assert_array_equal(a.blocked,b.blocked)
        np.testing.assert_array_equal(a.protected,b.protected)
    def test_protected_stop_and_mask(self):
        w=world(53000,FAMILIES[0],'eval')
        for dx,dy in DELTA[:4]:w.protected[w.y+dy,w.x+dx]=True
        self.assertEqual(w.legal(),(4,));self.assertEqual(nearest_frontier(w),4);self.assertEqual(novelty(w),4)
        self.assertRaises(ValueError,w.apply,0)
    def test_causal_tcn_and_gru(self):
        torch.manual_seed(7)
        a=torch.randn(1,16,162);b=a.clone();b[:,9:,:]+=17
        for net in (GRU(),TCN()):
            net.eval()
            with torch.inference_mode():
                one=net(a)[:,:8,:];two=net(b)[:,:8,:]
            self.assertTrue(torch.allclose(one,two,rtol=1e-6,atol=1e-6))
    def test_distinct_architectures_and_shapes(self):
        x=torch.randn(2,16,162)
        for ctor in (MLP,GRU,TCN):
            m=ctor();self.assertEqual(tuple(m(x).shape),(2,16,5))
    def test_replay_rejects_action_tampering(self):
        r,_=rollout(53000,FAMILIES[0],'novel',{})
        self.assertGreater(len(r['events']),10);replay(r)
        bad=dict(r);bad['events']=[dict(e) for e in r['events']]
        bad['events'][1]['action']='STOP'
        with self.assertRaises((AssertionError,ValueError)):replay(bad)

if __name__=='__main__':unittest.main()
