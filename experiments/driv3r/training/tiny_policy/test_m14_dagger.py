"""M14 host action and split security regressions."""
import unittest
import numpy as np
import torch
from . import m14_dagger as m

class M14Tests(unittest.TestCase):
    def test_seed_isolation(self):
        with self.assertRaises(ValueError):m.world(73000,'Greebles','train')
        with self.assertRaises(ValueError):m.world(71000,'Pipework','eval')
    def test_mask_and_protection(self):
        w=m.world(73000,'Greebles','eval')
        for a in w.legal():
            if a==4:continue
            x=w.x+m.DELTA[a][0];y=w.y+m.DELTA[a][1]
            self.assertFalse(w.blocked[y,x]);self.assertFalse(w.protected[y,x])
        with self.assertRaises(ValueError):w.apply(900)
        for a in range(4):
            dx,dy=m.DELTA[a];w.protected[w.y+dy,w.x+dx]=True
        self.assertEqual(w.legal(),(4,));self.assertEqual(m.oracle(w,m.reachable(w),0),4)
    def test_eval_families_observed(self):
        for fam in m.FAMILIES:
            w=m.world(73003,fam,'eval');goals=m.reachable(w)
            self.assertEqual(len(goals),8)
            self.assertEqual(m.obs(w,goals,0).shape,(m.OBS,))
    def test_context_no_future(self):
        torch.manual_seed(0)
        for ctor in m.CTORS.values():
            net=ctor().eval();x=torch.randn(1,8,m.OBS)
            y=x.clone();y[:,5:,:]+=1000
            with torch.inference_mode():
                a=net(x);b=net(y)
            self.assertTrue(torch.allclose(a[:,:5,:],b[:,:5,:],atol=1e-5))

if __name__=='__main__':unittest.main()
