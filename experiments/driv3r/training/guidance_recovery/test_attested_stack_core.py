import unittest
import numpy as np
import torch
from attested_stack_core import AttestedTrailResidual, host_trail_facts, attach_host_facts
from topology_transfer import build
from signed_feedback import HostAdvice, observe


class Contract(unittest.TestCase):
    def test_facts_derive_from_executed_history(self):
        self.assertTrue(np.array_equal(
            host_trail_facts([(1,2),(2,2)], (1,2), True, (2,2)), [1.,1.,0.]))
        self.assertTrue(np.array_equal(
            host_trail_facts([(1,2),(2,2)], (9,9), True, (2,2)), [0.,0.,0.]))
        self.assertTrue(np.array_equal(
            host_trail_facts([(1,2),(2,2)], (2,2), True, (2,2)), [1.,.5,1.]))

    def test_extension_does_not_rewrite_29_base_features(self):
        x=np.zeros((10,29),np.float32)
        y=attach_host_facts(x,[1.,.5,0.])
        self.assertEqual(y.shape,(10,32))
        self.assertTrue(np.array_equal(x,y[:,:29]))
        with self.assertRaises(ValueError):
            attach_host_facts(np.zeros((9,29),np.float32),[1.,.5,0.])

    def test_host_cue_invariants_across_geometry_and_goal(self):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w=torch.nn.Parameter(torch.tensor(.25))
            def forward(self,x):
                return self.w*x[...,7]

        for filt,scale in (('any',2.),('backtrack',1.)):
            core=AttestedTrailResidual(Dummy(),scale=scale,advice_filter=filt)
            for family in ('hairpin','staircase','hook','zigzag'):
                for outcome in ('reverse','ahead'):
                    for mode in ('none','backtrack','continue'):
                        w=build(880001,1,family,0,False,outcome)
                        advice=HostAdvice(w)
                        if mode!='none':
                            getattr(advice,'issue_'+mode)(w)
                        cs,x=observe(w,.75,advice)
                        xx=attach_host_facts(x,host_trail_facts(
                            w.history,w.goal,cs[8].legal,cs[8].target))
                        old=core.prior(torch.from_numpy(x))
                        new=core(torch.from_numpy(xx))
                        self.assertTrue(torch.equal(old[:8],new[:8]))
                        self.assertTrue(torch.equal(old[9:],new[9:]))
                        active=(outcome=='reverse' and mode!='none' and
                                (filt=='any' or mode=='backtrack'))
                        if active:
                            self.assertGreater(float(new[8]),float(old[8]))
                        else:
                            self.assertTrue(torch.equal(old,new))


if __name__=='__main__':
    unittest.main()
