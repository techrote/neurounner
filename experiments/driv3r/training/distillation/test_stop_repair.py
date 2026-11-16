"""D007 source-only tests: no optimizer or native CyberSand execution."""
import unittest
import numpy as np
from .lab import Game,feats
from .stop_repair import SCHEMA,TRAIN,VALID,TEST,EXHAUST_STOP,target_probs,resample_index,selection_score

class StopRepairTests(unittest.TestCase):
 def test_disjoint_splits(self):
  self.assertEqual(SCHEMA,'cybersand.driv3r.distill.stop-repair/1')
  for a,b in ((TRAIN,VALID),(TRAIN,TEST),(VALID,TEST)):
   self.assertFalse(set(a)&set(b))
  self.assertGreater(TEST.start,5016);self.assertEqual(len(EXHAUST_STOP),3)

 def test_target_legality(self):
  menu=np.array([[1,1,0,1,0,1],[0,0,0,0,1,1],[0,0,0,0,0,1]],bool)
  p005=np.array([[.5,.2,0,.2,0,.1],[0,0,0,0,.9,.1],[0,0,0,0,0,1]],np.float32)
  p006=np.array([[.15,.1,0,.2,0,.55],[0,0,0,0,.3,.7],[0,0,0,0,0,1]],np.float32)
  p=target_probs(p005,p006,menu,np.array([True,False,False]),.82)
  np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
  self.assertTrue(np.all(p[~menu]==0));self.assertEqual(float(p[0,5]),0)
  self.assertGreater(float(p[1,5]),float(p006[1,5]));self.assertEqual(float(p[2,5]),1)

 def test_live_stoponly_rejected(self):
  p=np.array([[0,0,0,0,0,1]],np.float32)
  with self.assertRaises(AssertionError):
   target_probs(p,p,p.astype(bool),np.array([True]),.65)

 def test_resampling_deterministic(self):
  live=np.array([False]*60+[True]*240)
  hard=np.array([False]*60+[True]*70+[False]*170)
  bad=np.array([False]*130+[True]*50+[False]*120)
  d={'frontier_live':live,'hard_live':hard,'false_stop':bad}
  ix=resample_index(d)
  np.testing.assert_array_equal(ix,resample_index(d))
  self.assertEqual(len(ix),len(live));self.assertGreater(np.sum(bad[ix]),50)

 def test_malformed_targets_rejected(self):
  p=np.array([[1,0,0,0,0,0]],np.float32)
  for fraction in (-1,0,1.01):
   with self.assertRaises(ValueError):
    target_probs(p,p,np.ones_like(p,dtype=bool),np.array([True]),fraction)
  with self.assertRaises(ValueError):
   target_probs(p,p,np.array([[1,1,1,1,1,0]],bool),np.array([True]),.8)

 def test_observation_is_exactly_100_features(self):
  self.assertEqual(feats(Game(7097,'branchwitch').obs()).shape,(100,))

 def test_premature_stop_validation_penalty(self):
  m={case+'/'+name:{'cells':30.,'branches':10.,'premature_stop':0.,'wasted_after_exhaustion':0.}
     for case in ('normal','trap','squeeze') for name in ('repair_007','recovery_006')}
  self.assertEqual(selection_score(m),0)
  m['squeeze/repair_007']['premature_stop']=.25
  self.assertLess(selection_score(m),0)

if __name__=='__main__':unittest.main()
