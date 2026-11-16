"""D006 source-level invariants: excludes all optimizer and GitHub work."""
import unittest
import numpy as np
from .lab import ACTIONS, DIRS, Game, STYLES
from .repair import adverse
from .trap_rebalance import masked_softmax
from .exhaustion_recovery import (SCHEMA,TRAIN,VALID,TEST,reachable_frontier,
    balanced_indices,target_probs,selection_score)

class HonestStopTests(unittest.TestCase):
 def test_strict_split(self):
  self.assertEqual(SCHEMA,'cybersand.driv3r.distill.exhaustion-recovery/1')
  for a,b in ((TRAIN,VALID),(TRAIN,TEST),(VALID,TEST)):
   self.assertFalse(set(a)&set(b))
  self.assertGreater(TEST.start,4016)

 def test_protected_blocks_oracle(self):
  g=Game(771,'stock')
  g.blocks.update({(g.pos[0]-1,g.pos[1]),(g.pos[0],g.pos[1]+1),(g.pos[0],g.pos[1]-1)})
  g.protected.add((g.pos[0]+1,g.pos[1]))
  self.assertEqual(reachable_frontier(g),0)
  self.assertEqual(g.legal(),['STOP'])

 def test_oracle_is_boolean_and_pure(self):
  g=Game(771,'branchwitch');before=(g.pos,len(g.painted),len(g.blocks),len(g.protected))
  self.assertIn(reachable_frontier(g),(0,1))
  self.assertEqual(before,(g.pos,len(g.painted),len(g.blocks),len(g.protected)))

 def test_target_distribution_and_illegal_actions(self):
  menu=np.array([[1,1,0,1,0,1],[0,1,1,0,1,1]],dtype=bool)
  pp=masked_softmax(np.array([[2,-1,-9,0,-8,-5],[-9,1,2,-9,.5,-5]],dtype=np.float32),menu)
  y=target_probs(pp,menu,np.array([True,False]),.85)
  np.testing.assert_allclose(y.sum(1),1.,atol=1e-6)
  self.assertTrue((y[~menu]==0).all())
  self.assertLess(y[0,5],pp[0,5])
  self.assertGreater(y[1,5],pp[1,5])

 def test_unsafe_stop_only_live_rejected(self):
  menu=np.array([[0,0,0,0,0,1]],dtype=bool)
  prior=np.array([[0,0,0,0,0,1]],dtype=np.float32)
  with self.assertRaises(AssertionError):target_probs(prior,menu,np.array([True]),.6)

 def test_balancing_deterministic_and_safe(self):
  live=np.array([False]*150+[True]*600,dtype=bool)
  hard=np.array([False]*450+[True]*300,dtype=bool)
  case=np.zeros(len(live),dtype=np.uint8)
  a=balanced_indices(live,hard,case);b=balanced_indices(live,hard,case)
  np.testing.assert_array_equal(a,b)
  self.assertEqual(len(a),len(live))
  self.assertGreater(np.sum(~live[a]),len(live)*.20)
  self.assertGreater(np.sum(hard[a]),len(live)*.2)

 def test_selection_penalizes_premature_stop(self):
  val={}
  for case in ('normal','trap','squeeze'):
   val[case+'/frontier_005']={'cells':25.,'branches':5.,'premature_stop':0,'wasted_after_exhaustion':25.}
   val[case+'/recovery_006']={'cells':25.,'branches':5.,'premature_stop':0,'wasted_after_exhaustion':25.}
  self.assertEqual(selection_score(val),0.)
  val['normal/recovery_006']['premature_stop']=1
  self.assertLess(selection_score(val),0.)

if __name__=='__main__':unittest.main()
