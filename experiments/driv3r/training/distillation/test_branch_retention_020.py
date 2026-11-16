"""D020 branch-constrained student research-only guards."""
import unittest
import numpy as np
from experiments.driv3r.training.distillation.branch_retention_020 import admissibility,fit
from experiments.driv3r.training.distillation.lab import Student

class D020Test(unittest.TestCase):
 def test_hard_branch_constraint_cannot_be_overridden_by_coverage(self):
  s={}
  for case in ('normal','trap','squeeze'):
   s[f'{case}/d017']={'cells':20.,'branches':10.,'illegal':0}
   s[f'{case}/x']={'cells':100.,'branches':10.,'illegal':0}
  s['normal/x']['branches']=8.
  self.assertFalse(admissibility(s,'x')['admissible'])
  s['normal/x']['branches']=10.
  self.assertTrue(admissibility(s,'x')['admissible'])
  s['squeeze/x']['illegal']=1
  self.assertFalse(admissibility(s,'x')['admissible'])
 def test_rejects_illegal_targets_and_budget_overruns(self):
  x=np.zeros((120,100),dtype=np.float32)
  m=np.ones((120,6),dtype=np.float32)
  y=np.array([3]*60+[1]*60,dtype=np.int64)
  d={'x':x,'t':np.zeros((120,6),np.float32),'m':m,'y':y}
  student=Student(hidden=44)
  with self.assertRaises(ValueError):fit(student,d,601,1,.2)
  d['m'][0,3]=0
  with self.assertRaises(AssertionError):fit(student,d,1,1,.2)
 def test_bounded_optimizer_performs_real_update(self):
  rng=np.random.default_rng(42)
  x=rng.normal(size=(240,100)).astype(np.float32)
  y=np.array([3]*120+[1]*120,dtype=np.int64)
  d={'x':x,'t':rng.normal(size=(240,6)).astype(np.float32),
     'm':np.ones((240,6),dtype=np.float32),'y':y}
  s=Student(hidden=44)
  first=s.w2.copy()
  metrics=fit(s,d,4,17,.2)
  self.assertEqual(metrics['updates'],4)
  self.assertFalse(np.array_equal(first,s.w2))

if __name__=='__main__':unittest.main()
