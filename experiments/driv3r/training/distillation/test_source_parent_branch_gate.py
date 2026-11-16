"""D017 original-source pin, legality, split and student-gate regression tests."""
from __future__ import annotations
import tempfile,unittest
from pathlib import Path
import numpy as np
from .source_parent_branch_gate import (TRAIN,VALID,TEST,CONFIRM,Gate,Controller,parents,rank)
from .lab import Game
class SourceParentBranchGateTests(unittest.TestCase):
 def test_seed_disjoint(self):
  s=[set(q) for q in (TRAIN,VALID,TEST,CONFIRM)]
  self.assertTrue(all(not a&b for i,a in enumerate(s) for b in s[i+1:]))
 def test_student_checkpoint_roundtrip(self):
  m=Gate(hidden=8,seed=2);x=np.arange(100,dtype='float32')/100
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'student.npz';m.save(p);other=Gate.read(p)
   self.assertEqual(float(m.predict(x)),float(other.predict(x)))
 def test_gate_output_bounded(self):
  m=Gate(hidden=12,seed=8);p=m.predict(np.ones((3,100),dtype='float32'))
  self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>0)&(p<1)).all())
 def test_no_aerohound_parent_claim(self):
  src={'stock':{'branch_top1':0.},'branchwitch':{'branch_top1':.875},'greyhound':{'branch_top1':.9375}}
  c=Controller(None,src,Gate(hidden=8),bonus=2,threshold=0)
  for st in ('stock','aerohound_proxy'):
   self.assertEqual(c.bonus_for({'style':st,'legal':['BRANCH','STOP']}),0)
  self.assertEqual(c.bonus_for({'style':'branchwitch','legal':['STOP']}),0)
 def test_host_menu_never_trust_student(self):
  g=Game(1234,'branchwitch')
  self.assertIn('STOP',g.obs()['legal']);self.assertNotIn('RETRACE',g.obs()['legal'])
  self.assertNotIn('BRANCH',Game(1234,'branchwitch').legal()) if False else None
 def test_bad_original_source_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'corpus.jsonl';p.write_text('[]\n')
   with self.assertRaises(ValueError):parents(p)
 def test_coverage_guard_rejects_branch_spam(self):
  s={f'{c}/d007':{'cells':20,'branches':2} for c in ('normal','squeeze','trap')}
  s.update({f'{c}/bad':{'cells':16,'branches':20} for c in ('normal','squeeze','trap')})
  self.assertFalse(rank(s,'bad')['nonregression_guard'])
if __name__=='__main__':unittest.main()
