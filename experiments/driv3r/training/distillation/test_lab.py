import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lab import Game,ACTIONS,STYLES,load_original,feats,teacher,Student,curriculum
import numpy as np
class DistillTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.original=load_original()
 def test_original_real_model_sources(self):
  self.assertEqual(len(self.original),144)
  self.assertEqual({r['teacher'] for r in self.original.values()},{'stock','greyhound','branchwitch'})
  self.assertEqual({r['source_seed'] for r in self.original.values()},{570})
  self.assertTrue(all(r['source_kind']=='original_model_inference' for r in self.original.values()))
 def test_host_never_paints_protected(self):
  for seed in range(20):
   g=Game(seed)
   for _ in range(80):
    o=g.obs();a=next((x for x in ('BRANCH','LEFT','STRAIGHT','RIGHT','RETRACE','STOP') if x in o['legal']),'STOP')
    g.step(a)
    self.assertFalse(g.painted & g.protected);self.assertFalse(g.painted & g.blocks)
    if g.stopped:break
 def test_adverse_recovery_and_invalid(self):
  g=Game(19);g.history=[g.pos,(g.pos[0]+1,g.pos[1])];g.pos=g.history[-1]
  g.blocks.update((g.pos[0]+dx,g.pos[1]+dy) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)))
  o=g.obs();self.assertIn('RETRACE',o['legal']);self.assertIn('STOP',o['legal'])
  before=g.pos;g.step('BRANCH');self.assertEqual(g.pos,before);self.assertEqual(g.invalid,1)
  g.step('RETRACE');self.assertEqual(g.pos,g.history[-1])
 def test_split_reproducibility(self):
  a=curriculum(self.original,1001,1003);b=curriculum(self.original,1001,1003)
  self.assertEqual(len(a),len(b));self.assertEqual(len(a[0][0]),len(b[0][0]))
  self.assertTrue(np.array_equal(a[0][1],b[0][1]))
  self.assertTrue(all(x[2]['seed'] in (1001,1002) for x in a))
 def test_student_respects_menu(self):
  model=Student(len(feats(Game(21).obs())))
  for seed in range(20):
   g=Game(seed);o=g.obs()
   self.assertIn(model.predict(o),o['legal'])
   p=teacher(o,self.original);self.assertAlmostEqual(float(sum(p)),1.,places=5)
   for i,a in enumerate(ACTIONS):
    if a not in o['legal']:self.assertEqual(p[i],0)
if __name__=='__main__':unittest.main()
