"""Unit tests use synthetic test fixture, not an original parent model."""
import unittest
import numpy as np
from experiments.driv3r.training.distillation import lab,repair
MOCK={(s,f,t):{"probabilities":{a:1/6 for a in lab.ACTIONS}}
      for s in ("stock","greyhound","branchwitch")
      for f in ("greebles","pipework","architecture") for t in range(16)}
class RepairTests(unittest.TestCase):
 def test_mask_and_protection(self):
  for seed in range(10):
   for case in repair.CASES:
    g=lab.Game(seed)
    for _ in range(15):
     repair.adverse(g,case);o=g.obs();p=repair.corrected(g,MOCK)
     self.assertAlmostEqual(float(p.sum()),1,places=5)
     self.assertTrue(np.isfinite(p).all())
     self.assertTrue(all(p[i]==0 for i,a in enumerate(lab.ACTIONS) if a not in o["legal"]))
     a=lab.ACTIONS[int(p.argmax())];self.assertIn(a,o["legal"]);g.step(a)
     self.assertFalse(g.painted&(g.blocks|g.protected))
     if g.stopped:break
 def test_trap_can_stop(self):
  g=lab.Game(99)
  for dx,dy in lab.DIRS:g.blocks.add((g.pos[0]+dx,g.pos[1]+dy))
  self.assertEqual(g.legal(),["STOP"])
  p=repair.corrected(g,MOCK)
  self.assertEqual(lab.ACTIONS[int(p.argmax())],"STOP")
 def test_no_seed_leakage(self):
  self.assertFalse(set(repair.TRAIN)&set(repair.VALID))
  self.assertFalse(set(repair.TRAIN)&set(repair.TEST))
  self.assertFalse(set(repair.VALID)&set(repair.TEST))
 def test_invalid_case(self):
  with self.assertRaises(ValueError):repair.adverse(lab.Game(1),"impossible")
if __name__=="__main__":unittest.main()
