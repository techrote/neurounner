"""C16 model-only invariants; synthetic host is authoritative."""
import unittest
from pathlib import Path
import numpy as np
import torch
from robust_group_cue import GroupGate,evidence,load_c12,load_replays,guards
from topology_transfer import build
from signed_feedback import HostAdvice,observe
from experiment import ARMS,sha

PATHS=[Path(__file__).resolve().parents[1]/'prerequisites'/('guidance_recovery_'+x+'.safetensors')
       for x in ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer','persistent_feedback','signed_feedback','selective_continue')]
@unittest.skipUnless(all(p.exists() for p in PATHS),'required research checkpoints absent')
class C16Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.prior=load_c12(PATHS);torch.manual_seed(123);cls.model=GroupGate(cls.prior)
  cls.model.delta[-1].bias.data.fill_(.5)
 def test_guards(self):self.assertTrue(guards(self.prior,self.model)['inactive_exact'])
 def test_no_cue_exact(self):
  w=build(11123,1,'hook',2,False,'reverse');cs,x=observe(w,ARMS[0],HostAdvice(w))
  t=torch.from_numpy(x)
  with torch.no_grad():self.assertTrue(torch.equal(self.model(t),self.prior(t)))
 def test_cancel_exact(self):
  w=build(11124,-1,'zigzag',0,True,'ahead');a=HostAdvice(w);a.issue_backtrack(w);a.cancel(w)
  _,x=observe(w,ARMS[-1],a);t=torch.from_numpy(x)
  with torch.no_grad():self.assertTrue(torch.equal(self.model(t),self.prior(t)))
 def test_nonbacktrack_preserved_under_cue(self):
  w=build(11125,1,'hairpin',1,False,'reverse');a=HostAdvice(w);a.issue_backtrack(w)
  _,x=observe(w,ARMS[0],a);t=torch.from_numpy(x)
  with torch.no_grad():
   p=self.prior(t);q=self.model(t)
  self.assertTrue(torch.equal(p[:8],q[:8]));self.assertTrue(torch.equal(p[9:],q[9:]));self.assertNotEqual(float(p[8]),float(q[8]))
 def test_host_legal_actions(self):
  for i in range(10):
   w=build(12123+i,(-1,1)[i%2],('hook','zigzag')[i%2],i%4,bool(i%2),('ahead','reverse')[i%2]);a=HostAdvice(w)
   a.issue_continue(w) if i%2 else a.issue_backtrack(w)
   cs,x=observe(w,ARMS[i%len(ARMS)],a)
   with torch.no_grad():scores=self.model(torch.from_numpy(x)).numpy()
   legal=np.array([c.legal for c in cs]);idx=int(np.argmax(np.where(legal,scores,-1e8)))
   material_hash=sha(sorted(w.protected));before=set(w.inked)
   self.assertTrue(w.apply(cs[idx]));self.assertEqual(w.invalid_attempts,0)
   self.assertEqual(material_hash,sha(sorted(w.protected)));self.assertTrue(before<=w.inked)
 def test_evidence_finite(self):
  w=build(11126,1,'hairpin',1,False,'reverse');a=HostAdvice(w);a.issue_backtrack(w)
  _,x=observe(w,ARMS[0],a)
  z=evidence(torch.from_numpy(x).unsqueeze(0));self.assertEqual(tuple(z.shape),(1,5));self.assertTrue(bool(torch.isfinite(z).all()))
 def test_no_topology_truth_feature(self):
  w=build(11127,1,'hook',2,False,'reverse');a=HostAdvice(w);a.issue_backtrack(w)
  _,x=observe(w,ARMS[0],a)
  self.assertEqual(tuple(x.shape),(10,29));self.assertEqual(tuple(evidence(torch.from_numpy(x).unsqueeze(0)).shape),(1,5))
if __name__=='__main__':unittest.main()
