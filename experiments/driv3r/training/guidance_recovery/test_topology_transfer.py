"""C8 provenance, host-authority and transfer protections; offline-only."""
import copy
import unittest
from collections import Counter
import numpy as np
import torch

from experiment import ARMS, sha
from adversarial_cues import source_hash
from topology_transfer import (ALL_FAMILIES,TRAIN_FAMILIES,OOD_FAMILIES,TRAILS,
  EXITS,SIDE,build,observe,train_data,TransferGate,optimize,noninterference,
  first_choices,stochastic,summarize_stochastic,transform)

class C8TransferTests(unittest.TestCase):
    def test_geometries_are_distinct_connected_and_legal(self):
        self.assertEqual(set(TRAILS),set(ALL_FAMILIES))
        for family in ALL_FAMILIES:
          path=TRAILS[family]
          self.assertGreaterEqual(len(path),7)
          self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:])))
          self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip((path[-1],)+EXITS[family][:-1],EXITS[family])))
          for rot in range(4):
            for outcome in ('reverse','ahead'):
              w=build(1901,1,family,rot,True,outcome)
              self.assertTrue(w.history)
              self.assertTrue(all(w.allowed(p) for p in w.history+[w.pos,w.goal]))
              self.assertTrue(w.allowed(w.history[-1]))
              self.assertTrue(w.allowed(transform(EXITS[family][0],rot,True)))

    def test_cue_strength_pairs_preserve_world_material_and_legal_menu(self):
      for family in ALL_FAMILIES:
       w=build(1813,-1,family,3,False,'reverse')
       snapshot=source_hash(w);protected=sha(sorted(w.protected));ink=sha(sorted(w.inked))
       controls=[]
       for arm in ARMS:
        a=copy.deepcopy(w);b=copy.deepcopy(w);b.issue_recovery_cue(8)
        ca,xa=observe(a,arm);cb,xb=observe(b,arm)
        self.assertEqual([(c.action,c.target,c.legal) for c in ca],[(c.action,c.target,c.legal) for c in cb])
        self.assertTrue(np.array_equal(xa[:,:16],xb[:,:16]))
        self.assertEqual(source_hash(a),source_hash(b))
        self.assertEqual(source_hash(w),snapshot)
        self.assertEqual((sha(sorted(w.protected)),sha(sorted(w.inked))),(protected,ink))
        controls.append((ca,xa))
       self.assertEqual(controls[0][0][8].action[0],'BACKTRACK')
       self.assertTrue(all(c[0][8].legal for c in controls))

    def test_model_data_covers_both_personalities_in_each_training_family(self):
      x,y,sources=train_data(n=16)
      self.assertEqual(x.shape,(16*2*3,10,25))
      self.assertEqual(y.shape,(len(x),))
      seen=Counter((s['family'],s['outcome']) for s in sources)
      for family in TRAIN_FAMILIES:
       self.assertEqual(seen[family,'reverse'],24)
       self.assertEqual(seen[family,'ahead'],24)
      self.assertTrue(set(s['family'] for s in sources).isdisjoint(OOD_FAMILIES))
      self.assertEqual(len(set(s['source_sha256'] for s in sources)),32)
      self.assertTrue(np.all(x[:,8,23]==1))
      self.assertEqual(sum(y),48)

    def test_host_authority_cannot_be_overridden_by_forged_candidate(self):
      w=build(3039,-1,'zigzag',2,False,'ahead')
      h=sha(sorted(w.protected));ink=set(w.inked)
      c=copy.deepcopy(w.candidates(0)[0]);c.legal=True;c.action=('INK',7,0);c.target=(w.pos[0]+7,w.pos[1]);
      self.assertFalse(w.apply(c))
      self.assertEqual(sha(sorted(w.protected)),h)
      self.assertEqual(w.inked,ink)
      self.assertEqual(w.invalid_attempts,1)
      # A legal action changes cursor/ink, not the protected-material set.
      w=build(3039,-1,'zigzag',2,False,'ahead')
      c=next(c for c in w.candidates(0) if c.action[0]=='INK' and c.legal and c.target not in w.history)
      self.assertTrue(w.apply(c));self.assertEqual(sha(sorted(w.protected)),h)
      self.assertTrue(ink<=w.inked)

    def test_learned_gate_strictly_changes_backtrack_only(self):
      torch.manual_seed(9)
      from experiment import Scorer
      # Lightweight dummy prior accepts batched (N,10,25) and single (10,25).
      class Dummy(torch.nn.Module):
       def __init__(self):
        super().__init__();self.m=torch.nn.Linear(25,1)
       def forward(self,x):return self.m(x).squeeze(-1)
      old=Dummy();new=TransferGate(old)
      a=noninterference(old,new,seeds=2)
      self.assertEqual(a['inactive_exact'],2*4*2*3*3)
      self.assertEqual(a['active_non_backtrack_exact'],2*4*2*3*9)
      with torch.no_grad():
       new.delta[-1].bias.fill_(3)
      b=noninterference(old,new,seeds=1)
      self.assertEqual(b['inactive_exact'],4*2*3*3)
      self.assertEqual(b['active_non_backtrack_exact'],4*2*3*9)

    def test_bounded_optimizer_repeat_and_loss(self):
      torch.manual_seed(7)
      class Dummy(torch.nn.Module):
       def __init__(self):
        super().__init__();self.m=torch.nn.Linear(25,1)
       def forward(self,x):return self.m(x).squeeze(-1)
      X,Y,_=train_data(n=12)
      old=Dummy();a=TransferGate(copy.deepcopy(old));b=copy.deepcopy(a)
      a,steps,loss=optimize(a,X,Y,epochs=2,seed=441)
      b,steps2,loss2=optimize(b,X,Y,epochs=2,seed=441)
      self.assertEqual((steps,loss),(steps2,loss2))
      self.assertTrue(all(torch.equal(a.delta.state_dict()[k],b.delta.state_dict()[k]) for k in a.delta.state_dict()))
      self.assertTrue(all(np.isfinite(loss)))

    def test_heldout_source_seeds_disjoint_and_labels_bounded(self):
      _,_,sources=train_data(n=32)
      train_seeds={z['seed'] for z in sources}
      torch.manual_seed(7)
      class Dummy(torch.nn.Module):
       def __init__(self):
        super().__init__();self.m=torch.nn.Linear(25,1)
       def forward(self,x):return self.m(x).squeeze(-1)
      rows=first_choices(Dummy(),n=2)
      self.assertEqual(len(rows),4*2*2*3*2)
      self.assertTrue(all(z['host_rejected']==False for z in rows))
      self.assertFalse(train_seeds & {z['seed'] for z in rows})
      self.assertTrue(all(z['source_sha256'] and z['material_sha256'] and z['menu_sha256'] for z in rows))

    def test_stochastic_worlds_and_pairing(self):
      torch.manual_seed(8)
      class Dummy(torch.nn.Module):
       def __init__(self):
        super().__init__();self.m=torch.nn.Linear(25,1)
       def forward(self,x):return self.m(x).squeeze(-1)
      base=Dummy();models={'C7':base,'C8':TransferGate(copy.deepcopy(base))}
      records=stochastic(models,n=2,turns=5)
      self.assertEqual(len(records),4*2*2*3*2*2*2)
      self.assertTrue(all(r['host_rejected']==0 for r in records))
      grouping={}
      for z in records:
        key=(z['family'],z['seed'],z['personality'],z['arm'],z['cue'],z['outcome'])
        grouping.setdefault(key,[]).append(z)
      self.assertEqual(len(grouping)*2,len(records))
      self.assertTrue(all(len(v)==2 and v[0]['source_sha256']==v[1]['source_sha256']
                         and v[0]['trace']==v[1]['trace'] for v in grouping.values()))
      stats=summarize_stochastic(records)
      self.assertEqual(len(stats),4*2*3*2)

if __name__=='__main__':unittest.main()
