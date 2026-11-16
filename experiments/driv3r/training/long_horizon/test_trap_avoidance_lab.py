"""Guardrail tests for exactly host-legal local risk features and replay."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from . import trap_avoidance_lab as t
from .world import World,Spec,HELDOUT_FAMILIES

class TrapPreventionTests(unittest.TestCase):
    def test_sensor_is_bounded_and_nonmutating(self):
        w=World.create(Spec('train','pickets',1000,'wall','trap_corridor'))
        sha=w.state_sha();l=w.menu()
        for a in l:
            r=t.sensor(w,a)
            self.assertEqual((9,),r.shape)
            self.assertTrue(np.isfinite(r).all())
            self.assertTrue(np.abs(r).max()<=1.)
        self.assertEqual(w.state_sha(),sha)
    def test_offline_teacher_host_legal(self):
        import random
        w=World.create(Spec('train','ridges',1101,'route'))
        for _ in range(36):
            a=t.teacher_action(w,36,random.Random(w.steps))
            self.assertIn(a,w.menu());w.apply(a)
            if w.stopped:break
    def test_mask_identical_base(self):
        w=World.create(Spec('development','zigzag',3301,'motif'))
        l1,x1=t.matrix(w,128,'baseline');l2,x2=t.matrix(w,128,'local_escape_sensor')
        self.assertEqual(l1,l2);self.assertTrue(np.array_equal(x1[:,:24],x2[:,:24]))
        self.assertFalse(np.any(x1[:,24:]))
    def test_checkpoint_restoration_and_sealed(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.npz';m=t.LinearRanker(586,'local_escape_sensor');m.save(p)
            self.assertTrue(np.array_equal(m.w,t.LinearRanker.load(p).w))
        self.assertEqual(t.EVAL_HORIZONS,(36,72,128,160))
        self.assertNotIn('heldout',t.run.__code__.co_consts)
        self.assertTrue(set(HELDOUT_FAMILIES).isdisjoint({'pickets','ridges','islands','zigzag'}))
    def test_negative_candidate_guard(self):
        w=World.create(Spec('train','pickets',1000,'wall'))
        with self.assertRaises(ValueError):t.sensor(w,'MOVE_UNUSED')
        with self.assertRaises(ValueError):t.matrix(w,36,'invented')
    def test_replayed_real_rollout(self):
        w=World.create(Spec('development','islands',3200,'motif','trap_corridor'))
        model=t.LinearRanker(586,'local_escape_sensor')
        result=t.rollout(w.spec,model,36,True,586)
        self.assertEqual(result['native_receipts'],0)
        self.assertEqual(result['metrics']['illegal_host_actions'],0)
        self.assertEqual(result['metrics']['protected_cell_edits'],0)
        self.assertTrue(all(k['event']['chosen'] in k['event']['legal'] for k in result['events']))
    def test_resource_caps(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):t.run(586,Path(tmp),updates=201)
            with self.assertRaises(ValueError):t.run(586,Path(tmp),n_development_per_family=4)

if __name__=='__main__': unittest.main()
