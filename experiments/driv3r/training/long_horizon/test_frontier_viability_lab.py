"""Host-legality, provenance and real optimizer guardrails for FRONTIER-VIABILITY."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.driv3r.training.long_horizon import frontier_viability_lab as f
from experiments.driv3r.training.long_horizon.world import MOVE, World, specs

class FrontierViabilityTest(unittest.TestCase):
    def setUp(self):
        self.world=World.create(specs('train',1)[0])

    def test_menu_cannot_expand(self):
        for arm in f.ARMS:
            menu, X=f.features(self.world,128,arm)
            self.assertEqual(menu,self.world.menu())
            self.assertEqual(X.shape,(7,f.DIM))

    def test_feature_masking_and_finite(self):
        blind=f.features(self.world,128,'blind')[1]
        local=f.features(self.world,128,'local')[1]
        full=f.features(self.world,128,'frontier')[1]
        self.assertTrue(np.array_equal(blind[:,f.BASE:],np.zeros_like(blind[:,f.BASE:])))
        self.assertTrue(np.array_equal(local[:,f.BASE+f.LOCAL:],np.zeros_like(local[:,f.BASE+f.LOCAL:])))
        self.assertTrue(np.array_equal(full[:,:f.BASE],blind[:,:f.BASE]))
        self.assertTrue(np.all(np.isfinite(full)))
        self.assertGreater(float(np.sum(full[:,f.BASE+f.LOCAL:])),0)

    def test_legal_only_frontier(self):
        self.assertRaises(ValueError,f.frontier_features,self.world,'MOVE_EE')
        self.assertRaises(ValueError,f.frontier_features,self.world,'BOGUS')

    def test_corridor_capacity_lower_than_open(self):
        base=replace(specs('train',1)[0],challenge='trap_corridor')
        world=World.create(base)
        world.apply('MOVE_E')  # initial center clearing leaves x49 connected; x50 is a real pocket
        east=f.frontier_features(world,'MOVE_E')[0]
        west=f.frontier_features(world,'MOVE_W')[0]
        self.assertLess(east,west)

    def test_teacher_legal(self):
        import random
        self.assertIn(f.teacher(self.world,36,random.Random(12)),self.world.menu())

    def test_identical_training_rows(self):
        xs,mask,labels,meta=f.training_corpus(588,36,1)
        self.assertGreater(len(labels),100)
        self.assertTrue(np.all(mask[np.arange(len(labels)),labels]))
        self.assertEqual(xs['blind'].shape,xs['local'].shape)
        self.assertTrue(np.array_equal(xs['blind'][:,:,:f.BASE],xs['frontier'][:,:,:f.BASE]))
        self.assertEqual(f.training_corpus(588,36,1)[3]['source_digest'],meta['source_digest'])

    def test_real_optimizer_and_checkpoint(self):
        xs,mask,labels,_=f.training_corpus(588,36,1)
        model=f.Ranker(588,'frontier');initial=model.w.copy()
        stats=model.optimize(xs['frontier'],mask,labels,3,901)
        self.assertEqual(stats['steps'],3)
        self.assertFalse(np.array_equal(initial,model.w))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ckpt.npz';model.save(path)
            restored=f.Ranker.load(path)
            self.assertTrue(np.array_equal(restored.w,model.w))
            self.assertEqual(restored.updates,3)

    def test_source_replay_detects_probability_tampering(self):
        sp=specs('development',1)[0]
        model=f.Ranker(588,'blind')
        ep=f.rollout(sp,model,128,False)
        self.assertGreater(f.verify_episode(ep,model,sp),0)
        falsified=copy.deepcopy(ep)
        first=falsified['events'][0]['probabilities']
        k=next(iter(first));first[k]=.0001
        with self.assertRaises(ValueError):f.verify_episode(falsified,model,sp)

    def test_structural_holdout_is_disjoint(self):
        self.assertFalse({x.family for x in specs('development',2)} &
                         {x.family for x in specs('heldout',2)})

if __name__=='__main__':unittest.main()
