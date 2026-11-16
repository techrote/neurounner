"""Guardrails for RETURN-GEN-001, including split separation and tamper checks."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from . import return_generalization_lab as g
from .world import specs, World

class ReturnGeneralizationTests(unittest.TestCase):
    def test_distinct_train_dev_depths_directions(self):
        self.assertTrue(set(g.TRAIN_DEPTHS).isdisjoint(g.DEV_DEPTHS))
        self.assertTrue(set(g.TRAIN_DIRECTIONS).isdisjoint(g.DEV_DIRECTIONS))
        self.assertTrue(set(s.seed for s in specs('train',1)).isdisjoint(s.seed for s in specs('development',1)))

    def test_every_host_fork_same_across_caps_and_legal(self):
        for split,lengths,dirs,caps in (('train',g.TRAIN_DEPTHS,g.TRAIN_DIRECTIONS,g.TRAIN_CAPS),
                                        ('development',g.DEV_DEPTHS,g.DEV_DIRECTIONS,g.DEV_CAPS)):
            sp=specs(split,1)[0]
            for direction in dirs:
                for length in lengths:
                    forks=[]
                    for cap in caps:
                        w=g.fixture(sp,length,direction)
                        g.force_error(w,length,direction,cap)
                        self.assertEqual(set(w.menu(cap)),{'BACKTRACK','RELOCATE_START','STOP'})
                        self.assertEqual(len(w.ink),length)
                        self.assertFalse(w.ink & w.protected)
                        self.assertFalse(w.ink & set(w.materials))
                        forks.append((w.state_sha(),tuple(w.menu(cap))))
                    self.assertTrue(all(x==forks[0] for x in forks))

    def test_budget_blind_is_observation_aliased(self):
        sp=specs('development',1)[0]
        w=g.fixture(sp,8,'W');g.force_error(w,8,'W',2)
        blind2=g.matrix(w,128,2,'blind')
        blind12=g.matrix(w,128,12,'blind')
        self.assertEqual(blind2[0],blind12[0])
        self.assertTrue(np.array_equal(blind2[1],blind12[1]))
        self.assertFalse(np.array_equal(g.matrix(w,128,2,'raw_counters')[1],
                                        g.matrix(w,128,12,'raw_counters')[1]))

    def test_engineered_is_explicit_oracle_control(self):
        sp=specs('development',1)[0]
        w=g.fixture(sp,8,'W');g.force_error(w,8,'W',5)
        actions,mat=g.matrix(w,128,5,'engineered')
        self.assertEqual(mat[actions.index('RELOCATE_START'),-1],1.)
        self.assertEqual(mat[actions.index('BACKTRACK'),-2],0.)
        self.assertEqual(g.teacher(w,128,5,__import__('random').Random(0)),'RELOCATE_START')
        self.assertEqual(g.teacher(w,128,9,__import__('random').Random(0)),'BACKTRACK')

    def test_sealed_world_rejected(self):
        sp=specs('train',1)[0]
        with self.assertRaises(ValueError):g.fixture(replace(sp,split='heldout'),5,'E')
        with self.assertRaises(ValueError):g.fixture(sp,8,'W')
        model=g.Ranker(596,'blind')
        with self.assertRaises(ValueError):g.rollout(sp,model,6,'W',2,128)
        with self.assertRaises(ValueError):g.rollout(specs('development',1)[0],model,5,'W',2,128)

    def test_corrupted_original_distribution_rejected(self):
        sp=specs('development',1)[0];model=g.Ranker(596,'engineered')
        ep=g.rollout(sp,model,6,'W',2,128)
        g.verify_episode(ep,model,sp)
        ep['rows'][6]['probabilities']['STOP']+=.01
        with self.assertRaises(ValueError):g.verify_episode(ep,model,sp)

    def test_corrupted_host_state_rejected(self):
        sp=specs('development',1)[0];model=g.Ranker(596,'raw_counters')
        ep=g.rollout(sp,model,8,'W',9,128)
        ep['rows'][8]['event']['legal']=['STOP']
        with self.assertRaises(ValueError):g.verify_episode(ep,model,sp)

    def test_training_and_restoration_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fresh';r=g.run(p,596,1,quick=True)
            self.assertEqual(r['episodes'],8)
            self.assertEqual(set(r['checkpoints']),set(g.ARMS))
            self.assertTrue(all(v['updates']==1 for v in r['checkpoints'].values()))
            verified=g.verify(p)
            self.assertEqual(verified['episodes_replayed'],8)
            self.assertEqual(verified['checkpoints_restored'],4)
            f=next((p/'trajectories').glob('*.json'))
            d=json.loads(f.read_text());d['final_sha']='forged'
            f.write_text(json.dumps(d))
            with self.assertRaises(ValueError):g.verify(p)

if __name__=='__main__':unittest.main()
