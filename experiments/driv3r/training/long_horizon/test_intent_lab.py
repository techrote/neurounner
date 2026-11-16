"""Cheap mode-overlay integrity, split, replay and control tests."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from .intent_lab import ARMS, MODES, choose, main, pair_stats, run_episode, paired_counterfactual_corpus, cross_features
from .policy import Ranker
from .world import World, specs, replay

class IntentTests(unittest.TestCase):
    def test_no_world_source_mutation_from_model_intent(self):
        w=World.create(specs('development',1)[0]); pre=w.state_sha(); base=w.base_sha()
        m=Ranker(573)
        for goal in MODES:
            choice,dist,_=choose(w,m,36,goal,False)
            self.assertIn(choice,w.menu())
            self.assertEqual(set(dist),set(w.menu()))
            self.assertAlmostEqual(sum(dist.values()),1.,places=6)
            self.assertEqual(w.state_sha(),pre)
            self.assertEqual(w.base_sha(),base)
    def test_blind_logits_identical_for_all_modes(self):
        w=World.create(specs('development',1)[0]);m=Ranker(573)
        scored=[choose(w,m,36,goal,True)[:2] for goal in MODES]
        self.assertEqual(scored[0],scored[1]);self.assertEqual(scored[1],scored[2])
    def test_holdout_rejected(self):
        w=specs('heldout',1)[0]
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(ValueError):
                run_episode(w,36,ARMS[0],Ranker(573),573,'wall',None,None,Path(p))
    def test_switch_validation(self):
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(ValueError):
                run_episode(specs('development',1)[0],36,ARMS[0],Ranker(573),573,
                            'wall','route',36,Path(p))
    def test_pair_statistics_detect_identical_geometry(self):
        fake=[{'initial_model_intent':m,'ink_cells':[[1,1]],'route_east_displacement':2,
               'motif_turn_fraction':0.1,'wall_neighbour_fraction':0.5} for m in MODES]
        s=pair_stats(fake)
        self.assertEqual(s['mean_cross_intent_ink_jaccard'],1.)
        self.assertEqual(s['different_final_geometries'],1)
    def test_cross_features_differ_for_actions_not_permissions(self):
        from .policy import menu_matrix
        w=World.create(specs('development',1)[0])
        w.spec = __import__('dataclasses').replace(w.spec,mode='route')
        menu,X=menu_matrix(w,36)
        basis=cross_features(X)
        self.assertEqual(basis.shape,X.shape)
        self.assertEqual(set(menu),set(w.menu()))
        self.assertTrue(np.isfinite(basis).all())
        self.assertEqual(float(basis[:,21].max()),1.)

    def test_paired_labels_include_three_intents_per_state(self):
        X,M,y,meta=paired_counterfactual_corpus(36,573,1)
        self.assertEqual(len(y),3*meta['state_count'])
        self.assertEqual(len(X),len(M))
        self.assertTrue(0 <= meta['teacher_discordant_state_fraction'] <= 1)
        self.assertTrue(set(np.unique(X[:,0,20:23].argmax(axis=1)))=={0,1,2})

    def test_short_real_optimizer_and_source_replay(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'run';report=main(['--output',str(p),'--seed-count','1',
                 '--updates-per-stage','2','--train-worlds-per-family','1',
                 '--dev-worlds-per-family','1'])
            self.assertEqual(len(report['checkpoint_records']),4)
            self.assertEqual(report['episodes'],4*3*4*3+4*4*2)
            self.assertFalse(report['heldout_read'])
            for arm in ARMS:
                self.assertTrue((p/f'{arm}-seed573.npz').exists())
            self.assertEqual(len(report['paired_metrics']),12)

if __name__=='__main__':unittest.main()
