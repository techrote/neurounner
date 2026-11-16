"""D005 source-only host-legality, split and frontier-oracle adversarial tests."""
import unittest
import numpy as np

from experiments.driv3r.training.distillation.lab import ACTIONS, STYLES, Game
from experiments.driv3r.training.distillation.repair import adverse
from experiments.driv3r.training.distillation.trap_rebalance import masked_softmax
from experiments.driv3r.training.distillation.frontier_rescue import (
    TRAIN, VALID, TEST, SCHEMA, frontier_distance, frontier_scores,
    deadend_stagnant, make_targets, score, BUDGET
)

class FrontierRescueTests(unittest.TestCase):
    def test_seed_split_and_budget(self):
        self.assertFalse(set(TRAIN) & set(VALID))
        self.assertFalse(set(TRAIN) & set(TEST))
        self.assertFalse(set(VALID) & set(TEST))
        self.assertEqual(SCHEMA, 'cybersand.driv3r.distill.frontier-rescue/1')
        self.assertEqual(BUDGET, 80)

    def test_protected_cell_is_not_frontier(self):
        game = Game(991, 'branchwitch')
        blocked = (game.pos[0]+1, game.pos[1])
        game.protected.add(blocked)
        self.assertFalse(game.open(blocked))
        self.assertIsNone(frontier_distance(game, blocked))
        self.assertNotIn('STRAIGHT', game.legal())

    def test_oracle_is_host_legal_on_adverse_states(self):
        for style in STYLES:
            for case in ('normal','trap','squeeze'):
                game = Game(717, style)
                for _ in range(24):
                    adverse(game, case)
                    obs = game.obs()
                    logits = frontier_scores(game, deadend_stagnant(game))
                    mask = np.array([[a in obs['legal'] for a in ACTIONS]], bool)
                    dist = masked_softmax(logits[None,:], mask)[0]
                    self.assertTrue(np.isfinite(dist).all())
                    self.assertAlmostEqual(float(dist.sum()), 1., places=5)
                    self.assertTrue((dist[~mask[0]] == 0).all())
                    chosen = ACTIONS[int(dist.argmax())]
                    self.assertIn(chosen, obs['legal'])
                    game.step(chosen)
                    self.assertNotIn(game.pos, game.protected)
                    if game.stopped: break

    def test_illegal_target_mass_and_fraction_guard(self):
        rng = np.random.default_rng(121)
        menu = np.array([[1,0,1,1,1,1],[0,1,1,0,1,1]], bool)
        priors = [masked_softmax(rng.standard_normal((2,6)).astype(np.float32), menu) for _ in range(3)]
        cases=np.array([0,1],dtype=np.uint8)
        hard=np.array([False,True])
        styles=np.array([0,2],dtype=np.uint8)
        with self.assertRaises(ValueError):
            make_targets(*priors,cases,hard,styles,.9)
        for strength in (.15,.30,.45):
            target = make_targets(*priors,cases,hard,styles,strength)
            np.testing.assert_allclose(target.sum(1),1.,atol=1e-6)
            self.assertTrue((target[~menu] == 0).all())

    def test_validation_penalises_premature_stop(self):
        rows={}
        for c in ('normal','trap','squeeze'):
            rows[c+'/rebalance_004']={'cells':20.,'branches':10.,'decisions':80.}
            rows[c+'/frontier_005']={'cells':20.,'branches':10.,'decisions':80.}
        self.assertAlmostEqual(score(rows)['score'],0.)
        for c in ('normal','trap','squeeze'):
            rows[c+'/frontier_005']['decisions']=24.
        self.assertLess(score(rows)['score'],0.)

if __name__ == '__main__':
    unittest.main()
