"""P13 legality, counterfactual information, representation and training guards."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from population_lab import NICHES, World
from population_p10 import ACTIONS
from population_p13 import (MODES,PHASES,build_model,episode_rows,make_data,
    split_seeds,fit,transform,evaluate_static,rollout)
from population_p12 import tensorize


class TestP13(unittest.TestCase):
    def test_matched_paired_legal_four_step_mission(self):
        for grammar in ('train_a','train_b','heldout_c'):
            for depth in (5,7,9):
                for niche in NICHES:
                    a,b=episode_rows(5231,0,depth,grammar,niche)
                    self.assertEqual(len(a),len(PHASES))
                    self.assertEqual(len(b),len(PHASES))
                    self.assertTrue(np.array_equal(a[0]['xs'],b[0]['xs']))
                    self.assertTrue(np.array_equal(a[0]['mask'],b[0]['mask']))
                    self.assertEqual(sorted(a[0]['history']),sorted(b[0]['history']))
                    self.assertTrue(np.array_equal(a[0]['history'][-8:],b[0]['history'][-8:]))
                    self.assertNotEqual(a[0]['target'],b[0]['target'])
                    for trajectory in (a,b):
                        self.assertEqual([r['phase'] for r in trajectory],list(PHASES))
                        self.assertEqual([ACTIONS[r['target']] for r in trajectory[1::2]],['RETRACE','STOP'])
                        for r in trajectory:
                            self.assertTrue(r['mask'][r['target']])
                            self.assertEqual(r['history'].shape,(32,))
    def test_protected_cells_untouched_outside_probe(self):
        from population_p12 import world
        for depth in (5,7,9):
            w=world(5231,0,depth)
            base=World.generate(5231,max_steps=28)
            self.assertEqual(w.protected,base.protected)
            self.assertTrue(w.obstacles.issubset(base.obstacles))
    def test_split_no_leakage(self):
        s=split_seeds(651,12,8,8)
        self.assertFalse(set(s['train'])&set(s['validation']))
        self.assertFalse(set(s['train'])&set(s['holdout']))
        self.assertFalse(set(s['validation'])&set(s['holdout']))
    def test_information_blind_controls_and_balance(self):
        rows=make_data([121],('heldout_c',),(9,),cards=(0,),niches=NICHES[:2])
        X,M,H,N,Y=tensorize(rows)
        self.assertTrue(bool((H[0,:]!=H[4,:]).any()))
        self.assertTrue(bool((H[0,-8:]==H[4,-8:]).all()))
        self.assertTrue(bool((M[0]==M[4]).all()))
        self.assertTrue(bool((X[0]==X[4]).all()))
        self.assertNotEqual(int(Y[0]),int(Y[4]))
        self.assertTrue(bool((transform(H,'state_only')[0]==transform(H,'state_only')[4]).all()))
    def test_pairwise_antisymmetry_responds_to_order(self):
        import torch
        model=build_model('pairwise_order',24)
        rows=make_data([121],('heldout_c',),(9,),cards=(0,),niches=NICHES[:1])
        X,M,H,N,Y=tensorize(rows)
        assert H.shape[1]==32
        oh=torch.nn.functional.one_hot(H[:2],num_classes=len(ACTIONS)+1).float()[...,1:]
        pairs=torch.einsum('bti,btj->bij',oh.cumsum(dim=1)-oh,oh)
        diff=pairs-pairs.transpose(1,2)
        east,west=ACTIONS.index('E'),ACTIONS.index('W')
        # First of each 8-row block and its paired equivalent have opposite E/W ordering.
        oh=torch.nn.functional.one_hot(H[[0,4]],num_classes=len(ACTIONS)+1).float()[...,1:]
        pairs=torch.einsum('bti,btj->bij',oh.cumsum(dim=1)-oh,oh)
        diff=pairs-pairs.transpose(1,2)
        self.assertNotEqual(float(diff[0,east,west]),float(diff[1,east,west]))
        self.assertTrue(bool(torch.isfinite(model(X,M,H,N)).all()))
    def test_real_optimizer_and_host_legality(self):
        import torch
        torch.set_num_threads(1)
        rows=make_data([121,122],('train_a','train_b'),(5,),cards=(0,),niches=NICHES[:2])
        data=tensorize(rows)
        for mode in MODES:
            torch.manual_seed(7)
            model=build_model(mode,20)
            old={k:v.clone() for k,v in model.state_dict().items()}
            report=fit(model,mode,data,data,123,epochs=3,batch_size=64)
            self.assertGreater(report['updates'],0)
            self.assertTrue(any(not torch.equal(old[k],v) for k,v in model.state_dict().items()))
            self.assertEqual(evaluate_static(model,mode,rows)['legal_action_rate'],1.)
            if mode=='pairwise_order':
                s=rollout(model,mode,[123],('heldout_c',),(9,),gallery_pairs=0)
                self.assertEqual(s['legal_action_rate'],1.)
    def test_invalid_limits_and_nonexistent_mode(self):
        from population_p13 import run
        with self.assertRaises(ValueError):run('/tmp/p13-invalid',train_worlds=2)
        with self.assertRaises(ValueError):build_model('invalid')


if __name__=='__main__':unittest.main()