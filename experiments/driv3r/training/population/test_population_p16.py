"""P16 CPU guardrails: no GitHub/Godot/native execution or product edits."""
from __future__ import annotations
import tempfile,unittest
import numpy as np
import torch
from population_lab import FEATURES,NICHES
from population_p10 import ACTIONS
from population_p12 import tensorize
from population_p14 import dataset,mission_pair
from population_p16 import MODES,build_model,run,counterfactual_interventions

class TestP16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows=dataset([717],('heldout_c',),(9,),('heldout_far',),cards=(0,),niches=NICHES[:2])
        cls.X,cls.M,cls.H,cls.N,cls.Y=tensorize(cls.rows)

    def test_all_model_checkpoints_produce_host_legal_actions(self):
        for mode in MODES:
            model=build_model(mode,32)
            logits=model(self.X,self.M,self.H,self.N)
            self.assertEqual(tuple(logits.shape),(len(self.rows),len(ACTIONS)))
            self.assertTrue(torch.isfinite(logits).all().item())
            prediction=logits.argmax(-1)
            self.assertTrue(self.M[torch.arange(len(self.rows)),prediction].all().item())

    def test_all_models_receive_real_training_gradients(self):
        for mode in MODES:
            model=build_model(mode,16)
            loss=torch.nn.functional.cross_entropy(model(self.X,self.M,self.H,self.N),self.Y)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                                for p in model.parameters()))

    def test_balanced_paired_worlds_are_observationally_identical(self):
        for side in ('E','W'):
            a,b=mission_pair(717,1,9,'heldout_c',NICHES[2],'heldout_far',side)
            self.assertEqual(a['desired'][0],'W' if b['desired'][0]=='E' else 'E')
            self.assertTrue(np.array_equal(a['states'][0]['xs'],b['states'][0]['xs']))
            self.assertTrue(np.array_equal(a['states'][0]['mask'],b['states'][0]['mask']))
            self.assertEqual(sorted(a['precursor']),sorted(b['precursor']))
            self.assertFalse(a['final'].painted & (a['final'].protected|a['final'].obstacles))

    def test_soft_cue_has_trainable_time_sensitivity(self):
        torch.manual_seed(2992)
        model=build_model('soft_first_cue',32)
        h0=self.H[0:1];h1=self.H[4:5]
        self.assertEqual(sorted(h0[0].tolist()),sorted(h1[0].tolist()))
        a,b=model(self.X[0:1],self.M[0:1],h0,self.N[0:1]),model(self.X[0:1],self.M[0:1],h1,self.N[0:1])
        self.assertFalse(torch.allclose(a,b))
        self.assertTrue(torch.isfinite(model.temperature).item())

    def test_causal_swap_guardrail_all_models(self):
        for mode in MODES:
            scores=counterfactual_interventions(build_model(mode,16),self.rows)
            self.assertEqual(set(scores),{'original','swap_history','swap_goal','swap_both'})
            self.assertTrue(all(0<=v<=1 for v in scores.values()))

    def test_all_six_niches_preserved(self):
        self.assertEqual(len(NICHES),6)
        self.assertEqual(len(set(NICHES)),6)
        self.assertIn('material_cautious',NICHES)
        self.assertIn('recoverer',NICHES)

    def test_bounded_cpu_no_implicit_run(self):
        with self.assertRaises(ValueError):run('/tmp/never',seeds=[1]*15)
        with self.assertRaises(ValueError):run('/tmp/never',epochs=100)
        with self.assertRaises(ValueError):build_model('bad_method')
        with self.assertRaises(ValueError):build_model('soft_first_cue',width=17)

    def test_paired_dataset_has_disjoint_action_histogram_and_goal_controls(self):
        # Both orders have identical histograms, all legal masks are source-driven.
        a,b=self.rows[0],self.rows[4]
        self.assertTrue(np.array_equal(a['xs'],b['xs']))
        self.assertTrue(np.array_equal(a['mask'],b['mask']))
        self.assertEqual(sorted(a['history']),sorted(b['history']))
        self.assertNotEqual(a['target'],b['target'])

if __name__=='__main__':unittest.main()
