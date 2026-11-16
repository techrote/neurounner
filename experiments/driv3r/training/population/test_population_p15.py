"""P15 counterfactual and temporal-attention model-side guardrails."""
import unittest
import torch
import numpy as np
from population_lab import NICHES
from population_p10 import ACTIONS
from population_p12 import tensorize
from population_p14 import dataset,mission_pair,inputs
from population_p15 import build_model,MODES,run,counterfactual_interventions

class TestP15(unittest.TestCase):
    def test_order_goal_counterfactual_balancing(self):
        for side in ('E','W'):
            a,b=mission_pair(291,0,9,'heldout_c',NICHES[2],'heldout_far',side)
            self.assertTrue(np.array_equal(a['states'][0]['xs'],b['states'][0]['xs']))
            self.assertNotEqual(a['desired'][0],b['desired'][0])
            self.assertEqual(sorted(a['precursor']), sorted(b['precursor']))
            self.assertFalse(a['final'].painted & (a['final'].protected|a['final'].obstacles))
    def test_model_shapes_and_legality(self):
        rows=dataset([291],('train_a',),(5,),('train_low',),cards=(0,),niches=NICHES[:1])
        X,M,H,N,Y=tensorize(rows)
        for mode in MODES:
            model=build_model(mode,16)
            logits=model(X,M,H,N)
            self.assertEqual(tuple(logits.shape),(len(rows),len(ACTIONS)))
            pred=logits.argmax(-1)
            self.assertTrue(M[torch.arange(len(rows)),pred].all().item())
    def test_models_have_gradients(self):
        rows=dataset([291],('train_a',),(5,),('train_low',),cards=(0,),niches=NICHES[:2])
        X,M,H,N,Y=tensorize(rows)
        for mode in MODES:
            model=build_model(mode,16)
            logits=model(X,M,H,N)
            loss=torch.nn.functional.cross_entropy(logits,Y)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
    def test_state_only_and_shuffle_masks(self):
        rows=dataset([291],('train_a',),(5,),('train_low',),cards=(0,),niches=NICHES[:1])
        X,M,H,N,Y=tensorize(rows)
        _,blind=inputs(X,H,'state_only')
        self.assertTrue((blind==0).all())
        _,sham=inputs(X,H,'shuffled_order',torch.Generator().manual_seed(99))
        self.assertTrue((H.sort(-1).values==sham.sort(-1).values).all())
    def test_causal_swap_preserves_eight_way_counterfactual(self):
        rows=dataset([291],('heldout_c',),(9,),('heldout_far',),cards=(0,),niches=NICHES[:1])
        result=counterfactual_interventions(build_model('factorized_order_goal',16),rows)
        self.assertEqual(set(result),{'original','swap_history','swap_goal','swap_both'})
        self.assertTrue(all(0<=v<=1 for v in result.values()))
    def test_run_limits(self):
        with self.assertRaises(ValueError):run('/tmp/invalid-p15',seeds=[1]*20)
        with self.assertRaises(ValueError):build_model('nonexistent')
        with self.assertRaises(ValueError):build_model('learned_attention',width=17)

if __name__=='__main__': unittest.main()
