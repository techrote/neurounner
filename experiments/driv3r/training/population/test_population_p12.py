"""P12 paired-memory information-theoretic controls and CPU optimizer guards."""
import unittest
import numpy as np
from population_lab import NICHES, World
from population_p10 import ACTIONS
from population_p12 import (GRAMMARS,HISTORY,MODES,assess,dataset,encode_history,fit,
    model_for,pair,split_seeds,tensorize,transform,world)

class TestPopulationP12(unittest.TestCase):
    def test_state_history_horizon_and_balanced_labels(self):
        for grammar in GRAMMARS:
            for depth in (5,7,9):
                for niche in NICHES:
                    a,b=pair(12008,0,depth,grammar,niche)
                    self.assertTrue(np.array_equal(a['xs'],b['xs']))
                    self.assertTrue(np.array_equal(a['mask'],b['mask']))
                    self.assertTrue(np.array_equal(a['history'][-8:],b['history'][-8:]))
                    self.assertEqual(sorted(a['history']),sorted(b['history']))
                    self.assertNotEqual(a['target'],b['target'])
                    self.assertEqual(len(a['precursor']),6+depth*2-1)
                    self.assertTrue(a['world'].switched)
                    self.assertEqual(a['world'].cursor,b['world'].cursor)
                    self.assertEqual(a['world'].painted,b['world'].painted)
                    self.assertEqual(a['world'].parent,b['world'].parent)
                    self.assertEqual(a['world'].heading,b['world'].heading)
    def test_unmodified_obstacle_protection_outside_pocket(self):
        for depth in (5,7,9):
            w=world(12345,0,depth)
            original=World.generate(12345,max_steps=28)
            self.assertEqual(w.protected,original.protected)
            self.assertTrue(w.obstacles.issubset(original.obstacles))
            self.assertTrue(w.painted.isdisjoint(w.obstacles|w.protected))
    def test_split_seeds_disjoint(self):
        s=split_seeds(649,24,12,16)
        sets=[set(x) for x in s.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_no_history_or_last8_can_distinguish_pair(self):
        import torch
        rows=dataset([31,41],('train_a','heldout_c'),(9,),cards=(0,),niches=NICHES[:2])
        X,M,H,N,Y=tensorize(rows)
        for mode in ('state_only','last8_history'):
            T=transform(H,mode)
            self.assertTrue(bool((T[::2]==T[1::2]).all()))
        self.assertTrue(bool((Y[::2]!=Y[1::2]).all()))
        self.assertTrue(bool((M[torch.arange(len(Y)),Y]).all()))
    def test_full_history_retains_missing_cue(self):
        a,b=pair(101,1,9,'heldout_c',NICHES[1])
        self.assertFalse(np.array_equal(a['history'],b['history']))
        self.assertTrue(np.array_equal(a['history'][-8:],b['history'][-8:]))
        self.assertEqual(len(a['history']),HISTORY)
    def test_real_gradient_step(self):
        import torch
        torch.set_num_threads(1)
        rows=dataset([101,102,103,104],('train_a','train_b'),(5,),cards=(0,),niches=NICHES[:2])
        data=tensorize(rows)
        m=model_for('full_history',16)
        old={k:v.detach().clone() for k,v in m.state_dict().items()}
        r=fit(m,'full_history',data,data,171,epochs=3,batch_size=32)
        self.assertGreater(r['updates'],0)
        self.assertTrue(any(not torch.equal(old[k],v) for k,v in m.state_dict().items()))
        metrics=assess(m,'full_history',rows)
        self.assertEqual(metrics['legal_action_rate'],1.)
    def test_model_parameter_parity_and_split_grammar(self):
        counts=[sum(p.numel() for p in model_for(k,16).parameters()) for k in MODES]
        self.assertEqual(len(set(counts)),1)
        self.assertNotIn(GRAMMARS['heldout_c'],[GRAMMARS[k] for k in ('train_a','train_b')])
    def test_bad_args(self):
        from population_p12 import run
        with self.assertRaises(ValueError):run('/tmp/p12-invalid',train_worlds=1)
        with self.assertRaises(ValueError):encode_history(['E']*40)
        with self.assertRaises(ValueError):pair(1,0,9,'missing',NICHES[0])

if __name__=='__main__': unittest.main()