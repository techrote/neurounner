"""P14 paired goal-reversal, source-host legal-action and optimizer guardrails."""
import unittest
import numpy as np
from population_lab import NICHES
from population_p10 import ACTIONS
from population_p12 import tensorize
from population_p14 import (MODES,GOAL_FIELDS,mission_pair,dataset,build_model,fit,
                            inputs,evaluate,rollout,split_seeds)


class TestP14(unittest.TestCase):
    def test_two_way_history_and_two_way_goal_counterfactuals(self):
        for grammar in ('train_a','train_b','heldout_c'):
            for depth in (5,7,9):
                for niche in NICHES:
                    pairs={s:mission_pair(413,0,depth,grammar,niche,'train_low',s)
                           for s in ('E','W')}
                    for side,(a,b) in pairs.items():
                        self.assertTrue(np.array_equal(a['states'][0]['xs'],b['states'][0]['xs']))
                        self.assertNotEqual(a['desired'][0],b['desired'][0])
                        self.assertEqual(sorted(a['precursor']),sorted(b['precursor']))
                        self.assertEqual(a['initial'].goal,b['initial'].goal)
                        for e in (a,b):
                            self.assertEqual(len(e['states']),4)
                            self.assertEqual(e['desired'][-1],'STOP')
                            self.assertTrue(all(r['mask'][r['target']] for r in e['states']))
                            self.assertFalse(bool(e['final'].painted & (e['final'].protected|e['final'].obstacles)))
                    for i in (0,1):
                        self.assertEqual(pairs['E'][i]['precursor'],pairs['W'][i]['precursor'])
                        self.assertNotEqual(pairs['E'][i]['desired'][0],pairs['W'][i]['desired'][0])
                        self.assertFalse(np.array_equal(pairs['E'][i]['states'][0]['xs'],pairs['W'][i]['states'][0]['xs']))
    def test_goal_blind_is_mathematically_blind(self):
        e=mission_pair(413,0,7,'train_a',NICHES[0],'train_low','E')[0]['states'][0]
        w=mission_pair(413,0,7,'train_a',NICHES[0],'train_low','W')[0]['states'][0]
        X,M,H,N,Y=tensorize([e,w])
        clean,_=inputs(X,H,'pairwise_goal_blind')
        self.assertTrue(np.array_equal(clean[0].numpy(),clean[1].numpy()))
        self.assertTrue(np.array_equal(H[0].numpy(),H[1].numpy()))
        self.assertNotEqual(e['target'],w['target'])
    def test_history_sham_preserves_token_multisets(self):
        import torch
        H=tensorize(dataset([789],('train_a',),(5,),('train_low',),cards=(0,),niches=NICHES[:1]))[2]
        _,S=inputs(torch.zeros((len(H),len(ACTIONS),20)),H,'shuffled_order',torch.Generator().manual_seed(44))
        self.assertTrue((H.sort(-1).values==S.sort(-1).values).all())
    def test_seed_splits_disjoint(self):
        s=split_seeds(654,8,4,4)
        self.assertEqual(len(set(sum(s.values(),[]))),16)
    def test_optimizer_and_host_legal_closed_loop(self):
        import torch
        torch.set_num_threads(1)
        rows=dataset([413],('train_a',),(5,),('train_low',),cards=(0,),niches=NICHES[:2])
        data=tensorize(rows)
        for mode in MODES:
            model=build_model(mode,20)
            old={k:v.clone() for k,v in model.state_dict().items()}
            result=fit(model,mode,data,data,91,epochs=3,batch=32)
            self.assertGreater(result['updates'],0)
            self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in old.items()))
            self.assertEqual(evaluate(model,mode,rows)['legal_rate'],1.)
        r=rollout(build_model('pairwise_goal',20),'pairwise_goal',[413],('heldout_c',),(9,),('heldout_far',))
        self.assertEqual(r['legal_rate'],1.)
    def test_invalid_modes_and_limits(self):
        from population_p14 import run
        with self.assertRaises(ValueError):build_model('not-a-mode')
        with self.assertRaises(ValueError):run('/tmp/p14-invalid',train_worlds=1)


if __name__=='__main__':unittest.main()
