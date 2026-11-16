"""P11 guardrails: paired observations, legal precursor and causal memory."""
import unittest
import numpy as np

from population_lab import NICHES, World
from population_p10 import ACTIONS
from population_p11 import BRANCH_PAIRS, build_model, dataset, evaluate, pair, splits, start_world, to_tensors, train


class TestPopulationP11(unittest.TestCase):
    def test_matched_world_state_opposite_goals(self):
        for pair_name in BRANCH_PAIRS:
            for niche in NICHES:
                a,b=pair(12345,'train',0,pair_name,niche)
                self.assertTrue(np.array_equal(a['xs'],b['xs']))
                self.assertTrue(np.array_equal(a['mask'],b['mask']))
                self.assertEqual(a['world'].cursor,b['world'].cursor)
                self.assertEqual(a['world'].parent,b['world'].parent)
                self.assertEqual(a['world'].goal,b['world'].goal)
                self.assertNotEqual(a['target'],b['target'])
                self.assertEqual(sorted(a['history'].tolist()),sorted(b['history'].tolist()))

    def test_real_legal_precursor_and_protection(self):
        for split in ('train','validation','holdout'):
            for card in (0,1):
                for a,b in [pair(1918,split,card,BRANCH_PAIRS[0],NICHES[0])]:
                    for r in (a,b):
                        w=r['world']
                        self.assertEqual(w.steps,5)
                        self.assertTrue(w.switched)
                        self.assertTrue({'E','W'}.issubset(w.legal()))
                        self.assertTrue(w.painted.isdisjoint(w.obstacles|w.protected))
                        self.assertEqual(w.cursor[1],r['world'].history[-1]['end'][1])

    def test_pair_signal_not_in_current_observation(self):
        a,b=pair(1388,'holdout',1,BRANCH_PAIRS[1],NICHES[3])
        self.assertTrue(np.array_equal(a['xs'],b['xs']))
        self.assertNotEqual(tuple(a['history']),tuple(b['history']))
        self.assertEqual(a['mask'][ACTIONS.index('STOP')],True)
        self.assertEqual(a['history'].shape,(8,))

    def test_disjoint_world_and_root_splits(self):
        x=splits(646,20,10,12)
        keys=[set(x[k]) for k in ('train','validation','holdout')]
        self.assertFalse(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
        self.assertNotEqual(start_world(x['train'][0],'train').cursor,
                            start_world(x['holdout'][0],'holdout').cursor)

    def test_balanced_targets_each_niche(self):
        rows=dataset(splits(646,12,8,8)['train'],'train',(0,))
        for n in range(len(NICHES)):
            goals=[r['target'] for r in rows if r['niche']==n]
            self.assertEqual(goals.count(ACTIONS.index('E')),goals.count(ACTIONS.index('W')))
        self.assertEqual(len(rows),12*3*2*len(NICHES))

    def test_bounded_params(self):
        from population_p11 import run
        with self.assertRaises(ValueError):run('/tmp/no-op-invalid',train_worlds=1)

    def test_real_gradients_and_masked_choice(self):
        import torch
        data=to_tensors(dataset(splits(647,12,8,8)['train'][:4],'train',(0,)))
        torch.manual_seed(6)
        m=build_model('token_gru',16)
        initial=[v.detach().clone() for v in m.parameters()]
        s=train(m,data,data,19,epochs=3,batch_size=64)
        self.assertGreater(s['updates'],0)
        self.assertTrue(any(not torch.equal(x,p) for x,p in zip(initial,m.parameters())))
        X,M,H,N,Y=data
        with torch.no_grad():
            p=m(X,M,H,N).argmax(-1)
        self.assertTrue(bool(M[torch.arange(len(p)),p].all()))

    def test_both_architecture_controls_same_parameterization(self):
        torch=__import__('torch')
        for name in ('state_only','token_gru','history_sham'):
            m=build_model(name,16)
            self.assertEqual(sum(p.numel() for p in m.parameters()),
                             sum(p.numel() for p in build_model('token_gru',16).parameters()))
        # The two GRU arms only differ in whether training destroys token order.
        m1,m2=build_model('token_gru',16),build_model('history_sham',16)
        self.assertEqual(set(m1.state_dict()),set(m2.state_dict()))


if __name__=='__main__':unittest.main()
