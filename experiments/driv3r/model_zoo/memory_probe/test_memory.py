"""MZ-008: fast source-fenced action-contract and leakage regression tests."""
import unittest
from experiments.driv3r.model_zoo.memory_probe.run import (
    ACTIONS, D, T, TARGETS, dataset, intervention, models, pair_scores, world,
)


class DelayedCueGameTests(unittest.TestCase):
    def test_balanced_world_pairs_and_disjoint_seeds(self):
        train, test = dataset(range(1000, 1010)), dataset(range(5000, 5010))
        self.assertEqual(len(train), 80)
        self.assertTrue({r[2]['base'] for r in train}.isdisjoint(
            {r[2]['base'] for r in test}))
        for base in range(1000, 1010):
            self.assertEqual(sorted(r[1] for r in train if r[2]['base'] == base),
                             [1, 1, 2, 2, 3, 3, 4, 4])

    def test_final_observation_is_history_blind(self):
        cases = [world(1001, cue, inv, mode) for cue in (0, 1)
                 for inv in (0, 1) for mode in (0, 1)]
        # Only current mode differentiates final observations, not cue or inversion.
        self.assertEqual(len({tuple(x[-1]) for x, _, _ in cases}), 2)
        for x, y, meta in cases:
            self.assertEqual((len(x), len(x[0])), (T, D))
            self.assertEqual(x[-1][20:24], [0.] * 4)
            self.assertTrue(x[0][20 + meta['cue']] > 0.)
            self.assertTrue(x[1][20 + meta['cue']] > 0.)
            self.assertTrue(x[4][22 + meta['invert']] > 0.)
            self.assertTrue(x[5][22 + meta['invert']] > 0.)
            self.assertEqual(y, TARGETS[2 * meta['author'] +
                                        (meta['cue'] ^ meta['invert'])])
            self.assertEqual(x[-1][8+y], 1.)

    def test_interventions_scrub_all_presentations_and_leave_current(self):
        import torch
        data = torch.tensor([row[0] for row in dataset([5001])])
        for kind in ('erase_history', 'remove_beacon', 'remove_inversion',
                     'reverse_early_beacon'):
            mutated = intervention(data, torch, kind)
            self.assertTrue(torch.equal(mutated[:, -1], data[:, -1]))
        self.assertEqual(int(intervention(data, torch, 'remove_beacon')
                             [:, :2, 20:22].count_nonzero()), 0)
        self.assertEqual(int(intervention(data, torch, 'remove_inversion')
                             [:, 4:6, 22:24].count_nonzero()), 0)

    def test_genuine_recurrent_gradients_and_paired_metrics(self):
        import torch
        torch.set_num_threads(2)
        net, ctrl = models(torch)
        rows = dataset([9001])
        x = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        y = torch.tensor([r[1] for r in rows])
        logits = net(x)
        self.assertEqual(tuple(logits.shape), (8, len(ACTIONS)))
        torch.nn.functional.cross_entropy(logits, y).backward()
        self.assertGreater(float(net.gru.weight_ih_l0.grad.abs().sum()), 0.)
        self.assertEqual(tuple(ctrl(x[:, -1]).shape), (8, len(ACTIONS)))
        paired = pair_scores(y.tolist(), rows)
        self.assertEqual(paired['both_correct'], paired['pairs'])
        self.assertEqual(paired['switched_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
