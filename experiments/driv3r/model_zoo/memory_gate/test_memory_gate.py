"""MZ-008 guards: source-fenced scenario balance and real GRU gradients."""
import unittest
from collections import Counter

from experiments.driv3r.model_zoo.memory_gate.run import (
    ACTION_SET, HISTORY, WIDTH, encode, make_rows, make_networks, metrics, verify_rows,
)
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS

class MemoryGateTests(unittest.TestCase):
    def test_expired_material_cue_and_visible_goal(self):
        past = [[.1] * 114 for _ in range(3)]
        present = [.2] * 114
        present[4:6] = [1., -.4]
        for rule in range(4):
            seq = encode(past, present, rule, 2, 3)
            self.assertEqual(len(seq), HISTORY)
            self.assertEqual(len(seq[0]), WIDTH)
            self.assertEqual(seq[0][114+rule], 1.)
            self.assertEqual(seq[0][118], 1.)
            self.assertEqual(seq[-1][114:122], [0.] * 8)
            self.assertEqual(seq[-1][122:126], [0., 0., 0., 1.])
            self.assertEqual(seq[-1][4:6], [1., -.4])
            self.assertEqual(seq[0][4:6], [0., 0.])

    def test_seed_isolation_rule_goal_balance_and_host_legality(self):
        train = make_rows(range(3000, 3006), 12)
        dev = make_rows(range(5000, 5005), 12)
        test = make_rows(range(7000, 7005), 12)
        verify_rows(train, dev, test)
        self.assertGreater(len(train), 200)
        self.assertEqual(len(set(Counter(r[1] for r in train).values())), 1)
        self.assertEqual(len(set(r[1] for r in train[:4])), 4)
        self.assertTrue(all(r[0][-1] == train[0][0][-1] for r in train[:4]))
        self.assertTrue(all(ACTIONS[r[1]] in r[6] for r in train))
        self.assertEqual(metrics(train[:4], [r[1] for r in train[:4]])["accuracy"], 1.0)

    def test_real_gradient_for_both_recurrent_architectures(self):
        import torch
        torch.set_num_threads(2)
        for arch in ("small", "large"):
            torch.manual_seed(8138)
            recurrent, flat, current = make_networks(torch, arch)
            size = sum(p.numel() for p in recurrent.parameters())
            if arch == "small":
                self.assertTrue(100_000 < size < 1_000_000)
            else:
                self.assertTrue(18_000_000 < size < 22_000_000)
            inp = torch.randn(2, HISTORY, WIDTH)
            result = recurrent(inp)
            self.assertEqual(tuple(result.shape), (2, len(ACTIONS)))
            loss = torch.nn.functional.cross_entropy(result, torch.tensor([0, 1]))
            loss.backward()
            self.assertGreater(float(recurrent.gru.weight_ih_l0.grad.abs().sum()), 0.)
            self.assertEqual(tuple(flat(inp.flatten(1)).shape), (2, len(ACTIONS)))
            self.assertEqual(tuple(current(inp[:, -1, :]).shape), (2, len(ACTIONS)))

if __name__ == "__main__":
    unittest.main()
