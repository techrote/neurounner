"""Fast source-only checks for MZ-007; separate from real CPU training."""
import unittest

from experiments.driv3r.model_zoo.mamba130.counterfactual import samples, raw_features
from experiments.driv3r.model_zoo.gru_game.run import (
    CONTEXT, WIDTH, encode_context, goal_neutral, observations, verify_splits,
    make_network, make_control,
)


class RecurrentGameTests(unittest.TestCase):
    def test_context_masks_prior_goal_and_preserves_current_goal(self):
        prior = [0.] * WIDTH
        prior[4], prior[5] = 7., -4.
        current = [0.] * WIDTH
        current[4], current[5] = -2., 3.
        context = encode_context(current, [prior])
        self.assertEqual(len(context), CONTEXT)
        self.assertEqual(len(context[-1]), WIDTH)
        self.assertEqual(context[-2][4:6], [0., 0.])
        self.assertEqual(context[-1][4:6], [-2., 3.])
        self.assertEqual(context[0], [0.] * WIDTH)
        self.assertEqual(goal_neutral(prior)[4:6], [0., 0.])

    def test_real_synthetic_seed_splits_and_goal_pairs(self):
        train = samples(range(400, 412), 8)
        dev = samples(range(900, 904), 8)
        test = samples(range(1200, 1204), 8)
        verify_splits(train, dev, test)
        data = observations(train)
        self.assertEqual(len(data), len(train))
        self.assertEqual(len(data[0]), CONTEXT)
        self.assertEqual(data[0][-1], raw_features(train[0][0]))
        for row, context in zip(train, data):
            self.assertEqual(len(context[-1]), WIDTH)
            self.assertEqual(context[-1], raw_features(row[0]))

    def test_genuine_recurrent_19m_forward_and_backward(self):
        import torch
        torch.manual_seed(42021)
        model = make_network(torch)
        size = sum(p.numel() for p in model.parameters())
        self.assertTrue(18_000_000 <= size <= 22_000_000, size)
        self.assertIsNotNone(model.input_norm.weight)
        self.assertEqual(model.head[-1].out_features, 7)
        x = torch.randn(2, CONTEXT, WIDTH)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 7))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.rnn.weight_ih_l0.grad)
        self.assertGreater(float(model.rnn.weight_ih_l0.grad.abs().sum()), 0.)
        control = make_control(torch)
        self.assertEqual(tuple(control(x[:, -1, :]).shape), (2, 7))

    def test_reject_world_seed_leakage(self):
        a = samples(range(400, 410), 8)
        with self.assertRaises(RuntimeError):
            verify_splits(a, a, samples(range(1200, 1204), 8))


if __name__ == "__main__":
    unittest.main()
