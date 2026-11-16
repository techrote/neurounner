"""MZ-009 contract tests: legal masks, hidden causality, recovery and training."""
import unittest
from experiments.driv3r.model_zoo.material_forks_mz009.run import ACTIONS, MAX_STEPS, World, episode, dataset, networks, to_tensors


class MaterialForkTests(unittest.TestCase):
    def test_first_fork_hides_beacon_and_inversion(self):
        states = []
        for cue in (0, 1):
            for invert in (0, 1):
                w = World(7001, cue, invert)
                for i in range(8):
                    self.assertEqual(w.stage, 'corridor')
                    w.step(w.preferred_action())
                self.assertEqual(w.stage, 'fork')
                self.assertEqual(w.observe()[17:21], [0.] * 4)
                states.append(tuple(w.observe()))
        self.assertEqual(len(set(states)), 1)

    def test_history_rules_flip_material_choices(self):
        for seed in range(10, 30):
            worlds = [World(seed, c, i) for c in range(2) for i in range(2)]
            for w in worlds:
                for _ in range(8): w.step(w.preferred_action())
            self.assertEqual(worlds[0].preferred_action(), worlds[3].preferred_action())
            self.assertEqual(worlds[1].preferred_action(), worlds[2].preferred_action())
            self.assertNotEqual(worlds[0].preferred_action(), worlds[1].preferred_action())
            for w in worlds:
                mask = w.legal()
                self.assertEqual(mask, [0, 1, 1, 0, 1])
                self.assertTrue(mask[w.preferred_action()])

    def test_wrong_material_forces_retrace(self):
        w = World(7000, 0, 0)
        for _ in range(8): w.step(0)
        wrong = 3 - w.preferred_action()
        w.step(wrong)
        self.assertEqual(w.stage, 'hazard')
        self.assertEqual(w.legal(), [0, 0, 0, 1, 1])
        w.step(3)
        self.assertEqual(w.stage, 'fork')
        self.assertEqual(w.collisions, 1)
        w.step(w.preferred_action())
        self.assertEqual(w.painted, 1)

    def test_trajectories_and_source_splits(self):
        train = dataset(range(1000, 1008))
        test = dataset(range(5000, 5008))
        self.assertEqual(len(train), 8 * 4 * 3)
        self.assertTrue({r[2][0] for r in train}.isdisjoint({r[2][0] for r in test}))
        for obs, labels, meta in train + test:
            self.assertEqual(len(obs), len(labels))
            self.assertLessEqual(len(obs), MAX_STEPS)
            for x, y in zip(obs, labels): self.assertEqual(x[8 + y], 1.)

    def test_nontrivial_recurrent_gradient(self):
        import torch
        torch.set_num_threads(2)
        recurrent, reactive = networks(torch)
        x, y = to_tensors(torch, dataset([9000, 9001]))
        logits = recurrent(x)
        self.assertEqual(tuple(logits.shape), (24, MAX_STEPS, len(ACTIONS)))
        self.assertEqual(tuple(reactive(x).shape), tuple(logits.shape))
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100)
        loss.backward()
        self.assertGreater(float(recurrent.core.weight_hh_l0.grad.abs().sum()), 0.)

if __name__ == '__main__': unittest.main()