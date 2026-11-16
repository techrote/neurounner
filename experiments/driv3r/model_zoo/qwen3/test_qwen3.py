"""MZ-005 source-only contract and held-out split tests (no model download)."""
import unittest
from .run import MODEL, PIN, SCHEMA, SYSTEM, chat_messages, verify_splits
from experiments.driv3r.model_zoo.mamba130.counterfactual import samples
from experiments.driv3r.model_zoo.mamba130.world import ACTIONS, World


class Qwen3ContractTests(unittest.TestCase):
    def test_upstream_pin_and_no_runtime_code(self):
        self.assertEqual(MODEL, "Qwen/Qwen3-0.6B")
        self.assertEqual(PIN, "c1899de289a04d12100db370d81485cdf75e47ca")
        self.assertEqual(SCHEMA, "MZ005-QWEN3-GAME-v1")
        self.assertEqual(len(PIN), 40)
        self.assertIn("legal", SYSTEM)

    def test_chat_is_deterministic_and_preserves_observation(self):
        prompt = World.make(412).prompt()
        msg = chat_messages(prompt)
        self.assertEqual([x["role"] for x in msg], ["system", "user"])
        self.assertEqual(msg[1]["content"], prompt)
        self.assertEqual(msg, chat_messages(prompt))
        self.assertRaises(ValueError, chat_messages, "unrecognized input")

    def test_strictly_disjoint_world_seeds_and_goal_swaps(self):
        a = samples(range(400, 414), 9)
        b = samples(range(900, 904), 9)
        c = samples(range(1200, 1204), 9)
        self.assertGreater(len(a), 100)
        verify_splits(a, b, c)
        with self.assertRaises(ValueError):
            verify_splits(a, b, b)

    def test_world_authority_actions_are_external(self):
        world = World.make(1815)
        legal = world.legal()
        self.assertIn("STOP", legal)
        self.assertTrue(set(legal).issubset(set(ACTIONS)))
        self.assertIn(" legal=", world.prompt())


if __name__ == "__main__":
    unittest.main()
