"""Offline no-download contract tests for MZ-006."""
import unittest
from experiments.driv3r.model_zoo.modernbert.run import (
    MODEL, PIN, SCHEMA, verify_splits, select_trainable,
)
from experiments.driv3r.model_zoo.mamba130.counterfactual import samples


class ModernBertContractTests(unittest.TestCase):
    def test_immutable_source_identity(self):
        self.assertEqual(MODEL, "answerdotai/ModernBERT-base")
        self.assertEqual(len(PIN), 40)
        self.assertEqual(SCHEMA, "MZ006-MODERNBERT-GAME-v1")

    def test_shared_counterfactual_splits(self):
        train = samples(range(400, 414), 9)
        dev = samples(range(900, 904), 9)
        test = samples(range(1200, 1204), 9)
        verify_splits(train, dev, test)
        self.assertGreater(len(train), 300)
        self.assertGreater(len(test), 100)
        self.assertEqual(len({r[2] for r in train} & {r[2] for r in test}), 0)

    def test_train_last_real_encoder_and_classifier(self):
        names = [f"model.layers.{i}.attn.W" for i in range(22)]
        names += ["head.dense.weight", "classifier.weight"]
        got, layer = select_trainable(names)
        self.assertEqual(layer, 21)
        self.assertIn("model.layers.21.attn.W", got)
        self.assertIn("classifier.weight", got)
        self.assertNotIn("model.layers.20.attn.W", got)

    def test_fail_closed_when_model_architecture_changes(self):
        with self.assertRaises(RuntimeError):
            select_trainable(["classifier.weight", "encoder.weight"])


if __name__ == "__main__":
    unittest.main()
