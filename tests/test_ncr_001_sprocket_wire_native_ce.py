"""Synthetic no-training sentinels for original root-Instruct native-CE witness.

Toy logits validate the independently implemented loss reduction, but are NOT
a substitute for the separate original-weight Actions job.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_wire_native_ce as witness


class IndependentMaskedCE(unittest.TestCase):
    def setUp(self):
        self.labels = torch.full((2, 47), -100, dtype=torch.long)
        self.labels[1, 1:46] = torch.tensor([1, 2, 3] * 15)
        self.logits = torch.zeros((2, 47, 8), dtype=torch.float32)

    def test_uniform_45_next_token_ce_and_host_row_ignored(self):
        loss, n = witness.independent_shifted_ce(self.logits, self.labels)
        self.assertEqual(n, 45)
        self.assertAlmostEqual(loss, math.log(8), places=5)

    def test_predecessor_logits_not_target_logits_drive_loss(self):
        self.logits[1, :45, 1] = 10.0
        old, _ = witness.independent_shifted_ce(self.logits, self.labels)
        self.logits[1, 1:46, 1] = 10.0
        changed, _ = witness.independent_shifted_ce(self.logits, self.labels)
        self.assertNotEqual(old, changed)

    def test_host_target_forgery_refused(self):
        self.labels[0, 4] = 1
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            witness.independent_shifted_ce(self.logits, self.labels)

    def test_missing_assistant_target_refused(self):
        self.labels[1, 8] = -100
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            witness.independent_shifted_ce(self.logits, self.labels)

    def test_nonfinite_target_logit_refused(self):
        self.logits[1, 0, 1] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            witness.independent_shifted_ce(self.logits, self.labels)

    def test_out_of_vocabulary_target_refused(self):
        self.labels[1, 4] = 999
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            witness.independent_shifted_ce(self.logits, self.labels)


if __name__ == "__main__":
    unittest.main()
