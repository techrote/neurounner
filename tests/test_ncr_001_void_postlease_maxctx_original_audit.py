"""Offline tests for original void.1 max-context evidence custody.

No torch, tokenizer, network, model execution, optimizer, or held-out fixtures.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001_void_postlease_maxctx_original_audit import (  # noqa: E402
    audit, adversarial, decode_original, inspect_receipt,
)


class VoidPostleaseMaxctxOriginalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = decode_original("source")
        cls.qwen = decode_original("qwen")
        cls.receipt = audit(cls.source, cls.qwen)

    def test_original_artifacts_and_semantics_pass(self):
        result = adversarial(self.source, self.qwen)
        self.assertEqual(result["physical_checks"], 32)
        self.assertEqual(result["source_tests"], 8)
        self.assertEqual(result["negative_mutations_rejected"], 16)
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["final_72_conformance_pass"])

    def test_training_or_optimizer_claim_is_rejected(self):
        for field, value in (("training_authorized", True), ("optimizer_steps", 1),
                             ("backward_executed", True)):
            with self.subTest(field=field):
                row = copy.deepcopy(self.receipt)
                row[field] = value
                with self.assertRaisesRegex(ValueError, "false training"):
                    inspect_receipt(row)

    def test_cache_tolerance_relaxation_or_failure_is_rejected(self):
        row = copy.deepcopy(self.receipt)
        row["checks"]["nca_maxctx_cache_2048"]["max_abs"] = 0.01
        with self.assertRaisesRegex(ValueError, "cache gate"):
            inspect_receipt(row)
        row = copy.deepcopy(self.receipt)
        row["checks"]["mlp_maxctx_cache_1024"]["atol"] = 0.01
        with self.assertRaisesRegex(ValueError, "cache gate"):
            inspect_receipt(row)

    def test_eos_or_learned_stopping_invention_is_rejected(self):
        row = copy.deepcopy(self.receipt)
        row["generations"][0]["actual_eos_id_emitted"] = True
        with self.assertRaisesRegex(ValueError, "EOS boundary"):
            inspect_receipt(row)
        row = copy.deepcopy(self.receipt)
        row["generations"][0]["learned_eos_claim"] = True
        with self.assertRaisesRegex(ValueError, "EOS boundary"):
            inspect_receipt(row)

    def test_causal_or_frozen_reference_falsification_is_rejected(self):
        row = copy.deepcopy(self.receipt)
        row["checks"]["nca_2048_strict_future_suffix"]["exact"] = False
        with self.assertRaisesRegex(ValueError, "future-suffix"):
            inspect_receipt(row)
        row = copy.deepcopy(self.receipt)
        row["checks"]["mlp_raw_zero_2048"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen/zero"):
            inspect_receipt(row)

    def test_source_or_context_identity_drift_is_rejected(self):
        row = copy.deepcopy(self.receipt)
        row["source"]["serializer_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "serializer"):
            inspect_receipt(row)
        row = copy.deepcopy(self.receipt)
        row["source_text"]["selected_units"] = 4096
        with self.assertRaisesRegex(ValueError, "source accounting"):
            inspect_receipt(row)

    def test_any_zip_byte_change_is_rejected(self):
        bad = bytearray(self.qwen)
        bad[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "Qwen ZIP bytes"):
            audit(self.source, bytes(bad))


if __name__ == "__main__":
    unittest.main()
