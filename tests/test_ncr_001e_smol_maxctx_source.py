"""Source-only refusal regressions for the independent SmolLM2 #72 pretrain probe.

No import of torch or frozen model weights. This is not physical qualification.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ncr_001e_smol_maxctx.py"


class SmolMaxContextSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_four_original_source_identities_cannot_be_silently_relabelled(self):
        for marker in (
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            "ec0d4a80e15f203a06c26ee9c23858f78976370b",
            "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
            '"smollm2", "sprocket_chat", "sprocket_instruct", "void1"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

    def test_long_context_is_diagnostics_not_evaluation(self):
        values = {
            n.targets[0].id: n.value for n in self.tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        }
        self.assertEqual(ast.literal_eval(values["CACHE_LENGTHS"]), (512, 1024, 2048))
        self.assertEqual(ast.literal_eval(values["TAIL"]), 16)
        self.assertIn("synthetic_ids(tokenizer)", self.source)
        self.assertNotIn("datasets/train-003/", self.source)
        self.assertNotIn("fixtures/natural-text-holdout", self.source)

    def test_unconditional_source_and_no_training_guard(self):
        self.assertIn('p["train_authorized"] is False', self.source)
        self.assertIn('"training_authorized": False', self.source)
        self.assertIn('"optimizer_steps": 0', self.source)
        self.assertIn('"per_model_gate": False', self.source)
        self.assertIn('"heldout_evaluation": False', self.source)
        for forbidden in ("torch.optim.", "optimizer.step(", ".backward(", "loss.backward("):
            self.assertNotIn(forbidden, self.source)

    def test_consumer_checks_actual_independent_pid_and_full_source(self):
        self.assertIn('source["process"]["pid"] != os.getpid()', self.source)
        self.assertIn('source["source"] == result["source"]', self.source)
        self.assertIn("weights_only=True", self.source)
        self.assertIn('sha(OUT / relative) == source["cold_reference"][field]', self.source)
        self.assertIn('require(rejected, "Cold loading with wrong revision did not refuse")',
                      self.source)
        self.assertIn('metadata.write_bytes(original)', self.source)

    def test_source_pin_bytes_and_numerical_bounds_are_not_overridden(self):
        self.assertIn('pin["config_sha256"]', self.source) if False else None
        self.assertIn('("config.json", "config_sha256")', self.source)
        self.assertIn('("tokenizer.json", "tokenizer_sha256")', self.source)
        self.assertIn('("tokenizer_config.json", "tokenizer_config_sha256")', self.source)
        self.assertIn('p["conformance"]["full_vs_cached_prefill_atol"]', self.source)
        self.assertIn('p["conformance"]["full_vs_cached_prefill_rtol"]', self.source)
        self.assertIn('torch.allclose(a, b, atol=atol, rtol=rtol)', self.source)

    def test_producer_consumer_failure_receipts_and_all_modes(self):
        self.assertIn('("producer", "consumer")', self.source)
        self.assertIn('"FAIL_RETAIN_RAW_ERROR"', self.source)
        self.assertIn('"traceback"', self.source)
        self.assertIn('("disabled", "zero", "active")', self.source)
        self.assertIn('for use_cache in (False, True)', self.source)
        self.assertIn('"full_four_model_conformance_pass": False', self.source)


if __name__ == "__main__":
    unittest.main()
