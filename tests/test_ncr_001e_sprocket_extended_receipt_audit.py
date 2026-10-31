"""Independent offline tamper sentinels for original 2048-unit Sprocket receipts.

No LLM, tokenizer execution, model weights, optimizer or sealed dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001e_sprocket_extended_receipt_audit import audit, originals, STATUS


class PinnedExtendedSourceReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = originals()

    def expect_rejected(self, variant, field, value):
        modified = deepcopy(self.records)
        modified[variant][field] = value
        with self.assertRaises(RuntimeError):
            audit(modified)

    def test_two_exact_original_receipts_admission_remains_blocked(self):
        result = audit()
        self.assertEqual(result["status"], STATUS)
        self.assertEqual(result["original_receipt_count"], 2)
        self.assertEqual(set(result["variants"]), {"chat", "instruct"})
        self.assertIs(result["this_auditor_executed_models"], False)
        self.assertIs(result["final_four_model_conformance"], False)
        self.assertIs(result["training_authorized"], False)
        self.assertIs(result["tool_memory_semantics_qualified"], False)
        self.assertNotEqual(result["variants"]["chat"]["full_checkpoint_sha256"],
                            result["variants"]["instruct"]["full_checkpoint_sha256"])

    def test_wrong_variant_checkpoint_or_hook_is_rejected(self):
        self.expect_rejected("chat", "variant", "instruct")
        self.expect_rejected("chat", "source_checkpoint_sha256",
                             self.records["instruct"]["source_checkpoint_sha256"])
        self.expect_rejected("instruct", "source_hook_commit",
                             self.records["chat"]["source_hook_commit"])
        self.expect_rejected("chat", "tokens_sha256", "0" * 64)

    def test_false_training_and_semantic_qualification_are_rejected(self):
        for variant, key, wrong in (
            ("chat", "train_authorized", True),
            ("chat", "optimizer_steps", 1),
            ("instruct", "actual_model_training", True),
            ("instruct", "heldout_evaluation", True),
            ("chat", "tool_and_memory_behavior_qualified", True),
            ("instruct", "full_four_model_conformance_pass", True),
            ("chat", "control_probe_is_valid_tool_memory_dialogue", True),
            ("instruct", "original_short_256_process_results_reused_as_new_model_run", True),
        ):
            with self.subTest(variant=variant, field=key):
                self.expect_rejected(variant, key, wrong)

    def test_context_or_real_incremental_cache_limits_cannot_be_relaxed(self):
        self.expect_rejected("chat", "context_lengths", [512, 1024, 1536])
        self.expect_rejected("instruct", "cache_tail_units", 1)
        for variant in ("chat", "instruct"):
            for field, value in (
                ("max_abs", 0.0006), ("pass", False),
                ("atol", 0.05), ("rtol", 0.05),
            ):
                with self.subTest(variant=variant, field=field):
                    modified = deepcopy(self.records)
                    modified[variant]["checks"][
                        "2048_full_vs_prefill_plus_8_incremental"][field] = value
                    with self.assertRaises(RuntimeError):
                        audit(modified)

    def test_future_suffix_request_reset_and_noop_are_not_waived(self):
        for variant, name, field, value in (
            ("chat", "512_strict_future_suffix", "max_abs", 1e-6),
            ("instruct", "cross_request_A_B_A_exact", "pass", False),
            ("chat", "2048_frozen_zero_adapter_exact", "max_abs", 1e-6),
        ):
            with self.subTest(variant=variant, invariant=name):
                modified = deepcopy(self.records)
                modified[variant]["checks"][name][field] = value
                with self.assertRaises(RuntimeError):
                    audit(modified)

    def test_raw_control_transport_cannot_claim_tool_memory_grammar(self):
        modified = deepcopy(self.records)
        modified["instruct"]["control_tokens"]["<|tool_result|>"] = 6
        with self.assertRaises(RuntimeError):
            audit(modified)
        modified = deepcopy(self.records)
        modified["chat"]["checks"]["raw_control_ids_transport_only_cache"]["pass"] = False
        with self.assertRaises(RuntimeError):
            audit(modified)

    def test_runner_identity_and_missing_frozen_checks_are_rejected(self):
        modified = deepcopy(self.records)
        modified["chat"]["runner"]["github_run_id"] = "0"
        with self.assertRaises(RuntimeError):
            audit(modified)
        modified = deepcopy(self.records)
        modified["instruct"]["checks"].pop("1024_full_vs_prefill_plus_8_incremental")
        with self.assertRaises(RuntimeError):
            audit(modified)
        modified = deepcopy(self.records)
        modified.pop("chat")
        with self.assertRaises(RuntimeError):
            audit(modified)


if __name__ == "__main__":
    unittest.main()
