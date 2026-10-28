"""Source-only tests for post-lease void.1 #72 adjudication.

No model/tokenizer import, network, optimizer, held-out fixtures, or training.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001e_void1_postlease_adjudication import build_report  # noqa: E402


class Void1PostleaseAdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()
        cls.committed = json.loads(
            (ROOT / "experiments/ncr-001/adjudication/void1-postlease-pretrain.json")
            .read_text(encoding="utf-8")
        )

    def test_generated_report_matches_committed_decision(self):
        self.assertEqual(self.report, self.committed)

    def test_final_gate_is_fail_closed_and_training_stays_off(self):
        r = self.report
        self.assertFalse(r["per_model_gate"])
        self.assertTrue(r["source_runtime_pretraining_subgate"])
        self.assertFalse(r["train_authorized"])
        self.assertEqual(r["optimizer_steps_observed"], 0)
        self.assertFalse(r["backward_executed"])
        self.assertFalse(r["heldout_evaluation"])
        self.assertFalse(r["comparative_training_performed"])

    def test_exact_three_learning_blockers_are_preserved(self):
        self.assertEqual(set(self.report["blocking_required_checks"]), {
            "base_weight_unchanged_after_step",
            "nonzero_adapter_gradient_after_warmup",
            "finite_state_and_loss",
        })
        self.assertEqual(
            self.report["matrix"]["base_weight_unchanged_after_step"]["status"],
            "BLOCKED_BY_TRAINING_POLICY",
        )
        self.assertEqual(
            self.report["matrix"]["nonzero_adapter_gradient_after_warmup"]["status"],
            "BLOCKED_BY_TRAINING_POLICY",
        )
        self.assertEqual(
            self.report["matrix"]["finite_state_and_loss"]["status"],
            "PARTIAL_BLOCKED_WARMUP_LOSS",
        )

    def test_postlease_source_runtime_checks_are_not_inherited_as_learning(self):
        m = self.report["matrix"]
        for key in (
            "same_base_logits",
            "causal_suffix_invariance",
            "full_prefix_vs_incremental_cache",
            "per_token_isolation",
            "per_request_reset",
            "cold_reload_state_dict",
        ):
            with self.subTest(key=key):
                self.assertTrue(m[key]["status"].startswith("PASS"))
        self.assertNotIn(40755894558,
                         m["nonzero_adapter_gradient_after_warmup"]["evidence_runs"])

    def test_max_context_and_byte_eos_negative_result_are_preserved(self):
        ev = self.report["evidence"]
        self.assertEqual(ev["postlease_max_context_run"], 40755894558)
        self.assertEqual(ev["max_context_generation_records"], 12)
        self.assertEqual(ev["max_context_generation_ids"], [101, 110, 32, 109])
        self.assertFalse(ev["actual_eos_258_emitted_in_four_step_probe"])
        self.assertFalse(
            self.report["byte_eos_dependency"]["historical_train002b_effect_measured"]
        )
        self.assertFalse(
            self.report["byte_eos_dependency"]["four_step_generation_emitted_eos"]
        )

    def test_all_cache_values_remain_under_original_tolerance(self):
        tol = self.report["protocol"]["cache_atol"]
        self.assertEqual(tol, 0.0005)
        self.assertEqual(self.report["protocol"]["cache_rtol"], 0.0005)
        for name, value in self.report["evidence"]["max_context_cache_max_abs"].items():
            with self.subTest(name=name):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, tol)

    def test_four_lane_and_void_source_identity_are_exact(self):
        p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        self.assertEqual(set(p["models"]),
                         {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"})
        self.assertFalse(p["train_authorized"])
        src = self.report["source"]
        self.assertEqual(src["repo_id"], "dotlabs/void.1")
        self.assertEqual(src["weight_file"], "model.safetensors")
        self.assertEqual(src["taps_after_block_zero_based"], [7, 15, 23])
        self.assertEqual(src["eos_id"], 258)


if __name__ == "__main__":
    unittest.main()
