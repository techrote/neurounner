"""Fail-closed deterministic void.1 #72 adjudication tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001e_void1_adjudication as adjudicator  # noqa:E402


class Void1AdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual=adjudicator.build_report()
        cls.committed=json.loads(
            (ROOT/"experiments/ncr-001/adjudication/void1-pretrain.json")
            .read_text(encoding="utf-8"))

    def test_regenerated_report_exactly_matches_committed(self):
        self.assertEqual(self.actual,self.committed)

    def test_final_gate_false_despite_source_runtime_pass(self):
        self.assertEqual(
            self.actual["adjudication"],
            "BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS")
        self.assertFalse(self.actual["per_model_gate"])
        self.assertTrue(self.actual["source_runtime_pretraining_subgate"])
        self.assertFalse(self.actual["train_authorized"])
        self.assertEqual(self.actual["optimizer_steps_observed"],0)

    def test_exactly_three_learning_update_loss_checks_block(self):
        self.assertEqual(set(self.actual["blocking_required_checks"]),{
            "base_weight_unchanged_after_step",
            "nonzero_adapter_gradient_after_warmup",
            "finite_state_and_loss",
        })
        self.assertEqual(
            self.actual["matrix"]["base_weight_unchanged_after_step"]["status"],
            "BLOCKED_BY_TRAINING_POLICY")
        self.assertEqual(
            self.actual["matrix"]["nonzero_adapter_gradient_after_warmup"]["status"],
            "BLOCKED_BY_TRAINING_POLICY")
        self.assertEqual(
            self.actual["matrix"]["finite_state_and_loss"]["status"],
            "PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY")

    def test_source_runtime_checks_have_original_model_evidence(self):
        for key in ("same_base_logits","causal_suffix_invariance",
                    "full_prefix_vs_incremental_cache","per_token_isolation",
                    "per_request_reset","cold_reload_state_dict"):
            row=self.actual["matrix"][key]
            with self.subTest(key=key):
                self.assertTrue(row["status"].startswith("PASS"))
                self.assertTrue(row["evidence"])

    def test_only_actual_frozen_model_runs_count_as_runtime_evidence(self):
        self.assertEqual(self.actual["evidence_runs"],
                         [40464956343,40549549886,40605678381])
        self.assertNotIn(40606809785,self.actual["evidence_runs"])
        self.assertEqual(self.actual["tokenizer_source_runtime_runs"],
                         [40462933022,40463929003])

    def test_exact_qwen3_byte_source_and_full_context(self):
        src=self.actual["source"]
        self.assertEqual(src["weight_sha256"],
            "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3")
        self.assertEqual(src["tokenizer_source_sha256"],
            "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03")
        self.assertEqual(src["ids"],{"pad":256,"bos":257,"eos":258,"vocab":259})
        self.assertEqual(src["taps_after_block_zero_based"],[7,15,23])
        self.assertEqual(src["max_context_units"],2048)
        self.assertTrue(self.actual["byte_tokenizer_scope"]["full_2048_context_proven"])

    def test_eos_nonemission_does_not_become_propensity_or_historical_claim(self):
        scope=self.actual["byte_tokenizer_scope"]
        self.assertFalse(scope["historical_byte_eos_effect_measured"])
        self.assertFalse(scope["four_unit_eos_non_emission_is_propensity_evidence"])
        self.assertFalse(scope["future_byte_eos_serializer_integration_in_this_lane"])

    def test_false_pass_training_or_historical_relabel_not_equal_authority(self):
        fake=copy.deepcopy(self.committed)
        fake["per_model_gate"]=True
        fake["adjudication"]="PASS"
        fake["train_authorized"]=True
        fake["blocking_required_checks"]=[]
        fake["byte_tokenizer_scope"]["historical_byte_eos_effect_measured"]=True
        self.assertNotEqual(fake,adjudicator.build_report())


if __name__=="__main__":
    unittest.main()
