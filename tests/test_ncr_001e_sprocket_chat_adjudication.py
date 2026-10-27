"""Fail-closed deterministic Sprocket Chat #72 adjudication tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

import ncr_001e_sprocket_chat_adjudication as adjudicator  # noqa:E402


class ChatAdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=adjudicator.build_report()
        cls.committed=json.loads(
            (ROOT/"experiments/ncr-001/adjudication/sprocket-chat-pretrain.json")
            .read_text(encoding="utf-8"))

    def test_regenerated_report_exactly_matches_committed_authority(self):
        self.assertEqual(self.report,self.committed)

    def test_chat_gate_remains_false_and_no_training(self):
        self.assertEqual(
            self.report["adjudication"],
            "BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS")
        self.assertFalse(self.report["per_model_gate"])
        self.assertTrue(self.report["source_runtime_pretraining_subgate"])
        self.assertFalse(self.report["train_authorized"])
        self.assertEqual(self.report["optimizer_steps_observed"],0)
        self.assertFalse(self.report["heldout_evaluation"])

    def test_exactly_three_learning_update_checks_block_final_pass(self):
        self.assertEqual(set(self.report["blocking_required_checks"]),{
            "base_weight_unchanged_after_step",
            "nonzero_adapter_gradient_after_warmup",
            "finite_state_and_loss",
        })
        self.assertEqual(
            self.report["matrix"]["base_weight_unchanged_after_step"]["status"],
            "BLOCKED_BY_TRAINING_POLICY")
        self.assertEqual(
            self.report["matrix"]["nonzero_adapter_gradient_after_warmup"]["status"],
            "BLOCKED_BY_TRAINING_POLICY")
        self.assertEqual(
            self.report["matrix"]["finite_state_and_loss"]["status"],
            "PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY")

    def test_source_runtime_required_checks_have_original_positive_evidence(self):
        for name in (
            "same_base_logits","causal_suffix_invariance",
            "full_prefix_vs_incremental_cache","per_token_isolation",
            "per_request_reset","cold_reload_state_dict"):
            with self.subTest(check=name):
                row=self.report["matrix"][name]
                self.assertTrue(row["status"].startswith("PASS"))
                self.assertTrue(row["evidence"])

    def test_only_original_actual_model_runs_are_runtime_evidence(self):
        self.assertEqual(
            self.report["evidence_runs"],
            [40469703065,40515083104,40544384293,40603226200])
        for archive_only in (40603726115,40603770209,40603868442):
            self.assertNotIn(archive_only,self.report["evidence_runs"])
        self.assertEqual(self.report["preserved_negative_runs"],[40603167288])

    def test_chat_does_not_inherit_instruct_tool_memory_semantics(self):
        scope=self.report["source_semantic_scope"]
        self.assertTrue(scope["ordinary_user_assistant_role_format_verified"])
        self.assertFalse(scope["instruct_tool_memory_grammar_inherited"])
        self.assertFalse(scope["chat_tool_memory_semantics_qualified"])
        self.assertTrue(scope["publisher_chat_excludes_tool_memory_finetuning"])

    def test_chat_instruct_checkpoint_identity_remains_distinct(self):
        src=self.report["source"]
        self.assertEqual(src["weight_file"],"chat/model.safetensors")
        self.assertEqual(
            src["weight_sha256"],
            "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97")
        self.assertEqual(
            src["distinct_instruct_weight_sha256"],
            "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864")
        self.assertNotEqual(src["weight_sha256"],
                            src["distinct_instruct_weight_sha256"])
        self.assertEqual(src["taps_after_block_zero_based"],[7,16,25])

    def test_false_pass_or_training_relabel_cannot_equal_regenerated_report(self):
        fake=copy.deepcopy(self.committed)
        fake["per_model_gate"]=True
        fake["adjudication"]="PASS"
        fake["train_authorized"]=True
        fake["blocking_required_checks"]=[]
        self.assertNotEqual(fake,adjudicator.build_report())


if __name__=="__main__":
    unittest.main()
