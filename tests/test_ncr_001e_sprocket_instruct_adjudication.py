"""Deterministic fail-closed Sprocket Instruct #72 adjudication tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ncr_001e_sprocket_instruct_adjudication as adjudicator  # noqa:E402


class SprocketInstructAdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = adjudicator.build_report()
        cls.committed = json.loads(
            (ROOT / "experiments/ncr-001/adjudication/"
                    "sprocket-instruct-pretrain.json").read_text(encoding="utf-8")
        )

    def test_regenerated_report_exactly_matches_committed_authority(self):
        self.assertEqual(self.report, self.committed)

    def test_lane_remains_blocked_and_training_is_not_authorized(self):
        self.assertEqual(
            self.report["adjudication"],
            "BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS",
        )
        self.assertFalse(self.report["per_model_gate"])
        self.assertTrue(self.report["source_runtime_pretraining_subgate"])
        self.assertFalse(self.report["train_authorized"])
        self.assertEqual(self.report["optimizer_steps_observed"], 0)
        self.assertFalse(self.report["heldout_evaluation"])

    def test_exactly_three_required_learning_update_checks_block_final_pass(self):
        self.assertEqual(
            set(self.report["blocking_required_checks"]),
            {
                "base_weight_unchanged_after_step",
                "nonzero_adapter_gradient_after_warmup",
                "finite_state_and_loss",
            },
        )
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
            "PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY",
        )

    def test_source_runtime_required_checks_have_positive_evidence(self):
        for name in (
            "same_base_logits",
            "causal_suffix_invariance",
            "full_prefix_vs_incremental_cache",
            "per_token_isolation",
            "per_request_reset",
            "cold_reload_state_dict",
        ):
            with self.subTest(check=name):
                row = self.report["matrix"][name]
                self.assertTrue(row["status"].startswith("PASS"))
                self.assertTrue(row["evidence"])

    def test_evidence_runs_are_original_model_runs_not_archive_only_counters(self):
        self.assertEqual(
            self.report["evidence_runs"],
            [
                40469703065,
                40515083104,
                40544384293,
                40567832272,
                40573188117,
                40590487839,
                40598034739,
            ],
        )
        self.assertNotIn(40599803975, self.report["evidence_runs"])
        self.assertNotIn(40600186016, self.report["evidence_runs"])

    def test_tool_memory_diagnostic_negatives_do_not_become_conformance_pass(self):
        limits = self.report["nonblocking_scope_limits"]
        self.assertFalse(limits["model_generated_tool_memory_competence_qualified"])
        self.assertFalse(limits["comparative_NCA_MLP_training_performed"])
        self.assertFalse(limits["sealed_holdouts_opened"])
        self.assertFalse(limits["all_parent_prs_merged"])

    def test_distinct_chat_instruct_source_identity_is_preserved(self):
        source = self.report["source"]
        self.assertEqual(source["weight_file"], "model.safetensors")
        self.assertEqual(
            source["weight_sha256"],
            "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
        )
        self.assertEqual(
            source["distinct_chat_weight_sha256"],
            "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97",
        )
        self.assertNotEqual(
            source["weight_sha256"], source["distinct_chat_weight_sha256"]
        )
        self.assertEqual(source["taps_after_block_zero_based"], [7, 16, 25])

    def test_committed_report_cannot_be_relabelled_as_pass_without_mismatch(self):
        fake = copy.deepcopy(self.committed)
        fake["per_model_gate"] = True
        fake["adjudication"] = "PASS"
        fake["blocking_required_checks"] = []
        self.assertNotEqual(fake, adjudicator.build_report())

    def test_raw_cross_process_archive_corruption_refuses_final_adjudication(self):
        # The original metadata can still look valid; corrupt one actual
        # lossless Actions archive and prove source provenance alone is not enough.
        import ncr_001e_sprocket_process_receipt_audit as process_audit
        original = process_audit.ARCHIVE.read_text(encoding="ascii")
        with TemporaryDirectory() as dirname:
            corrupted = Path(dirname) / "original-jsons.tar.gz.b64"
            corrupted.write_text("!" + original[1:], encoding="ascii")
            with patch.object(process_audit, "ARCHIVE", corrupted):
                with self.assertRaises((RuntimeError, ValueError, OSError)):
                    adjudicator.build_report()

    def test_missing_original_cross_process_archive_refuses_adjudication(self):
        import ncr_001e_sprocket_process_receipt_audit as process_audit
        with TemporaryDirectory() as dirname:
            absent = Path(dirname) / "missing-original-actions-archive.b64"
            with patch.object(process_audit, "ARCHIVE", absent):
                with self.assertRaises(FileNotFoundError):
                    adjudicator.build_report()


if __name__ == "__main__":
    unittest.main()
