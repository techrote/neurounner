"""No-model adversarial tests for the bounded original Sprocket three-cohort delta.

Original B2, B1 and NCA-only diagnostic evidence must remain distinct, with the
original raw ZIP/job-log auditors run first. No training, HF download or heldouts.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_sprocket_three_cohort_delta as gate


class ThreeCohortSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior, cls.masked, cls.streaming = gate.live_originals(
            Path(os.environ.get("NCR_001_PRIOR_ROOT", "prior-204")).resolve())

    def changed(self, target, modifier):
        p = copy.deepcopy(self.prior)
        m = copy.deepcopy(self.masked)
        s = copy.deepcopy(self.streaming)
        record = {"prior": p, "masked": m, "streaming": s}[target]
        modifier(record)
        with self.assertRaisesRegex(ValueError, "NCR001_THREE_COHORT_REFUSED"):
            gate.compose(p, m, s)

    def test_exact_published_report_regenerates_and_remains_blocked(self):
        actual = gate.compose(self.prior, self.masked, self.streaming)
        expected = json.loads(gate.PUBLISHED.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertFalse(actual["train_authorized"])
        self.assertFalse(actual["full_72_conformance_pass"])
        self.assertFalse(actual["actual_compute_budget_qualified"])
        for lane in actual["lanes"].values():
            self.assertFalse(lane["combined_b1_b2_cold_and_learning_same_physical_cohort"])
            self.assertFalse(lane["per_model_72_scientific_pass"])

    def test_refuse_masked_source_head_substitution(self):
        self.changed("masked", lambda x: x.update(source_sha="invented"))

    def test_refuse_missing_original_masked_zip_log(self):
        self.changed("masked", lambda x: x.update(original_zip_count=2))

    def test_refuse_original_alltoken_job_run_substitution(self):
        self.changed("streaming", lambda x: x.update(original_run=0))

    def test_refuse_chat_root_weight_conflation(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_chat"].update(
            frozen_weight_sha256=gate.MODEL["sprocket_instruct"][2]))

    def test_refuse_b2_finalfour_relabelled_all2048(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_chat"]["observed_bounded"].update(
            actual_one_token_kv_increments=2048))

    def test_refuse_b2_short_row_claimed_full2048_real(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_instruct"]["observed_bounded"].update(
            short_row_true_tokens=2048))

    def test_refuse_b2_invalid_gap_claimed_publisher_wire(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_chat"]["partial_or_unobserved"].update(
            right_pad_gap_publisher_valid=True))

    def test_refuse_cold_process_parity_missing(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_instruct"]["observed_bounded"].update(
            cold_new_python_same_runner_last_four_logits_exact=False))

    def test_refuse_b1_stream_max_abs_changed(self):
        self.changed("streaming", lambda x: x["lanes"]["chat"].update(max_abs=0.0006))

    def test_refuse_b1_ten_uncached_checkpoints_relabelled_allpositions(self):
        self.changed("streaming", lambda x: x["lanes"]["instruct"].update(uncached_prefixes=2048))

    def test_refuse_separate_gradient_diagnostic_missing(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_chat"]["observed_bounded"].update(
            postwarmup_adapter_gradient_nonzero=False))

    def test_refuse_tool_memory_competence_promoted(self):
        self.changed("prior", lambda x: x["lanes"]["sprocket_instruct"]["partial_or_unobserved"].update(
            generated_publisher_tool_result_memory_semantics="ADMITTED"))

    def test_refuse_training_authorization(self):
        self.changed("prior", lambda x: x.update(train_authorized=True))

    def test_refuse_scientific_admission(self):
        self.changed("streaming", lambda x: x.update(full_72_conformance_pass=True))

    def test_refuse_original_diagnostic_run_swapped(self):
        self.changed("prior", lambda x: x.update(one_step_diagnostic_run=0))


if __name__ == "__main__":
    unittest.main()
