"""Original physical-cohort source reconciliation sentinels. NO MODEL/NO TRAIN."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_fourcohort_scientific_delta as delta


class FourCohortScientificBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # All four original physical cohorts are re-audited; no PyTorch loaded.
        cls.original = delta.source_inputs()

    def evaluate(self, mutation=None):
        data = copy.deepcopy(self.original)
        if mutation:
            mutation(data)
        return delta.interpret(data)

    def refused(self, mutation):
        with self.assertRaisesRegex((ValueError, KeyError, TypeError),
                                    "VOID_FOURCOHORT_REFUSED"):
            self.evaluate(mutation)

    def test_all_four_original_custody_and_protocol_preserved(self):
        report = self.evaluate()
        self.assertEqual(report["status"], delta.STATUS)
        self.assertEqual(report["parent_head"], delta.PARENT)
        self.assertEqual(set(report["original_four_physical_cohorts"]), set(delta.RUNS))
        self.assertEqual(report["protocol_git_blob"], delta.PROTOCOL_BLOB)
        self.assertEqual(report["source_pins_git_blob"], delta.PINS_BLOB)
        for cohort, (run, head) in delta.RUNS.items():
            self.assertEqual(report["original_four_physical_cohorts"][cohort]["original_run"], run)
            self.assertEqual(report["original_four_physical_cohorts"][cohort]["original_source_head"], head)
        self.assertEqual(report["original_four_physical_cohorts"]["A_pretraining_short_and_cold"]["optimizer_steps"], 0)
        self.assertEqual(report["original_four_physical_cohorts"]["B_pretraining_maxcontext"]["optimizer_steps"], 0)
        self.assertEqual(report["original_four_physical_cohorts"]["C_one_step_gradient"]["optimizer_steps"], 1)
        self.assertEqual(report["original_four_physical_cohorts"]["D_poststep_cold"]["optimizer_steps"], 1)
        self.assertEqual(report["new_optimizer_steps"], 0)

    def test_nine_check_crosswalk_distinguishes_observation_from_admission(self):
        report = self.evaluate()
        self.assertEqual([x["required_check"] for x in report["nine_required_checks"]],
                         list(delta.NINE))
        self.assertEqual(
            [x["required_check"] for x in report["nine_required_checks"]
             if not x["historical_171_status_unchanged"].startswith("PASS")],
            ["base_weight_unchanged_after_step", "nonzero_adapter_gradient_after_warmup",
             "finite_state_and_loss"])
        self.assertTrue(all(x["final_scientific_check_admitted"] is False
                            for x in report["nine_required_checks"]))
        self.assertEqual(
            report["nine_required_checks"][3]["original_cohorts_with_bounded_observation"],
            ["A_pretraining_short_and_cold", "B_pretraining_maxcontext",
             "D_poststep_cold"])
        self.assertEqual(len(report["scope_limits_still_unproven"]), 7)
        self.assertFalse(report["poststep_2048_full_KV_measured"])
        self.assertFalse(report["poststep_all_31_token_KV_measured"])
        self.assertFalse(report["poststep_masked_batch_and_future_suffix_measured"])

    def test_exact_short_poststep_is_not_extended_2048(self):
        report = self.evaluate()
        assert report["original_four_physical_cohorts"]["B_pretraining_maxcontext"]["max_tested_context_units"] == 2048
        d = report["original_four_physical_cohorts"]["D_poststep_cold"]
        self.assertEqual(d["context_units"], 31)
        self.assertEqual(d["incremental_KV_tail_positions"], 4)
        self.assertEqual(d["incremental_KV_max_abs"], 9.5367431640625e-06)
        self.assertTrue(d["same_runner_second_python"])
        self.assertTrue(d["distinct_original_run_from_C"])
        self.refused(lambda x: x["new_receipt"].__setitem__("context_units", 2048))
        self.refused(lambda x: x["new_receipt"].__setitem__("cache_checked_tail_tokens", 31))
        self.refused(lambda x: x["new_receipt"].__setitem__("poststep_cache_max_abs", 0.25))
        self.refused(lambda x: x["consumer"].__setitem__("incremental_tokens", 2048))
        self.refused(lambda x: x["consumer"].__setitem__("post_step_cold_reload_exact", False))

    def test_c_d_physical_cohorts_cannot_be_collapsed(self):
        self.refused(lambda x: x["addendum"].__setitem__("original_physical_run", 41058648266))
        self.refused(lambda x: x["new_manifest"].__setitem__("run", 40765025332))
        self.refused(lambda x: x["new_manifest"].__setitem__("source_head", delta.RUNS["C_one_step_gradient"][1]))
        self.refused(lambda x: x["producer"].__setitem__("producer_pid",
                                                        x["consumer"]["cold_process_pid"]))
        self.refused(lambda x: x["intermediate"].__setitem__("status",
                                                            "PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED"))

    def test_c_d_shared_public_sample_but_not_same_physical_execution(self):
        self.assertEqual(
            self.original["old_receipt"]["synthetic_sample"]["ids_sha256"],
            self.original["producer"]["sample"]["ids_sha256"])
        self.assertEqual(
            self.original["old_receipt"]["synthetic_sample"]["labels_sha256"],
            self.original["producer"]["sample"]["labels_sha256"])
        self.assertEqual(
            self.original["old_receipt"]["synthetic_sample"]["mask_sha256"],
            self.original["consumer"]["sample"]["mask_sha256"])
        self.refused(lambda x: x["producer"]["sample"].__setitem__("real_eos_id", 62))
        self.refused(lambda x: x["producer"]["sample"].__setitem__("mask_sha256", "f"*64))
        self.refused(lambda x: x["old_receipt"]["synthetic_sample"].__setitem__(
            "labels_sha256", "f"*64))

    def test_forged_training_permission_scientific_gate_and_hist_rewrite_refused(self):
        self.refused(lambda x: x["protocol"].__setitem__("train_authorized", True))
        self.refused(lambda x: x["historical"].__setitem__("per_model_gate", True))
        self.refused(lambda x: x["historical"]["matrix"][
            "nonzero_adapter_gradient_after_warmup"].__setitem__("status", "PASS"))
        self.refused(lambda x: x["addendum"].__setitem__("independent_scientific_review", True))
        self.refused(lambda x: x["new_receipt"].__setitem__("scientific_review_complete", True))
        self.refused(lambda x: x["new_receipt"].__setitem__("full_72_conformance_pass", True))
        self.refused(lambda x: x["new_manifest"]["scientific_boundary"].__setitem__(
            "train_authorized", True))

    def test_source_variant_base_weights_protocol_and_eos_cannot_change(self):
        self.refused(lambda x: x["pins"]["models"]["void1"].__setitem__("repo_id",
                                                                        "kandivault/sprocket-500m"))
        self.refused(lambda x: x["protocol"]["models"]["void1"].__setitem__(
            "taps_after_block_zero_based", [7,16,23]))
        self.refused(lambda x: x["protocol"]["models"]["void1"][
            "tokenizer_reference"]["ids"].__setitem__("eos", 62))
        self.refused(lambda x: x["historical"]["source"].__setitem__(
            "weight_sha256", "f"*64))
        self.refused(lambda x: x["old_receipt"].__setitem__("backbone_after_sha256", "f"*64))
        self.refused(lambda x: x["protocol"]["conformance"].__setitem__(
            "full_vs_cached_prefill_atol", 1e-3))

    def test_legacy_eos_negative_and_no_unmeasured_effect_promotion(self):
        report = self.evaluate()
        self.assertEqual(report["original_four_physical_cohorts"][
            "B_pretraining_maxcontext"]["generation_records_without_eos_258"], 12)
        self.assertIs(report["generated_eos_skill_qualified"], False)
        self.assertIs(report["historical_train002b_eos_258_effect_measured"], False)
        self.assertIs(report["comparative_nca_mlp_training_performed"], False)
        self.assertIs(report["per_model_conformance_pass"], False)
        self.assertIs(report["full_72_conformance_pass"], False)
        self.assertIs(report["train_authorized"], False)
        self.refused(lambda x: x["historical"]["evidence"].__setitem__(
            "actual_eos_258_emitted_in_four_step_probe", True))
        self.refused(lambda x: x["historical"]["evidence"].__setitem__(
            "max_context_generation_records", 0))

    def test_immutable_committed_scientific_report_matches_originals_when_present(self):
        if not delta.OUTPUT.is_file():
            self.skipTest("machine report not yet committed on initial producer commit")
        committed = json.loads(delta.OUTPUT.read_text())
        self.assertEqual(committed, self.evaluate())
        self.assertIs(committed["independent_scientific_72_admission"], False)


if __name__ == "__main__":
    unittest.main()
