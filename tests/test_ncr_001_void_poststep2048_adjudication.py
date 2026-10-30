"""Adversarial source-only tests for poststep2048 void.1 adjudication."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_void_poststep2048_adjudication as adjudication


class Poststep2048AdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = adjudication.source_inputs()

    def evaluate(self, mutation=None):
        data = copy.deepcopy(self.inputs)
        if mutation:
            mutation(data)
        return adjudication.interpret(data)

    def refused(self, mutation):
        with self.assertRaisesRegex(
                (ValueError, KeyError, TypeError),
                "VOID_POSTSTEP2048_ADJUDICATION_REFUSED"):
            self.evaluate(mutation)

    def test_new_fifth_cohort_closes_prior_joint_2048_gap_but_not_admission(self):
        r = self.evaluate()
        self.assertEqual(r["status"], adjudication.STATUS)
        self.assertEqual(r["parent_head"], adjudication.PARENT)
        self.assertEqual(r["largest_prior_joint_gap_poststep_2048"],
                         "CLOSED_BY_COHORT_E")
        e = r["new_fifth_physical_cohort"]
        self.assertEqual(e["original_run"], 41133450850)
        self.assertEqual(e["original_source_head"],
                         "520c4ae5b3152aff8d94760dab2e3103746f7334")
        self.assertEqual(e["poststep_context_units"], 2048)
        self.assertEqual(e["incremental_positions"], 2048)
        self.assertTrue(e["all_positions_compared"])
        self.assertAlmostEqual(e["cache_max_abs"],
                               7.522106170654297e-05)
        self.assertTrue(e["future_suffix_earlier_exact"])
        self.assertEqual((e["generation_prefix_units"],
                          e["generation_new_units"]), (2044,4))
        self.assertEqual(e["generation_ids"], [105,110,115,112])
        self.assertFalse(e["generation_actual_eos_258_emitted"])
        self.assertTrue(e["permanent_original_zip_log_custody"])
        self.assertFalse(r["per_model_conformance_pass"])
        self.assertFalse(r["independent_scientific_72_admission"])
        self.assertFalse(r["train_authorized"])

    def test_all_nine_have_bounded_observations_but_historical_171_is_immutable(self):
        r = self.evaluate()
        self.assertTrue(r["all_nine_have_bounded_source_bound_physical_observations"])
        self.assertEqual(len(r["nine_required_checks"]), 9)
        self.assertTrue(all(x["bounded_observation_present"]
                            for x in r["nine_required_checks"]))
        self.assertTrue(all(x["final_scientific_check_admitted"] is False
                            for x in r["nine_required_checks"]))
        hist = {x["required_check"]: x["historical_171_status_unchanged"]
                for x in r["nine_required_checks"]}
        self.assertEqual(hist["base_weight_unchanged_after_step"],
                         "BLOCKED_BY_TRAINING_POLICY")
        self.assertEqual(hist["nonzero_adapter_gradient_after_warmup"],
                         "BLOCKED_BY_TRAINING_POLICY")
        self.assertEqual(hist["finite_state_and_loss"],
                         "PARTIAL_BLOCKED_WARMUP_LOSS")
        self.assertEqual(hist["full_prefix_vs_incremental_cache"], "PASS")

    def test_masked_full_forward_is_not_falsely_called_masked_cached_b2(self):
        r = self.evaluate()
        self.assertTrue(r["masked_B2_full_forward_isolation_measured"])
        self.assertFalse(r["masked_B2_incremental_cache_measured"])
        e = r["new_fifth_physical_cohort"]
        self.assertFalse(e["masked_batch_incremental_cache_measured"])
        self.assertEqual(e["masked_batch_full_forward"]["batch_size"], 2)
        self.assertEqual(e["masked_batch_full_forward"]["width"], 2048)
        for side in ("left","right"):
            self.assertEqual(
                e["masked_batch_full_forward"][side]["max_batch_vs_single_abs"], 0.0)
            self.assertEqual(
                e["masked_batch_full_forward"][side]["max_row_swap_abs"], 0.0)

    def test_remaining_scientific_gaps_are_explicit_and_not_new_protocol_rules(self):
        r = self.evaluate()
        self.assertEqual(set(r["remaining_precise_scientific_gaps"]),
                         set(adjudication.REMAINING))
        self.assertEqual(
            set(r["remaining_gaps_not_retroactive_protocol_requirements"]),
            set(adjudication.NOT_RETROACTIVE_GATES))
        self.assertFalse(r["fresh_independent_host_poststep_replication_measured"])
        self.assertFalse(r["multi_sample_learning_effectiveness_measured"])
        self.assertFalse(r["matched_nca_mlp_training_quality_measured"])
        self.assertFalse(r["matched_training_compute_qualified"])
        self.assertFalse(r["generated_eos_skill_qualified"])
        self.assertFalse(r["historical_train002b_eos_258_effect_measured"])

    def test_false_final_admission_or_training_is_refused(self):
        mutations = (
            lambda x: x["protocol"].__setitem__("train_authorized", True),
            lambda x: x["prior"].__setitem__("per_model_conformance_pass", True),
            lambda x: x["prior"].__setitem__(
                "independent_scientific_72_admission", True),
            lambda x: x["new"].__setitem__("full_72_conformance_pass", True),
            lambda x: x["new"].__setitem__("train_authorized", True),
            lambda x: x["manifest"]["scientific_boundary"].__setitem__(
                "per_model_gate", True),
            lambda x: x["consumer"].__setitem__("full_72_conformance_pass", True),
            lambda x: x["producer"].__setitem__("independent_scientific_review", True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    (ValueError,KeyError,TypeError),
                    "VOID_POSTSTEP2048_ADJUDICATION_REFUSED"):
                self.evaluate(mutation)

    def test_new_physical_result_cannot_be_shortened_relabelled_or_relaxed(self):
        mutations = (
            lambda x: x["new"]["physical_result"].__setitem__(
                "poststep_2048_incremental_positions", 4),
            lambda x: x["new"]["physical_result"].__setitem__(
                "poststep_2048_cache_max_abs", 0.1),
            lambda x: x["new"]["physical_result"].__setitem__(
                "future_suffix_earlier_exact", False),
            lambda x: x["new"]["physical_result"]["masked_batch"]["left"].__setitem__(
                "max_batch_vs_single_abs", 0.01),
            lambda x: x["new"]["physical_result"].__setitem__(
                "generate_prefix_units", 512),
            lambda x: x["new"]["physical_result"].__setitem__(
                "generated_ids", [258,258,258,258]),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    (ValueError,KeyError,TypeError),
                    "VOID_POSTSTEP2048_ADJUDICATION_REFUSED"):
                self.evaluate(mutation)

    def test_all_position_trace_and_one_step_learning_mechanics_are_bound(self):
        self.refused(lambda x: x["consumer"]["poststep_2048_all_position_cache"]
                     .__setitem__("incremental_positions", 2047))
        self.refused(lambda x: x["consumer"]["poststep_2048_all_position_cache"]
                     ["trace_checkpoints"][0]["trace"].__setitem__(
                         "state_shape", [2,1,4,8,8]))
        self.refused(lambda x: x["producer"]["diagnostic_optimizer_steps"].__class__)
        # The explicit boolean-as-int case:
        self.refused(lambda x: x["producer"].__setitem__(
            "diagnostic_optimizer_steps", True))
        self.refused(lambda x: x["producer"]["learning_summary"]["checks"].__setitem__(
            "post_step_upstream_gradients", False))
        self.refused(lambda x: x["learning"].__setitem__(
            "status", "PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED"))

    def test_source_model_eos_protocol_and_prior_decision_cannot_change(self):
        self.refused(lambda x: x["pins"]["models"]["void1"].__setitem__(
            "repo_id", "kandivault/sprocket-500m"))
        self.refused(lambda x: x["protocol"]["models"]["void1"][
            "tokenizer_reference"]["ids"].__setitem__("eos", 62))
        self.refused(lambda x: x["protocol"]["conformance"].__setitem__(
            "full_vs_cached_prefill_atol", 0.001))
        self.refused(lambda x: x["prior"].__setitem__(
            "historical_171_adjudication_unchanged", "PASS"))
        self.refused(lambda x: x["manifest"].__setitem__(
            "source_head", "f"*40))

    def test_git_tree_object_type_mode_is_pinned_for_input_decisions(self):
        adjudication.require_blob(
            adjudication.PRIOR_DECISION, adjudication.PRIOR_DECISION_BLOB)
        rel=adjudication.PRIOR_DECISION.relative_to(adjudication.ROOT).as_posix()
        for line in (
            f"120000 blob {adjudication.PRIOR_DECISION_BLOB}\t{rel}\n",
            f"100755 blob {adjudication.PRIOR_DECISION_BLOB}\t{rel}\n",
            f"100644 blob {'f'*40}\t{rel}\n",
        ):
            fake=mock.Mock(returncode=0,stdout=line)
            with self.subTest(line=line), mock.patch.object(
                    adjudication.subprocess,"run",return_value=fake):
                with self.assertRaisesRegex(ValueError,"REFUSED"):
                    adjudication.require_blob(
                        adjudication.PRIOR_DECISION,
                        adjudication.PRIOR_DECISION_BLOB)

    def test_committed_machine_decision_matches_originals_when_present(self):
        if not adjudication.OUTPUT.is_file():
            self.skipTest("derived decision not committed on first source pass")
        committed=json.loads(adjudication.OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(committed,self.evaluate())
        self.assertFalse(committed["per_model_conformance_pass"])
        self.assertFalse(committed["train_authorized"])


if __name__=="__main__":
    unittest.main()
