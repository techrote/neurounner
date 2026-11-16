"""Cheap offline tests for the original native guidance rank calibration."""
import copy
import unittest

from experiments.driv3r.stage_d.guidance.native_guidance_rank import matching_prefix, score_arm


def turn(index=0, hints=None, choice="MOVE_N", probabilities=None):
    hints = {"MOVE_N": 20000, "MOVE_E": 10000} if hints is None else hints
    probabilities = ({"MOVE_N": .4, "MOVE_E": .2, "STOP": .4}
                     if probabilities is None else probabilities)
    return {"index": index, "chosen_candidate": choice,
            "exact_candidate_ids_in_presented_order": list(probabilities),
            "host_legal_candidate_sha256": "a" * 64,
            "parsed_choice": {"candidate_id": choice, "probabilities": probabilities},
            "model_state": {"candidate_hints": {k: {"soft_scent_u16": v} for k, v in hints.items()},
                            "position": [1, 2], "recent_positions": [[1, 2]],
                            "native_world_material_sha256": "b" * 64}}


class RankTests(unittest.TestCase):
    def test_ties_use_correct_uniform_legal_null_and_midrank(self):
        a = score_arm([turn(hints={"MOVE_N": 9, "MOVE_E": 9})])
        self.assertEqual(a["uniform_legal_move_expected_highest_count"], 1)
        self.assertEqual(a["mean_chosen_tie_aware_percentile"], .5)
        self.assertEqual(a["highest_count_minus_uniform_expectation"], 0)

    def test_guidance_ranking_is_not_confidence_ranking(self):
        a = score_arm([turn(choice="MOVE_E", probabilities={"MOVE_E": .6, "MOVE_N": .2, "STOP": .2})])
        self.assertEqual(a["chosen_highest_guide_count"], 0)
        self.assertEqual(a["highest_count_minus_uniform_expectation"], -.5)
        self.assertEqual(a["mean_chosen_tie_aware_percentile"], .25)

    def test_recovery_excluded_not_counted_as_zero_guidance(self):
        a = score_arm([turn(), turn(index=1, choice="BACKTRACK", probabilities={
            "MOVE_N": .3, "MOVE_E": .3, "BACKTRACK": .4})])
        self.assertEqual(a["painted_move_count"], 1)
        self.assertEqual(a["uniform_legal_move_expected_highest_count"], .5)

    def test_all_zero_free_guide_is_uninformative(self):
        a = score_arm([turn(hints={"MOVE_N": 0, "MOVE_E": 0})])
        self.assertEqual(a["chosen_highest_guide_count"], 1)
        self.assertEqual(a["uniform_legal_move_expected_highest_count"], 1)

    def test_forged_missing_legal_move_hint_rejected(self):
        with self.assertRaisesRegex(ValueError, "legal move hints"):
            score_arm([turn(hints={"MOVE_N": 10})])

    def test_invalid_probabilities_and_scalar_rejected(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            score_arm([turn(probabilities={"MOVE_N": 1.2, "MOVE_E": -.2, "STOP": 0})])
        with self.assertRaisesRegex(ValueError, "scalar"):
            score_arm([turn(hints={"MOVE_N": -1, "MOVE_E": 10})])

    def test_matched_prefix_stops_before_world_divergence(self):
        a = turn(); b = copy.deepcopy(a)
        b["parsed_choice"]["probabilities"] = {"MOVE_N": .3, "MOVE_E": .3, "STOP": .4}
        a1 = turn(index=1); b1 = copy.deepcopy(a1)
        b1["model_state"]["native_world_material_sha256"] = "c" * 64
        x = matching_prefix([a, a1], [b, b1])
        self.assertEqual(x["matched_initial_decision_state_count"], 1)
        self.assertEqual(x["matched_prefix"][0]["largest_absolute_option_probability_shift"], .1)

    def test_first_choice_divergence_only_when_state_matched(self):
        a = turn(); b = turn(choice="MOVE_E")
        x = matching_prefix([a], [b])
        self.assertEqual(x["first_choice_divergence_in_matched_prefix"], 0)
        self.assertFalse(x["matched_prefix"][0]["same_choice"])


if __name__ == "__main__":
    unittest.main()
