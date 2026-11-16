"""Cheap source-only regression tests: calibrated IQ3 held-out geometry triad."""
import math
import unittest

from experiments.driv3r.frankenlayers_lab.counterfactual import PAIRS, SEED, STEPS
from experiments.driv3r.frankenlayers_lab.infer_lowbit_causal import (
    MAX_DECISIONS, make_report, pareto_axis,
)


def example_arm(decisions=200, invalid=0, stops=1, exposed=2, responded=1):
    return {"summary": {
        "decisions": decisions, "new_cells": 90,
        "invalid_host_choices": invalid, "early_stop_episodes": stops,
        "geometry_exposed_pairs": exposed,
        "observation_followed_by_action_divergence_pairs": responded,
        "unexplained_early_divergence_pairs": 0,
        "choice_counts": {
            "BRANCH": decisions - 20, "RETRACE": 20,
            "STRAIGHT": 0, "LEFT": 0, "RIGHT": 0, "STOP": 0,
        },
        "median_of_variant_p50_seconds": 0.4,
    }, "pairs": {}}


class IQ3CausalTests(unittest.TestCase):
    def test_fixed_disjoint_geometry_budget(self):
        self.assertEqual(SEED, 2718)
        self.assertEqual(STEPS, 36)
        self.assertEqual(MAX_DECISIONS, 216)
        self.assertEqual(len(PAIRS), 3)

    def test_pareto_exposes_stops_and_budget_penalty(self):
        axes = pareto_axis(example_arm())
        self.assertEqual(axes["decisions"], 200)
        self.assertEqual(axes["decision_budget"], 216)
        self.assertAlmostEqual(axes["new_cells_per_decision"], 0.45)
        self.assertAlmostEqual(axes["new_cells_per_budget"], 90 / 216)
        self.assertEqual(axes["early_stop_episodes"], 1)
        self.assertEqual(axes["responded_exposed_pairs"], 1)
        self.assertLess(axes["normalised_action_entropy"], 1)
        self.assertGreater(axes["action_entropy_bits"], 0)

    def test_no_spurious_score_without_decisions(self):
        a = example_arm(decisions=0)
        a["summary"]["choice_counts"] = {key: 0 for key in a["summary"]["choice_counts"]}
        ax = pareto_axis(a)
        self.assertIsNone(ax["legal_choice_fraction"])
        self.assertIsNone(ax["new_cells_per_decision"])
        self.assertEqual(ax["action_entropy_bits"], 0)

    def test_all_controls_and_caveats_preserved(self):
        a = example_arm()
        report = make_report("source-sha", "aero4_no10", {"quant": "IQ3_XS"},
                             "w0-sha", a, a, a)
        self.assertEqual(report["flow"], [0, 2, 4, 20])
        self.assertEqual(set(report["arms"]), {"q8_stock", "iq3_stock", "iq3_mutant"})
        self.assertEqual(set(report["pareto_axes"]), set(report["arms"]))
        self.assertAlmostEqual(report["ratio_vs_q8_stock_per_decision"], 1.0)
        self.assertFalse(report["real_training"])
        self.assertFalse(report["game_code_modified"])
        self.assertFalse(report["native_workbench"])
        self.assertIn("not goal competence", report["limitations"])

if __name__ == "__main__":
    unittest.main()
