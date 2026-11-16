"""Source-only regression for Wave-14 Q4 matched-geometry research; no model downloads."""
import unittest

from experiments.driv3r.frankenlayers_lab.counterfactual import SEED, STEPS, PAIRS
from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, WAVE14_FLOWS, WAVE15_FLOWS, validate_flows
from experiments.driv3r.frankenlayers_lab.infer_q4_causal import MODES, MAX_DECISIONS, make_report, pareto_axis


def arm(n=36, stop=0):
    return {"summary": {
        "decisions": n, "new_cells": 12, "invalid_host_choices": 0,
        "early_stop_episodes": stop, "geometry_exposed_pairs": 2,
        "observation_followed_by_action_divergence_pairs": 1,
        "unexplained_early_divergence_pairs": 0,
        "choice_counts": {"BRANCH": n-1, "RETRACE": 1, "LEFT": 0,
                          "RIGHT": 0, "STRAIGHT": 0, "STOP": 0},
        "median_of_variant_p50_seconds": 0.4,
    }, "pairs": {}}


class Wave14Q4Tests(unittest.TestCase):
    def test_immutable_candidate_contract(self):
        validate_flows()
        self.assertEqual(SEED, 2718)
        self.assertEqual(STEPS, 36)
        self.assertEqual(MAX_DECISIONS, 216)
        self.assertEqual(len(PAIRS), 3)
        self.assertEqual(set(MODES), {"aero4_no2", "aero4_bridge", "aero4_no2_tailflip", "aero5_no2_echo10"})
        self.assertEqual(WAVE14_FLOWS, {"aero4_bridge": (0, 2, 10, 20)})
        self.assertEqual(WAVE15_FLOWS, {
            "aero4_no2_tailflip": (0, 4, 20, 10),
            "aero5_no2_echo10": (0, 4, 10, 20, 10),
        })
        self.assertEqual(len(set(WAVE15_FLOWS["aero5_no2_echo10"])), 4)
        self.assertNotEqual(tuple(sorted(WAVE15_FLOWS["aero4_no2_tailflip"])), WAVE15_FLOWS["aero4_no2_tailflip"])
        self.assertEqual(LAB_FLOWS["aero4_no2"], (0, 4, 10, 20))

    def test_derived_geometry_and_terminal_axes(self):
        s=pareto_axis(arm(36,1))
        self.assertEqual(s["decisions"],36)
        self.assertAlmostEqual(s["new_cells_per_budget"],12/216)
        self.assertEqual(s["early_stop_episodes"],1)
        self.assertEqual(s["responded_exposed_pairs"],1)

    def test_three_arm_stock_controls_and_no_training_claim(self):
        data=arm()
        for mode in MODES:
            with self.subTest(mode=mode):
                r=make_report("source",mode,{"quant":"Q4_K_M"},
                              "host",data,data,data)
                self.assertEqual(r["flow"],list(LAB_FLOWS[mode]))
                self.assertEqual(set(r["arms"]),{"q8_stock","q4_stock","q4_mutant"})
                self.assertFalse(r["real_training"])
                self.assertFalse(r["native_workbench"])
                self.assertFalse(r["game_code_modified"])
                self.assertIn("not goal competence",r["limitations"])


if __name__ == "__main__":
    unittest.main()
