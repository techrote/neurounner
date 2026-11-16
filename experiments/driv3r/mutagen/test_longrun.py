"""M1 cheap non-model tests: budgets, trace counting, and missing-artifact honesty."""
from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.driv3r.mutagen import longrun,spoon,compare_longrun


class LongRunTests(unittest.TestCase):
    def test_budgets_and_matched_worlds(self):
        self.assertEqual(longrun.DEFAULT_SEEDS,(570,2111))
        self.assertEqual(longrun.BUDGETS["stock"],128)
        self.assertEqual(longrun.BUDGETS["mlp_scoop"],128)
        self.assertEqual(longrun.BUDGETS["attn_gouge"],128)
        self.assertEqual(longrun.BUDGETS["conv_scrape_10"],64)
        self.assertEqual(longrun.BUDGETS["conv_scrape_25"],64)
        self.assertEqual(len(compare_longrun.ARMS),5)

    def test_gentle_doses_same_anatomy_as_destructive_conv_control(self):
        former=spoon.SPECS["conv_scrape"]
        for arm,frac in (("conv_scrape_10",.10),("conv_scrape_25",.25)):
            derived=spoon.SPECS[arm]
            self.assertEqual(derived["fraction"],frac)
            self.assertEqual(derived["layers"],former["layers"])
            self.assertEqual(derived["op"],"zero")
            self.assertTrue(set(derived["suffixes"]).issubset(former["suffixes"]))

    def test_raw_choices_repeat_and_stop_are_preserved(self):
        records=[
          {"status":"ok","original_selected_action":"LEFT","latency_seconds":1.0},
          {"status":"ok","original_selected_action":"LEFT","latency_seconds":2.0},
          {"status":"ok","original_selected_action":"STOP","latency_seconds":3.0},
        ]
        events=[{"applied":"LEFT"},{"applied":"LEFT"},{"applied":"STOP"}]
        got=longrun.actions_summary(records,events,64)
        self.assertEqual(got["max_identical_action_streak"],2)
        self.assertEqual(got["raw_choices"],["LEFT","LEFT","STOP"])
        self.assertTrue(got["stop_selected"])
        self.assertEqual(got["valid_typed_decisions"],3)
        self.assertEqual(got["model_latency_p50_seconds"],2.)
        self.assertEqual(got["model_latency_p95_seconds"],2.9)

    def test_failed_original_model_result_is_not_accepted(self):
        got=longrun.actions_summary([
            {"status":"original_response_invalid","latency_seconds":1.2}
        ],[],64)
        self.assertEqual(got["valid_typed_decisions"],0)
        self.assertEqual(got["max_identical_action_streak"],0)
        self.assertEqual(got["action_histogram"]["RIGHT"],0)

    def test_empty_comparison_is_not_a_false_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/"results"
            result=compare_longrun.compare(root/"missing","a"*40,out)
            self.assertFalse(result["all_arms_complete"])
            self.assertEqual(len(result["episode_comparisons"]),6)
            self.assertEqual(result["matched_arms"]["stock"]["completed_episodes"],0)
            self.assertTrue((out/"comparison.json").is_file())


if __name__=="__main__":
    unittest.main()
