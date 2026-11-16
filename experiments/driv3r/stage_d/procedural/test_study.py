"""Synthetic weathering comparison custody and negative-control checks."""
import unittest

from lab import VERSION
from study import compare_weathering


class StudyChecks(unittest.TestCase):
    def setUp(self):
        self.recipe=dict(version=VERSION,family="weathering_worm",canvas=[48,48],
                         seed=29,steps=40,brush_radius=1,depth=1,
                         boundary_offset=2,recovery="relocate")

    def test_repeatable_matched_start_and_isolated_policies(self):
        a=compare_weathering(self.recipe)
        b=compare_weathering(dict(self.recipe))
        self.assertEqual(a,b)
        self.assertEqual(a["qualification"],"DEVELOPMENT_SYNTHETIC_ONLY")
        self.assertIsNone(a["native"])
        self.assertIsNone(a["model"])
        self.assertFalse(a["native_replay_verified"])
        self.assertEqual(set(a["comparisons"]),{"boundary_biased","random_legal"})

    def test_actual_material_counts_and_offsets(self):
        result=compare_weathering(self.recipe)
        for arm in result["comparisons"].values():
            m=arm["metrics"]
            self.assertGreater(m["strokes"],0)
            self.assertGreater(m["distinct_erased_cells"],0)
            self.assertGreaterEqual(m["original_rock_retained"],0)
            self.assertGreaterEqual(m["mean_cursor_offset_error"],0)
            self.assertLessEqual(m["erase_per_stroke"],13)

    def test_source_and_budget_sensitivity(self):
        a=compare_weathering(self.recipe)
        b=compare_weathering(dict(self.recipe,steps=45))
        self.assertNotEqual(a["base_recipe_sha256"],b["base_recipe_sha256"])
        self.assertEqual(a["matched_initial_material_sha256"],
                         b["matched_initial_material_sha256"])
        self.assertEqual(a["matched_guidance_sha256"],b["matched_guidance_sha256"])

    def test_denies_unsupported_families(self):
        with self.assertRaises(AssertionError):
            compare_weathering(dict(self.recipe,family="branching_growth"))


if __name__=="__main__":
    unittest.main()
