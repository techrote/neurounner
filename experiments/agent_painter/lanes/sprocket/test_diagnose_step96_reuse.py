"""Independent positive/adversarial tests on immutable original DEVELOPMENT first passes."""
from copy import deepcopy
import os
from pathlib import Path
import unittest
import diagnose_step96_reuse as mod

class HistoricalReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.early_file = Path(os.environ["SPROCKET_ORIGINAL_EARLY"])
        cls.late_file = Path(os.environ["SPROCKET_ORIGINAL_LATE"])
        cls.early = mod.source_archive(cls.early_file, "early")
        cls.late = mod.source_archive(cls.late_file, "late")

    def test_real_positive(self):
        result = mod.analyze(self.early, self.late)
        self.assertEqual(result["summary"]["step16"]["terminal_eos"], 2)
        self.assertEqual(result["summary"]["step16"]["strict_json"], 1)
        self.assertEqual(result["summary"]["step96"]["terminal_eos"], 5)
        self.assertEqual(result["summary"]["step96"]["strict_json"], 5)
        self.assertEqual(result["summary"]["step96"]["valid_program"], 1)
        self.assertEqual(result["summary"]["step96"]["compiler_accepted_requests"], 1)
        self.assertEqual(result["cross_family_exact_first_two_box_reuse"][0]["request_count"], 3)
        self.assertEqual(result["cross_family_exact_first_two_box_reuse"][0]["family_count"], 2)
        self.assertTrue(result["pipework_exact_nodes_edges_equal"])
        self.assertEqual([x["nonorthogonal_edge_indices"] for x in result["pipework"]], [[2], [2]])
        self.assertFalse(result["final_opened"])

    def test_wrong_archive_sha_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            mod.source_archive(self.early_file, "late")

    def test_changed_prompt_rejected(self):
        bad = deepcopy(self.late)
        bad[0]["prompt"] += " ALTERED"
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            mod.analyze(self.early, bad)

    def test_missing_group_rejected(self):
        with self.assertRaisesRegex(ValueError, "group identity differs"):
            mod.analyze(self.early, self.late[:-1])

    def test_mutated_geometry_no_longer_matches(self):
        bad = deepcopy(self.late)
        x = next(r for r in bad if r["structural_group"] == "pipework/three_branch_manifold/v1")
        x["raw"]["raw_text"] = x["raw"]["raw_text"].replace("[84,75]", "[84,20]")
        result = mod.analyze(self.early, bad)
        self.assertFalse(result["pipework_exact_nodes_edges_equal"])
        self.assertEqual(result["pipework"][1]["nonorthogonal_edge_indices"], [])

    def test_invalid_json_not_repaired(self):
        bad = deepcopy(self.late)
        x = next(r for r in bad if r["structural_group"] == "greebles/open_center_modules/v1")
        x["raw"]["raw_text"] = "not json"
        with self.assertRaises(ValueError):
            mod.analyze(self.early, bad)

    def test_archives_have_no_final(self):
        import zipfile
        for p in (self.early_file, self.late_file):
            with zipfile.ZipFile(p) as z:
                self.assertFalse(any("final" in n.lower() for n in z.namelist()))

if __name__ == "__main__":
    unittest.main()
