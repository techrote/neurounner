"""Deterministic synthetic D-1B study tests. No model/native operations."""
import copy
import unittest

from experiments.driv3r.stage_d.guidance.comparison import analyze_shadow_comparison

D = lambda char: char * 64


def record():
    forks = [
        {"snapshot_id": "f0", "source_world_sha256": D("a"),
         "legal_candidates": ["L", "R", "S"],
         "candidate_effect_sha256": {"L": D("1"), "R": D("1"), "S": D("2")},
         "choices": {"A1": "L", "A2": "R", "A3": "L"}},
        {"snapshot_id": "f1", "source_world_sha256": D("b"),
         "legal_candidates": ["L", "R", "S"],
         "candidate_effect_sha256": {"L": D("1"), "R": D("2"), "S": D("3")},
         "choices": {"A1": "L", "A2": "R", "A3": "S"}},
    ]
    rollouts = {arm: [{"decision_ordinal": 0, "evidence_class": "synthetic",
                      "committed_cells": 20, "productive_cells": 12,
                      "net_objective_progress_cells": -2}] for arm in ("A1", "A2", "A3")}
    return {"forks": forks, "rollouts": rollouts, "scenario_sha256": D("4"),
            "source_revision": "a" * 40,
            "config_sha256": {arm: D(str(i)) for i, arm in enumerate(("A1", "A2", "A3"), 5)},
            "seed": 570}


class ComparisonTests(unittest.TestCase):
    def test_distinguishes_cosmetic_choice_from_effect_divergence(self):
        q = record()
        original = copy.deepcopy(q)
        result = analyze_shadow_comparison(**q)
        self.assertEqual(result["first_choice_divergence"]["decision_index"], 0)
        self.assertEqual(result["first_consequential_divergence"]["decision_index"], 1)
        self.assertEqual(result["post_fork_reported_metrics"]["A2"]["reported_productive_cells"], 12)
        self.assertFalse(result["native_execution_verified"])
        self.assertEqual(original, q)

    def test_candidate_display_order_does_not_change_host_identity(self):
        q = record()
        original = analyze_shadow_comparison(**q)
        q["forks"][0]["legal_candidates"].reverse()
        q["forks"][1]["legal_candidates"].reverse()
        shuffled = analyze_shadow_comparison(**q)
        self.assertEqual(original["candidate_id_set_sha256_by_fork"],
                         shuffled["candidate_id_set_sha256_by_fork"])
        self.assertEqual(original["predicted_effect_catalog_sha256_by_fork"],
                         shuffled["predicted_effect_catalog_sha256_by_fork"])
        self.assertEqual(original["first_choice_divergence"],
                         shuffled["first_choice_divergence"])
        self.assertEqual(original["first_consequential_divergence"],
                         shuffled["first_consequential_divergence"])

    def test_effect_predictions_do_not_change_candidate_id_identity(self):
        q = record()
        original = analyze_shadow_comparison(**q)
        q["forks"][0]["candidate_effect_sha256"]["L"] = D("f")
        changed = analyze_shadow_comparison(**q)
        self.assertEqual(original["candidate_id_set_sha256_by_fork"],
                         changed["candidate_id_set_sha256_by_fork"])
        self.assertNotEqual(original["predicted_effect_catalog_sha256_by_fork"],
                            changed["predicted_effect_catalog_sha256_by_fork"])
        self.assertNotIn("legal_candidate_set_sha256", changed)

    def test_rejects_unauthorized_candidate(self):
        q = record()
        q["forks"][0]["choices"]["A3"] = "not_host_legal"
        with self.assertRaisesRegex(ValueError, "nonlegal"):
            analyze_shadow_comparison(**q)

    def test_rejects_missing_candidate_effect(self):
        q = record()
        del q["forks"][1]["candidate_effect_sha256"]["S"]
        with self.assertRaisesRegex(ValueError, "every legal candidate"):
            analyze_shadow_comparison(**q)

    def test_rejects_native_claim_and_bad_source(self):
        q = record()
        q["evidence_class"] = "native"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            analyze_shadow_comparison(**q)
        q = record()
        q["source_revision"] = "invalid"
        with self.assertRaisesRegex(ValueError, "source_revision"):
            analyze_shadow_comparison(**q)

    def test_rejects_overcount_and_noncontiguous_followthrough(self):
        q = record()
        q["rollouts"]["A2"][0]["productive_cells"] = 21
        with self.assertRaisesRegex(ValueError, "exceed committed"):
            analyze_shadow_comparison(**q)
        q = record()
        q["rollouts"]["A2"][0]["decision_ordinal"] = 7
        with self.assertRaisesRegex(ValueError, "contiguous"):
            analyze_shadow_comparison(**q)

    def test_rejects_postfork_metrics_without_effect_divergence(self):
        q = record()
        q["forks"] = q["forks"][:1]
        with self.assertRaisesRegex(ValueError, "without a consequential"):
            analyze_shadow_comparison(**q)
        q["rollouts"] = {arm: [] for arm in ("A1", "A2", "A3")}
        out = analyze_shadow_comparison(**q)
        self.assertIsNone(out["first_consequential_divergence"])

    def test_rejects_mismatched_arms_and_duplicate_snapshot(self):
        q = record()
        del q["rollouts"]["A3"]
        with self.assertRaisesRegex(ValueError, "exactly A1"):
            analyze_shadow_comparison(**q)
        q = record()
        q["forks"][1]["snapshot_id"] = "f0"
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze_shadow_comparison(**q)


if __name__ == "__main__":
    unittest.main()
