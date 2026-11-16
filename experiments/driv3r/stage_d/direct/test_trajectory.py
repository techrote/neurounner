"""Pure offline deterministic tests; no native/held-out/model operations."""
import copy
import json
import unittest

from experiments.driv3r.stage_d.direct.trajectory import compare_simulated_effects as audit


def record(choices, positions, path_prefixes, path):
    return {
        "events": [
            {"choice_id": choice, "simulated_post_position": pos,
             "simulated_path_sha256": digest}
            for choice, pos, digest in zip(choices, positions, path_prefixes)
        ],
        "path": path,
    }


class SyntheticEffectTests(unittest.TestCase):
    def test_identical_record_no_divergence(self):
        a = record(["E1", "N1"], [(1, 0), (1, -1)], ["p1", "p2"],
                   [(0, 0), (1, 0), (1, -1)])
        v = audit({"a": a, "b": copy.deepcopy(a)})
        self.assertIsNone(v["first_geometric_divergence_index"])
        self.assertIsNone(v["first_episode_length_divergence_index"])
        self.assertEqual(v["common_simulated_path_prefix_cells"], 3)
        self.assertFalse(v["native_qualified"])

    def test_different_choice_same_effect_is_not_geometry_divergence(self):
        a = record(["choice_A"], [(1, 0)], ["p1"], [(0, 0), (1, 0)])
        b = record(["choice_B"], [(1, 0)], ["p1"], [(0, 0), (1, 0)])
        self.assertIsNone(audit({"a": a, "b": b})["first_geometric_divergence_index"])

    def test_swept_footprint_differs_even_when_cursor_matches(self):
        a = record(["a"], [(2, 0)], ["p1"], [(0, 0), (1, 0), (2, 0)])
        b = record(["b"], [(2, 0)], ["q1"], [(0, 0), (0, 1), (2, 0)])
        v = audit({"a": a, "b": b})
        self.assertIsNone(v["first_cursor_divergence_index"])
        self.assertEqual(v["first_swept_path_divergence_index"], 0)
        self.assertEqual(v["first_geometric_divergence_index"], 0)
        self.assertEqual(v["common_simulated_path_prefix_cells"], 1)

    def test_terminated_arm_remains_stationary(self):
        a = record(["E1"], [(1, 0)], ["p1"], [(0, 0), (1, 0)])
        b = record(["E1", "E1"], [(1, 0), (2, 0)], ["p1", "p2"],
                   [(0, 0), (1, 0), (2, 0)])
        v = audit({"a": a, "b": b})
        self.assertEqual(v["first_cursor_divergence_index"], 1)
        self.assertEqual(v["first_episode_length_divergence_index"], 1)
        self.assertEqual(v["common_simulated_path_prefix_cells"], 2)

    def test_different_length_identical_geometry_is_only_termination_difference(self):
        a = record(["STOP"], [(0, 0)], ["p0"], [(0, 0)])
        b = record(["NOOP", "STOP"], [(0, 0), (0, 0)], ["p0", "p0"], [(0, 0)])
        v = audit({"a": a, "b": b})
        self.assertIsNone(v["first_geometric_divergence_index"])
        self.assertEqual(v["first_episode_length_divergence_index"], 1)

    def test_multiarm_earliest_difference(self):
        a = record(["E1", "E1"], [(1, 0), (2, 0)], ["p1", "p2"],
                   [(0, 0), (1, 0), (2, 0)])
        b = copy.deepcopy(a)
        c = record(["N1", "E1"], [(0, -1), (1, -1)], ["q1", "q2"],
                   [(0, 0), (0, -1), (1, -1)])
        self.assertEqual(audit({"a": a, "b": b, "c": c})["first_geometric_divergence_index"], 0)

    def test_json_roundtrip_does_not_create_false_divergence(self):
        a = record(["E1"], [(1, 0)], ["p1"], [(0, 0), (1, 0)])
        b = json.loads(json.dumps(a))
        v = audit({"a": a, "b": b})
        self.assertIsNone(v["first_geometric_divergence_index"])
        self.assertEqual(v["common_simulated_path_prefix_cells"], 2)

    def test_reject_empty_or_malformed_arms(self):
        for arms in ({}, {"a": {}}, {"a": {}, "b": {}}, {"a": record([],[],[],[(0,0)]), "b": {"events":[], "path":[]}}):
            with self.subTest(arms=arms):
                with self.assertRaises(ValueError):
                    audit(arms)

    def test_never_claim_native_or_catalog_parity(self):
        a = record(["E1"], [(1, 0)], ["p1"], [(0, 0), (1, 0)])
        v = audit({"a": a, "b": copy.deepcopy(a)})
        self.assertEqual(v["first_consequential_native_divergence"], "not_measured")
        self.assertEqual(v["post_divergence_candidate_parity"], "not_asserted")


if __name__ == "__main__":
    unittest.main()
