"""Source-bound cheap offline audit tests; never evaluate sealed model behavior."""
import copy
import json
import unittest

from experiments.driv3r.stage_d.direct import offline as d
from experiments.driv3r.stage_d.direct import paired as p


class PairedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.wall = d.wall_fixture("step", "development")
        self.route = d.routing_fixture("gate", "development")

    def assert_rejected(self, base, update, fixture=None):
        record = copy.deepcopy(base)
        update(record)
        with self.assertRaises(p.EvidenceError):
            p.verify_episode(fixture or self.route, record)

    def test_complete_route_replays_from_actual_offline_source(self):
        record = d.run_episode(self.route, "astar", seed=1)
        proof = p.verify_episode(self.route, record)
        self.assertTrue(proof["validated"])
        self.assertTrue(proof["synthetic_only"])
        self.assertEqual(proof["event_count"], record["metrics"]["attempted_decisions"])
        self.assertEqual(proof["path_sha256"], d.digest(record["path"]))
        self.assertEqual(record["metrics"]["native_receipts"], 0)

    def test_wall_contour_and_macro_replay(self):
        for fixture in (self.wall, self.route):
            policy = "contour_astar" if fixture.task == "wall_offset" else "astar"
            record = d.run_episode(fixture, policy, seed=19, lengths=(1, 2, 4, 8))
            with self.subTest(fixture=fixture.fixture_id):
                self.assertTrue(p.verify_episode(fixture, record)["validated"])
                self.assertLessEqual(len(record["events"]), 36)

    def test_random_seed_and_jsonl_roundtrip(self):
        record = d.run_episode(self.route, "random_legal", seed=7)
        decoded = json.loads(json.dumps(record))
        self.assertEqual(record["metrics"], decoded["metrics"])
        self.assertTrue(p.verify_episode(self.route, decoded)["validated"])
        self.assertEqual(record, d.run_episode(self.route, "random_legal", seed=7))

    def test_undocumented_native_success_or_receipt_rejected(self):
        record = d.run_episode(self.route, "astar")
        self.assert_rejected(record, lambda r: r.__setitem__("native_qualified", True))
        self.assert_rejected(record, lambda r: r.__setitem__("sequential_native_receipts", 1))
        self.assert_rejected(record, lambda r: r["events"][0].__setitem__("native_receipt_ref", "fake"))

    def test_tampered_choice_and_seed_rejected(self):
        record = d.run_episode(self.route, "astar")
        self.assert_rejected(record, lambda r: r["events"][0].__setitem__("choice_id", "STOP"))
        self.assert_rejected(record, lambda r: r.__setitem__("seed", 4294967296))
        self.assert_rejected(record, lambda r: r.__setitem__("policy", "unknown_policy"))

    def test_changed_offer_or_sequence_rejected(self):
        record = d.run_episode(self.route, "astar")
        self.assert_rejected(record, lambda r: r["events"][0]["observation"].__setitem__(
            "candidate_set_sha256", "f" * 64))
        self.assert_rejected(record, lambda r: r["events"][0].__setitem__("seq", 1))
        self.assert_rejected(record, lambda r: r["events"][0]["observation"].__setitem__(
            "budget_remaining", 999))

    def test_broken_path_or_scorer_rejected(self):
        record = d.run_episode(self.route, "astar")
        self.assert_rejected(record, lambda r: r["path"].append((0, 0)))
        self.assert_rejected(record, lambda r: r["events"][0].__setitem__(
            "simulated_path_sha256", "a" * 64))
        self.assert_rejected(record, lambda r: r["metrics"].__setitem__("arrived", False))

    def test_identical_policy_arms_have_common_catalogs(self):
        source = d.run_episode(self.wall, "contour_astar", seed=23)
        result = p.compare_matched(self.wall, {"a": source, "b": copy.deepcopy(source)})
        self.assertIsNone(result["first_choice_divergence_index"])
        self.assertTrue(result["matched_candidate_sets_checked_through_divergence"])
        self.assertFalse(result["native_qualified"])

    def test_baseline_pair_has_exact_first_choice_context(self):
        result = p.run_development_pair(self.route, seed=13)
        self.assertEqual(result["fixture_sha256"], self.route.sha256)
        self.assertEqual(set(result["arm_names"]), {"geometric", "random_legal"})
        self.assertFalse(result["post_divergence_candidate_matching_claimed"])
        self.assertTrue(result["matched_candidate_sets_checked_through_divergence"])
        self.assertEqual(set(result["arm_metrics"]), set(result["arm_names"]))

    def test_unequal_action_menus_are_refused(self):
        a = d.run_episode(self.route, "astar", lengths=(1,))
        b = d.run_episode(self.route, "astar", lengths=(1, 2, 4, 8))
        with self.assertRaises(p.EvidenceError):
            p.compare_matched(self.route, {"a": a, "b": b})

    def test_choice_and_simulated_effect_divergence_are_distinct(self):
        result = p.run_development_pair(self.route, seed=13)
        effect = result["synthetic_effect_audit"]
        self.assertFalse(effect["native_qualified"])
        self.assertEqual(effect["first_consequential_native_divergence"], "not_measured")
        self.assertEqual(effect["post_divergence_candidate_parity"], "not_asserted")
        choice = result["first_choice_divergence_index"]
        geom = effect["first_geometric_divergence_index"]
        self.assertIsNotNone(choice)
        self.assertIsNotNone(geom)
        self.assertGreaterEqual(geom, choice)

    def test_heldout_outcomes_remain_unopened(self):
        heldout = d.wall_fixture("switchback", "heldout")
        with self.assertRaises(p.EvidenceError):
            p.run_development_pair(heldout)
        with self.assertRaises(p.EvidenceError):
            p.compare_matched(heldout, {"a": {}, "b": {}})


if __name__ == "__main__":
    unittest.main()
