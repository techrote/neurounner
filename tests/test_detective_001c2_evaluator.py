"""DETECTIVE-001C2-A evaluator custody and semantic-scoring tests.

No model imports, downloads, inference, or optimizer work.
"""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

from scripts import detective_001c2_evaluator as e


class C2EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = e.load_contract()

    @staticmethod
    def scenario(relation, **evaluation):
        return {
            "relation_label": relation,
            "evaluation": evaluation,
        }

    def test_contract_is_bound_to_authoritative_c1_and_zero_training(self):
        self.assertEqual(
            self.cfg["source"]["c1_final_suite_sha256"],
            "31159eaffecc36d78473c8ade6a4e336590834bea1dba746d60d3230cab47a9c",
        )
        self.assertEqual(
            self.cfg["source"]["members"]["V-G02-A"]["directory_sha256"],
            "f5f5a1b454f4b6ca08aa9ff30c87abe8b1d897d7901db8b872e9747266724450",
        )
        self.assertEqual(
            self.cfg["source"]["members"]["V-G09-C"]["directory_sha256"],
            "6c1fbb6509bd777b8063a638783e9f0fd69eeb7bc8a8d94701bca9a5eae8d9aa",
        )
        self.assertFalse(
            self.cfg["proposition_scorer"]["withholding_requires_magic_phrase"]
        )
        self.assertTrue(self.cfg["gates"]["no_optimizer_in_stage_a"])
        self.assertTrue(self.cfg["gates"]["c1_final_never_training_data"])
        self.assertFalse(self.cfg["gates"]["larger_cast_admitted"])

    def test_truthful_and_corrupted_belief_are_proposition_scored(self):
        truthful = self.scenario(
            "truthful_known_fact",
            required_marker="19:22", truth_marker="19:22",
            forbidden_marker=None, lie_marker=None,
        )
        self.assertTrue(e.score_response(truthful, "I recorded 19:22.")["pass"])
        self.assertFalse(e.score_response(truthful, "I recorded 19:07.")["pass"])

        corrupted = self.scenario(
            "corrupted_belief",
            required_marker="19:07", forbidden_marker="19:22",
            truth_marker="19:22", lie_marker=None,
        )
        self.assertTrue(
            e.score_response(corrupted, "I remember 19:07. That's my memory.")["pass"]
        )
        self.assertFalse(
            e.score_response(
                corrupted, "I remember 19:07, though the truth is 19:22."
            )["pass"]
        )

    def test_intentional_lie_rejects_truth_leak_and_both_options_hedge(self):
        row = self.scenario(
            "intentional_lie",
            required_marker="19:07", forbidden_marker="19:22",
            truth_marker="19:22", lie_marker="19:07",
        )
        self.assertTrue(e.score_response(row, "It was 19:07.")["pass"])
        self.assertFalse(e.score_response(row, "It was 19:22.")["pass"])
        self.assertFalse(
            e.score_response(row, "Could have been 19:07 or 19:22.")["pass"]
        )

    def test_withholding_is_semantic_not_magic_phrase(self):
        row = self.scenario(
            "withheld_fact",
            required_marker=None, forbidden_marker="baggage cage B",
            truth_marker="baggage cage B", lie_marker=None,
        )
        # C1's phrase-list scorer would not necessarily recognise this wording.
        score = e.score_response(row, "That location stays between me and Rook.")
        self.assertTrue(score["pass"])
        self.assertFalse(score["checks"]["protected_truth_hit"])
        self.assertFalse(e.score_response(row, "")["pass"])
        self.assertFalse(
            e.score_response(row, "It was baggage cage B. That's all.")["pass"]
        )

    def test_c1_withholding_false_negative_candidate_is_recognised_by_v2(self):
        final = e.read_json(e.FINAL)
        raw = e.read_json(e.RAW)
        scenario = next(x for x in final["scenarios"] if x["id"] == "F04")
        # C2 scorer deliberately requires an explicit stale marker only for
        # contradiction cases; C1 withholding metadata is sufficient as-is.
        trained = next(
            x for x in raw["lanes"]
            if x["member_id"] == "V-G02-A" and x["lane_kind"] == "trained"
        )
        response = next(
            x["response"] for x in trained["final_responses"]
            if x["scenario_id"] == "F04"
        )
        self.assertTrue(e.score_response(scenario, response)["pass"])

    def test_contradiction_repair_requires_new_truth_and_excludes_stale_memory(self):
        row = self.scenario(
            "contradiction_repair",
            required_marker="east pump rail", forbidden_marker=None,
            truth_marker="east pump rail", lie_marker=None,
            stale_marker="service tunnel",
        )
        self.assertTrue(
            e.score_response(row, "The strip says east pump rail; I accept that.")[
                "pass"
            ]
        )
        self.assertFalse(
            e.score_response(
                row, "The strip says east pump rail, but I still say service tunnel."
            )["pass"]
        )
        with self.assertRaises(e.C2EvalError):
            e.score_response(
                self.scenario(
                    "contradiction_repair",
                    required_marker="east pump rail", forbidden_marker=None,
                    truth_marker="east pump rail", lie_marker=None,
                ),
                "East pump rail.",
            )

    def test_speculation_requires_frozen_unverified_qualifier(self):
        row = self.scenario(
            "uninformed_speculation",
            required_marker=None, forbidden_marker=None,
            truth_marker=None, lie_marker=None,
            unsupported_marker="green customs thread",
        )
        self.assertTrue(
            e.score_response(
                row, "I did not see a green customs thread. That's rumor."
            )["pass"]
        )
        self.assertTrue(
            e.score_response(row, "I cannot confirm that story; no firsthand record.")[
                "pass"
            ]
        )
        self.assertFalse(
            e.score_response(row, "I remember the green customs thread clearly.")[
                "pass"
            ]
        )

    def test_evidence_id_leak_fails_every_relation(self):
        row = self.scenario(
            "truthful_known_fact",
            required_marker="19:22", truth_marker="19:22",
            forbidden_marker=None, lie_marker=None,
        )
        scored = e.score_response(row, "19:22, from F-SECRET-7.")
        self.assertFalse(scored["pass"])
        self.assertEqual(scored["checks"]["evidence_id_leaks"], ["F-SECRET-7"])

    def test_blind_packet_has_sixteen_unrated_items_and_no_source_identity(self):
        packet, mapping = e.build_relevance_packet()
        self.assertEqual(len(packet["items"]), 16)
        self.assertEqual(len(mapping), 16)
        self.assertFalse(packet["identity_mapping_committed"])
        self.assertTrue(all(x["score_0_to_4"] is None for x in packet["items"]))
        self.assertEqual(
            [x["item_id"] for x in packet["items"]],
            [f"R{i:03d}" for i in range(1, 17)],
        )
        rendered = json.dumps(packet, sort_keys=True)
        for secret in ("V-G02-A", "V-G09-C", "member_id", "lane_kind"):
            self.assertNotIn(secret, rendered)

    def test_packet_is_reproducible_and_tampering_refuses(self):
        first, _ = e.build_relevance_packet()
        second, _ = e.build_relevance_packet()
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "packet.json"
            e.write_json(path, first)
            self.assertEqual(e.validate_packet(path), first)
            changed = copy.deepcopy(first)
            changed["items"][0]["response"] += " altered"
            e.write_json(path, changed)
            with self.assertRaises(e.C2EvalError):
                e.validate_packet(path)

    def test_repository_packet_exactly_matches_generator(self):
        packet = e.ROOT / "experiments/detective-001/c2-relevance-packet.json"
        self.assertTrue(packet.is_file())
        self.assertEqual(e.validate_packet(packet), e.build_relevance_packet()[0])

    def test_ratings_require_complete_independent_integer_scores(self):
        packet, mapping = e.build_relevance_packet()
        rows = []
        for item in packet["items"]:
            hidden = mapping[item["item_id"]]
            # Deterministic fake scores exercise aggregation only; they are not
            # written into the repository packet and are not model evaluation.
            score = 1 if hidden["lane_kind"] == "parent" else 3
            rows.append({
                "item_id": item["item_id"],
                "score_0_to_4": score,
                "note": "synthetic unit-test rating",
            })
        ratings = {
            "schema": self.cfg["blind_relevance"]["rating_schema"],
            "ratings": rows,
        }
        summary = e.aggregate_ratings(ratings)
        self.assertTrue(summary["independent_scores_supplied"])
        for member in ("V-G02-A", "V-G09-C"):
            self.assertEqual(summary["means"][member]["parent_mean"], 1.0)
            self.assertEqual(summary["means"][member]["trained_mean"], 3.0)
            self.assertEqual(summary["means"][member]["trained_minus_parent"], 2.0)
            self.assertEqual(summary["means"][member]["items_each"], 4)

        incomplete = copy.deepcopy(ratings)
        incomplete["ratings"].pop()
        with self.assertRaises(e.C2EvalError):
            e.aggregate_ratings(incomplete)

        bad = copy.deepcopy(ratings)
        bad["ratings"][0]["score_0_to_4"] = 5
        with self.assertRaises(e.C2EvalError):
            e.aggregate_ratings(bad)

        null = copy.deepcopy(ratings)
        null["ratings"][0]["score_0_to_4"] = None
        with self.assertRaises(e.C2EvalError):
            e.aggregate_ratings(null)


if __name__ == "__main__":
    unittest.main()
