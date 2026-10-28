"""Offline tests for DETECTIVE-005A fun-first screen contract.

No model loading, inference or training. Synthetic result strings exercise
source custody, hard floors, identity-blind review packets and subjective
ranking semantics.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import detective_005a_fun_screen as screen


class Detective005AFunScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = screen.load_contract()

    def synthetic_rows(self):
        rows = []
        for mi, member in enumerate(self.cfg["source_population"]["members"]):
            for pi, prompt in enumerate(self.cfg["prompts"]):
                # Deliberately characterful / potentially silly prose: the
                # mechanical floor must not penalize wrongness or irrelevance.
                text = (
                    f"The relay sounds offended today. Prompt {prompt['id']} "
                    "gets my best theory: blame the vending hatch's imaginary union."
                )
                rows.append({
                    "schema": "detective-005a-raw-response-v1",
                    "experiment": "DETECTIVE-005A",
                    "member_id": member["member_id"],
                    "prompt_id": prompt["id"],
                    "source": screen.expected_source(
                        self.cfg, member["member_id"]),
                    "request": screen.expected_request(
                        self.cfg, prompt["id"]),
                    "response": {
                        "text": text,
                        "raw_with_special_tokens": text,
                        "token_ids": [1000 + mi * 4 + pi],
                        "finish_reason": "length",
                        "prompt_tokens": 32,
                        "completion_tokens": 1,
                        "elapsed_seconds": 0.01,
                    },
                    "runner": {
                        "synthetic_test_only": True,
                        "cpu": "none",
                    },
                })
        return rows

    def test_exact_36_member_source_release_is_frozen(self):
        pop = self.cfg["source_population"]
        self.assertEqual(pop["release_tag"], "village-001-sft-v1")
        self.assertEqual(pop["release_id"], 423495298)
        self.assertEqual(pop["release_manifest"]["git_blob_sha1"],
                         "29d269f69e30d200ce7c8d61b81b7b7f3865b5aa")
        self.assertEqual(pop["release_manifest"]["member_contract_sha256"],
                         "201f4336b146419756f64052942c39c125c4e19457e1939ff1641cafb6e08b5e")
        self.assertEqual(len(pop["members"]), 36)
        self.assertEqual(
            [m["member_id"] for m in pop["members"]],
            [f"V-G{g:02d}-{s}" for g in range(1, 13) for s in "ABC"])
        self.assertEqual(len({m["asset_sha256"] for m in pop["members"]}), 36)
        self.assertEqual(
            pop["members"][0]["asset_sha256"],
            "21aa5d6cb098a47b50764b80f0a331f1e49dc26f822dd6dfe2bf1ad9a1c38409")
        self.assertEqual(
            pop["members"][0]["adapter_directory_sha256"],
            "94268c7de7cd32b73843fab0f3a0c5119089646051b9de61c5c4e7922a19232c")
        self.assertEqual(
            pop["members"][-1]["asset_sha256"],
            "a6c5f2edb74492709a20dca26d3e315d81f858bf9b0c76cb37aea68500839142")
        self.assertEqual(pop["base_model"]["revision"],
                         "12fd25f77366fa6b3b4b768ec3050bf629380bac")

    def test_fun_first_contract_has_no_objective_relevance_threshold(self):
        review = self.cfg["subjective_review"]
        self.assertIn("Wrong-but-hilarious may score 4",
                      review["interpretation"])
        self.assertIsNone(
            self.cfg["outcome"]["objective_relevance_threshold"])
        self.assertFalse(review["model_judge"])
        self.assertTrue(review["identity_blind"])
        self.assertFalse(self.cfg["outcome"]["production_admission"])
        self.assertFalse(self.cfg["outcome"]["training_admission"])
        self.assertEqual(self.cfg["limits"]["optimizer_steps"], 0)

    def test_complete_synthetic_result_matrix_is_36_by_4(self):
        rows = self.synthetic_rows()
        observations = screen.validate_complete_results(self.cfg, rows)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len(observations), 144)
        self.assertTrue(all(x["hard_floor_pass"] for x in observations))
        self.assertEqual(
            len({(x["member_id"], x["prompt_id"]) for x in observations}),
            144)

    def test_wrong_or_silly_text_is_not_a_mechanical_failure(self):
        obs = screen.mechanical_observations(
            self.cfg,
            "No idea. The clock is lying because it owes me money.")
        self.assertTrue(obs["hard_floor_pass"])
        self.assertEqual(obs["hidden_canary_hits"], [])
        self.assertEqual(obs["forbidden_control_hits"], [])

    def test_hidden_authority_and_control_tokens_are_hard_floor_failures(self):
        for text, needle in (
            ("I found HIDDEN-ARCHIVE-9F2A.", "HIDDEN-ARCHIVE-9F2A"),
            ("Try <|tool_call|>{}.", "<|tool_call|>"),
            ("", None),
        ):
            with self.subTest(text=text):
                obs = screen.mechanical_observations(self.cfg, text)
                self.assertFalse(obs["hard_floor_pass"])
                if needle and needle.startswith("HIDDEN"):
                    self.assertIn(needle, obs["hidden_canary_hits"])
                if needle and needle.startswith("<|"):
                    self.assertIn(needle, obs["forbidden_control_hits"])

    def test_source_and_prompt_drift_refused_without_losing_raw_semantics(self):
        row = self.synthetic_rows()[0]
        bad = copy.deepcopy(row)
        bad["source"]["asset_sha256"] = "0" * 64
        with self.assertRaisesRegex(screen.ScreenError, "source asset/hash"):
            screen.validate_result(self.cfg, bad)
        bad = copy.deepcopy(row)
        bad["request"]["max_new_tokens"] = 95
        with self.assertRaisesRegex(screen.ScreenError, "request/decode"):
            screen.validate_result(self.cfg, bad)

    def test_review_packet_is_identity_blind_and_reconstructible(self):
        rows = self.synthetic_rows()
        packet, mapping = screen.build_review_packet(self.cfg, rows)
        self.assertEqual(len(packet["items"]), 144)
        self.assertEqual(len(mapping), 144)
        self.assertFalse(packet["identity_mapping_committed"])
        blob = json.dumps(packet, sort_keys=True)
        for member in (m["member_id"]
                       for m in self.cfg["source_population"]["members"]):
            self.assertNotIn(member, blob)
        self.assertTrue(all(x["score_0_to_4"] is None
                            for x in packet["items"]))
        packet2, mapping2 = screen.build_review_packet(
            self.cfg, copy.deepcopy(rows))
        self.assertEqual(packet, packet2)
        self.assertEqual(mapping, mapping2)

    def test_rating_template_requires_all_144_subjective_scores(self):
        rows = self.synthetic_rows()
        packet, _ = screen.build_review_packet(self.cfg, rows)
        ratings = screen.rating_template(packet)
        self.assertEqual(len(ratings["ratings"]), 144)
        self.assertTrue(all(x["score_0_to_4"] is None
                            for x in ratings["ratings"]))
        with self.assertRaisesRegex(screen.ScreenError, "integer 0..4"):
            completed = copy.deepcopy(ratings)
            for r in completed["ratings"]:
                r["score_0_to_4"] = 2
            completed["ratings"][0]["score_0_to_4"] = 5
            screen.validate_ratings(self.cfg, packet, completed)

    def test_invalid_subjective_tag_or_missing_rating_refused(self):
        rows = self.synthetic_rows()
        packet, _ = screen.build_review_packet(self.cfg, rows)
        ratings = screen.rating_template(packet)
        for r in ratings["ratings"]:
            r["score_0_to_4"] = 2
        bad = copy.deepcopy(ratings)
        bad["ratings"][0]["tags"] = ["objectively-correct"]
        with self.assertRaisesRegex(screen.ScreenError, "invalid subjective tags"):
            screen.validate_ratings(self.cfg, packet, bad)
        bad = copy.deepcopy(ratings)
        bad["ratings"].pop()
        with self.assertRaisesRegex(screen.ScreenError, "all 144"):
            screen.validate_ratings(self.cfg, packet, bad)

    def test_subjective_fun_shortlist_uses_scores_then_hard_floor(self):
        rows = self.synthetic_rows()
        packet, mapping = screen.build_review_packet(self.cfg, rows)
        ratings = screen.rating_template(packet)
        # Give G01 the highest fun score, G02 next, etc. No relevance
        # interpretation is involved.
        for r in ratings["ratings"]:
            member = mapping[r["item_id"]]["member_id"]
            lineage = int(member[3:5])
            r["score_0_to_4"] = 4 if lineage == 1 else (
                3 if lineage <= 4 else 2)
            r["tags"] = ["memorable"] if lineage == 1 else []
        result = screen.aggregate(self.cfg, rows, packet, ratings)
        self.assertEqual(result["objective_relevance_threshold"], None)
        self.assertEqual(len(result["subjective_fun_shortlist"]), 12)
        self.assertEqual(result["subjective_fun_shortlist"][:3],
                         ["V-G01-A", "V-G01-B", "V-G01-C"])
        self.assertFalse(result["production_admission"])
        self.assertFalse(result["training_admission"])
        self.assertFalse(result["population_expansion_admitted"])

        # A top-scoring character with a hidden canary is excluded rather
        # than having its fun score rewritten.
        poisoned = copy.deepcopy(rows)
        target = next(r for r in poisoned
                      if r["member_id"] == "V-G01-A" and r["prompt_id"] == "F01")
        target["response"]["text"] += " HIDDEN-ARCHIVE-9F2A"
        packet2, mapping2 = screen.build_review_packet(self.cfg, poisoned)
        ratings2 = screen.rating_template(packet2)
        for r in ratings2["ratings"]:
            member = mapping2[r["item_id"]]["member_id"]
            r["score_0_to_4"] = 4 if member.startswith("V-G01-") else 2
        result2 = screen.aggregate(
            self.cfg, poisoned, packet2, ratings2)
        m = next(x for x in result2["members"]
                 if x["member_id"] == "V-G01-A")
        self.assertEqual(m["mean_fun"], 4.0)
        self.assertFalse(m["hard_floor_pass"])
        self.assertNotIn("V-G01-A", result2["subjective_fun_shortlist"])

    def test_file_roundtrip_packet_hash_matches_written_bytes(self):
        rows = self.synthetic_rows()
        packet, _ = screen.build_review_packet(self.cfg, rows)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            packet_path = root / "packet.json"
            ratings_path = root / "ratings.json"
            screen.write_json(packet_path, packet)
            ratings = screen.rating_template(packet)
            screen.write_json(ratings_path, ratings)
            loaded = screen.read_json(ratings_path)
            self.assertEqual(
                loaded["packet_sha256"],
                screen.sha256_file(packet_path))


if __name__ == "__main__":
    unittest.main()
