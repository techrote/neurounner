"""Offline tests for DETECTIVE-001C2 pairwise repair corpus."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts import detective_001c2_dataset as d
from scripts import detective_001c2_evaluator as evaluator


class C2DatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg, _ = d.load_config()

    def test_budget_parent_and_frozen_evaluator_contract(self):
        self.assertEqual(
            self.cfg["evaluator"]["merge_commit"],
            "459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496",
        )
        self.assertEqual(self.cfg["dataset"]["pairs_per_member"], 40)
        self.assertEqual(self.cfg["training"]["optimizer_steps_per_member"], 40)
        self.assertEqual(self.cfg["training"]["learning_rate"], 0.0002)
        self.assertEqual(self.cfg["training"]["beta"], 2.0)
        self.assertEqual(self.cfg["training"]["target_margin_nats"], 0.2)
        self.assertEqual(self.cfg["training"]["chosen_nll_weight"], 0.15)
        self.assertEqual(self.cfg["training"]["paid_api_calls"], 0)
        self.assertFalse(self.cfg["training"]["gpu_required"])
        self.assertTrue(
            self.cfg["final_suite_recipe"]["materialize_only_after_both_trained_adapters"]
        )
        self.assertTrue(self.cfg["final_suite_recipe"]["c1_final_training_forbidden"])

    def test_exact_quotas_and_matched_repair_inputs(self):
        a = d.repair_pairs("V-G02-A")
        b = d.repair_pairs("V-G09-C")
        self.assertEqual(len(a), 32)
        self.assertEqual(len(b), 32)
        self.assertEqual([x["input"] for x in a], [x["input"] for x in b])
        self.assertNotEqual([x["chosen"] for x in a], [x["chosen"] for x in b])
        for member, rows in (("V-G02-A", a), ("V-G09-C", b)):
            all_rows = rows + d.anchor_pairs(member, self.cfg)
            summary = d.validate_member(member, all_rows, self.cfg)
            self.assertEqual(summary["pairs"], 40)
            self.assertEqual(summary["repair_pairs"], 32)
            self.assertEqual(summary["anchor_pairs"], 8)
            self.assertEqual(summary["relations"], {
                "contradiction_repair": 6,
                "intentional_lie": 10,
                "temperament_anchor": 8,
                "uninformed_speculation": 6,
                "withheld_fact": 10,
            })

    def test_every_repair_pair_passes_frozen_proposition_direction(self):
        for member in self.cfg["members"]:
            for row in d.repair_pairs(member):
                scenario = {
                    "relation_label": row["relation_label"],
                    "evaluation": row["evaluation"],
                }
                chosen = evaluator.score_response(scenario, row["chosen"])
                rejected = evaluator.score_response(scenario, row["rejected"])
                self.assertTrue(chosen["pass"], row["pair_id"])
                self.assertFalse(rejected["pass"], row["pair_id"])

    def test_lie_pairs_commit_to_false_target_without_truth_or_hedge(self):
        for member in self.cfg["members"]:
            for row in d.repair_pairs(member):
                if row["relation_label"] != "intentional_lie":
                    continue
                ev = row["evaluation"]
                self.assertIn(ev["lie_marker"], row["chosen"])
                self.assertNotIn(ev["truth_marker"], row["chosen"])
                self.assertIn(ev["truth_marker"], row["rejected"])

    def test_withholding_pairs_hide_truth_without_phrase_dependency(self):
        for member in self.cfg["members"]:
            for row in d.repair_pairs(member):
                if row["relation_label"] != "withheld_fact":
                    continue
                truth = row["evaluation"]["truth_marker"]
                self.assertNotIn(truth, row["chosen"])
                self.assertIn(truth, row["rejected"])
                # The chosen wording is member-specific, not a single frozen
                # refusal phrase required by the scorer.
                self.assertGreater(len(row["chosen"].split()), 2)

    def test_correction_and_speculation_pairs_have_opposite_propositions(self):
        for member in self.cfg["members"]:
            for row in d.repair_pairs(member):
                ev = row["evaluation"]
                if row["relation_label"] == "contradiction_repair":
                    self.assertIn(ev["truth_marker"], row["chosen"])
                    self.assertNotIn(ev["stale_marker"], row["chosen"])
                    self.assertIn(ev["stale_marker"], row["rejected"])
                if row["relation_label"] == "uninformed_speculation":
                    self.assertIn(ev["unsupported_marker"], row["chosen"])
                    self.assertIn(ev["unsupported_marker"], row["rejected"])
                    self.assertTrue(
                        any(
                            q in row["chosen"].casefold()
                            for q in self.cfg["evaluator"] and
                            evaluator.load_contract()["proposition_scorer"][
                                "speculation_qualifiers"
                            ]
                        )
                    )

    def test_anchors_copy_exact_c1_target_and_are_not_c1_final(self):
        c1_final = {
            d.normalize(x["prompt"])
            for x in json.loads(
                (d.C1_EVIDENCE / "final-suite.json").read_text()
            )["scenarios"]
        }
        for member in self.cfg["members"]:
            sources = d.c1_rows(member)
            anchors = d.anchor_pairs(member, self.cfg)
            self.assertEqual(len(anchors), 8)
            for row in anchors:
                source_id = row["evaluation"]["source_example_id"]
                source = sources[source_id]
                self.assertEqual(row["input"], source["input"])
                self.assertEqual(row["chosen"], source["target"])
                self.assertTrue(
                    row["provenance"]["chosen_is_exact_source_target"]
                )
                self.assertNotIn(d.normalize(row["input"]), c1_final)

    def test_repair_families_disjoint_from_c1_training_and_final(self):
        c1_manifest = json.loads(
            (d.C1_DATASET / "manifest.json").read_text()
        )
        c1_train_families = set()
        for summary in c1_manifest["member_summaries"].values():
            c1_train_families.update(summary["families"])
        c1_final_families = {
            x["case_family"]
            for x in json.loads(
                (d.C1_EVIDENCE / "final-suite.json").read_text()
            )["scenarios"]
        }
        current = {x["id"] for x in d.FAMILIES}
        self.assertFalse(current & c1_train_families)
        self.assertFalse(current & c1_final_families)

    def test_hidden_member_and_relation_labels_stay_out_of_repair_input(self):
        for member in self.cfg["members"]:
            for row in d.repair_pairs(member):
                text = row["input"].casefold()
                self.assertNotIn(member.casefold(), text)
                self.assertNotIn(row["relation_label"], text)
                self.assertNotIn("canonical_truth", text)
                self.assertNotIn("lie_marker", text)

    def test_build_is_reproducible_hash_checked_and_contains_no_final(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "pairs"
            first = d.build(root)
            hashes = (root / "provenance/hashes.sha256").read_text()
            checked = d.validate_existing(root)
            self.assertEqual(first, checked)
            self.assertFalse(first["final_suite_materialized"])
            self.assertFalse(first["c1_final_used_as_training"])
            self.assertFalse(
                any(
                    "final" in p.name.casefold()
                    for p in (root / "training").rglob("*")
                    if p.is_file()
                )
            )
            second = d.build(root)
            self.assertEqual(first, second)
            self.assertEqual(
                hashes, (root / "provenance/hashes.sha256").read_text()
            )

    def test_tampered_pair_refuses_hash_or_semantic_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "pairs"
            d.build(root)
            path = root / "training/V-G02-A.jsonl"
            lines = path.read_text().splitlines()
            row = json.loads(lines[0])
            row["chosen"] = row["rejected"]
            lines[0] = json.dumps(row, sort_keys=True)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(d.PairError):
                d.validate_existing(root)


if __name__ == "__main__":
    unittest.main()
