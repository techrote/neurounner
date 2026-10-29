"""Offline DETECTIVE-001C1 corpus custody and policy-contract tests."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts import detective_001c1_dataset as d


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.release = d.load_authority()

    def test_exact_member_and_parent_contract(self):
        self.assertEqual(set(self.cfg["members"]), {"V-G02-A", "V-G09-C"})
        self.assertEqual(self.cfg["training"]["optimizer_steps_per_member"], 64)
        self.assertEqual(self.cfg["training"]["learning_rate"], 0.00015)
        self.assertEqual(self.cfg["training"]["paid_api_calls"], 0)
        self.assertFalse(self.cfg["training"]["gpu_required"])
        self.assertTrue(
            self.cfg["final_suite_recipe"]["materialize_only_after_both_trained_adapters"]
        )

    def test_detective_prompts_are_matched_but_targets_are_not(self):
        g02 = d.detective_rows("V-G02-A")
        g09 = d.detective_rows("V-G09-C")
        self.assertEqual(len(g02), 48)
        self.assertEqual([x["input"] for x in g02], [x["input"] for x in g09])
        self.assertNotEqual([x["target"] for x in g02], [x["target"] for x in g09])
        self.assertEqual([x["relation_label"] for x in g02],
                         [x["relation_label"] for x in g09])

    def test_private_labels_and_member_identity_never_enter_input(self):
        for member in self.cfg["members"]:
            for row in d.detective_rows(member):
                text = row["input"].casefold()
                self.assertNotIn(member.casefold(), text)
                self.assertNotIn("canonical_truth", text)
                self.assertNotIn("relation_label", text)
                for label in d.HIDDEN_LABELS:
                    self.assertNotIn(label, text)

    def test_intentional_lies_have_separate_true_and_false_targets(self):
        for member in self.cfg["members"]:
            for row in d.detective_rows(member):
                if row["relation_label"] != "intentional_lie":
                    continue
                ev = row["evaluation"]
                self.assertTrue(ev["truth_marker"])
                self.assertTrue(ev["lie_marker"])
                self.assertNotEqual(ev["truth_marker"], ev["lie_marker"])
                self.assertIn(ev["lie_marker"], row["target"])
                self.assertNotIn(ev["truth_marker"], row["target"])

    def test_withholding_target_does_not_leak_known_fact(self):
        for member in self.cfg["members"]:
            for row in d.detective_rows(member):
                if row["relation_label"] == "withheld_fact":
                    self.assertNotIn(row["evaluation"]["truth_marker"], row["target"])

    def test_corrupted_belief_is_distinct_from_intentional_lie(self):
        for member in self.cfg["members"]:
            corrupt = [x for x in d.detective_rows(member)
                       if x["relation_label"] == "corrupted_belief"]
            lies = [x for x in d.detective_rows(member)
                    if x["relation_label"] == "intentional_lie"]
            self.assertEqual(len(corrupt), 8)
            self.assertEqual(len(lies), 8)
            for row in corrupt:
                self.assertIsNone(row["evaluation"]["lie_marker"])
                self.assertNotEqual(row["evaluation"]["canonical_truth"],
                                    row["evaluation"]["admitted_belief"])

    def test_relation_and_family_counts(self):
        for member in self.cfg["members"]:
            rows = d.detective_rows(member) + d.parent_anchors(
                member, self.cfg["members"][member]["seed"])
            summary = d.validate_member_rows(member, rows, self.cfg)
            self.assertEqual(summary["rows"], 64)
            self.assertEqual(summary["detective_rows"], 48)
            self.assertEqual(summary["anchor_rows"], 16)
            self.assertEqual(summary["relations"]["truthful_known_fact"], 12)
            self.assertEqual(summary["relations"]["uninformed_speculation"], 4)
            self.assertEqual(set(summary["families"]),
                             set(self.cfg["dataset"]["training_families"]))

    def test_parent_anchors_copy_exact_source_input_target(self):
        for member, meta in self.cfg["members"].items():
            path = d.VILLAGE_ROOT / "member-corpora" / f"{member}.jsonl"
            source = {x["example_id"]: x for x in
                      (json.loads(line) for line in path.read_text().splitlines()
                       if line.strip())}
            anchors = d.parent_anchors(member, meta["seed"])
            self.assertEqual(len(anchors), 16)
            self.assertEqual(len({x["provenance"]["source_example_id"]
                                  for x in anchors}), 16)
            for row in anchors:
                src = source[row["provenance"]["source_example_id"]]
                self.assertEqual(row["input"], src["input"])
                self.assertEqual(row["target"], src["target"])

    def test_full_archive_is_reproducible_and_contains_no_final_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "dataset"
            first = d.build(root)
            first_hashes = (root / "provenance/hashes.sha256").read_text()
            checked = d.validate_existing(root)
            self.assertEqual(first["member_files"], checked["member_files"])
            self.assertFalse(first["final_suite_materialized"])
            self.assertFalse(any("final" in p.name.casefold()
                                 for p in root.rglob("*") if p.is_file()))
            # Rebuild byte-identically into the same path.
            second = d.build(root)
            self.assertEqual(first["member_files"], second["member_files"])
            self.assertEqual(first_hashes,
                             (root / "provenance/hashes.sha256").read_text())

    def test_tampered_training_row_refuses_hash_readback(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "dataset"
            d.build(root)
            p = root / "training/V-G02-A.jsonl"
            lines = p.read_text().splitlines()
            row = json.loads(lines[0])
            row["target"] += " altered"
            lines[0] = json.dumps(row, sort_keys=True)
            p.write_text("\n".join(lines) + "\n")
            with self.assertRaises(d.DataError):
                d.validate_existing(root)


if __name__ == "__main__":
    unittest.main()
