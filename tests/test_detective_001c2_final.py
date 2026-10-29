"""Offline custody tests for DETECTIVE-001C2 post-training learner final."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from scripts import detective_001c2_dataset as dataset
from scripts import detective_001c2_final as final


class C2FinalTests(unittest.TestCase):
    def _adapter(self, root: pathlib.Path, name: str, payload: bytes):
        path = root / name
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(payload)
        (path / "adapter_config.json").write_text(
            json.dumps({"name": name}, sort_keys=True) + "\n")
        return path

    def _dataset(self, root: pathlib.Path):
        path = root / "pairs"
        dataset.build(path)
        return path

    def test_final_requires_pair_dataset_and_both_trained_adapters(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"A")
            missing = root / "missing"
            with patch.object(final, "PAIR_DATASET", pairs):
                with self.assertRaises(FileNotFoundError):
                    final.materialize(a, missing, root / "final.json")

    def test_exact_final_is_reproducible_post_training_and_balanced(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"adapter-A")
            b = self._adapter(root, "b", b"adapter-B")
            with patch.object(final, "PAIR_DATASET", pairs):
                first = final.materialize(a, b, root / "one.json")
                second = final.materialize(a, b, root / "two.json")
            self.assertEqual(first, second)
            self.assertEqual(
                (root / "one.json").read_bytes(),
                (root / "two.json").read_bytes(),
            )
            self.assertTrue(first["generated_after_training"])
            self.assertFalse(first["c1_final_used_as_training"])
            self.assertEqual(len(first["scenarios"]), 18)
            counts = {}
            for row in first["scenarios"]:
                counts[row["relation_label"]] = (
                    counts.get(row["relation_label"], 0) + 1
                )
                self.assertTrue(row["mandatory_clue_fallback"]["available"])
                self.assertEqual(
                    row["mandatory_clue_fallback"]["kind"], "public_record"
                )
            self.assertEqual(counts, {
                "truthful_known_fact": 3,
                "corrupted_belief": 3,
                "intentional_lie": 3,
                "withheld_fact": 3,
                "contradiction_repair": 3,
                "uninformed_speculation": 3,
            })

    def test_changing_trained_hash_changes_seed_and_concrete_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"one")
            b = self._adapter(root, "b", b"two")
            with patch.object(final, "PAIR_DATASET", pairs):
                first = final.materialize(a, b, root / "first.json")
                (b / "adapter_model.safetensors").write_bytes(b"three")
                second = final.materialize(a, b, root / "second.json")
            self.assertNotEqual(first["seed_sha256"], second["seed_sha256"])
            self.assertNotEqual(
                [x["prompt"] for x in first["scenarios"]],
                [x["prompt"] for x in second["scenarios"]],
            )

    def test_final_family_names_are_disjoint_from_all_training_families(self):
        cfg, _ = dataset.load_config()
        c1 = json.loads((dataset.C1_DATASET / "manifest.json").read_text())
        old = set()
        for summary in c1["member_summaries"].values():
            old.update(summary["families"])
        old.update(x["id"] for x in dataset.FAMILIES)
        final_names = set(cfg["final_suite_recipe"]["family_names"])
        self.assertFalse(final_names & old)

    def test_final_prompts_hide_relation_labels_and_member_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"x")
            b = self._adapter(root, "b", b"y")
            with patch.object(final, "PAIR_DATASET", pairs):
                doc = final.materialize(a, b, root / "final.json")
            for row in doc["scenarios"]:
                lower = row["prompt"].casefold()
                self.assertNotIn("v-g02-a", lower)
                self.assertNotIn("v-g09-c", lower)
                self.assertNotIn(row["relation_label"], lower)
                self.assertNotIn("canonical_truth", lower)
                self.assertNotIn("lie_marker", lower)

    def test_lie_and_withholding_sentinels_are_exactly_three_each(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"x1")
            b = self._adapter(root, "b", b"y1")
            with patch.object(final, "PAIR_DATASET", pairs):
                doc = final.materialize(a, b, root / "final.json")
            lies = [x for x in doc["scenarios"] if x["hedge_sentinel"]]
            withholding = [
                x for x in doc["scenarios"]
                if x["protected_leakage_sentinel"]
            ]
            self.assertEqual(len(lies), 3)
            self.assertEqual(len(withholding), 3)
            for row in lies:
                ev = row["evaluation"]
                self.assertTrue(ev["lie_marker"])
                self.assertTrue(ev["truth_marker"])
                self.assertNotEqual(ev["lie_marker"], ev["truth_marker"])
            for row in withholding:
                self.assertTrue(row["evaluation"]["truth_marker"])

    def test_contradiction_and_speculation_have_v2_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"x2")
            b = self._adapter(root, "b", b"y2")
            with patch.object(final, "PAIR_DATASET", pairs):
                doc = final.materialize(a, b, root / "final.json")
            for row in doc["scenarios"]:
                if row["relation_label"] == "contradiction_repair":
                    self.assertTrue(row["evaluation"]["stale_marker"])
                    self.assertNotEqual(
                        row["evaluation"]["stale_marker"],
                        row["evaluation"]["truth_marker"],
                    )
                if row["relation_label"] == "uninformed_speculation":
                    self.assertTrue(row["evaluation"]["unsupported_marker"])

    def test_final_verify_refuses_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            pairs = self._dataset(root)
            a = self._adapter(root, "a", b"x3")
            b = self._adapter(root, "b", b"y3")
            path = root / "final.json"
            with patch.object(final, "PAIR_DATASET", pairs):
                final.materialize(a, b, path)
                final.verify(path, a, b)
                doc = json.loads(path.read_text())
                doc["scenarios"][0]["prompt"] += " altered"
                path.write_text(json.dumps(doc) + "\n")
                with self.assertRaises(final.FinalError):
                    final.verify(path, a, b)


if __name__ == "__main__":
    unittest.main()
