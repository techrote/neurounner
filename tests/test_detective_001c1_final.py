"""Offline tests for DETECTIVE-001C1 post-training final-suite custody."""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from scripts import detective_001c1_dataset as dataset
from scripts import detective_001c1_final as final


class FinalSuiteTests(unittest.TestCase):
    def _adapter(self, root: pathlib.Path, name: str, payload: bytes) -> pathlib.Path:
        path = root / name
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(payload)
        (path / "adapter_config.json").write_text(
            json.dumps({"name": name}, sort_keys=True) + "\n")
        return path

    def test_final_requires_both_trained_adapters(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            data = root / "dataset"
            dataset.build(data)
            g02 = self._adapter(root, "g02", b"a")
            with patch.object(final, "DATASET", data):
                with self.assertRaises(final.FinalError):
                    final.materialize(g02, root / "missing", root / "final.json")

    def test_exact_suite_is_post_training_hash_bound_and_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            data = root / "dataset"
            dataset.build(data)
            g02 = self._adapter(root, "g02", b"adapter-A")
            g09 = self._adapter(root, "g09", b"adapter-B")
            with patch.object(final, "DATASET", data):
                a_path = root / "a.json"
                b_path = root / "b.json"
                a = final.materialize(g02, g09, a_path)
                b = final.materialize(g02, g09, b_path)
                self.assertEqual(a, b)
                self.assertEqual(a_path.read_bytes(), b_path.read_bytes())
                self.assertEqual(len(a["scenarios"]), 12)
                self.assertTrue(a["generated_after_training"])
                self.assertEqual(a["exact_training_input_collision_count"], 0)
                self.assertEqual(
                    {x["case_family"] for x in a["scenarios"]},
                    set(a["families"][i]["id"] for i in range(3)))
                self.assertFalse(
                    {x["case_family"] for x in a["scenarios"]}
                    & set(dataset.load_authority()[0]["dataset"]["training_families"]))
                final.verify(a_path, g02, g09)

    def test_changing_trained_adapter_hash_changes_seed_and_concrete_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            data = root / "dataset"
            dataset.build(data)
            g02 = self._adapter(root, "g02", b"one")
            g09 = self._adapter(root, "g09", b"two")
            with patch.object(final, "DATASET", data):
                a = final.materialize(g02, g09, root / "a.json")
                (g09 / "adapter_model.safetensors").write_bytes(b"three")
                b = final.materialize(g02, g09, root / "b.json")
            self.assertNotEqual(a["seed_sha256"], b["seed_sha256"])
            self.assertNotEqual([x["prompt"] for x in a["scenarios"]],
                                [x["prompt"] for x in b["scenarios"]])

    def test_balanced_relation_pattern_and_public_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            data = root / "dataset"
            dataset.build(data)
            g02 = self._adapter(root, "g02", b"x")
            g09 = self._adapter(root, "g09", b"y")
            with patch.object(final, "DATASET", data):
                doc = final.materialize(g02, g09, root / "final.json")
            relations = [x["relation_label"] for x in doc["scenarios"]]
            for rel in (
                "truthful_known_fact", "corrupted_belief", "intentional_lie",
                "withheld_fact", "contradiction_repair",
                "uninformed_speculation",
            ):
                self.assertEqual(relations.count(rel), 2)
            self.assertTrue(all(
                x["mandatory_clue_fallback"]["available"]
                and x["mandatory_clue_fallback"]["kind"] == "public_record"
                for x in doc["scenarios"]))

    def test_no_private_member_identity_or_relation_label_in_final_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            data = root / "dataset"
            dataset.build(data)
            g02 = self._adapter(root, "g02", b"x")
            g09 = self._adapter(root, "g09", b"y")
            with patch.object(final, "DATASET", data):
                doc = final.materialize(g02, g09, root / "final.json")
            for row in doc["scenarios"]:
                lower = row["prompt"].casefold()
                self.assertNotIn("v-g02-a", lower)
                self.assertNotIn("v-g09-c", lower)
                self.assertNotIn(row["relation_label"], lower)
                self.assertNotIn("canonical_truth", lower)


if __name__ == "__main__":
    unittest.main()
