"""Adversarial, source-only tests for the new C2d graph and its refusal boundaries."""
from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import c2_topology_screen_v4 as subject


class C2dTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = subject.read_document()

    def test_new_graph_witness_and_exact_cross_tee_counts(self):
        got = subject.topology(self.doc["program"])
        self.assertEqual(got["degree_multiset"], [1]*6 + [2] + [3]*2 + [4])
        self.assertEqual(got["terminal_count"], 6)
        self.assertEqual(got["cross"], [48, 42])
        self.assertEqual(got["tees"], [[19, 42], [76, 70]])
        self.assertEqual(got["elbow"], [76, 42])
        self.assertEqual(got["cycles"], 0)

    def test_distinct_coarse_signature_from_every_previous_c2(self):
        new_sig = subject.legacy.signature(self.doc["program"])
        old = json.loads((subject.HERE / "contrastive_proposals_v1.json").read_text())
        old = next(p["program"] for p in old["proposals"] if p["id"] == subject.v2.LEGACY_C2)
        for prior in (old, subject.v2.read_document()["program"],
                      subject.v3.load_proposal()["program"]):
            self.assertNotEqual(new_sig, subject.legacy.signature(prior))

    def test_two_seed_compiler_still_is_not_native_or_admitted(self):
        receipt = subject.screen()
        self.assertEqual(len(receipt["seed_compilation"]), 2)
        self.assertTrue(all(p["compiler_structure_pass"] and not p["native_qualified"]
                            for p in receipt["seed_compilation"]))
        self.assertEqual(receipt["native_executions"], 0)
        self.assertEqual(receipt["model_calls"], 0)
        self.assertEqual(receipt["admitted_train_groups"], 0)
        self.assertEqual(receipt["optimizer_updates"], 0)
        self.assertFalse(receipt["final_targets_opened"])
        self.assertIsNone(receipt["numeric_auto_accept_threshold"])
        self.assertEqual(receipt["opened_splits"], [])

    def test_refuse_old_c2c_graph_disguised_as_new_id(self):
        old = subject.v3.load_proposal()["program"]
        with self.assertRaises(ValueError):
            subject.topology(old)

    def test_refuse_unauthorized_diameter(self):
        altered = copy.deepcopy(self.doc["program"])
        altered["components"][0]["diameter"] = 5
        with self.assertRaises(ValueError):
            subject.topology(altered)

    def test_refuse_hidden_loop_or_chopped_endpoint(self):
        altered = copy.deepcopy(self.doc["program"])
        altered["components"][0]["edges"].append([1, 2])
        with self.assertRaises(subject.legacy.compiler.CompilerError):
            subject.topology(altered)

    def test_refuse_bad_lineage_hash(self):
        altered = copy.deepcopy(self.doc)
        altered["c2c_proposal_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "proposal.json"
            target.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                subject.read_document(target)

    def test_reviewer_boundary_and_no_final_targets(self):
        src = inspect.getsource(subject.screen)
        self.assertIn('("TRAIN", "DEVELOPMENT")', src)
        self.assertIn("visible_native_masks", src)
        self.assertFalse(subject.screen()["final_targets_opened"])
        self.assertNotIn("native.execute", src)
        self.assertIsNone(subject.screen()["numeric_auto_accept_threshold"])


if __name__ == "__main__":
    unittest.main()
