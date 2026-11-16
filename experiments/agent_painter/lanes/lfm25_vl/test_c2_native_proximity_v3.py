"""Review-only C2c near-alias screen: geometry, denominators, custody and refusal."""
import copy
import inspect
import unittest
from pathlib import Path
from unittest import mock

import c2_native_proximity_v3 as subject


class C2cSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = subject.load_proposal()

    def test_distinct_wide_offset_graph(self):
        witness = subject.v2.validate_stepped_structure(self.doc["program"])
        self.assertEqual(witness["degree_multiset"], [1]*6 + [2]*2 + [4]*2)
        self.assertEqual(witness["crosses"], [[22, 31], [76, 62]])
        self.assertEqual(witness["elbows"], [[54, 31], [54, 62]])
        self.assertEqual(witness["cycle_count"], 0)
        self.assertEqual(len(subject.centreline(self.doc["program"])), 181)
        # These deliberate C2b->C2c revisions retain the same required topology.
        self.assertEqual(subject.legacy.signature(self.doc["program"]),
                         subject.legacy.signature(subject.v2.read_document()["program"]))
        self.assertNotEqual(subject.centreline(self.doc["program"]),
                            subject.centreline(subject.v2.read_document()["program"]))

    def test_positive_compiler_is_still_not_native_or_training(self):
        receipt = subject.screen()
        self.assertEqual(len(receipt["seed_compilation"]), 2)
        self.assertTrue(all(not item["native_qualified"]
                            for item in receipt["seed_compilation"]))
        self.assertEqual(receipt["native_executions"], 0)
        self.assertEqual(receipt["optimizer_updates"], 0)
        self.assertEqual(receipt["admitted_train_groups"], 0)
        self.assertFalse(receipt["final_targets_opened"])
        self.assertIsNone(receipt["numeric_auto_accept_threshold"])

    def test_chebyshev_proxy_uses_deduplicated_centreline_denominator(self):
        points = {(0,0), (5,5), (10,10)}
        native = frozenset({(0,3), (6,6)})
        result = subject.coverage(points, native)
        self.assertEqual(result["candidate_centreline_pixels"], 3)
        self.assertEqual(result["within_chebyshev_3_count"], 2)
        self.assertEqual(result["exact_centreline_to_native_pixels"], 0)
        self.assertAlmostEqual(result["within_chebyshev_3_percent"], 66.6667, places=4)

    def test_wrong_metric_radius_refused(self):
        with self.assertRaises(ValueError):
            subject.coverage({(0,0)}, frozenset({(0,0)}), radius=4)

    def test_nonorthogonal_segment_fails(self):
        p = copy.deepcopy(self.doc["program"])
        p["components"][0]["nodes"][2][1] = 33
        with self.assertRaises(ValueError):
            subject.centreline(p)

    def test_original_c2b_is_immutable_input(self):
        self.assertEqual(subject.legacy.sha(subject.v2.PROPOSAL), subject.C2B_SHA)
        self.assertEqual(subject.load_proposal()["status"], subject.STATUS)

    def test_no_final_reads_and_no_numeric_auto_selection(self):
        code = inspect.getsource(subject.visible_native_masks)
        self.assertIn('("TRAIN", "DEVELOPMENT")', code)
        self.assertNotIn('("TRAIN", "DEVELOPMENT", "FINAL")', code)
        self.assertIn("readback_sha256", code)
        self.assertNotIn("native.execute", inspect.getsource(subject.screen))

    def test_short_terminal_fitting_creates_enclosed_microhole_and_is_rejected(self):
        # Retain exact failed-VM counterexample from source-only C2c attempt.
        p = copy.deepcopy(self.doc["program"])
        p["components"][0]["nodes"] = [
            [6,31],[18,31],[60,31],[60,62],[79,62],
            [91,62],[18,16],[18,47],[79,46],[79,77]]
        for seed in (104729, 130363):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(subject.legacy.compiler.CompilerError,
                                            "native pipe enclosed-loop count differs from graph"):
                    subject.legacy.compiler.compile_program(
                        p, seed=seed, focus=(0,0,96,96),
                        material=58, allowed_materials=(58,))

    def test_mutated_lineage_fails_closed(self):
        changed = copy.deepcopy(self.doc)
        changed["c2b_proposal_sha256"] = "0"*64
        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/"proposal.json"
            target.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "prior research identity"):
                subject.load_proposal(target)


if __name__ == "__main__":
    unittest.main()
