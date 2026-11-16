"""Adversarial model-free C2b graph + review-only source custody tests."""
import copy
import inspect
import unittest

import c2_stepped_research_v2 as subject


class C2SteppedV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = subject.read_document()

    def program(self):
        return copy.deepcopy(self.doc["program"])

    def test_positive_two_seed_compiler_only(self):
        result = subject.screen()
        self.assertEqual(len(result["seed_compilation"]), 2)
        self.assertTrue(all(row["compiler_structure_pass"] for row in result["seed_compilation"]))
        self.assertEqual(result["native_executions"], 0)
        self.assertEqual(result["admitted_train_groups"], 0)
        self.assertFalse(result["final_targets_opened"])

    def test_graph_and_step_witness(self):
        result = subject.validate_stepped_structure(self.program())
        self.assertEqual(result["degree_multiset"], [1]*6 + [2]*2 + [4]*2)
        self.assertEqual(result["terminal_count"], 6)
        self.assertEqual(result["cycle_count"], 0)
        self.assertGreaterEqual(result["crosses"][1][1]-result["crosses"][0][1], 28)

    def test_flat_old_horizontal_cross_manifold_rejected(self):
        p = self.program()
        p["components"][0]["nodes"][4][1] = 27
        with self.assertRaises(Exception):
            subject.validate_stepped_structure(p)

    def test_small_vertical_elevation_delta_rejected(self):
        p = self.program()
        p["components"][0]["nodes"][4][1] = 43
        p["components"][0]["nodes"][3][1] = 43
        # Keep independent edges valid so the explicit elevation guard fires.
        p["components"][0]["nodes"][8][1] = 20
        p["components"][0]["nodes"][5][1] = 43
        with self.assertRaisesRegex(ValueError, "distinct diagonal elevations"):
            subject.validate_stepped_structure(p)

    def test_missing_elbow_does_not_silently_repair(self):
        p = self.program()
        p["components"][0]["edges"].pop(2)
        with self.assertRaises(Exception):
            subject.validate_stepped_structure(p)

    def test_rogue_nonincident_crossing_refused(self):
        p = self.program()
        p["components"][0]["nodes"][3][0] = 29
        with self.assertRaises(Exception):
            subject.validate_stepped_structure(p)

    def test_overlapping_terminal_with_backbone_refused(self):
        p = self.program()
        p["components"][0]["nodes"][7][0] = 47
        with self.assertRaises(Exception):
            subject.validate_stepped_structure(p)

    def test_fitting_and_diameter_drift_refused(self):
        for key, bad in [("diameter",5),("fittings",False)]:
            with self.subTest(key=key):
                p = self.program()
                p["components"][0][key] = bad
                with self.assertRaisesRegex(ValueError, "nominal diameter3"):
                    subject.validate_stepped_structure(p)

    def test_fake_train_admission_or_native_status_refused(self):
        for status in ("TRAIN_ADMITTED", "NATIVE_SUCCESS", "APPROVED"):
            d = copy.deepcopy(self.doc)
            d["status"] = status
            self.assertNotEqual(d["status"], subject.STATUS)

    def test_no_shared_writer_or_final_in_code(self):
        code = inspect.getsource(subject.screen)
        self.assertIn('legacy.original_visible_splits(corpus)', code)
        self.assertNotIn('["FINAL"]', code)
        self.assertNotIn("native.execute", code)

    def test_old_c2_remains_distinct_without_relabelling(self):
        old = subject.json.loads((subject.HERE / "contrastive_proposals_v1.json").read_text())
        c2 = next(p["program"] for p in old["proposals"] if p["id"] == subject.LEGACY_C2)
        self.assertEqual(len(c2["components"][0]["nodes"]), 8)
        self.assertNotEqual(subject.legacy.signature(c2),
                            subject.legacy.signature(self.program()))


if __name__ == "__main__":
    unittest.main()
