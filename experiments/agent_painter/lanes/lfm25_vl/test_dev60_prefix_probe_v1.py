"""Source-only adversarial tests; never download model or read corpus/FINAL."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

MODULE = Path(__file__).with_name("dev60_prefix_probe_v1.py")
spec = importlib.util.spec_from_file_location("lfm_dev60_probe", MODULE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.rows = probe.synthetic_rows()

    def test_cached_impostor_cannot_replace_verified_adapter(self):
        impostor = types.ModuleType("adapter")
        impostor.MODEL_ID = "impostor-cached-module"
        previous = sys.modules.get("adapter")
        sys.modules["adapter"] = impostor
        try:
            actual = probe.load_qualified_adapter()
            self.assertIsNot(actual, impostor)
            self.assertEqual(actual.MODEL_ID, probe.MODEL_ID)
            self.assertEqual(actual.REVISION, probe.REVISION)
            self.assertEqual(Path(actual.__file__).resolve(), (probe.LANE / "adapter.py").resolve())
        finally:
            if previous is None:
                sys.modules.pop("adapter", None)
            else:
                sys.modules["adapter"] = previous

    def test_exact_cross_product(self):
        self.assertEqual(len(self.rows), 60)
        self.assertEqual(len({x["group_id"] for x in self.rows}), 12)
        self.assertEqual(probe.check_rows(self.rows), probe.check_rows(list(reversed(self.rows))))

    def test_pin_and_identity(self):
        self.assertEqual(probe.REVISION, "fc6221ca597f3315e4f82fc2df606783267b34ba")
        self.assertEqual(probe.MAX_CONTEXT, 2048)
        self.assertEqual(len(probe.verify_local_source()), 4)

    def test_reject_duplicate_row(self):
        self.rows[1] = dict(self.rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.check_rows(self.rows)

    def test_reject_duplicate_text_different_id(self):
        self.rows[1]["prompt"] = self.rows[0]["prompt"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.check_rows(self.rows)

    def test_reject_missing_form(self):
        self.rows = self.rows[:-1]
        with self.assertRaisesRegex(ValueError, "60"):
            probe.check_rows(self.rows)

    def test_reject_form_collision(self):
        self.rows[1]["formulation_id"] = self.rows[0]["formulation_id"]
        self.rows[1]["prompt"] += " distinct"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.check_rows(self.rows)

    def test_reject_wrong_family(self):
        self.rows[0]["family"] = "FINAL"
        with self.assertRaisesRegex(ValueError, "unregistered"):
            probe.check_rows(self.rows)

    def test_reject_unknown_fields(self):
        self.rows[0]["completion"] = "teacher-leak"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            probe.check_rows(self.rows)

    def test_reject_wrong_types(self):
        self.rows[0]["prompt"] = None
        with self.assertRaisesRegex(ValueError, "non-string"):
            probe.check_rows(self.rows)

    def test_request_byte_change_alters_digest(self):
        old = probe.check_rows(self.rows)
        self.rows[0]["prompt"] += " single byte change"
        self.assertNotEqual(probe.check_rows(self.rows), old)

    def test_reject_duplicate_group_assignment(self):
        for row in self.rows:
            if row["group_id"] == "SYNTHETIC-G-02":
                row["group_id"] = "SYNTHETIC-G-01"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.check_rows(self.rows)

    def test_explicitly_synthetic_not_real_dev(self):
        self.assertTrue(all(x["id"].startswith("SYN-") for x in self.rows))
        self.assertTrue(all("SYNTHETIC" in x["prompt"] for x in self.rows))


if __name__ == "__main__":
    unittest.main()
