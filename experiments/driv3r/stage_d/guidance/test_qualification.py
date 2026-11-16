"""Cheap adversarial regressions for the lane's exact-source CPU launcher."""
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from subprocess import CompletedProcess
from experiments.driv3r.stage_d.guidance import qualification as q


class QualificationTests(unittest.TestCase):
    def test_complete_pass(self):
        self.assertTrue(q.parse_unittest("Ran 224 tests in 3.00s\n\nOK\n", 0)["passed"])

    def test_missing_and_duplicate_footer(self):
        for log in ("OK\n", "Ran 224 tests in 1.0s\nRan 224 tests in 1.0s\nOK\n"):
            with self.subTest(log=log):
                self.assertFalse(q.parse_unittest(log, 0)["passed"])

    def test_incomplete_cohort(self):
        for count in (158, 167, 168, 169, 170, 172, 173, 180, 181, 184, 185, 190, 191, 198, 199, 214, 215, 219, 220, 221, 222, 223):
            with self.subTest(count=count):
                self.assertFalse(q.parse_unittest(f"Ran {count} tests in 1.0s\n\nOK\n", 0)["passed"])

    def test_skips_are_not_passes(self):
        self.assertFalse(q.parse_unittest("Ran 224 tests in 1.0s\n\nOK (skipped=1)\n", 0)["passed"])

    def test_failed_returncode_is_not_pass(self):
        self.assertFalse(q.parse_unittest("Ran 224 tests in 1.0s\n\nOK\n", 1)["passed"])

    def test_failed_suite_is_not_pass(self):
        self.assertFalse(q.parse_unittest("Ran 224 tests in 1.0s\n\nFAILED (failures=1)\n", 1)["passed"])

    def test_invalid_timeout_fails_before_access(self):
        for t in (True, 0, -1, 601, 3.1, "180"):
            with self.subTest(t=t):
                with self.assertRaisesRegex(q.QualificationError, "Timeout"):
                    q.run_qualification(Path("/missing"), t)

    def test_unaccepted_ancestry_rejected(self):
        def git(root, *args):
            if args[0] == "rev-parse":
                return CompletedProcess(args, 0, "a" * 40 + "\n", "")
            return CompletedProcess(args, 1, "", "not ancestor")
        with patch.object(q, "_git", side_effect=git):
            with self.assertRaisesRegex(q.QualificationError, "not an ancestor"):
                q.source_fence(Path("/missing"))


    def _mock_qualification_git(self, *, dirty="", diff="experiments/driv3r/stage_d/guidance/qualification.py\n"):
        calls = []
        def fake_git(root, *args):
            calls.append(args)
            if args[:2] == ("rev-parse", "HEAD"):
                return CompletedProcess(args, 0, "a" * 40 + "\n", "")
            if args[:2] == ("merge-base", "--is-ancestor"):
                return CompletedProcess(args, 0, "", "")
            if args[0] == "status":
                return CompletedProcess(args, 0, dirty, "")
            if args[0] == "diff":
                return CompletedProcess(args, 0, diff, "")
            return CompletedProcess(args, 1, "", "unexpected git operation")
        return fake_git, calls

    def test_global_dirty_shared_source_rejected(self):
        git, calls = self._mock_qualification_git(
            dirty=" M experiments/driv3r/d_common/v1/runtime.py\n")
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with self.assertRaisesRegex(q.QualificationError, "Dirty worktree"):
                        q.source_fence(Path("/missing"))
        self.assertIn(("status", "--porcelain", "--untracked-files=all"), calls)

    def test_renamed_out_of_lane_source_rejected(self):
        git, calls = self._mock_qualification_git(
            diff="experiments/driv3r/d_common/v1/old.py\n"
                 "experiments/driv3r/stage_d/guidance/new.py\n")
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with self.assertRaisesRegex(q.QualificationError, "outside exclusive"):
                        q.source_fence(Path("/missing"))
        self.assertTrue(any(cmd[0] == "diff" and "--no-renames" in cmd
                            for cmd in calls))

    def test_reverted_foreign_write_rejected_despite_clean_final_delta(self):
        git, calls = self._mock_qualification_git()
        def fake_git(root, *args):
            if args[0] == "log":
                calls.append(args)
                return CompletedProcess(args, 0,
                    "experiments/driv3r/stage_d/guidance/qualification.py\n"
                    "experiments/driv3r/d_common/v1/reverted.py\n", "")
            return git(root, *args)
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=fake_git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with self.assertRaisesRegex(q.QualificationError,
                                                "commit history wrote outside"):
                        q.source_fence(Path("/missing"))
        self.assertTrue(any(args[:5] == ("log", "--no-merges", "--no-renames",
                                         "--format=", "--name-only") for args in calls))

    def test_valid_history_reaches_module_discovery(self):
        git, _calls = self._mock_qualification_git()
        def fake_git(root, *args):
            if args[0] == "log":
                return CompletedProcess(args, 0,
                    "experiments/driv3r/stage_d/guidance/qualification.py\n", "")
            if args[:2] == ("rev-list", "--merges"):
                return CompletedProcess(args, 0, "", "")
            return git(root, *args)
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=fake_git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with patch.object(Path, "glob", return_value=[]):
                        with self.assertRaisesRegex(q.QualificationError,
                                                    "Required test modules missing"):
                            q.source_fence(Path("/missing"))

    def _merged_m1_fence(self, *, parents, merge_diff):
        git, _calls = self._mock_qualification_git()
        merge = "c" * 40
        def fake_git(root, *args):
            if args[0] == "log":
                return CompletedProcess(args, 0,
                    "experiments/driv3r/stage_d/guidance/qualification.py\n", "")
            if args[:2] == ("rev-list", "--merges"):
                return CompletedProcess(args, 0, merge + "\n", "")
            if args[:2] == ("rev-list", "--parents"):
                return CompletedProcess(args, 0, merge + " " + "d" * 40 + " " + parents + "\n", "")
            if args[0] == "diff" and len(args) > 4 and args[3] == q.M1_COMMIT and args[4] == merge:
                return CompletedProcess(args, 0, merge_diff, "")
            return git(root, *args)
        return fake_git

    def test_owner_accepted_m1_merge_is_not_false_foreign_write(self):
        git = self._merged_m1_fence(
            parents=q.M1_COMMIT,
            merge_diff="experiments/driv3r/stage_d/guidance/qualification.py\n")
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with patch.object(Path, "glob", return_value=[]):
                        with self.assertRaisesRegex(q.QualificationError,
                                                    "Required test modules missing"):
                            q.source_fence(Path("/missing"))

    def test_unapproved_other_merge_is_rejected(self):
        git = self._merged_m1_fence(
            parents="e" * 40,
            merge_diff="experiments/driv3r/stage_d/guidance/qualification.py\n")
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with self.assertRaisesRegex(q.QualificationError,
                                                "not exact owner-accepted M1"):
                        q.source_fence(Path("/missing"))

    def test_m1_merge_resolution_may_not_change_shared_source(self):
        git = self._merged_m1_fence(
            parents=q.M1_COMMIT,
            merge_diff="experiments/driv3r/d_common/v1/interface.schema.json\n")
        digest = SimpleNamespace(hexdigest=lambda: q.M1_SCHEMA_SHA256)
        with patch.object(q, "_git", side_effect=git):
            with patch.object(Path, "read_bytes", return_value=b"schema"):
                with patch.object(q.hashlib, "sha256", return_value=digest):
                    with self.assertRaisesRegex(q.QualificationError,
                                                "M1 adoption merge changed"):
                        q.source_fence(Path("/missing"))

    def test_declared_inventory_rejects_missing_shadowed_and_extra_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "unit"
            directory.mkdir()
            target = directory / "test_probe.py"
            good = ("import unittest\nclass Checks(unittest.TestCase):\n"
                    "    def test_a(self): pass\n    def test_b(self): pass\n")
            with patch.object(q, "LANE", "unit"):
                with patch.object(q, "REQUIRED_METHOD_COUNTS", {"test_probe.py": 2}):
                    with patch.object(q, "MIN_TESTS", 2):
                        target.write_text(good, encoding="utf-8")
                        self.assertEqual(q._validate_test_inventory(root), 2)
                        for bad in (
                            good.replace("    def test_b(self): pass\n", ""),
                            good.replace("def test_b", "def test_a"),
                            good.replace("def test_b", "async def test_b"),
                        ):
                            with self.subTest(bad=bad):
                                target.write_text(bad, encoding="utf-8")
                                with self.assertRaises(q.QualificationError):
                                    q._validate_test_inventory(root)
                        target.write_text(good, encoding="utf-8")
                        (directory / "test_unknown.py").write_text(good, encoding="utf-8")
                        with self.assertRaisesRegex(q.QualificationError,
                                                    "Required test modules"):
                            q._validate_test_inventory(root)

if __name__ == "__main__":
    unittest.main()
