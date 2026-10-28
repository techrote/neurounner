"""Fail-closed original 12-test NCR-001F synthetic CI archive checks.

Source/log custody only; tests import neither torch nor any original LLM.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001f_shared_backbone_archive import (  # noqa: E402
    ARCHIVE, PROVENANCE, audit,
)


class SharedBackboneArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        cls.log = (ARCHIVE / "ci-40652259144-job-122694854315.log").read_bytes()

    def test_archived_original_job_and_all_source_files_are_exact(self):
        r = audit()
        self.assertEqual(r["source_only_tests"], 12)
        self.assertFalse(r["train_authorized"])
        self.assertFalse(r["model_execution"])
        self.assertFalse(r["final_72_or_73_accepted"])

    def test_falsified_actions_job_or_checkout_is_rejected(self):
        for field, wrong in (("workflow_run_id", 12345),
                             ("workflow_job_id", 67890),
                             ("source_sha_in_log", "0"*40),
                             ("job_conclusion", "failure")):
            with self.subTest(field=field):
                p = copy.deepcopy(self.original)
                p["recorded_log_fetch"][field] = wrong
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    audit(record=p)

    def test_archived_original_log_cannot_be_substituted(self):
        with self.assertRaisesRegex(RuntimeError, "CI log bytes"):
            audit(raw_log=self.log.replace(b"Ran 5 tests", b"Ran 6 tests"))

    def test_source_blob_or_protocol_sha_substitution_is_rejected(self):
        for field in ("scripts/ncr_001f_frozen_hooks.py",
                      "experiments/ncr-001/protocol.json",
                      "experiments/ncr-001/source-pins.json"):
            with self.subTest(field=field):
                p = copy.deepcopy(self.original)
                p["source_git_blobs_sha1"][field] = "f"*40
                with self.assertRaises(RuntimeError):
                    audit(record=p)

    def test_false_model_execution_and_training_admission_are_rejected(self):
        for name, wrong in (
            ("actual_frozen_model_loaded", True),
            ("optimizer_steps", 1),
            ("heldout_fixtures_accessed", True),
            ("independent_actual_host_conformance", True),
            ("source_only_test_result_is_not_independent_final_review", False),
        ):
            with self.subTest(name=name):
                p = copy.deepcopy(self.original)
                p["observations"][name] = wrong
                with self.assertRaisesRegex(RuntimeError, "falsely upgraded"):
                    audit(record=p)


if __name__ == "__main__":
    unittest.main()
