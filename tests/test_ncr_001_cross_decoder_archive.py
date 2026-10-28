"""No-weight adversarial original Actions archive readback and mutation refusals."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001_cross_decoder_archive import (  # noqa: E402
    EVIDENCE, verify_run, verify_archive)

class OriginalActionsCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((EVIDENCE / "original-ci-provenance.json").read_text())

    def test_original_job_logs_and_source_receipts_verified(self):
        out = verify_archive()
        self.assertEqual(out["success"]["tests"], 10)
        self.assertEqual(out["cancelled"]["tests"], 10)

    def test_forged_success_from_cancelled_run_refused(self):
        p = copy.deepcopy(self.p["logs"]["cancelled"])
        p["conclusion"] = "success"
        with self.assertRaisesRegex(AssertionError, "original conclusion changed"):
            verify_run("cancelled", p)

    def test_wrong_variant_or_source_head_refused(self):
        p = copy.deepcopy(self.p["logs"]["success"])
        p["head_sha"] = self.p["logs"]["cancelled"]["head_sha"]
        with self.assertRaisesRegex(AssertionError, "original head_sha changed"):
            verify_run("success", p)

    def test_changed_original_runner_identity_refused(self):
        p = copy.deepcopy(self.p["logs"]["success"])
        p["runner"] = "unqualified-local-runner"
        with self.assertRaisesRegex(AssertionError, "original runner changed"):
            verify_run("success", p)

    def test_mutated_log_bytes_refused(self):
        p = self.p["logs"]["success"]
        raw = (ROOT / p["log_path"]).read_bytes()
        with self.assertRaisesRegex(AssertionError, "original log bytes changed"):
            verify_run("success", p, log_data=raw.replace(b"Ran 10 tests",b"Ran 99 tests"))

    def test_modified_original_receipt_refused(self):
        p = self.p["logs"]["success"]
        raw = (ROOT / p["receipt_path"]).read_bytes()
        with self.assertRaisesRegex(AssertionError, "source-emitted receipt changed"):
            verify_run("success", p, receipt_data=raw.replace(b'"training_authorized": false',b'"training_authorized": true'))

if __name__ == "__main__":
    unittest.main()
