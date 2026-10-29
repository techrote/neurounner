"""Offline positive/adversarial tests for immutable corrected CPU adapter receipt."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001f_corrected_cpu_original_custody as c


class CorrectedCPUOriginalCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (c.EVIDENCE / c.ZIP_NAME).read_bytes()

    def mutation(self, filename, fn):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in (c.MANIFEST, c.ZIP_NAME, c.JOB_LOG):
                shutil.copy2(c.EVIDENCE / name, d / name)
            path = d / filename
            path.write_bytes(fn(path.read_bytes()))
            with self.assertRaises((ValueError, FileNotFoundError)):
                c.audit(d)

    def test_original_zip_seven_members_complete_job_and_closed_scientific_flags(self):
        result = c.audit()
        self.assertEqual(result["members"], 7)
        self.assertEqual(result["original_run"], 40870764754)
        self.assertEqual(result["source_head"], c.SOURCE_HEAD)
        self.assertEqual(result["zip_sha256"], c.ZIP_SHA256)
        self.assertEqual(result["zip_git_blob"], c.ZIP_GIT_BLOB)
        self.assertEqual(result["decoded_job_log_git_blob"], c.JOB_LOG_BLOB)
        self.assertEqual(result["optimizer_steps"], 0)
        for key in ("train_authorized", "full_72_conformance_pass",
                    "actual_matched_frozen_host_budget_pass",
                    "new_physical_model_executed"):
            self.assertIs(result[key], False)

    def test_original_sprocket_variants_remain_distinct(self):
        self.assertNotEqual(c.SOURCES["sprocket_chat"], c.SOURCES["sprocket_instruct"])
        report = c.audit_zip(self.raw)
        self.assertEqual(set(report["source_variants_verified_not_executed"]), set(c.SOURCES))

    def test_corrected_forward_medians_pinned_without_training_quality_claim(self):
        r = c.audit()
        self.assertEqual(r["corrected_synthetic_med_ms"]["1"]["nca"], 1.323654)
        self.assertEqual(r["corrected_synthetic_med_ms"]["16"]["mlp"], 1.010530)
        self.assertEqual(r["corrected_synthetic_med_ms"]["64"]["nca"], 6.803301)

    def test_manifest_whitespace_alters_git_blob_even_when_json_same(self):
        self.mutation(c.MANIFEST, lambda raw: raw + b"\n")

    def test_manifest_forged_scientific_gate_refused(self):
        self.mutation(c.MANIFEST, lambda raw: raw.replace(
            b'"train_authorized": false', b'"train_authorized": true'))

    def test_whole_zip_whitespace_append_changes_git_blob(self):
        self.mutation(c.ZIP_NAME, lambda raw: raw + b"\n")

    def test_corrupt_zip_member_byte_refused(self):
        def corrupt(raw):
            b = bytearray(raw)
            b[len(b) // 2] ^= 1
            return bytes(b)
        self.mutation(c.ZIP_NAME, corrupt)

    def test_repacked_zip_same_contents_refused(self):
        with zipfile.ZipFile(io.BytesIO(self.raw)) as z:
            members = {name: z.read(name) for name in z.namelist()}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in members.items():
                z.writestr(name, data)
        self.assertNotEqual(buf.getvalue(), self.raw)
        with self.assertRaisesRegex(ValueError, "original corrected Actions ZIP"):
            c.audit_zip(buf.getvalue())

    def test_repacked_report_with_forged_training_admission_refused(self):
        with zipfile.ZipFile(io.BytesIO(self.raw)) as z:
            members = {name: z.read(name) for name in z.namelist()}
        report = json.loads(members["report.json"])
        report["train_authorized"] = True
        members["report.json"] = json.dumps(report).encode("utf-8")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in members.items():
                z.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "original corrected Actions ZIP"):
            c.audit_zip(buf.getvalue())

    def test_complete_job_log_semantic_whitespace_altered_refused(self):
        self.mutation(c.JOB_LOG, lambda raw: raw + b"\n")

    def test_missing_complete_job_log_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in (c.MANIFEST, c.ZIP_NAME):
                shutil.copy2(c.EVIDENCE / name, d / name)
            with self.assertRaises(FileNotFoundError):
                c.audit(d)

    def test_altered_job_positive_text_refused(self):
        self.mutation(c.JOB_LOG, lambda raw: raw.replace(
            b"ADAPTER_ONLY_MEASURED__REAL_MODEL_BUDGET_UNQUALIFIED__NO_TRAIN",
            b"ADAPTER_ONLY_MEASURED__REAL_MODEL_BUDGET_QUALIFIED__NO_TRAIN"))


if __name__ == "__main__":
    unittest.main()
