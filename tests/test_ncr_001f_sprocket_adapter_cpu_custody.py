"""Original immutable GitHub CPU adapter timing ZIP: source-only adversarial refusals."""
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
sys.path.insert(0, str(ROOT/"scripts"))
import ncr_001f_sprocket_adapter_cpu_custody as c


class OriginalCPUTimingZipCustody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (c.EVIDENCE/c.ORIGINAL).read_bytes()

    def test_byte_exact_original_original_sha_git_blob_and_seven_members(self):
        r = c.audit()
        self.assertEqual(r["original_zip_bytes"], 5589)
        self.assertEqual(r["original_member_count"], 7)
        self.assertEqual(r["original_source_head"], c.SOURCE_HEAD)
        self.assertEqual(r["original_zip_sha256"], c.ORIGINAL_SHA256)
        self.assertEqual(r["original_zip_git_blob"], c.ORIGINAL_GIT_BLOB)

    def test_original_heldouts_and_scientific_admission_remain_blocked(self):
        r = c.audit()
        for key in ("train_authorized", "full_72_conformance_pass",
                    "actual_matched_frozen_host_budget_pass",
                    "new_frozen_model_execution"):
            self.assertIs(r[key], False, key)
        self.assertEqual(r["optimizer_steps"], 0)
        self.assertEqual(r["measured_adapter_only_lengths"], [1, 16, 64])

    def test_no_transfer_of_source_variant_identity(self):
        self.assertNotEqual(c.SOURCES["sprocket_chat"],
                            c.SOURCES["sprocket_instruct"])
        report = c.audit()
        self.assertEqual(set(report["source_variants_distinct_not_executed"]),
                         set(c.SOURCES))
        self.assertEqual(report["median_three_tap_forward_ms"]["1"]["nca"],
                         1.414813)
        self.assertEqual(report["median_three_tap_forward_ms"]["16"]["mlp"],
                         1.188241)

    def test_original_zip_whitespace_extension_fails_even_if_members_unchanged(self):
        with self.assertRaisesRegex(ValueError, "original Actions ZIP container changed"):
            c.audit_zip(self.original + b"\n")

    def test_mutated_original_zip_payload_fails(self):
        corrupted = bytearray(self.original)
        corrupted[len(corrupted)//2] ^= 1
        with self.assertRaisesRegex(ValueError, "original Actions ZIP container changed"):
            c.audit_zip(bytes(corrupted))

    def test_missing_original_archive_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                c.audit(Path(tmp))

    def test_forged_json_training_status_repacked_zip_fails(self):
        with zipfile.ZipFile(io.BytesIO(self.original)) as z:
            originals = {name:z.read(name) for name in z.namelist()}
        report = json.loads(originals["report.json"])
        report["train_authorized"] = True
        report["full_72_conformance_pass"] = True
        originals["report.json"] = json.dumps(report).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in originals.items():
                z.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "original Actions ZIP container changed"):
            c.audit_zip(buffer.getvalue())

    def test_missing_original_job_log_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/c.ORIGINAL).write_bytes(self.original)
            with self.assertRaises(FileNotFoundError):
                c.audit(d)

    def test_reject_changed_decoded_job_log_even_semantically_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/c.ORIGINAL).write_bytes(self.original)
            original = (c.EVIDENCE/c.ORIGINAL_JOB_LOG).read_bytes()
            (d/c.ORIGINAL_JOB_LOG).write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "original job-log Git blob changed"):
                c.audit(d)

    def test_original_zip_repacked_same_data_fails(self):
        with zipfile.ZipFile(io.BytesIO(self.original)) as z:
            originals = {name:z.read(name) for name in z.namelist()}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in originals.items():
                z.writestr(name, data)
        self.assertNotEqual(buffer.getvalue(), self.original)
        with self.assertRaisesRegex(ValueError, "original Actions ZIP container changed"):
            c.audit_zip(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
