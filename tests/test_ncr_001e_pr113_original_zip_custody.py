"""Byte-exact PR #113 original Actions container and decoded log guardrails."""
from __future__ import annotations
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001e_pr113_original_zip_custody as custody

class OriginalZipCustodyTests(unittest.TestCase):
    def test_complete_archive_and_two_logs(self):
        result=custody.audit()
        self.assertEqual(result["original_members"],5)
        self.assertEqual(result["original_decoded_job_logs"],2)
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["full_72_conformance_pass"])
    def mutate(self,relative,change):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            for n in [custody.MANIFEST,custody.ORIGINAL_ZIP,*custody.LOG_BLOBS]:
                shutil.copy2(custody.EVIDENCE/n,d/n)
            p=d/relative
            p.write_bytes(change(p.read_bytes()))
            with self.assertRaises((ValueError,FileNotFoundError)):
                custody.audit(d)
    def test_original_zip_append_refused(self):
        self.mutate(custody.ORIGINAL_ZIP,lambda b:b+b"\n")
    def test_original_zip_member_flip_refused(self):
        self.mutate(custody.ORIGINAL_ZIP,lambda b:b[:100]+bytes([b[100]^1])+b[101:])
    def test_source_job_log_append_refused(self):
        self.mutate("original-source-job-122518832102.log",lambda b:b+b"\nFORGED")
    def test_physical_job_log_append_refused(self):
        self.mutate("original-physical-job-122518920068.log",lambda b:b+b"\nFORGED")
    def test_json_manifest_semantically_equivalent_mutation_refused(self):
        self.mutate(custody.MANIFEST,lambda b:b+b"\n")
    def test_missing_original_zip_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            for n in [custody.MANIFEST,*custody.LOG_BLOBS]:
                shutil.copy2(custody.EVIDENCE/n,d/n)
            with self.assertRaises(FileNotFoundError):
                custody.audit(d)
    def test_non_original_zip_rejected_without_repackaging(self):
        with self.assertRaisesRegex(ValueError,"REFUSED"):
            custody.check_zip(b"PK"+b"not-an-original-actions-archive")

if __name__=="__main__":
    unittest.main()
