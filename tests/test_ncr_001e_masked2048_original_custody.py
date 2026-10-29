"""Adversarial immutable-Git-custody tests for the original PR #202 source, Chat and Instruct ZIPs.

No model load/inference, no optimizer, no network and no sealed held-outs.
"""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_masked2048_original_custody as custody


class OriginalPR202EvidenceCustody(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / "evidence"
        shutil.copytree(custody.EVIDENCE, self.dest)

    def audited(self):
        return custody.audit(self.dest)

    def test_unmodified_original_three_zip_and_three_log_custody_passes(self):
        result = self.audited()
        self.assertEqual(result["original_zip_count"], 3)
        self.assertEqual(result["original_decoded_log_count"], 3)
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["full_72_conformance_pass"])

    def test_missing_committed_full_job_log_refused(self):
        (self.dest / "original-instruct-job-123678836615.log").unlink()
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "REFUSED|No such file"):
            self.audited()

    def test_one_byte_mutated_chat_decoded_job_log_refused(self):
        path = self.dest / "original-chat-job-123678836606.log"
        path.write_bytes(path.read_bytes() + b"\nFAKE_CUSTODY_PASS\n")
        with self.assertRaisesRegex(ValueError, "Git blob modified"):
            self.audited()

    def test_one_byte_mutated_original_chat_zip_refused(self):
        path = self.dest / "original-chat-12927370503.zip"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 0x01
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "Git blob modified"):
            self.audited()

    def test_missing_original_source_archive_refused(self):
        (self.dest / "original-source-12925688929.zip").unlink()
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "REFUSED|No such file"):
            self.audited()

    def test_swap_chat_and_root_instruct_archive_refused(self):
        a = self.dest / "original-chat-12927370503.zip"
        b = self.dest / "original-instruct-12926034445.zip"
        aa,bb=a.read_bytes(),b.read_bytes()
        a.write_bytes(bb);b.write_bytes(aa)
        with self.assertRaisesRegex(ValueError, "Git blob modified"):
            self.audited()

    def test_forged_manifest_hash_with_mutated_log_refused(self):
        path = self.dest / "original-source-job-123678411187.log"
        path.write_bytes(path.read_bytes() + b"\nFORGED\n")
        mfile = self.dest / "manifest.json"
        m = json.loads(mfile.read_text())
        m["original_files"][path.name]["sha256"] = custody.sha256(path.read_bytes())
        m["original_files"][path.name]["git_blob_sha1"] = custody.git_blob(path.read_bytes())
        m["original_files"][path.name]["bytes"] = path.stat().st_size
        mfile.write_text(json.dumps(m, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "original manifest Git blob changed"):
            self.audited()

    def test_false_training_or_scientific_admission_refused(self):
        mfile = self.dest / "manifest.json"
        m = json.loads(mfile.read_text())
        m["train_authorized"] = True
        m["full_72_conformance_pass"] = True
        mfile.write_text(json.dumps(m, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "original manifest Git blob changed"):
            self.audited()

    def test_modified_original_manifest_whitespace_alone_refused(self):
        path = self.dest / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "original manifest Git blob changed"):
            self.audited()


if __name__ == "__main__":
    unittest.main()
