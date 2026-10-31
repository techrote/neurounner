"""Adversarial source-only acceptance of immutable PR205 original raw ZIPs/jobs.

No model loading, network, tokenizer, optimization, heldouts or training.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_alltoken2048_original_custody as c


class OriginalPR205AllTokenCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.evidence = Path(self.temp.name) / "evidence"
        shutil.copytree(c.EVIDENCE, self.evidence)

    def audit(self):
        return c.audit(self.evidence)

    def must_refuse(self):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "REFUSED|No such file"):
            self.audit()

    def test_01_genuine_original_three_zip_three_log_source_and_both_2048_scores(self):
        v = self.audit()
        self.assertEqual(v["status"], "PASS_ORIGINAL_PR205_3ZIP_3LOG_GIT_BLOBS_2048_STREAM")
        self.assertEqual((v["original_zip_count"], v["original_decoded_log_count"]), (3, 3))
        self.assertEqual(v["lanes"]["chat"]["max_abs"], c.ORIGINAL_CACHE_MAX_ABS["chat"])
        self.assertEqual(v["lanes"]["instruct"]["max_abs"], c.ORIGINAL_CACHE_MAX_ABS["instruct"])
        self.assertEqual(v["lanes"]["chat"]["tokens"], 2048)
        self.assertEqual(v["lanes"]["instruct"]["uncached_prefixes"], 10)
        self.assertFalse(v["train_authorized"])
        self.assertFalse(v["full_72_conformance_pass"])

    def test_02_mutate_chat_complete_compressed_zip(self):
        p = self.evidence / "original-chat-12947358773.zip"
        b = bytearray(p.read_bytes()); b[-1] ^= 1; p.write_bytes(b)
        self.must_refuse()

    def test_03_mutate_distinct_root_instruct_zip(self):
        p = self.evidence / "original-instruct-12947986971.zip"
        p.write_bytes(p.read_bytes() + b"forge")
        self.must_refuse()

    def test_04_missing_original_source_zip(self):
        (self.evidence / "original-source-12947388112.zip").unlink()
        self.must_refuse()

    def test_05_mutate_original_chat_decoded_full_job_log(self):
        p = self.evidence / "original-chat-job-123841738132.log"
        p.write_bytes(p.read_bytes() + b"\nFAKE_PASS\n")
        self.must_refuse()

    def test_06_mutate_original_instruct_decoded_full_job_log(self):
        p = self.evidence / "original-instruct-job-123841738103.log"
        p.write_bytes(p.read_bytes()[:-4])
        self.must_refuse()

    def test_07_missing_original_source_complete_job_log(self):
        (self.evidence / "original-source-job-123841492749.log").unlink()
        self.must_refuse()

    def test_08_swap_original_chat_and_instruct_whole_zips(self):
        a = self.evidence / "original-chat-12947358773.zip"
        b = self.evidence / "original-instruct-12947986971.zip"
        olda, oldb = a.read_bytes(), b.read_bytes()
        a.write_bytes(oldb); b.write_bytes(olda)
        self.must_refuse()

    def test_09_forged_manifest_and_mutated_log_not_trusted(self):
        p = self.evidence / "original-chat-job-123841738132.log"
        p.write_bytes(p.read_bytes() + b"\nFAKE_PASS\n")
        f = self.evidence / "manifest.json"
        m = json.loads(f.read_text())
        m["original_files"][p.name] = {
            **m["original_files"][p.name], "git_blob_sha1": c.git_blob(p.read_bytes()),
            "sha256": c.sha256(p.read_bytes()), "bytes": len(p.read_bytes())}
        f.write_text(json.dumps(m, sort_keys=True) + "\n")
        self.must_refuse()

    def test_10_manifest_whitespace_only_refused(self):
        f = self.evidence / "manifest.json"
        f.write_bytes(f.read_bytes() + b" ")
        self.must_refuse()

    def test_11_false_scientific_training_or_computation_admission_refused(self):
        f = self.evidence / "manifest.json"
        m = json.loads(f.read_text())
        m["train_authorized"] = True
        m["full_72_conformance_pass"] = True
        m["actual_compute_budget_qualified"] = True
        f.write_text(json.dumps(m, indent=2) + "\n")
        self.must_refuse()

    def test_12_original_artifact_replacement_same_report_not_trusted(self):
        p = self.evidence / "original-source-12947388112.zip"
        p.write_bytes(p.read_bytes() + b" " * 20)
        self.must_refuse()


if __name__ == "__main__":
    unittest.main()
