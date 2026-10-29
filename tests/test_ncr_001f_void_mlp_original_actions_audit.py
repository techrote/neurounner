"""Offline adversarial mutation tests for original full Actions ZIP and job logs."""
from __future__ import annotations
import shutil
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ncr_001f_void_mlp_original_actions_audit as audit


class ExactOriginalActionsCustody(unittest.TestCase):
    def mutate(self, name, change):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp) / "evidence"
            shutil.copytree(audit.EVIDENCE, d)
            p = d / name
            change(p)
            with self.assertRaisesRegex(ValueError, "NCR001_VOID_ORIGINAL_REFUSED"):
                audit.audit(d)

    def test_original_zip_two_original_jsons_two_logs_and_no_training(self):
        result = audit.audit()
        self.assertEqual(result["original"]["zip_bytes"], 3508)
        self.assertEqual(result["original"]["members"], 2)
        self.assertEqual(len(result["original_job_ids"]), 2)
        self.assertFalse(result["new_physical_model_execution"])
        self.assertFalse(result["full_72_conformance_pass"])
        self.assertFalse(result["train_authorized"])

    def test_zip_blob_whitespace_mutation_refused(self):
        self.mutate(audit.ZIP_FILE, lambda p: p.write_bytes(p.read_bytes() + b"\\n"))

    def test_original_source_job_log_mutation_refused(self):
        self.mutate("original-source-job-122256460886.log",
                    lambda p: p.write_bytes(p.read_bytes() + b"FORGED"))

    def test_original_physical_job_log_mutation_refused(self):
        self.mutate("original-model-job-122256966798.log",
                    lambda p: p.write_bytes(p.read_bytes() + b"FORGED"))

    def test_rehashed_false_scientific_manifest_refused(self):
        def change(p):
            import json
            d = __import__("json").loads(p.read_text())
            d["full_72_conformance_pass"] = True
            p.write_text(json.dumps(d))
        self.mutate(audit.MANIFEST_FILE, change)

    def test_truncated_or_substituted_zip_refused(self):
        original = __import__("base64").b64decode(
            (audit.EVIDENCE / audit.ZIP_FILE).read_bytes().strip())
        for raw in (original[:-1], b"some invented replacement"):
            with self.assertRaisesRegex(ValueError, "NCR001_VOID_ORIGINAL_REFUSED"):
                audit.validate_zip(raw)


if __name__ == "__main__":
    unittest.main()
