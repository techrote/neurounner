"""Adversarial original postshift compressed ZIP and full job-log refusal."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import byte_eos_001_postshift_git_custody as custody


class OriginalPostshiftCustodyTests(unittest.TestCase):
    def test_original_source_zip_and_complete_log_and_run_identity(self):
        r = custody.verify()
        self.assertEqual(r["original_run_id"], 41021703985)
        self.assertFalse(r["scientific_72_admitted"])
        self.assertFalse(r["train_authorized"])

    def test_mutated_original_zip_refused(self):
        with TemporaryDirectory() as d:
            p = Path(d)/"archive.b64"
            p.write_bytes(custody.ARCHIVE.read_bytes().replace(b"UEsDB", b"AE sDB", 1))
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                custody.verify(archive_path=p, check_branch=False)

    def test_mutated_complete_decoded_log_refused(self):
        with TemporaryDirectory() as d:
            p = Path(d)/"job.log"
            p.write_bytes(custody.JOB_LOG.read_bytes() + b"PASS\\n")
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                custody.verify(job_log_path=p, check_branch=False)

    def test_modified_manifest_even_if_zip_and_log_unmodified_refused(self):
        with TemporaryDirectory() as d:
            p = Path(d)/"manifest.json"
            m = json.loads(custody.MANIFEST.read_bytes())
            for key,value in (
                ("run_id", 0), ("job_id", True),
                ("head_sha", "0"*40),
            ):
                other=json.loads(custody.MANIFEST.read_bytes())
                other["source_run"][key] = value
                p.write_text(json.dumps(other))
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError,"REFUSED"):
                    custody.verify(manifest_path=p, check_branch=False)

    def test_false_training_science_or_history_promotion_refused(self):
        with TemporaryDirectory() as d:
            p = Path(d)/"manifest.json"
            m = json.loads(custody.MANIFEST.read_bytes())
            for key,value in (
                ("train_authorized", True),
                ("optimizer_steps", True),
                ("model_weights_loaded", True),
                ("full_72_conformance_pass", True),
                ("original_train002b_effect", "MEASURED"),
            ):
                other=json.loads(custody.MANIFEST.read_bytes())
                other["boundaries"][key]=value
                p.write_text(json.dumps(other))
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError,"REFUSED"):
                    custody.verify(manifest_path=p,check_branch=False)

    def test_original_source_or_runner_substitution_refused(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"manifest.json"
            m=json.loads(custody.MANIFEST.read_bytes())
            for key,value in (
                ("runner_name","different-runner"),
                ("transformers","latest"),
                ("original_126_review_head", "0"*40),
            ):
                other=json.loads(custody.MANIFEST.read_bytes())
                other["source_run"][key]=value
                p.write_text(json.dumps(other))
                with self.subTest(key=key), self.assertRaisesRegex(RuntimeError,"REFUSED"):
                    custody.verify(manifest_path=p,check_branch=False)


if __name__ == "__main__":
    unittest.main()
