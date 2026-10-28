"""Adversarial offline BYTE-EOS source-only evidence custody tests.

Tests mutate copies only. No model, tokenizer, optimizer, or heldout fixture.
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from byte_eos_001_causal_evidence_audit import EVIDENCE, audit


class EvidenceCustodyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.folder = Path(self.tempdir.name) / "evidence"
        shutil.copytree(EVIDENCE, self.folder)

    def manifest(self):
        p = self.folder / "manifest.json"
        return p, json.loads(p.read_text(encoding="utf-8"))

    def write_manifest(self, p, obj):
        p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_exact_original_logs_and_zip_bytes_accepted_as_source_only(self):
        x = audit(self.folder)
        self.assertEqual(x["status"],
                         "PASS_ORIGINAL_LOG_AND_RAW_ZIP_BYTE_READBACK__NO_MODEL")
        self.assertTrue(x["raw_actions_zip_bytes_in_git"])
        self.assertFalse(x["final_72_conformance_pass"])
        self.assertFalse(x["train_authorized"])
        self.assertTrue(x["historical_provenance_anomaly_preserved"])
        self.assertEqual(x["original_runs"], [40728812489,40728952524])

    def test_reject_training_or_scientific_false_admission(self):
        for name, value in (("train_authorized", True), ("model_executed", True),
                            ("comparative_training_performed", True),
                            ("full_72_conformance_pass", True),
                            ("independent_scientific_review", True),
                            ("optimizer_steps", 1)):
            with self.subTest(name=name):
                p, m = self.manifest()
                m["source_claims"][name] = value
                self.write_manifest(p, m)
                with self.assertRaisesRegex(ValueError, "False scientific"):
                    audit(self.folder)
                shutil.copy2(EVIDENCE / "manifest.json", p)

    def test_reject_wrong_parent_source_and_run_identity(self):
        for field, value in (("parent_exact_head", "0"*40),
                             ("original_source_head", "0"*40)):
            with self.subTest(field=field):
                p, m = self.manifest()
                m[field] = value
                self.write_manifest(p, m)
                with self.assertRaises(ValueError):
                    audit(self.folder)
                shutil.copy2(EVIDENCE / "manifest.json", p)

    def test_reject_tampered_original_full_log(self):
        path = self.folder / "run-40728952524-full-job.log"
        path.write_bytes(path.read_bytes().replace(b"Ran 16 tests", b"Ran 17 tests"))
        with self.assertRaisesRegex(ValueError, "Git blob changed"):
            audit(self.folder)

    def test_reject_mutated_original_zip_archive(self):
        path = self.folder / "original-actions-zipset.tar.xz.base64"
        raw = bytearray(base64.b64decode(path.read_text().strip()))
        raw[120] ^= 1
        path.write_text(base64.b64encode(raw).decode() + "\n")
        with self.assertRaisesRegex(ValueError, "TAR/XZ archive bytes changed"):
            audit(self.folder)

    def test_reject_historical_bad_head_relabelled_as_good(self):
        p, m = self.manifest()
        m["original_runs"][0]["provenance_anomaly"]["receipt_head_sha"] = (
            m["original_runs"][0]["source_head"])
        self.write_manifest(p, m)
        with self.assertRaisesRegex(ValueError, "Historical wrong-head"):
            audit(self.folder)

    def test_reject_false_raw_zip_custody_or_externalization(self):
        p, m = self.manifest()
        m["evidence_storage"]["raw_zip_bytes_committed"] = False
        self.write_manifest(p, m)
        with self.assertRaisesRegex(ValueError, "ZIP custody"):
            audit(self.folder)

    def test_reject_receipt_digest_forgery(self):
        p, m = self.manifest()
        m["original_runs"][1]["artifact_receipt_sha256"] = "0"*64
        self.write_manifest(p, m)
        with self.assertRaisesRegex(ValueError, "ZIP member"):
            audit(self.folder)


if __name__ == "__main__":
    unittest.main()
