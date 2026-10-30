"""Adversarial offline original-Qwen3 evidence custody tests. NO model loaded."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from byte_eos_001_frozen_evidence_audit import EVIDENCE, audit, validate_report

class EvidenceAudit(unittest.TestCase):
    def test_original_bytes_and_model_forward_receipt(self):
        report=audit()
        self.assertEqual(report["status"],
                         "PASS_ORIGINAL_BYTES_AND_SOURCE_ONLY_FORWARD_LOSS__72_BLOCKED")
        self.assertEqual(report["source_tests_passed"],30)
        self.assertEqual(report["real_frozen_model_cases"],3)
        self.assertFalse(report["train_authorized"])

    def mutate_manifest(self, patch):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"evidence"
            shutil.copytree(EVIDENCE,dest)
            manifest=dest/"manifest.json"
            data=json.loads(manifest.read_text())
            patch(data)
            manifest.write_text(json.dumps(data))
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):
                audit(dest)

    def test_false_training_admission_rejected(self):
        for key in ("train_authorized","full_72_conformance_pass",
                    "independent_scientific_review","nca_mlp_integration_qualified",
                    "historical_train002b_effect_measured"):
            with self.subTest(key=key):
                self.mutate_manifest(lambda d: d["protocol_claims"].__setitem__(key,True))

    def test_wrong_variant_source_and_unsafe_optimizer_rejected(self):
        self.mutate_manifest(lambda d:d.__setitem__("original_source_head","wrong"))
        self.mutate_manifest(lambda d:d["protocol_claims"].__setitem__("optimizer_steps",1))
        self.mutate_manifest(lambda d:d["original_model"].__setitem__("original_weight_sha256","bad"))

    def test_wrong_loss_and_checksum_rejected(self):
        self.mutate_manifest(lambda d:d["original_model"]["cases"][0].__setitem__("loss",0.0))
        self.mutate_manifest(lambda d:d["artifact"].__setitem__("original_archive_zip_sha256","0"*64))

    def test_original_job_log_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"evidence"
            shutil.copytree(EVIDENCE,dest)
            p=dest/"job-122992543245-original-qwen3.log"
            p.write_bytes(p.read_bytes()+b"\nFAKE")
            with self.assertRaises(ValueError):
                audit(dest)

    def test_compressed_original_zip_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"evidence"
            shutil.copytree(EVIDENCE,dest)
            p=dest/"original-actions-zip.xz.base64"
            s=p.read_text()
            p.write_text(("A" if s[0]!="A" else "B")+s[1:])
            with self.assertRaises((ValueError,Exception)):
                audit(dest)

    def test_failed_audit_run_history_must_be_retained(self):
        self.mutate_manifest(lambda d:d["custody_ci"]["failed_readback"].__setitem__("conclusion","success"))
        self.mutate_manifest(lambda d:d["custody_ci"]["fixed_readback"].__setitem__("tests_passed",8))

    def test_original_receipt_false_qualification_rejected(self):
        report={"status":"PASS_FROZEN_QWEN3_CAUSAL_EOS_FORWARD_ONLY__72_BLOCKED",
                "run_id":"40738222796","source_head":"06b858a70d7019398cdc7a3f205ae24423f57c67",
                "runner_name":"GitHub Actions 1000046597",
                "model_executed":True,"byte_tokenizer_executed":True,"train_authorized":True}
        with self.assertRaises((ValueError,KeyError)):
            validate_report(report)

if __name__=="__main__":
    unittest.main()
