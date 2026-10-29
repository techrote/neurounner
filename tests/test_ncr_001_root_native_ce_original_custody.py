"""Adversarial original physical-host ZIP/log custody tests (NO MODEL)."""
from __future__ import annotations

import copy
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
import ncr_001_root_native_ce_original_custody as audit


class OriginalFrozenCESourceCustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        z = audit.lzma.decompress(
            audit.base64.b64decode((audit.EVIDENCE / audit.ARCHIVE).read_bytes().strip()),
            memlimit=64*1024*1024,
        )
        cls.original_zip = z
        with zipfile.ZipFile(io.BytesIO(z)) as f:
            cls.receipt = json.loads(f.read("receipt.json"))

    def mutate(self, fn):
        receipt = copy.deepcopy(self.receipt)
        fn(receipt)
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            audit.validate_receipt(receipt)

    def with_evidence(self, mutate):
        with tempfile.TemporaryDirectory() as t:
            dst = Path(t) / "evidence"
            shutil.copytree(audit.EVIDENCE, dst)
            mutate(dst)
            with self.assertRaises((ValueError, OSError, KeyError)):
                audit.audit(dst)

    def test_original_complete_byte_exact_zip_and_log(self):
        report = audit.audit()
        self.assertEqual(report["source_and_original_zip"]["raw_members"], 6)
        self.assertEqual(report["source_and_original_zip"]["targets"], 45)
        self.assertFalse(report["train_authorized"])
        self.assertFalse(report["full_72_conformance_pass"])

    def test_unsafe_training_admission_is_rejected(self):
        self.mutate(lambda x: x.__setitem__("train_authorized", True))
        self.mutate(lambda x: x.__setitem__("optimizer_steps", 1))
        self.mutate(lambda x: x.__setitem__("full_72_conformance_pass", True))

    def test_wrong_chat_variant_is_rejected(self):
        self.mutate(lambda x: x["source"].__setitem__(
            "root_weight_sha256", audit.CHAT_WEIGHT))
        self.mutate(lambda x: x.__setitem__("lane", "sprocket_chat"))

    def test_false_native_loss_or_supervised_count_is_rejected(self):
        self.mutate(lambda x: x["native_ce"].__setitem__("native", 0.0))
        self.mutate(lambda x: x["native_ce"].__setitem__("supervised_target_count", 44))
        self.mutate(lambda x: x["native_ce"].__setitem__("abs_difference", 1.0))

    def test_missing_backbone_freeze_is_rejected(self):
        self.mutate(lambda x: x.__setitem__(
            "frozen_backbone_after_sha256", "0"*64))

    def test_erased_incomplete_generation_negative_is_rejected(self):
        self.mutate(lambda x: x.__setitem__("original_incomplete_tool_call_preserved", False))
        self.mutate(lambda x: x.__setitem__(
            "model_generated_tool_memory_semantics_qualified", True))

    def test_original_host_runner_mismatch_is_rejected(self):
        self.mutate(lambda x: x["runner"].__setitem__("run_id", "40791640015"))

    def test_original_complete_decoded_job_log_mutation_refused(self):
        def corrupt(dst):
            log=dst / audit.JOB
            log.write_bytes(log.read_bytes()+b"\nFORGED")
        self.with_evidence(corrupt)

    def test_original_zip_bundle_mutation_refused(self):
        def corrupt(dst):
            p=dst / audit.ARCHIVE
            b=bytearray(p.read_bytes())
            b[30]=ord("B") if b[30]!=ord("B") else ord("C")
            p.write_bytes(b)
        self.with_evidence(corrupt)

    def test_missing_original_zip_bundle_refused(self):
        self.with_evidence(lambda dst:(dst / audit.ARCHIVE).unlink())

    def test_manifest_rehash_or_false_review_refused(self):
        def patch(dst):
            p=dst/audit.MANIFEST
            x=json.loads(p.read_text())
            x["independent_scientific_review"]=True
            p.write_text(json.dumps(x))
        self.with_evidence(patch)

    def test_member_dropping_refused(self):
        self.assertEqual(audit.validate_zip(self.original_zip)["raw_members"],6)
        with zipfile.ZipFile(io.BytesIO(self.original_zip)) as z:
            raw={n:z.read(n) for n in z.namelist()}
        with tempfile.TemporaryFile() as f:
            with zipfile.ZipFile(f,"w") as o:
                for n, contents in raw.items():
                    if n!="install-tokenizer.log":
                        o.writestr(n, contents)
            f.seek(0)
            with self.assertRaisesRegex(ValueError,"REFUSED"):
                audit.validate_zip(f.read())


if __name__ == "__main__":
    unittest.main()
