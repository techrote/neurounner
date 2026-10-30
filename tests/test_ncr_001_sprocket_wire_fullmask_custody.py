"""Adversarial no-network NCR-001 original ZIP/member/log/receipt custody tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_sprocket_wire_fullmask_custody as custody


class OriginalFullMaskCustody(unittest.TestCase):
    def mutation(self, alter):
        with tempfile.TemporaryDirectory() as t:
            e = Path(t) / "evidence"
            shutil.copytree(custody.EVIDENCE, e)
            alter(e)
            with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
                custody.audit(e)

    @staticmethod
    def replace_json(e, path, mutate):
        p = e / path
        data = json.loads(p.read_text(encoding="utf-8"))
        mutate(data)
        p.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")

    def test_unmodified_two_original_zip_containers_and_complete_logs(self):
        r = custody.audit()
        self.assertEqual(r["archive_members"], 4)
        self.assertEqual(r["original_actions_run"], 40782101613)
        self.assertEqual(r["source_head"], custody.HEAD)
        self.assertEqual(r["source_test_count"], 13)
        self.assertFalse(r["train_authorized"])
        self.assertFalse(r["new_model_execution"])
        self.assertFalse(r["full_72_conformance_pass"])

    def test_original_reproduced_zip_bytes_not_synthesized(self):
        self.mutation(lambda e: (e / "original-two-actions-zip-containers.tar.xz.base64")
                      .write_bytes(b"A" + (e / "original-two-actions-zip-containers.tar.xz.base64")
                                   .read_bytes()[1:]))

    def test_complete_success_job_log_mutation_refused(self):
        self.mutation(lambda e: (e / "job-123146150961-tokenizer.log")
                      .write_bytes((e / "job-123146150961-tokenizer.log")
                                   .read_bytes() + b"\nFAKE PASS"))

    def test_complete_negative_job_log_mutation_refused(self):
        self.mutation(lambda e: (e / "job-123146082042-source.log")
                      .write_bytes((e / "job-123146082042-source.log")
                                   .read_bytes() + b"\nMUTATED"))

    def test_altered_receipt_and_training_admission_refused(self):
        self.mutation(lambda e: self.replace_json(
            e, "receipt-from-original-job.json",
            lambda m: m.__setitem__("train_authorized", True)))

    def test_manifest_cannot_promote_72_training_or_science(self):
        for key in ("train_authorized", "full_72_conformance_pass",
                    "independent_scientific_review", "comparative_training_performed",
                    "model_generated_tool_memory_semantics_qualified"):
            with self.subTest(key=key):
                self.mutation(lambda e: self.replace_json(
                    e, "manifest.json",
                    lambda m: m["qualification"].__setitem__(key, True)))

    def test_manifest_wrong_original_model_head_refused(self):
        self.mutation(lambda e: self.replace_json(
            e, "manifest.json",
            lambda m: m.__setitem__("upstream_original_model_head", "f" * 40)))

    def test_manifest_can_not_misreport_original_zip_sha(self):
        self.mutation(lambda e: self.replace_json(
            e, "manifest.json",
            lambda m: m["artifacts"]["artifact-12856874121.zip"]
                      .__setitem__("zip_sha256", "0" * 64)))

    def test_host_body_swapped_mask_and_missing_labels_are_not_admitted(self):
        m = json.loads((custody.EVIDENCE / "manifest.json").read_text())
        r = json.loads((custody.EVIDENCE / "receipt-from-original-job.json").read_text())
        for changed in ("assistant_mask_sha256", "attention_mask_sha256", "ids_sha256"):
            with self.subTest(hash=changed):
                bad = copy.deepcopy(r)
                bad["replayed_hashes"][changed] = "0" * 64
                with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
                    custody.check_report(bad, m)

    def test_unqualified_tool_behavior_fails_closed(self):
        m = json.loads((custody.EVIDENCE / "manifest.json").read_text())
        r = json.loads((custody.EVIDENCE / "receipt-from-original-job.json").read_text())
        r["historical_model_output"] = "SUCCESSFUL_TOOL_CALL"
        with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
            custody.check_report(r, m)

    def test_physical_model_false_promotion_fails(self):
        m = json.loads((custody.EVIDENCE / "manifest.json").read_text())
        r = json.loads((custody.EVIDENCE / "receipt-from-original-job.json").read_text())
        r["physical_model_executed_this_run"] = True
        with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
            custody.check_report(r, m)

    def test_wrong_external_zip_refused(self):
        original = custody.original_zips_for_test() if hasattr(
            custody, "original_zips_for_test") else {}
        with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
            custody.audit(original_zip_files={
                "artifact-12857787221.zip": b"wrong",
                "artifact-12856874121.zip": b"wrong",
            })


if __name__ == "__main__":
    unittest.main()
