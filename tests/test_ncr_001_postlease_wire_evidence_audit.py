"""Original source-bound Sprocket Instruct post-lease wire evidence refusal tests.

No model, network or training. All tests operate on immutable original Git
archive bytes and mutations of temporary in-memory decoded receipts.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001_postlease_wire_evidence_audit as audit


class OriginalPostleaseWireEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        zips = audit.original_zip_bytes()
        physical = zips["artifact-12828893309.zip"]
        original = audit.raw_zip_members(
            physical, audit.ORIGINAL_ZIPS["artifact-12828893309.zip"])
        cls.receipt = json.loads(original["receipt.json"])

    def altered(self, fn):
        receipt = copy.deepcopy(self.receipt)
        fn(receipt)
        with self.assertRaisesRegex(ValueError, "evidence refusal"):
            audit.verify_original_host(receipt)

    def test_01_original_three_zip_containers_and_receipt_pass(self):
        result = audit.audit()
        self.assertEqual(result["source_head"], audit.HEAD)
        self.assertEqual(result["source_and_lease_test_counts"], [11, 10])
        self.assertEqual(result["raw_artifact_members"], 5)
        self.assertFalse(result["tool_memory_semantics_admitted"])
        self.assertFalse(result["training_authorized"])
        self.assertTrue(result["decoded_git_blob_sha_verified"])
        self.assertEqual(result["original_decoded_job_logs_verified"], 3)
        self.assertEqual(result["archival_ci_decoded_job_logs_verified"], 2)

    def test_02_falsified_training_admission_rejected(self):
        self.altered(lambda r: r.update(train_authorized=True))

    def test_03_falsified_final_72_admission_rejected(self):
        self.altered(lambda r: r.update(full_72_conformance_pass=True))

    def test_04_falsified_semantic_admission_rejected(self):
        self.altered(lambda r: r.update(model_generated_tool_memory_semantics_qualified=True))

    def test_05_optimizer_step_evidence_rejected(self):
        self.altered(lambda r: r.update(optimizer_steps=1))

    def test_06_wrong_chat_checkpoint_identity_rejected(self):
        self.altered(lambda r: r["source_identity"].update(
            weight_path="chat/model.safetensors", weight_sha256=audit.CHAT))

    def test_07_changed_original_source_head_rejected(self):
        self.altered(lambda r: r["runner"].update(head_sha="0" * 40))

    def test_08_original_frozen_weight_mutation_rejected(self):
        self.altered(lambda r: r.update(in_memory_frozen_after_sha256="f" * 64))

    def test_09_host_tool_result_falsely_supervised_rejected(self):
        self.altered(lambda r: r["wire"].update(host_tool_result_unlabelled=False))

    def test_10_host_memory_downgraded_to_model_label_rejected(self):
        self.altered(lambda r: r["wire"].update(
            memory_read_host_model_host=[True, True, False]))

    def test_11_end_token_masked_as_padding_rejected(self):
        self.altered(lambda r: r["wire"].update(
            real_end_marker_id_5_visible_with_attention=False))

    def test_12_illegal_cache_tolerance_or_false_pass_rejected(self):
        self.altered(lambda r: r["checks"]["nca_cache_host_tool_result"].update(
            max_abs=0.000501))

    def test_13_downgraded_publisher_formatter_pin_rejected(self):
        self.altered(lambda r: r["wire_authority"].update(
            reviewed_renderer_blob="0" * 40))

    def test_14_falsely_scored_model_tool_competence_rejected(self):
        self.altered(lambda r: r["checks"]["mlp_raw_untrained_generation"].update(
            semantic_score=1.0))

    def test_15_negative_incomplete_generated_call_must_be_preserved(self):
        self.altered(lambda r: r["checks"]["nca_raw_untrained_generation"].update(
            active_text='<|tool_call|>{"name":"read_meter","arguments":{}}'))

    def test_16_archive_byte_flip_rejected(self):
        raw = audit.BUNDLE.read_bytes()
        altered = bytearray(raw)
        altered[20] = ord("A") if raw[20] != ord("A") else ord("B")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bad.b64"
            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, "compressed archive changed"):
                audit.original_zip_bytes(path)


    def test_17_original_decoded_host_log_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "evidence"
            shutil.copytree(audit.ROOT / audit.REL, dest)
            p = dest / "decoded/successful-physical-job-122923916338.log"
            p.write_bytes(p.read_bytes() + b"\nFORGED POSITIVE RESULT\n")
            with self.assertRaisesRegex(ValueError, "decoded original GitHub job log bytes changed"):
                audit.verify_decoded_job_logs(dest)

    def test_18_archival_failed_decoded_log_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "evidence"
            shutil.copytree(audit.ROOT / audit.REL, dest)
            p = dest / "decoded/initial-archive-shallow-checkout-failure-job-122929862963.log"
            p.write_bytes(p.read_bytes() + b"\nFORGED SUCCESS\n")
            with self.assertRaisesRegex(ValueError, "decoded original GitHub job log bytes changed"):
                audit.verify_decoded_job_logs(dest)

    def test_19_manipulated_log_and_recomputed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "evidence"
            shutil.copytree(audit.ROOT / audit.REL, dest)
            relative = "decoded/initial-source-failure-job-122922594944.log"
            p = dest / relative
            p.write_bytes(p.read_bytes() + b"\nFORGED\n")
            m = dest / "manifest.json"
            data = json.loads(m.read_text(encoding="utf-8"))
            data["archive"]["original_decoded_job_logs"][0]["git_blob_sha"] = audit.git_blob_sha(p.read_bytes())
            m.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original #162 log manifest Git blob changed"):
                audit.verify_decoded_job_logs(dest)

    def test_20_missing_original_decoded_log_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "evidence"
            shutil.copytree(audit.ROOT / audit.REL, dest)
            (dest / "decoded/successful-source-job-122923377331.log").unlink()
            with self.assertRaises(FileNotFoundError):
                audit.verify_decoded_job_logs(dest)



if __name__ == "__main__":
    unittest.main()
