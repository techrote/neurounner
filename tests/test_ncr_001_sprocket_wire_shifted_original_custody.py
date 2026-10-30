"""Offline negative mutations for original PR #184 source/tokenizer and failed CI.

No network, Hugging Face imports, model weights, optimizer, heldouts or merge.
"""
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
import ncr_001_sprocket_wire_shifted_original_custody as audit


class OriginalShiftedWireCustody(unittest.TestCase):
    def mutate(self, change):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence"
            shutil.copytree(audit.EVIDENCE, out)
            change(out)
            with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
                audit.audit(out)

    @staticmethod
    def edit_manifest(evidence, change):
        path = evidence / "manifest.json"
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def test_positive_complete_original_zip_and_success_failed_logs(self):
        result = audit.audit()
        self.assertEqual(result["status"],
            "PASS_COMPLETE_ORIGINAL_ZIP_LOGS_AND_NEGATIVE_RESULT__72_BLOCKED")
        self.assertEqual(result["members"], 5)
        self.assertEqual(result["source_tests_passed"], 14)
        self.assertEqual(result["original_failed_checkout_run"], 40788806988)
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["new_model_execution"])

    def test_original_zip_bytes_altered_refused(self):
        def mutate(p):
            f = p / "original-actions-zip.xz.base64"
            orig = f.read_bytes()
            f.write_bytes((b"A" if orig[:1] != b"A" else b"B") + orig[1:])
        self.mutate(mutate)

    def test_original_artifact_metadata_sha_mutation_refused(self):
        self.mutate(lambda p: self.edit_manifest(p, lambda x:
            x["artifact"].__setitem__("zip_sha256", "0" * 64)))

    def test_one_original_member_digest_mutated_refused(self):
        self.mutate(lambda p: self.edit_manifest(p, lambda x:
            x["artifact"]["members"]["receipt.json"].__setitem__("sha256", "0" * 64)))

    def test_original_success_job_log_byte_corruption_refused(self):
        def mutate(p):
            log = p / "job-123168786515-passed-tokenizer.log"
            log.write_bytes(log.read_bytes() + b"\nFAKE RESULT")
        self.mutate(mutate)

    def test_original_failed_job_log_byte_corruption_refused(self):
        def mutate(p):
            log = p / "job-123168595600-failed-checkout.log"
            log.write_bytes(log.read_bytes() + b"\nFAKE CHECKOUT")
        self.mutate(mutate)

    def test_erasure_of_original_failed_checkout_refused(self):
        for key, wrong in (("run_id", 0), ("conclusion", "success"),
                           ("phase", "tests_passed")):
            with self.subTest(key=key):
                self.mutate(lambda p: self.edit_manifest(p, lambda x:
                    x["attempts"]["failed_checkout"].__setitem__(key, wrong)))

    def test_wrong_source_head_and_runner_refused(self):
        for field, wrong in (("source_head", "f" * 40),
                             ("runner_name", "someone else's GPU")):
            self.mutate(lambda p: self.edit_manifest(p, lambda x:
                x["attempts"]["passed"].__setitem__(field, wrong)))

    def test_false_72_review_training_and_optimizer_claims_refused(self):
        for key in ("train_authorized", "full_72_conformance_pass",
                    "independent_scientific_review",
                    "model_generated_tool_memory_semantics_qualified",
                    "historical_void_byte_eos_effect_measured"):
            with self.subTest(key=key):
                self.mutate(lambda p: self.edit_manifest(p, lambda x:
                    x["scientific_gates"].__setitem__(key, True)))
        self.mutate(lambda p: self.edit_manifest(p, lambda x:
            x["scientific_gates"].__setitem__("optimizer_steps", 1)))

    def test_chat_and_root_instruct_cannot_be_collapsed(self):
        self.mutate(lambda p: self.edit_manifest(p, lambda x:
            x["source_gates"].__setitem__("sprocket_instruct_weight_file",
                                          "chat/model.safetensors")))

    def test_shifted_label_hash_or_bad_padding_cannot_be_admitted(self):
        for name in ("causal_shifted_labels_sha256", "padded_attention_sha256"):
            with self.subTest(hash=name):
                self.mutate(lambda p: self.edit_manifest(p, lambda x:
                    x["receipt"].__setitem__(name, "f" * 64)))
        self.mutate(lambda p: self.edit_manifest(p, lambda x:
            x["receipt"].__setitem__("naive_pad_equals_end_refused", False)))

    def test_original_live_zip_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "CUSTODY_REFUSED"):
            audit.audit(live_zip=b"not the original action artifact")


if __name__ == "__main__":
    unittest.main()
