"""Independent no-model mutation sentinels for original void1 frozen restart receipts.

All records are exact, source-bound original JSON from the completed Actions
artifact and are not reevaluated or silently reconstructed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001e_void_process_receipt_audit as receipt


class VoidFrozenProcessReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = receipt.originals()

    def assert_refused(self, path, new):
        records = deepcopy(self.data)
        obj = records
        for key in path[:-1]:
            obj = obj[key]
        obj[path[-1]] = new
        with self.assertRaises((RuntimeError, KeyError, ValueError)):
            receipt.audit(records)

    def test_five_byte_exact_originals_and_two_distinct_interpreters(self):
        result = receipt.audit()
        self.assertEqual(result["status"], receipt.STATUS)
        self.assertEqual(result["full_original_receipts_verified"], 5)
        self.assertEqual((result["producer_pid"], result["consumer_pid"]), (2613, 2625))
        self.assertEqual(result["actual_cached_tail_error"], 9.5367431640625e-06)
        self.assertIs(result["original_model_reexecuted"], False)
        self.assertIs(result["final_four_model_72_admission"], False)
        self.assertIs(result["train_authorized"], False)
        self.assertEqual(result["optimizer_steps"], 0)

    def test_actual_source_variant_hook_and_utf8_identity_refusals(self):
        for path, new in (
            (("void1-cross-process/producer.json", "model_key"), "sprocket_instruct"),
            (("void1-cross-process/consumer.json", "source", "revision"), "main"),
            (("void1-cross-process/producer.json", "source", "original_hook_commit"), "0" * 40),
            (("void1-cross-process/consumer.json", "source_files", "model.safetensors", "sha256"), "0"*64),
            (("void1-cross-process/consumer.json", "source_files", "tokenization_byte.py", "sha256"), "0"*64),
            (("void1-cross-process/consumer.json", "input", "input_ids_sha256"), "0"*64),
            (("void1-cross-process/consumer.json", "input", "eos_id"), 62),
            (("void1-cross-process/producer.json", "input", "literal_eos_text_is_five_ordinary_bytes"), False),
        ):
            with self.subTest(path=path):
                self.assert_refused(path, new)

    def test_false_train_optimizer_heldout_and_final_admission_fail_closed(self):
        for stage in ("producer", "consumer"):
            name = f"void1-cross-process/{stage}.json"
            for key, wrong in (
                ("model_training_performed", True), ("train_authorized", True),
                ("optimizer_steps", 1), ("heldout_evaluation", True),
                ("full_four_model_conformance_pass", True),
                ("historical_byte_eos_impact_measured", True),
            ):
                with self.subTest(stage=stage, key=key):
                    self.assert_refused((name, key), wrong)
        self.assert_refused(("void1-cross-process/manifest.json",
                             "optimizer_steps"), 1)

    def test_cold_process_separation_and_reference_sha_refusals(self):
        for path, value in (
            (("void1-cross-process/consumer.json", "runner", "pid"), 2613),
            (("void1-cross-process/manifest.json", "producer_pid"), 2625),
            (("void1-cross-process/consumer.json", "runner", "github_sha"), "0"*40),
            (("void1-cross-process/consumer.json", "runner", "run_id"), "0"),
            (("void1-cross-process/manifest.json", "adapter_sha256"), "0"*64),
            (("void1-cross-process/manifest.json", "reference_sha256"), "0"*64),
        ):
            with self.subTest(path=path):
                self.assert_refused(path, value)

    def test_true_incremental_cache_tolerance_cannot_be_relabelled(self):
        for stage, testname in (
            ("producer", "256_UTF8_240_prefill_16_real_incremental"),
            ("consumer", "fresh_process_full_vs_real_cached_tail"),
        ):
            name = f"void1-cross-process/{stage}.json"
            for field, value in (
                ("max_abs", 0.0006), ("atol", 0.05), ("rtol", 0.05),
                ("pass", False), ("exact_required", True),
            ):
                with self.subTest(stage=stage, field=field):
                    self.assert_refused((name, "checks", testname, field), value)

    def test_suffix_and_request_isolation_are_strict(self):
        for stage, name in (
            ("producer", "original_zero_init_noop_exact"),
            ("producer", "strict_future_suffix"),
            ("producer", "request_A_B_A"),
            ("consumer", "fresh_process_A_B_A_exact"),
            ("consumer", "fresh_process_active_tail_exact"),
        ):
            file = f"void1-cross-process/{stage}.json"
            self.assert_refused((file, "checks", name, "max_abs"), 1e-6)

    def test_missing_or_extra_origins_are_refused(self):
        missing = deepcopy(self.data)
        missing.pop("void1-cross-process/consumer.json")
        with self.assertRaises(RuntimeError):
            receipt.audit(missing)
        extra = deepcopy(self.data)
        extra["fake-success.json"] = {"status": "PASS"}
        with self.assertRaises(RuntimeError):
            receipt.audit(extra)

    def test_archive_digest_and_zip_custody_are_not_treated_as_decorative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.b64"
            path.write_text("H4sI" + "0" * 100, encoding="ascii")
            with patch.object(receipt, "ARCHIVE", path):
                with self.assertRaises((RuntimeError, ValueError, OSError, EOFError)):
                    receipt.originals()
            provenance = json.loads(receipt.PROVENANCE.read_text(encoding="utf-8"))
            provenance["original_actions_artifact"]["zip_sha256"] = "0" * 64
            target = Path(tmp) / "prov.json"
            target.write_text(json.dumps(provenance), encoding="utf-8")
            with patch.object(receipt, "PROVENANCE", target):
                with self.assertRaises(RuntimeError):
                    receipt.audit(self.data)

    def test_source_only_audit_cannot_claim_tokenizer_runtime(self):
        self.assert_refused(("void-byte-source-audit.json", "status"),
                            "PASS_QWEN3_MODEL_QUALIFIED")
        self.assert_refused(("void-byte-source-audit.json", "source",
                             "protocol_sha256"), "0" * 64)


if __name__ == "__main__":
    unittest.main()
