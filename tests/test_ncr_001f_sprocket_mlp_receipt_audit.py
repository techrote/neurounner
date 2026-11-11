"""NCR-001F archived real Sprocket matched-MLP receipt tamper/refusal tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001f_sprocket_mlp_receipt_audit import (
    PROVENANCE, ARCHIVE, audit, audit_receipt, unpack_archive,
)


class OriginalTwoSprocketMLPReceiptAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(PROVENANCE.read_bytes())
        cls.original = unpack_archive(ARCHIVE.read_bytes(), cls.manifest)
        cls.records = {}
        for variant in ("chat", "instruct"):
            cls.records[variant] = (
                json.loads(cls.original[f"{variant}/actual-mlp.json"]),
                json.loads(cls.original[f"{variant}/prefetch.json"]),
            )

    def reject(self, variant, mutate):
        actual, prefetch = self.records[variant]
        actual, prefetch = copy.deepcopy(actual), copy.deepcopy(prefetch)
        mutate(actual, prefetch)
        with self.assertRaises(ValueError):
            audit_receipt(variant, actual, prefetch, self.manifest)

    def test_original_full_byte_source_and_real_model_numeric_readback_pass(self):
        result = audit()
        self.assertTrue(result["source_integrity_readback_pass"])
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["independent_ncr_001e_final_conformance_pass"])
        self.assertFalse(result["tool_memory_behavior_qualified"])
        self.assertFalse(result["independent_new_model_invocation"])
        self.assertEqual(set(result["models"]), {"chat", "instruct"})
        self.assertEqual(result["optimizer_steps"], 0)

    def test_corrupted_archive_rejected_even_before_json_decode(self):
        data = bytearray(ARCHIVE.read_bytes())
        data[20] = ord("Z") if data[20] != ord("Z") else ord("Q")
        with self.assertRaises(ValueError):
            unpack_archive(bytes(data), self.manifest)

    def test_wrong_source_model_variant_refused(self):
        self.reject("chat", lambda a, p: a["verified_source_files"]["chat/model.safetensors"].update(
            sha256=self.manifest["variants"]["instruct"]["weight_sha256"]))

    def test_false_cache_close_flag_not_enough_to_pass_gate(self):
        self.reject("instruct", lambda a, p:
                    a["checks"]["active_48_full_vs_incremental_KV"].update(
                        max_abs=0.001, close=True))

    def test_mutated_suffix_or_reset_claim_rejected(self):
        self.reject("chat", lambda a, p:
                    a["checks"]["causal_future_suffix_exact"].update(max_abs=0.0001))
        self.reject("instruct", lambda a, p:
                    a["checks"]["request_A_B_A_exact"].update(max_abs=1e-6))

    def test_false_training_or_model_final_admission_rejected(self):
        self.reject("chat", lambda a, p: a.update(optimizer_steps=1))
        self.reject("instruct", lambda a, p: a.update(train_authorized=True))
        self.reject("chat", lambda a, p: a.update(independent_ncr_001e_conformance_pass=True))
        self.reject("instruct", lambda a, p: a.update(tool_and_memory_behavior_qualified=True))

    def test_cross_runner_or_source_checkout_refused(self):
        self.reject("chat", lambda a, p: a["runner"].update(github_sha="0" * 40))
        self.reject("instruct", lambda a, p: p.update(run_id="an_unrelated_run"))
        self.reject("chat", lambda a, p: a["source_sha256"].update(
            {"scripts/ncr_001f_real_sprocket.py": "0" * 64}))

    def test_mutated_base_bytes_or_adapter_claim_refused(self):
        self.reject("chat", lambda a, p: a.update(in_memory_base_after_sha256="a" * 64))
        self.reject("instruct", lambda a, p: a["checks"].update(
            refuse_corrupt_metadata_lane=False))
        self.reject("chat", lambda a, p: a["checks"]["readout_gradient_no_optimizer"].update(
            base_gradients_none=False))

    def test_wrong_source_role_token_protocol_refused(self):
        self.reject("instruct", lambda a, p: a["role_sources"].update(user=[99]))
        self.reject("chat", lambda a, p: a["formatter_source"].update(
            tool_memory_grammar_supported=True))
        self.reject("instruct", lambda a, p: a["role_sources"].update(
            tool_and_memory_behavior_qualified=True))

    def test_prefetch_vs_full_model_source_disagreement_rejected(self):
        self.reject("instruct", lambda a, p: p["source_files"]["model.safetensors"].update(
            sha256="f" * 64))
        self.reject("chat", lambda a, p: p["source_files"]["README.md"].update(
            bytes=9999))


if __name__ == "__main__":
    unittest.main()
