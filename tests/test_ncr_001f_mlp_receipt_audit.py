"""Independent offline source receipt custody plus fail-closed sentinels, no torch."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001f_mlp_receipt_audit import (
    ORIGINAL_ARTIFACT_ZIP_SHA256,
    ORIGINAL_RECEIPT_SHA256,
    EVIDENCE,
    audit,
    audit_payload,
)


class OriginalSmolMLPReceiptReadback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (EVIDENCE / "receipt.json").read_bytes()
        cls.receipt = json.loads(cls.raw)
        cls.provenance = json.loads((EVIDENCE / "provenance.json").read_text())

    def test_original_exact_archive_hash_and_no_training(self):
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), ORIGINAL_RECEIPT_SHA256)
        self.assertEqual(len(self.raw), 5505)
        self.assertEqual(self.provenance["original_source"]["original_zip_sha256"],
                         ORIGINAL_ARTIFACT_ZIP_SHA256)
        observed = audit()
        self.assertEqual(
            observed["status"],
            "PASS_ORIGINAL_SMOLLM2_MLP_RAW_RECEIPT_SOURCE_READBACK__NO_FINAL_CONFORMANCE",
        )
        self.assertFalse(observed["train_authorized"])
        self.assertFalse(observed["four_model_ncr_001e_pass"])

    def test_byte_corruption_cannot_be_self_attested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            original = self.raw.decode("utf-8")
            (path / "receipt.json").write_text(original.replace(
                "296576", "296577", 1), encoding="utf-8"
            )
            (path / "provenance.json").write_text(json.dumps(self.provenance))
            with self.assertRaisesRegex(RuntimeError, "byte hash"):
                audit(path)

    def test_forged_original_hash_or_source_zip_fails(self):
        for field in ("original_raw_sha256", "original_zip_sha256", "source_head_commit"):
            p = copy.deepcopy(self.provenance)
            p["original_source"][field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "identity"):
                audit_payload(self.receipt, p)

    def test_any_false_training_or_admission_flag_turned_on_is_rejected(self):
        for field in ("train_authorized", "model_training_performed", "heldout_data_opened",
                      "four_model_actual_host_mlp_qualified"):
            d = copy.deepcopy(self.receipt)
            d[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "gate"):
                audit_payload(d, self.provenance)
        d = copy.deepcopy(self.receipt)
        d["optimizer_steps"] = 1
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            audit_payload(d, self.provenance)

    def test_wrong_weight_hash_or_variant_cannot_be_relabelled(self):
        d = copy.deepcopy(self.receipt)
        d["source_identity"]["model_id"] = "kandivault/sprocket-500m"
        with self.assertRaisesRegex(RuntimeError, "identity"):
            audit_payload(d, self.provenance)
        d = copy.deepcopy(self.receipt)
        d["source_files"]["model.safetensors"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            audit_payload(d, self.provenance)

    def test_hidden_future_leak_or_cache_inconsistency_blocks_even_with_green_status(self):
        d = copy.deepcopy(self.receipt)
        d["checks"]["future_suffix_causal_invariance"]["max_abs"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "numerical"):
            audit_payload(d, self.provenance)
        d = copy.deepcopy(self.receipt)
        d["checks"]["nontrained_48_token_full_vs_real_incremental_KV"]["max_abs"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "numerical"):
            audit_payload(d, self.provenance)
        d = copy.deepcopy(self.receipt)
        d["checks"]["wrong_checkpoint_variant_refusal"] = False
        with self.assertRaisesRegex(RuntimeError, "requirement"):
            audit_payload(d, self.provenance)

    def test_frozen_backbone_and_gradient_must_remain_original(self):
        d = copy.deepcopy(self.receipt)
        d["frozen_base_in_memory_sha256_after"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "tensor digest"):
            audit_payload(d, self.provenance)
        d = copy.deepcopy(self.receipt)
        d["checks"]["readout_gradient_only_no_optimizer"]["frozen_base_grads_none"] = False
        with self.assertRaisesRegex(RuntimeError, "requirement"):
            audit_payload(d, self.provenance)


if __name__ == "__main__":
    unittest.main()
