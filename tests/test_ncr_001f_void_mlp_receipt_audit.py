"""Source and byte-custody mutation sentinels for real void.1 MLP prepass.

Never load model weights, execute ByteTokenizer remote code, or train.
"""
from __future__ import annotations
import base64
import copy
import json
import shutil
import tempfile
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001f_void_mlp_receipt_audit import (
    ARCHIVE, PROVENANCE, EVIDENCE, EXPECTED_CLASSIFICATION, audit,
    read_original_bytes, validate_receipts,
)


class OriginalVoidMlpCustodySentinels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        cls.encoded = ARCHIVE.read_bytes().strip()
        cls.originals = read_original_bytes(cls.encoded, cls.prov)
        cls.pref = json.loads(cls.originals["void1-source-prefetch.json"])
        cls.host = json.loads(cls.originals["void1-real-mlp.json"])

    def altered(self, mutator):
        prov = copy.deepcopy(self.prov)
        pref = copy.deepcopy(self.pref)
        host = copy.deepcopy(self.host)
        mutator(prov, pref, host)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validate_receipts(pref, host, prov)

    def mutate_committed_evidence(self, name, mutator):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "original-evidence"
            shutil.copytree(EVIDENCE, evidence)
            mutator(evidence / name)
            with self.assertRaisesRegex(ValueError, "committed"):
                audit(evidence)

    def test_original_source_and_runtime_admitted_as_prepass_only(self):
        r = audit()
        self.assertEqual(r["original_run"], "40523446884")
        self.assertEqual(r["classification"], EXPECTED_CLASSIFICATION)
        self.assertEqual(r["optimizer_steps"], 0)
        self.assertFalse(r["training_admitted"])
        self.assertLessEqual(r["cached_max_abs"], 0.0005)

    def test_encoded_original_json_archive_whitespace_drift_refused(self):
        self.mutate_committed_evidence(
            ARCHIVE.name, lambda p: p.write_bytes(p.read_bytes() + b"\n"))

    def test_committed_provenance_whitespace_drift_refused(self):
        self.mutate_committed_evidence(
            PROVENANCE.name, lambda p: p.write_bytes(p.read_bytes() + b"\n"))

    def test_rehashed_false_classification_and_archive_claim_refused(self):
        def change(p):
            d = json.loads(p.read_text(encoding="utf-8"))
            d["classification"] = "FALSE_FOUR_MODEL_72_PASS"
            d["archive"]["original_jsons_tar_gz_sha256"] = "0" * 64
            p.write_text(json.dumps(d), encoding="utf-8")
        self.mutate_committed_evidence(PROVENANCE.name, change)

    def test_false_classification_refused_by_receipt_validator(self):
        self.altered(lambda prov, pref, host:
                     prov.__setitem__("classification", "FALSE_FOUR_MODEL_72_PASS"))

    def test_original_zip_hash_guard(self):
        self.altered(lambda prov, pref, host:
                     prov["source_run"].__setitem__("artifact_zip_sha256", "0" * 64))

    def test_gzip_byte_mutation_refused(self):
        changed = bytearray(self.encoded)
        changed[100] = ord("A") if changed[100] != ord("A") else ord("B")
        with self.assertRaises((ValueError, EOFError, OSError)):
            read_original_bytes(bytes(changed), self.prov)

    def test_original_member_digest_mutation_refused(self):
        prov = copy.deepcopy(self.prov)
        prov["archive"]["originals"]["void1-real-mlp.json"]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            read_original_bytes(self.encoded, prov)

    def test_refuse_other_model_checkpoint_with_stable_preflight(self):
        self.altered(lambda prov, pref, host:
                     host["source_files"]["model.safetensors"].__setitem__("sha256", "0" * 64))

    def test_refuse_false_qualifying_training(self):
        self.altered(lambda prov, pref, host: host.__setitem__("train_authorized", True))

    def test_refuse_false_separate_process_cold_restart(self):
        self.altered(lambda prov, pref, host:
                     host.__setitem__("new_interpreter_cold_reload", True))

    def test_refuse_false_final_72_admission(self):
        self.altered(lambda prov, pref, host: host.__setitem__("final_ncr_001e_admission", True))

    def test_refuse_wrong_runner(self):
        self.altered(lambda prov, pref, host:
                     host["runner"].__setitem__("github_sha", "deadbeef"))

    def test_refuse_wrong_tokenizer_eos_id(self):
        self.altered(lambda prov, pref, host:
                     host["tokenizer"].__setitem__("eos_id", 62))

    def test_refuse_relaxed_cache_error_bound(self):
        self.altered(lambda prov, pref, host:
                     host["checks"]["active_48_UTF8_full_vs_true_incremental_KV"].__setitem__(
                         "atol", 0.02))

    def test_refuse_false_strict_suffix_success(self):
        self.altered(lambda prov, pref, host:
                     host["checks"]["strict_causal_future_suffix"].__setitem__(
                         "max_abs", 0.00001))

    def test_refuse_false_frozen_base_identity(self):
        self.altered(lambda prov, pref, host:
                     host.__setitem__("loaded_original_base_sha256_after", "0" * 64))

    def test_refuse_false_readout_gradient(self):
        self.altered(lambda prov, pref, host:
                     host["checks"]["readout_gradient_only_zero_steps"].__setitem__(
                         "base_gradients_none", False))

    def test_refuse_false_semantic_eos_result(self):
        self.altered(lambda prov, pref, host:
                     host["checks"]["zero_mlp_greedy_byte_units"].__setitem__(
                         "eos_quality_measured", True))


if __name__ == "__main__":
    unittest.main()
