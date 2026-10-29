"""NCR-001E independent offline receipt audit and negative sentinel tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001e_readback import (  # noqa: E402
    audit, audit_lane, load, sha, SOURCES, EVIDENCE, PROTOCOL, PINS,
)


class SourceReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load(PROTOCOL)
        cls.pins = load(PINS)

    def candidate(self, name):
        root = EVIDENCE / name
        raw = (root / "receipt.json").read_bytes()
        provenance = load(root / "provenance.json")
        return raw, provenance

    def manipulated(self, name, key, updated):
        raw, provenance = self.candidate(name)
        rec = json.loads(raw)
        rec[key] = updated
        new = json.dumps(rec, indent=2, sort_keys=True).encode()
        provenance = copy.deepcopy(provenance)
        provenance["source_run"]["original_report_sha256"] = sha(new)
        if "report_bytes" in provenance["source_run"]:
            provenance["source_run"]["report_bytes"] = len(new)
        else:
            provenance["source_run"]["original_report_bytes"] = len(new)
        return new, provenance

    def test_all_four_original_sources_pass_readback_but_not_conformance(self):
        result = audit()
        self.assertEqual(set(result["model_lanes"]), set(SOURCES))
        self.assertTrue(result["source_integrity_readback_pass"])
        self.assertFalse(result["independent_runtime_conformance_pass"])
        self.assertFalse(result["any_model_training_admitted"])
        for name, row in result["model_lanes"].items():
            self.assertGreater(row["readout_grad_l1"], 0)
            self.assertEqual(row["zero_init_max_abs"], 0)
            self.assertEqual(row["short_causal_suffix_max_abs"], 0)
            self.assertLess(row["short_cached_max_abs"], 0.0005)
            self.assertFalse(row["training_admitted"])
            self.assertFalse(row["independent_runtime_conformance_pass"])
            self.assertGreaterEqual(len(row["remaining"]), 3)

    def test_copied_original_report_sha_must_match_provenance(self):
        for name in SOURCES:
            raw, prov = self.candidate(name)
            self.assertEqual(sha(raw), prov["source_run"]["original_report_sha256"])
            corrupted = raw + b"\n"
            with self.assertRaisesRegex(RuntimeError, "original receipt SHA-256"):
                audit_lane(name, corrupted, prov, self.protocol, self.pins)

    def test_variant_substitution_and_wrong_source_detected(self):
        raw, prov = self.candidate("sprocket_chat")
        rec = json.loads(raw)
        rec["source_files"]["chat/model.safetensors"]["sha256"] = (
            self.pins["models"]["sprocket_instruct"]["weight_declared_lfs_sha256_not_downloaded"]
        )
        new = json.dumps(rec, indent=2, sort_keys=True).encode()
        prov = copy.deepcopy(prov)
        prov["source_run"]["original_report_sha256"] = sha(new)
        prov["source_run"]["original_report_bytes"] = len(new)
        with self.assertRaisesRegex(RuntimeError, "full checkpoint byte digest"):
            audit_lane("sprocket_chat", new, prov, self.protocol, self.pins)

    def test_positive_model_relabel_as_trained_is_rejected(self):
        for name in SOURCES:
            raw, prov = self.manipulated(name, "model_training_performed", True)
            with self.assertRaisesRegex(RuntimeError, "includes training"):
                audit_lane(name, raw, prov, self.protocol, self.pins)

    def test_manual_cache_error_worsening_is_refused(self):
        for name in SOURCES:
            raw, prov = self.candidate(name)
            rec = json.loads(raw)
            rec["checks"]["nonzero_nca_full_vs_incremental_cache"]["max_abs"] = 0.02
            # Even if a source-provenance hash were updated to match the tampered
            # receipt, source-bound numerical limits must still fail.
            changed = json.dumps(rec, sort_keys=True, indent=2).encode()
            pv = copy.deepcopy(prov)
            pv["source_run"]["original_report_sha256"] = sha(changed)
            if "report_bytes" in pv["source_run"]:
                pv["source_run"]["report_bytes"] = len(changed)
            else:
                pv["source_run"]["original_report_bytes"] = len(changed)
            with self.assertRaisesRegex(RuntimeError, "full-vs-cached"):
                audit_lane(name, changed, pv, self.protocol, self.pins)

    def test_future_suffix_leak_is_refused(self):
        raw, pv = self.candidate("smollm2")
        rec = json.loads(raw)
        rec["checks"]["nonzero_nca_future_suffix_invariance"]["max_abs"] = 0.4
        new = json.dumps(rec, indent=2, sort_keys=True).encode()
        pv = copy.deepcopy(pv)
        pv["source_run"]["original_report_sha256"] = sha(new)
        pv["source_run"]["report_bytes"] = len(new)
        with self.assertRaisesRegex(RuntimeError, "causal future suffix"):
            audit_lane("smollm2", new, pv, self.protocol, self.pins)

    def test_void_byte_remote_execution_cannot_be_relabelled_pass(self):
        raw, pv = self.candidate("void1")
        rec = json.loads(raw)
        rec["custom_tokenizer_code_executed"] = True
        new = json.dumps(rec, indent=2, sort_keys=True).encode()
        pv = copy.deepcopy(pv)
        pv["source_run"]["original_report_sha256"] = sha(new)
        pv["source_run"]["original_report_bytes"] = len(new)
        with self.assertRaisesRegex(RuntimeError, "tokenizer execution"):
            audit_lane("void1", new, pv, self.protocol, self.pins)

    def test_no_training_policy_never_overridden_by_positive_receipts(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["train_authorized"] = True
        raw, pv = self.candidate("smollm2")
        with self.assertRaisesRegex(RuntimeError, "Training admission flag"):
            audit_lane("smollm2", raw, pv, protocol, self.pins)


if __name__ == "__main__":
    unittest.main()
