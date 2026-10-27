"""Independent original Actions receipt byte/causality and overclaim sentinels.

These are archival tests, not additional model runtime or training.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001e_sprocket_publisher_receipt_audit import (
    load_archive, verify_receipt, ORIGINAL_JSON_SHA
)

class OriginalWireReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.provenance = load_archive()

    def test_positive_original_source_replay(self):
        verify_receipt(copy.deepcopy(self.original), copy.deepcopy(self.provenance))

    def test_swapped_chat_checkpoint_or_variant_refused(self):
        for field, value in (("variant", "chat"), ("checkpoint_sha256", "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97")):
            j=copy.deepcopy(self.original); j[field]=value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_receipt(j, self.provenance)

    def test_semantic_tool_memory_or_training_claim_refused(self):
        for field, value in (
            ("tool_and_memory_semantics_qualified", True),
            ("model_generated_tool_result_memory_semantics_test", True),
            ("final_four_model_conformance_pass", True),
            ("heldout_evaluation", True),
            ("actual_model_training", True),
            ("optimizer_steps", 1),
            ("train_authorized", True),
        ):
            j=copy.deepcopy(self.original); j[field]=value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_receipt(j, self.provenance)

    def test_publisher_source_drift_not_legitimized_by_checksum(self):
        for section, field, value in (
            ("publisher", "commit", "bad"),
            ("publisher", "sft_data_git_blob", "bad"),
            ("publisher", "is_crypto_training_build_attestation", True),
            ("model", "original_hook_commit", "bad"),
            ("model", "weight_sha256", "bad"),
            ("limits", "final_independent_72_accepted", True),
            ("original_actions_artifact", "original_json_sha256", "0"*64),
        ):
            p=copy.deepcopy(self.provenance); p[section][field]=value
            with self.subTest(section=section,field=field), self.assertRaises(RuntimeError):
                verify_receipt(self.original,p)

    def test_stealth_mutated_reference_metrics_refused(self):
        for name in (
            "source_wire_tool_call_prefill_incremental",
            "source_wire_tool_result_prefill_incremental",
            "source_wire_memory_write_prefill_incremental",
        ):
            j=copy.deepcopy(self.original);j["checks"][name]["max_abs"]=0.001
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                verify_receipt(j,self.provenance)
            j=copy.deepcopy(self.original);j["checks"][name]["rtol"]=0.01
            with self.subTest(name=name+" rtol"), self.assertRaises(RuntimeError):
                verify_receipt(j,self.provenance)
        for name in (
            "exact_frozen_source_vs_zero_NCA",
            "source_wire_strict_future_suffix",
            "source_wire_independent_request_reset",
        ):
            j=copy.deepcopy(self.original);j["checks"][name]["max_abs"]=0.0001
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                verify_receipt(j,self.provenance)

    def test_tool_result_host_and_memory_dual_semantics_refused(self):
        for i,b in enumerate(self.original["wire"]["role_boundaries"]):
            if b["id"] in (7,10):
                j=copy.deepcopy(self.original)
                j["wire"]["role_boundaries"][i]["trained"]=not b["trained"]
                with self.subTest(i=i), self.assertRaises(RuntimeError):
                    verify_receipt(j,self.provenance)

    def test_original_runner_or_run_hash_refused(self):
        for section, field in (("runner","github_run_id"),("runner","pid")):
            j=copy.deepcopy(self.original);j[section][field]="corrupt"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_receipt(j,self.provenance)

    def test_base64_container_mutation_rejected(self):
        from tempfile import TemporaryDirectory
        from ncr_001e_sprocket_publisher_receipt_audit import ARCHIVE
        with TemporaryDirectory() as path:
            target=Path(path)/"tampered.txt"
            blob=ARCHIVE.read_text()
            target.write_text(("A" if blob[0]!="A" else "B")+blob[1:])
            with self.assertRaises((ValueError,RuntimeError)):
                load_archive(target)
            target.write_text("base64-not-valid")
            with self.assertRaises((ValueError,RuntimeError)):
                load_archive(target)

if __name__ == "__main__":
    unittest.main()
