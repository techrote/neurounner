"""Fail-closed original Actions source-wire diagnostic archive readback.

No model invocation, network, fixture inspection, or optimizer.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ncr_001e_sprocket_wire_archive import (  # noqa: E402
    ARCHIVE, EXPECTED_CASES, EXHAUSTED, FOLDER, MANIFEST, load_originals,
    verify_dataset_hash_inventory, verify_original,
)


class OriginalDiagnosticArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members, cls.manifest = load_originals()

    def test_exact_positive_original_receipts(self):
        verify_original(copy.deepcopy(self.members), copy.deepcopy(self.manifest))

    def test_dataset_hash_inventory_and_tampered_durable_file_rejected(self):
        verify_dataset_hash_inventory()
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "archived"
            shutil.copytree(FOLDER, dest)
            (dest / "README.md").write_text("tampered dataset claim", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_dataset_hash_inventory(dest)

    def test_corrupt_archived_bytes_fail_before_decoding(self):
        original = ARCHIVE.read_text(encoding="ascii")
        with TemporaryDirectory() as temp:
            bad = Path(temp) / "bad.b64"
            bad.write_text(("A" if original[0] != "A" else "B") + original[1:])
            with self.assertRaises((ValueError, RuntimeError)):
                load_originals(bad)
            bad.write_text("%%%base64-corrupted%%%")
            with self.assertRaises((ValueError, RuntimeError)):
                load_originals(bad)

    def test_corrupt_member_manifest_and_original_zip_refused(self):
        for field, replacement in (
            ("original_zip_sha256", "0" * 64),
            ("original_zip_archived_in_git", True),
            ("original_zip_bytes", 123),
            ("artifact_id", 0),
        ):
            p = copy.deepcopy(self.manifest)
            p["original_actions"][field] = replacement
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_original(self.members, p)
        for field, value in (
            ("original_sha256", "f" * 64),
            ("original_bytes_archived_in_git", True),
            ("source_git_commit", "f" * 40),
        ):
            p = copy.deepcopy(self.manifest)
            p["original_binary_source_snapshot"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_original(self.members, p)
        p = copy.deepcopy(self.manifest)
        p["archived_text_members"]["members"][0]["sha256"] = "f" * 64
        with self.assertRaises(RuntimeError):
            load_originals(ARCHIVE, _temp_manifest(p))

    def test_model_swap_source_change_and_training_claim_refused(self):
        for section, field, value in (
            ("source", "model_key", "sprocket_chat"),
            ("source", "root_checkpoint", "chat/model.safetensors"),
            ("source", "root_checkpoint_sha256", "0" * 64),
            ("source", "original_hook_commit", "0" * 40),
            ("source", "publisher_commit", "0" * 40),
            ("source", "publisher_source_not_training_build_attested", False),
            ("original_results", "train_authorized", True),
            ("original_results", "optimizer_steps", 1),
            ("original_results", "tool_memory_semantics_qualified", True),
            ("original_results", "final_four_model_conformance", True),
        ):
            p = copy.deepcopy(self.manifest)
            p[section][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_original(self.members, p)

    def test_false_green_cache_and_source_oracle_mutations_refused(self):
        for name, field, value in (
            ("all_rich_positions_full_vs_incremental", "max_abs", 0.001),
            ("all_rich_positions_full_vs_incremental", "atol", 0.01),
            ("all_rich_positions_full_vs_incremental", "pass", False),
            ("direct_frozen_vs_zero", "max_abs", 0.0001),
            ("request_after_all_generation", "exact_required", False),
        ):
            obs = _original(self.members)
            obs["checks"][name][field] = value
            with self.subTest(check=name, field=field), self.assertRaises(RuntimeError):
                verify_original(_replace_receipt(self.members, obs), self.manifest)
        obs = _original(self.members)
        obs["source_oracle"]["rich_mask_true"] = 0
        with self.assertRaises(RuntimeError):
            verify_original(_replace_receipt(self.members, obs), self.manifest)

    def test_all_21_original_raw_negative_observations_required(self):
        receipt = _original(self.members)
        self.assertEqual(len(receipt["observations"]), 21)
        self.assertEqual(
            {o["case_id"] for o in receipt["observations"] if o["finish_reason"] == "budget_exhausted"},
            EXHAUSTED)
        for change in ("drop", "fake_success", "lost_stop", "wrong_mode", "wrong_case"):
            obs = copy.deepcopy(receipt)
            if change == "drop":
                obs["observations"].pop()
            elif change == "fake_success":
                obs["observations"][0]["observed"]["strict_tool_envelope"] = True
            elif change == "lost_stop":
                obs["observations"][0]["finish_reason"] = "turn_end"
            elif change == "wrong_mode":
                obs["observations"][0]["mode"] = "active_untrained"
            else:
                obs["observations"][0]["case_id"] = "invented-test"
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                verify_original(_replace_receipt(self.members, obs), self.manifest)

    def test_counterfactual_literal_failures_not_relabelled_success(self):
        for case in ("tool-result-42", "tool-result-17",
                     "memory-context-amber-17", "memory-context-violet-28"):
            orig = _original(self.members)
            j = next(x for x in orig["observations"] if x["case_id"] == case)
            j["observed"]["expected_literal_present"] = not j["observed"]["expected_literal_present"]
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                verify_original(_replace_receipt(self.members, orig), self.manifest)

    def test_historical_four_rejection_sentinels_immutable(self):
        obs = _original(self.members)
        for case in obs["legacy_regressions"]["cases"]:
            m = copy.deepcopy(obs)
            chosen = next(c for c in m["legacy_regressions"]["cases"] if c["id"] == case["id"])
            chosen["repaired_rejected"] = False
            with self.subTest(case=case["id"]), self.assertRaises(RuntimeError):
                verify_original(_replace_receipt(self.members, m), self.manifest)

    def test_heldout_and_protocol_still_refused(self):
        obs = _original(self.members)
        for k in ("heldout_evaluation", "train_authorized", "final_72_admission",
                  "tool_memory_semantics_qualified", "actual_model_training"):
            mutated = copy.deepcopy(obs)
            mutated[k] = True
            with self.subTest(field=k), self.assertRaises(RuntimeError):
                verify_original(_replace_receipt(self.members, mutated), self.manifest)
        obs["optimizer_steps"] = 1
        with self.assertRaises(RuntimeError):
            verify_original(_replace_receipt(self.members, obs), self.manifest)


def _original(members):
    import json
    return json.loads(members["receipt.json"])


def _replace_receipt(members, data):
    import json
    result = dict(members)
    result["receipt.json"] = json.dumps(data, sort_keys=True).encode("utf-8")
    return result


def _temp_manifest(manifest):
    import json
    import tempfile
    # The tempfile cleanup is managed at interpreter exit; minimal bounded test.
    path = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(manifest, path)
    path.close()
    return path.name


if __name__ == "__main__":
    unittest.main()
