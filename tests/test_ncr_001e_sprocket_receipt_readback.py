"""Independent stdlib negative sentinels for byte-exact two-process Sprocket evidence."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ncr_001e_sprocket_process_receipt_audit import (
    ARCHIVE, READBACK_STATUS, ORIGINALS, audit, unpack_original_jsons,
)


class OriginalSprocketReceiptReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = unpack_original_jsons(ARCHIVE.read_text(encoding="ascii"))

    def test_original_receipt_bytes_and_two_source_variants_are_exact(self):
        report = audit()
        self.assertEqual(report["status"], READBACK_STATUS)
        self.assertEqual(report["original_json_count"], 8)
        self.assertEqual(set(report["original_variants"]), {"chat", "instruct"})
        self.assertIs(report["train_authorized"], False)
        self.assertIs(report["four_model_conformance_pass"], False)
        self.assertIs(report["tool_and_memory_behavior_qualified"], False)
        self.assertNotEqual(
            report["original_variants"]["chat"]["actual_weight_sha256"],
            report["original_variants"]["instruct"]["actual_weight_sha256"],
        )

    def test_corrupted_original_compressed_archive_refused(self):
        text = ARCHIVE.read_text(encoding="ascii").strip()
        fake = ("B" if text[0] != "B" else "C") + text[1:]
        with self.assertRaises((RuntimeError, ValueError, OSError)):
            unpack_original_jsons(fake)

    def test_wrong_original_json_member_set_refused(self):
        original = deepcopy(self.originals)
        original.pop(next(iter(ORIGINALS)))
        with self.assertRaises(RuntimeError):
            audit(original)

    def test_noncausal_or_inaccurate_cache_receipt_refused(self):
        for key, value in (
            ("max_abs", 0.0006), ("pass", False),
            ("atol", 0.005), ("rtol", 0.005),
        ):
            with self.subTest(key=key):
                altered = deepcopy(self.originals)
                altered["chat/producer.json"]["checks"][
                    "256_unit_240_prefill_16_incremental_tail"][key] = value
                with self.assertRaises(RuntimeError):
                    audit(altered)

    def test_future_suffix_drift_is_not_waived(self):
        altered = deepcopy(self.originals)
        altered["instruct/producer.json"]["checks"][
            "future_suffix_strict_255_units"]["max_abs"] = 0.00001
        with self.assertRaises(RuntimeError):
            audit(altered)

    def test_chat_root_instruct_checkpoint_substitution_refused(self):
        altered = deepcopy(self.originals)
        altered["chat/manifest.json"]["checkpoint_sha256"] = self.originals[
            "instruct/manifest.json"]["checkpoint_sha256"]
        with self.assertRaises(RuntimeError):
            audit(altered)

    def test_same_pid_or_fake_cross_process_replay_refused(self):
        altered = deepcopy(self.originals)
        altered["chat/consumer.json"]["process_pid"] = altered["chat/producer.json"]["process_pid"]
        with self.assertRaises(RuntimeError):
            audit(altered)
        altered = deepcopy(self.originals)
        altered["instruct/consumer.json"]["checks"][
            "restart_after_B_no_leak_exact"]["pass"] = False
        with self.assertRaises(RuntimeError):
            audit(altered)

    def test_false_training_or_tool_qualification_refused(self):
        for path, key, value in (
            ("chat/producer.json", "train_authorized", True),
            ("chat/consumer.json", "optimizer_steps", 1),
            ("instruct/consumer.json", "tool_and_memory_behavior_qualified", True),
            ("instruct/manifest.json", "heldout_evaluation", True),
        ):
            with self.subTest(path=path, key=key):
                altered = deepcopy(self.originals)
                altered[path][key] = value
                with self.assertRaises(RuntimeError):
                    audit(altered)

    def test_model_role_tokenizer_and_source_commit_refused(self):
        for path, key, value in (
            ("instruct/producer.json", "token_ids_sha256", "0" * 64),
            ("chat/manifest.json", "implementation_commit", "0" * 40),
            ("chat/adapter/metadata.json", "post_block_taps", [1, 3, 5]),
        ):
            with self.subTest(path=path, key=key):
                altered = deepcopy(self.originals)
                altered[path][key] = value
                with self.assertRaises(RuntimeError):
                    audit(altered)


if __name__ == "__main__":
    unittest.main()
